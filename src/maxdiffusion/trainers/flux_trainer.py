"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os
from functools import partial
import datetime
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PositionalSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning
from maxdiffusion.checkpointing.flux_checkpointer import (
    FluxCheckpointer,
    FLUX_CHECKPOINT,
    FLUX_STATE_KEY,
    FLUX_STATE_SHARDINGS_KEY,
    VAE_STATE_KEY,
    VAE_STATE_SHARDINGS_KEY,
)

from maxdiffusion.input_pipeline.input_pipeline_interface import (make_data_iterator)

from maxdiffusion import (max_utils, max_logging)

from maxdiffusion.train_utils import (
    get_first_step,
    load_next_batch,
    record_scalar_metrics,
    write_metrics,
)

from maxdiffusion.maxdiffusion_utils import calculate_flux_tflops

from ..schedulers import (FlaxEulerDiscreteScheduler)


class FluxTrainer(FluxCheckpointer):

  def __init__(self, config):
    FluxCheckpointer.__init__(self, config, FLUX_CHECKPOINT)

    self.text_encoder_2_learning_rate_scheduler = None

    if config.train_text_encoder:
      raise ValueError("this script currently doesn't support training text_encoders")

  def post_training_steps(self, pipeline, params, train_states, msg=""):
    pass

  def create_scheduler(self, pipeline, params):
    noise_scheduler, noise_scheduler_state = FlaxEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=self.config.pretrained_model_name_or_path, subfolder="scheduler", dtype=jnp.float32
    )
    noise_scheduler_state = noise_scheduler.set_timesteps(
        state=noise_scheduler_state, num_inference_steps=self.config.num_inference_steps, timestep_spacing="flux"
    )
    return noise_scheduler, noise_scheduler_state

  def calculate_tflops(self, pipeline):
    per_device_tflops = calculate_flux_tflops(self.config, pipeline, self.total_train_batch_size, self.rng, train=True)
    max_logging.log(f"JFLUX per device TFLOPS: {per_device_tflops}")
    return per_device_tflops

  def start_training(self):

    # Hook
    # self.pre_training_steps()
    # Load checkpoint - will load or create states
    pipeline, params = self.load_checkpoint()

    # create train states
    train_states = {}
    state_shardings = {}

    # move params to accelerator
    encoders_sharding = PositionalSharding(self.devices_array).replicate()
    partial_device_put_replicated = partial(max_utils.device_put_replicated, sharding=encoders_sharding)
    pipeline.clip_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), pipeline.clip_encoder.params)
    pipeline.clip_encoder.params = jax.tree_util.tree_map(partial_device_put_replicated, pipeline.clip_encoder.params)
    pipeline.t5_encoder.params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), pipeline.t5_encoder.params)
    pipeline.t5_encoder.params = jax.tree_util.tree_map(partial_device_put_replicated, pipeline.t5_encoder.params)

    vae_state, vae_state_mesh_shardings = self.create_vae_state(
        pipeline=pipeline, params=params, checkpoint_item_name=VAE_STATE_KEY, is_training=False
    )
    train_states[VAE_STATE_KEY] = vae_state
    state_shardings[VAE_STATE_SHARDINGS_KEY] = vae_state_mesh_shardings

    # Load dataset
    data_iterator = self.load_dataset(pipeline, params, train_states)
    if self.config.dataset_type == "grain":
      data_iterator = self.restore_data_iterator_state(data_iterator)

    # don't need this anymore, clear some memory.
    del pipeline.t5_encoder

    # evaluate shapes

    flux_state, flux_state_mesh_shardings, flux_learning_rate_scheduler = self.create_flux_state(
        # ambiguous here, but if params=None
        # Then its 1 of 2 scenarios:
        # 1. flux state will be loaded directly from orbax
        # 2. a new flux is being trained from scratch.
        pipeline=pipeline,
        params=None,  # Params are loaded inside create_flux_state
        checkpoint_item_name=FLUX_STATE_KEY,
        is_training=True,
    )
    flux_state = jax.device_put(flux_state, flux_state_mesh_shardings)
    train_states[FLUX_STATE_KEY] = flux_state
    state_shardings[FLUX_STATE_SHARDINGS_KEY] = flux_state_mesh_shardings
    # self.post_training_steps(pipeline, params, train_states, msg="before_training")

    # Create scheduler
    noise_scheduler, noise_scheduler_state = self.create_scheduler(pipeline, params)
    pipeline.scheduler = noise_scheduler
    train_states["scheduler"] = noise_scheduler_state

    # Calculate tflops
    per_device_tflops = self.calculate_tflops(pipeline)
    self.per_device_tflops = per_device_tflops

    data_shardings = self.get_data_shardings()
    # Compile train_step
    p_train_step = self.compile_train_step(pipeline, params, train_states, state_shardings, data_shardings)
    # Start training
    train_states = self.training_loop(
        p_train_step, pipeline, params, train_states, data_iterator, flux_learning_rate_scheduler
    )
    # 6. save final checkpoint
    # Hook
    self.post_training_steps(pipeline, params, train_states, "after_training")

  def get_shaped_batch(self, config, pipeline=None):
    """Return the shape of the batch - this is what eval_shape would return for the
    output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078.
    """

    scale_factor = 16  # hardcoded in jflux.get_noise
    h = config.resolution // scale_factor
    w = config.resolution // scale_factor
    c = 16
    ph = pw = 2
    batch_image_shape = (self.total_train_batch_size, h * w, c * ph * pw)  # b
    img_ids_shape = (self.total_train_batch_size, (2 * h // 2) * (2 * w // 2), 3)
    text_shape = (
        self.total_train_batch_size,
        config.max_sequence_length,
        4096,  # Sequence length of text encoder, how to get this programmatically?
    )
    text_ids_shape = (
        self.total_train_batch_size,
        config.max_sequence_length,
        3,
    )
    prompt_embeds_shape = (
        self.total_train_batch_size,
        768,  # Sequence length of clip, how to get this programmatically?
    )
    input_ids_dtype = self.config.activations_dtype

    shaped_batch = {}
    shaped_batch["pixel_values"] = jax.ShapeDtypeStruct(batch_image_shape, input_ids_dtype)
    shaped_batch["text_embeds"] = jax.ShapeDtypeStruct(text_shape, input_ids_dtype)
    shaped_batch["input_ids"] = jax.ShapeDtypeStruct(text_ids_shape, input_ids_dtype)
    shaped_batch["prompt_embeds"] = jax.ShapeDtypeStruct(prompt_embeds_shape, input_ids_dtype)
    shaped_batch["img_ids"] = jax.ShapeDtypeStruct(img_ids_shape, input_ids_dtype)
    return shaped_batch

  def get_data_shardings(self):
    data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))
    data_sharding = {
        "text_embeds": data_sharding,
        "input_ids": data_sharding,
        "prompt_embeds": data_sharding,
        "pixel_values": data_sharding,
        "img_ids": data_sharding,
    }

    return data_sharding

  # adapted from max_utils.tokenize_captions_xl
  @staticmethod
  def tokenize_captions(examples, caption_column, encoder):
    prompt = list(examples[caption_column])

    prompt_embeds, pooled_prompt_embeds, text_ids = encoder(prompt, prompt)

    examples["text_embeds"] = jnp.float16(prompt_embeds)
    examples["input_ids"] = jnp.float16(text_ids)
    examples["prompt_embeds"] = jnp.float16(pooled_prompt_embeds)

    return examples

  @staticmethod
  def transform_images(examples, image_column, image_resolution, vae_encode, pack_latents, prepare_latent_imgage_ids):
    """Preprocess images to latents."""
    images = list(examples[image_column])

    images = [
        jax.image.resize(
            jnp.asarray(image) / 127.5 - 1.0, [image_resolution, image_resolution, 3], method="bilinear", antialias=True
        )
        for image in images
    ]

    images = jnp.stack(images, axis=0, dtype=jnp.float16)
    batch_size = 8
    num_batches = len(images) // batch_size + int(len(images) % batch_size != 0)
    encoded_images = []
    for i in range(num_batches):
      batch_images = images[i * batch_size : (i + 1) * batch_size]
      batch_images = jnp.transpose(batch_images, (0, 3, 1, 2))
      batch_images = vae_encode(batch_images)
      batch_images = jnp.transpose(batch_images, (0, 3, 1, 2))
      encoded_images.append(batch_images)

    images = jnp.concatenate(encoded_images, axis=0, dtype=jnp.float16)
    b, c, h, w = images.shape
    images = pack_latents(latents=images, batch_size=b, num_channels_latents=c, height=h, width=w)

    img_ids = prepare_latent_imgage_ids(h // 2, w // 2)
    img_ids = jnp.tile(img_ids, (b, 1, 1))

    examples["pixel_values"] = jnp.float16(images)
    examples["img_ids"] = jnp.float16(img_ids)

    return examples

  def load_dataset(self, pipeline, params, train_states):
    config = self.config
    total_train_batch_size = self.total_train_batch_size
    mesh = self.mesh

    encode_fn = partial(
        pipeline.encode_prompt,
        clip_tokenizer=pipeline.clip_tokenizer,
        t5_tokenizer=pipeline.t5_tokenizer,
        clip_text_encoder=pipeline.clip_encoder,
        t5_text_encoder=pipeline.t5_encoder,
        encode_in_batches=True,
        encode_batch_size=16,
    )
    pack_latents_p = partial(pipeline.pack_latents)
    prepare_latent_image_ids_p = partial(pipeline.prepare_latent_image_ids)
    vae_encode_p = partial(pipeline.vae_encode, vae=pipeline.vae, state=train_states["vae_state"])

    tokenize_fn = partial(FluxTrainer.tokenize_captions, caption_column=config.caption_column, encoder=encode_fn)
    image_transforms_fn = partial(
        FluxTrainer.transform_images,
        image_column=config.image_column,
        image_resolution=config.resolution,
        vae_encode=vae_encode_p,
        pack_latents=pack_latents_p,
        prepare_latent_imgage_ids=prepare_latent_image_ids_p,
    )

    data_iterator = make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        total_train_batch_size,
        tokenize_fn=tokenize_fn,
        image_transforms_fn=image_transforms_fn,
    )

    return data_iterator

  def compile_train_step(self, pipeline, params, train_states, state_shardings, data_shardings):
    self.rng, train_rngs = jax.random.split(self.rng)
    guidance_vec = jnp.full((self.total_train_batch_size,), self.config.guidance_scale, dtype=self.config.activations_dtype)
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      p_train_step = jax.jit(
          partial(
              _train_step,
              guidance_vec=guidance_vec,
              pipeline=pipeline,
              scheduler=train_states["scheduler"],
              config=self.config,
          ),
          in_shardings=(
              state_shardings["flux_state_shardings"],
              data_shardings,
              None,
          ),
          out_shardings=(state_shardings["flux_state_shardings"], None, None),
          donate_argnums=(0,),
      )
      max_logging.log("Precompiling...")
      s = time.time()
      dummy_batch = self.get_shaped_batch(self.config, pipeline)
      p_train_step = p_train_step.lower(train_states[FLUX_STATE_KEY], dummy_batch, train_rngs)
      p_train_step = p_train_step.compile()
      max_logging.log(f"Compile time: {(time.time() - s )}")
      return p_train_step

  def training_loop(self, p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler):

    writer = max_utils.initialize_summary_writer(self.config)
    flux_state = train_states[FLUX_STATE_KEY]
    num_model_parameters = max_utils.calculate_num_params_from_pytree(flux_state.params)

    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ.get("LIBTPU_INIT_ARGS", ""), writer)
    max_utils.add_config_to_summary_writer(self.config, writer)

    if jax.process_index() == 0:
      max_logging.log("***** Running training *****")
      max_logging.log(f"  Instantaneous batch size per device = {self.config.per_device_batch_size}")
      max_logging.log(f"  Total train batch size (w. parallel & distributed) = {self.total_train_batch_size}")
      max_logging.log(f"  Total optimization steps = {self.config.max_train_steps}")

    last_step_completion = datetime.datetime.now()
    local_metrics_file = open(self.config.metrics_file, "a", encoding="utf8") if self.config.metrics_file else None
    running_gcs_metrics = [] if self.config.gcs_metrics else None
    example_batch = None

    first_profiling_step = self.config.skip_first_n_steps_for_profiler
    if self.config.enable_profiler and first_profiling_step >= self.config.max_train_steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
    last_profiling_step = np.clip(
        first_profiling_step + self.config.profiler_steps - 1, first_profiling_step, self.config.max_train_steps - 1
    )
    start_step = get_first_step(train_states[FLUX_STATE_KEY])
    _, train_rngs = jax.random.split(self.rng)
    times = []
    for step in np.arange(start_step, self.config.max_train_steps):
      if self.config.enable_profiler and step == first_profiling_step:
        max_utils.activate_profiler(self.config)

      example_batch = load_next_batch(data_iterator, example_batch, self.config)
      example_batch = {key: jnp.asarray(value, dtype=self.config.activations_dtype) for key, value in example_batch.items()}

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        with self.mesh:
          flux_state, train_metric, train_rngs = p_train_step(flux_state, example_batch, train_rngs)

      samples_count = self.total_train_batch_size * (step + 1)
      new_time = datetime.datetime.now()

      record_scalar_metrics(
          train_metric, new_time - last_step_completion, self.per_device_tflops, unet_learning_rate_scheduler(step)
      )
      if self.config.write_metrics:
        write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config)
      times.append(new_time - last_step_completion)
      last_step_completion = new_time

      if step != 0 and self.config.checkpoint_every != -1 and samples_count % self.config.checkpoint_every == 0:
        max_logging.log(f"Saving checkpoint for step {step}")
        train_states[FLUX_STATE_KEY] = flux_state
        self.save_checkpoint(step, pipeline, train_states)

      if self.config.enable_profiler and step == last_profiling_step:
        max_utils.deactivate_profiler(self.config)

    train_states[FLUX_STATE_KEY] = flux_state
    if len(times) > 0:
      max_logging.log(f"Average time per step: {sum(times[2:], datetime.timedelta(0)) / len(times[2:])}")
    if self.config.save_final_checkpoint:
      max_logging.log(f"Saving checkpoint for step {step}")
      self.save_checkpoint(step, pipeline, train_states)
      self.checkpoint_manager.wait_until_finished()
    return train_states


def _train_step(flux_state, batch, train_rng, guidance_vec, pipeline, scheduler, config):
  _, gen_dummy_rng = jax.random.split(train_rng)
  sample_rng, timestep_bias_rng, new_train_rng = jax.random.split(gen_dummy_rng, 3)
  state_params = {FLUX_STATE_KEY: flux_state.params}

  def compute_loss(state_params):
    latents = batch["pixel_values"]
    text_embeds_ids = batch["input_ids"]
    text_embeds = batch["text_embeds"]
    prompt_embeds = batch["prompt_embeds"]
    img_ids = batch["img_ids"]

    # Sample noise that we'll add to the latents
    noise_rng, timestep_rng = jax.random.split(sample_rng)
    noise = jax.random.normal(
        key=noise_rng,
        shape=latents.shape,
        dtype=latents.dtype,
    )
    # Sample a random timestep for each image
    bsz = latents.shape[0]
    timesteps = jax.random.randint(timestep_rng, shape=(bsz,), minval=0, maxval=len(scheduler.timesteps) - 1)
    noisy_latents = pipeline.scheduler.add_noise(scheduler, latents, noise, timesteps, flux=True)

    model_pred = pipeline.flux.apply(
        {"params": state_params[FLUX_STATE_KEY]},
        hidden_states=noisy_latents,
        img_ids=img_ids,
        encoder_hidden_states=text_embeds,
        txt_ids=text_embeds_ids,
        timestep=scheduler.timesteps[timesteps],
        guidance=guidance_vec,
        pooled_projections=prompt_embeds,
    ).sample

    target = noise - latents
    loss = (target - model_pred) ** 2

    loss = jnp.mean(loss)

    return loss

  grad_fn = jax.value_and_grad(compute_loss)
  loss, grad = grad_fn(state_params)

  new_state = flux_state.apply_gradients(grads=grad[FLUX_STATE_KEY])

  metrics = {"scalar": {"learning/loss": loss}, "scalars": {}}

  return new_state, metrics, new_train_rng
