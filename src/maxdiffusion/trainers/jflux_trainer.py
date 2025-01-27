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

import jax
import jax.numpy as jnp
import datetime
import time
import flax.linen as nn
from functools import partial
from jax.sharding import PartitionSpec as P
from flax.linen import partitioning as nn_partitioning
import optax
from einops import repeat, rearrange

from maxdiffusion import (FlaxEulerDiscreteScheduler, maxdiffusion_utils, train_utils, max_utils, max_logging)
from maxdiffusion.input_pipeline.input_pipeline_interface import (make_data_iterator)

from maxdiffusion.checkpointing.jflux_checkpointer import (JfluxCheckpointer)


# pulls in code from BaseStableDiffusionTrainer and StableDiffusionTrainer
class JFluxTrainer(JfluxCheckpointer):

  def __init__(self, config):
    JfluxCheckpointer.__init__(self, config)

    # sharding
    self.data_sharding = None

    self.per_device_tflops = None

    self.writer = max_utils.initialize_summary_writer(config)

    self.p_train_step = None

  def pre_training_steps(self):
    pass

  def post_training_steps(self, pipeline, state):
    if self.config.run_inference_after_training:
      import os
      from glob import iglob
      import re
      import numpy as np
      from PIL import Image

      seed = jax.random.PRNGKey(seed=102333)
      with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        state = jax.device_put(state, pipeline.data_sharding)
        img = pipeline.create_noise(
            len(jax.devices()), self.config.resolution, self.config.resolution, self.config.activations_dtype, seed
        )
        (txt, txt_ids, vec, img) = pipeline.prepare_inputs([self.config.prompt for _ in range(len(jax.devices()))], img)

        def do_inference():
          return pipeline(
              state,
              txt,
              txt_ids,
              vec,
              self.config.num_inference_steps,
              self.config.resolution,
              self.config.resolution,
              self.config.guidance_scale,
              img,
              shift=self.config.model_name != "flux-schnell",
          )

      max_logging.log("Inference")
      t0 = time.perf_counter()
      x = do_inference()
      t1 = time.perf_counter()
      output_dir = "output"
      output_name = os.path.join(output_dir, "maxdiff_img_{idx}.jpg")
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
      else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"maxdiff_img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
          idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
          idx = 0
      fn = output_name.format(idx=idx)
      max_logging.log(f"Done in {t1 - t0:.1f}s. Saving {fn}")
      # bring into PIL format and save
      x = x.clip(-1, 1)
      x = rearrange(x[0], "c h w -> h w c")

      x = 127.5 * (x + 1.0)
      x_numpy = np.array(x.astype(jnp.uint8))
      img = Image.fromarray(x_numpy)

      img.save(fn, quality=95, subsampling=0)

  def calculate_tflops(self, pipeline):
    per_device_tflops = maxdiffusion_utils.calculate_flux_tflops(
        self.config, pipeline, self.total_train_batch_size, self.rng, train=True
    )
    max_logging.log(f"JFLUX per device TFLOPS: {per_device_tflops}")
    return per_device_tflops

  def start_training(self):
    # Hook
    self.pre_training_steps()
    # Load checkpoint - will load or create states
    pipeline = self.load_checkpoint()
    # create train states
    train_states = {}
    state_shardings = {}
    flux_state, flux_state_mesh_shardings, flux_learning_rate_scheduler = self.create_flux_state(
        # ambiguous here, but if self.params.get("flux") doesn't exist
        # Then its 1 of 2 scenarios:
        # 1. flux state will be loaded directly from orbax
        # 2. a new flux is being trained from scratch.
        flux=pipeline.flux,
        init_flux_weights=pipeline.init_flux_weights,
        params=None,
        is_training=True,
    )
    train_states[JfluxCheckpointer.flux_state_item_name] = flux_state
    state_shardings["flux_state_shardings"] = flux_state_mesh_shardings

    # Create scheduler
    max_logging.log("Creating scheduler")
    noise_scheduler, noise_scheduler_state = self.create_scheduler(pipeline, None)
    pipeline.scheduler = noise_scheduler
    train_states["scheduler"] = noise_scheduler_state

    # Calculate tflops
    per_device_tflops = self.calculate_tflops(pipeline)
    self.per_device_tflops = per_device_tflops

    # Load dataset
    max_logging.log("Loading data set")
    data_iterator = self.load_dataset(pipeline)
    max_logging.log("Data set loaded")

    data_shardings = self.get_data_shardings()
    max_logging.log("Data sharding created")
    # Compile train_step
    p_train_step = self.compile_train_step(pipeline, train_states, state_shardings, data_shardings)
    max_logging.log("Train loop compiled")
    # Start training
    max_logging.log("Starting training loop")
    train_states = self.training_loop(p_train_step, pipeline, train_states, data_iterator, flux_learning_rate_scheduler)
    max_logging.log("End training loop")
    # 6. save final checkpoint
    # Hook
    self.post_training_steps(pipeline, train_states[JfluxCheckpointer.flux_state_item_name].params)

  def get_shaped_batch(self, config, pipeline):
    """Return the shape of the batch - this is what eval_shape would return for the
    output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078.
    """

    if config.dataset_type == "tf" and config.cache_latents_text_encoder_outputs:
      scale_factor = 16  # hardcoded in jflux.get_noise
      h = config.resolution // scale_factor
      w = config.resolution // scale_factor
      c = 16
      ph = pw = 2
      batch_image_shape = (self.total_train_batch_size, h * w, c * ph * pw)  # b
      img_ids_shape = (self.total_train_batch_size, (2 * h // 2) * (2 * w // 2), 3)
      text_shape = (
          self.total_train_batch_size,
          256 if config.model_name == "flux-schnell" else 512,
          4096,  # Sequence length of text encoder, how to get this programmatically?
      )
      text_ids_shape = (
          self.total_train_batch_size,
          256 if config.model_name == "flux-schnell" else 512,
          3,
      )
      prompt_embeds_shape = (
          self.total_train_batch_size,
          768,  # Sequence length of clip, how to get this programmatically?
      )
      input_ids_dtype = self.config.activations_dtype
    else:
      batch_image_shape = (self.total_train_batch_size, 3, config.resolution, config.resolution)
      text_shape = (
          self.total_train_batch_size,
          pipeline.t5.max_length,
      )
      text_ids_shape = (
          self.total_train_batch_size,
          pipeline.t5.max_length,
      )
      prompt_embeds_shape = (
          self.total_train_batch_size,
          pipeline.clip.max_length,
      )
      input_ids_dtype = self.config.activations_dtype

    shaped_batch = {}
    shaped_batch["pixel_values"] = jax.ShapeDtypeStruct(batch_image_shape, input_ids_dtype)
    shaped_batch["text_embeds"] = jax.ShapeDtypeStruct(text_shape, input_ids_dtype)
    shaped_batch["input_ids"] = jax.ShapeDtypeStruct(text_ids_shape, input_ids_dtype)
    shaped_batch["prompt_embeds"] = jax.ShapeDtypeStruct(prompt_embeds_shape, input_ids_dtype)
    shaped_batch["img_ids"] = jax.ShapeDtypeStruct(img_ids_shape, input_ids_dtype)
    return shaped_batch

  def create_scheduler(self, pipeline, params):
    noise_scheduler, noise_scheduler_state = FlaxEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=self.config.pretrained_model_name_or_path, subfolder="scheduler", dtype=jnp.float32
    )
    noise_scheduler_state = noise_scheduler.set_timesteps(
        state=noise_scheduler_state, num_inference_steps=self.config.num_inference_steps, timestep_spacing="flux"
    )
    return noise_scheduler, noise_scheduler_state

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
  def tokenize_captions(examples, caption_column, tokenizer_t5, tokenizer_clip):
    prompt = list(examples[caption_column])
    bs = len(prompt)

    text_embeds = tokenizer_t5(prompt)
    prompt_embeds = tokenizer_clip(prompt)

    examples["text_embeds"] = jnp.float16(text_embeds)
    examples["input_ids"] = jnp.float16(jnp.zeros((bs, text_embeds.shape[1], 3)))
    examples["prompt_embeds"] = jnp.float16(prompt_embeds)

    return examples

  @staticmethod
  def transform_images(
      examples,
      image_column,
      image_resolution,
      encoder,
  ):
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
      batch_images = rearrange(batch_images, "b h w c -> b c h w")
      batch_images = encoder.encode(batch_images)
      encoded_images.append(batch_images)

    images = jnp.concatenate(encoded_images, axis=0, dtype=jnp.float16)

    batch_size, _, h, w = images.shape
    images = rearrange(images, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_ids = jnp.zeros((h // 2, w // 2, 3))
    img_ids = img_ids.at[..., 1].set(jnp.arange(h // 2)[:, None])
    img_ids = img_ids.at[..., 2].set(jnp.arange(w // 2)[None, :])
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

    examples["pixel_values"] = images
    examples["img_ids"] = img_ids

    return examples

  @staticmethod
  def encode_image(pipeline, x):
    return pipeline.pack_img(pipeline.encode(x))

  def load_dataset(self, pipeline):
    p_encode = None
    rng = None

    if self.config.dataset_type == "tf" and self.config.cache_latents_text_encoder_outputs:
      ...

    tokenize_fn = partial(
        JFluxTrainer.tokenize_captions,
        caption_column=self.config.caption_column,
        tokenizer_t5=pipeline.t5,
        tokenizer_clip=pipeline.clip,
    )
    max_logging.log("Creating image transforms")
    image_transforms_fn = partial(
        JFluxTrainer.transform_images,
        image_column=self.config.image_column,
        image_resolution=self.config.resolution,
        encoder=pipeline.ae,
    )
    max_logging.log("Creating data iterator")
    data_iterator = make_data_iterator(
        self.config,
        jax.process_index(),
        jax.process_count(),
        self.mesh,
        self.total_train_batch_size,
        tokenize_fn=tokenize_fn,
        image_transforms_fn=image_transforms_fn,
    )
    return data_iterator

  def compile_train_step(self, pipeline, train_states, state_shardings, data_shardings):
    self.rng, train_rngs = jax.random.split(self.rng)
    guidance_vec = jnp.full((self.total_train_batch_size,), self.config.guidance, dtype=self.config.activations_dtype)
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
      p_train_step = p_train_step.lower(train_states["flux_state"], dummy_batch, train_rngs)
      p_train_step = p_train_step.compile()
      max_logging.log(f"Compile time: {(time.time() - s )}")
      return p_train_step

  def training_loop(self, p_train_step, pipeline, train_states, data_iterator, unet_learning_rate_scheduler):
    writer = max_utils.initialize_summary_writer(self.config)
    flux_state = train_states["flux_state"]

    num_model_parameters = max_utils.calculate_num_params_from_pytree(flux_state.params)

    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
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
    last_profiling_step = jnp.clip(
        first_profiling_step + self.config.profiler_steps - 1, first_profiling_step, self.config.max_train_steps - 1
    )

    start_step = train_utils.get_first_step(train_states["flux_state"])
    _, train_rngs = jax.random.split(self.rng)
    times = []
    for step in jnp.arange(start_step, self.config.max_train_steps):
      if self.config.enable_profiler and step == first_profiling_step:
        max_utils.activate_profiler(self.config)

      example_batch = train_utils.load_next_batch(data_iterator, example_batch, self.config)
      example_batch = {key: jnp.asarray(value, dtype=self.config.activations_dtype) for key, value in example_batch.items()}

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        with self.mesh:
          flux_state, train_metric, train_rngs = p_train_step(flux_state, example_batch, train_rngs)
      samples_count = self.total_train_batch_size * (step + 1)
      new_time = datetime.datetime.now()

      train_utils.record_scalar_metrics(
          train_metric, new_time - last_step_completion, self.per_device_tflops, unet_learning_rate_scheduler(step)
      )
      times.append(new_time - last_step_completion)
      if self.config.write_metrics:
        train_utils.write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config)
      last_step_completion = new_time

      if step != 0 and self.config.checkpoint_every != -1 and samples_count % self.config.checkpoint_every == 0:
        max_logging.log(f"Saving checkpoint for step {step}")
        train_states["flux_state"] = flux_state
        self.save_checkpoint(step, pipeline, train_states)

      if self.config.enable_profiler and step == last_profiling_step:
        max_utils.deactivate_profiler(self.config)

    if self.config.write_metrics and start_step < self.config.max_train_steps:
      train_utils.write_metrics(
          writer, local_metrics_file, running_gcs_metrics, train_metric, self.config.max_train_steps - 1, self.config
      )

    train_states["flux_state"] = flux_state
    max_logging.log(f"Average time per step: {sum(times[2:], datetime.timedelta(0)) / len(times[2:])}")
    if self.config.save_final_checkpoint:
      max_logging.log(f"Saving checkpoint for step {step}")
      self.save_checkpoint(step, pipeline, train_states)
      self.checkpoint_manager.wait_until_finished()
    return train_states


def _train_step(flux_state, batch, train_rng, guidance_vec, pipeline, scheduler, config):
  _, gen_dummy_rng = jax.random.split(train_rng)
  sample_rng, timestep_bias_rng, new_train_rng = jax.random.split(gen_dummy_rng, 3)
  state_params = {"flux": flux_state.params}

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
    if config.timestep_bias["strategy"] == "none":
      timesteps = jax.random.randint(timestep_rng, shape=(bsz,), minval=0, maxval=len(scheduler.timesteps))
    else:
      weights = train_utils.generate_timestep_weights(config, pipeline.scheduler.config.num_train_timesteps)
      timesteps = jax.random.categorical(timestep_bias_rng, logits=jnp.log(weights), shape=(bsz,))

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = pipeline.scheduler.add_noise(scheduler, latents, noise, timesteps, flux=True)

    # Predict the noise residual and compute loss
    # for flux, encoder_hidden_states = (txt, txt_ids, vec)
    model_pred = pipeline.flux.apply(
        {"params": state_params["flux"]},
        hidden_states=noisy_latents,
        img_ids=img_ids,
        encoder_hidden_states=text_embeds,
        txt_ids=text_embeds_ids,
        pooled_projections=prompt_embeds,
        timestep=scheduler.timesteps[timesteps],
        guidance=guidance_vec,
    ).sample

    target = noise - latents
    loss = (target - model_pred) ** 2

    loss = nn.with_logical_constraint(loss, ("activation_embed_and_logits_batch", "activation_length"))
    loss = jnp.mean(loss)

    return loss

  grad_fn = jax.value_and_grad(compute_loss)
  loss, grad = grad_fn(state_params)

  if config.max_grad_norm > 0:
    grad, _ = optax.clip_by_global_norm(config.max_grad_norm).update(grad, flux_state, None)

  new_state = flux_state.apply_gradients(grads=grad["flux"])

  metrics = {"scalar": {"learning/loss": loss}, "scalars": {}}

  return new_state, metrics, new_train_rng
