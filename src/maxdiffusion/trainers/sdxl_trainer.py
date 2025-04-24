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
from jax.sharding import PartitionSpec as P
from flax.linen import partitioning as nn_partitioning
from maxdiffusion.trainers.stable_diffusion_trainer import (StableDiffusionTrainer)

from maxdiffusion.input_pipeline.input_pipeline_interface import (make_data_iterator)

from maxdiffusion import (max_utils, maxdiffusion_utils, max_logging)

from maxdiffusion.train_utils import (
    compute_snr,
    generate_timestep_weights,
    get_first_step,
    load_next_batch,
    record_scalar_metrics,
    write_metrics,
)

from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import (STABLE_DIFFUSION_XL_CHECKPOINT)


class StableDiffusionXLTrainer(StableDiffusionTrainer):

  def __init__(self, config):
    StableDiffusionTrainer.__init__(self, config, STABLE_DIFFUSION_XL_CHECKPOINT)

    self.text_encoder_2_learning_rate_scheduler = None

    if config.train_text_encoder:
      raise ValueError("this script currently doesn't support training text_encoders")

  def post_training_steps(self, pipeline, params, train_states):
    return super().post_training_steps(pipeline, params, train_states)

  def get_shaped_batch(self, config, pipeline):
    """Return the shape of the batch - this is what eval_shape would return for the
    output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078."""

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    total_train_batch_size = config.total_train_batch_size
    shaped_batch = {}

    if self.config.dataset_type == "tf" and self.config.cache_latents_text_encoder_outputs:
      batch_image_shape = (
          total_train_batch_size,
          pipeline.unet.config.in_channels,
          config.resolution // vae_scale_factor,
          config.resolution // vae_scale_factor,
      )

      text_embeds_dim = pipeline.unet.config.projection_class_embeddings_input_dim - (
          6 * pipeline.unet.config.addition_time_embed_dim
      )
      text_embeds_shape = (total_train_batch_size, text_embeds_dim)
      prompt_embeds_shape = (total_train_batch_size, pipeline.text_encoder.config.max_position_embeddings, 2048)
      shaped_batch["prompt_embeds"] = jax.ShapeDtypeStruct(prompt_embeds_shape, jnp.float32)
      shaped_batch["text_embeds"] = jax.ShapeDtypeStruct(text_embeds_shape, jnp.float32)
    else:
      batch_image_shape = (total_train_batch_size, 3, self.config.resolution, self.config.resolution)

    input_ids_shape = (total_train_batch_size, 2, pipeline.text_encoder.config.max_position_embeddings)
    shaped_batch["pixel_values"] = jax.ShapeDtypeStruct(batch_image_shape, jnp.float32)
    shaped_batch["input_ids"] = jax.ShapeDtypeStruct(input_ids_shape, jnp.int32)
    return shaped_batch

  def get_data_shardings(self):
    data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))
    if self.config.dataset_type == "tf" and self.config.cache_latents_text_encoder_outputs:
      data_sharding = {
          "input_ids": data_sharding,
          "pixel_values": data_sharding,
          "prompt_embeds": data_sharding,
          "text_embeds": data_sharding,
      }
    else:
      data_sharding = {
          "input_ids": data_sharding,
          "pixel_values": data_sharding,
      }
    return data_sharding

  def load_dataset(self, pipeline, params, train_states):
    config = self.config
    total_train_batch_size = self.total_train_batch_size
    mesh = self.mesh
    p_encode = None
    p_vae_apply = None
    rng = None
    if config.dataset_type == "tf" and config.cache_latents_text_encoder_outputs:
      p_encode = jax.jit(
          partial(
              maxdiffusion_utils.encode_xl,
              text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2],
              text_encoder_params=[train_states["text_encoder_state"].params, train_states["text_encoder_2_state"].params],
          )
      )
      p_vae_apply = jax.jit(
          partial(maxdiffusion_utils.vae_apply, vae=pipeline.vae, vae_params=train_states["vae_state"].params)
      )
      rng = self.rng

    tokenize_fn = partial(
        maxdiffusion_utils.tokenize_captions_xl,
        caption_column=config.caption_column,
        tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2],
        p_encode=p_encode,
    )
    image_transforms_fn = partial(
        maxdiffusion_utils.transform_images,
        image_column=config.image_column,
        image_resolution=config.resolution,
        rng=rng,
        global_batch_size=total_train_batch_size,
        p_vae_apply=p_vae_apply,
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
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      p_train_step = jax.jit(
          max_utils.get_train_step_partial_with_signature(_train_step, pipeline=pipeline, params=params, config=self.config),
          in_shardings=(
              state_shardings["unet_state_shardings"],
              state_shardings["vae_state_shardings"],
              None,
              None,
              data_shardings,
              None,
          ),
          out_shardings=(state_shardings["unet_state_shardings"], None, None),
          donate_argnums=(0,),
      )
      max_logging.log("Precompiling...")
      s = time.time()
      dummy_batch = self.get_shaped_batch(self.config, pipeline)
      p_train_step = p_train_step.lower(
          train_states["unet_state"],
          train_states["vae_state"],
          train_states["text_encoder_state"],
          train_states["text_encoder_2_state"],
          dummy_batch,
          train_rngs,
      )
      p_train_step = p_train_step.compile()
      max_logging.log(f"Compile time: {(time.time() - s )}")
      return p_train_step

  def training_loop(self, p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler):

    writer = max_utils.initialize_summary_writer(self.config)
    unet_state = train_states["unet_state"]
    vae_state = train_states["vae_state"]
    text_encoder_state = train_states["text_encoder_state"]
    text_encoder_2_state = train_states["text_encoder_2_state"]
    num_model_parameters = max_utils.calculate_num_params_from_pytree(unet_state.params)

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
    start_step = get_first_step(train_states["unet_state"])
    _, train_rngs = jax.random.split(self.rng)

    for step in np.arange(start_step, self.config.max_train_steps):
      if self.config.enable_profiler and step == first_profiling_step:
        max_utils.activate_profiler(self.config)

      example_batch = load_next_batch(data_iterator, example_batch, self.config)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        (unet_state, train_metric, train_rngs) = p_train_step(
            unet_state, vae_state, text_encoder_state, text_encoder_2_state, example_batch, train_rngs
        )

      samples_count = self.total_train_batch_size * (step + 1)
      new_time = datetime.datetime.now()

      record_scalar_metrics(
          train_metric, new_time - last_step_completion, self.per_device_tflops, unet_learning_rate_scheduler(step)
      )
      if self.config.write_metrics:
        write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config)
      last_step_completion = new_time

      if step != 0 and self.config.checkpoint_every != -1 and samples_count % self.config.checkpoint_every == 0:
        train_states["unet_state"] = unet_state
        train_states["vae_state"] = vae_state
        train_states["text_encoder_state"] = text_encoder_state
        train_states["text_encoder_2_state"] = text_encoder_2_state
        self.save_checkpoint(step, pipeline, params, train_states)

      if self.config.enable_profiler and step == last_profiling_step:
        max_utils.deactivate_profiler(self.config)

    if self.config.write_metrics:
      write_metrics(
          writer, local_metrics_file, running_gcs_metrics, train_metric, self.config.max_train_steps - 1, self.config
      )

    train_states["unet_state"] = unet_state
    train_states["text_encoder_state"] = text_encoder_state
    train_states["text_encoder_2_state"] = text_encoder_2_state
    self.save_checkpoint(self.config.max_train_steps - 1, pipeline, params, train_states)
    self.checkpoint_manager.wait_until_finished()


def _train_step(unet_state, vae_state, text_encoder_state, text_encoder_2_state, batch, train_rng, pipeline, params, config):
  _, gen_dummy_rng = jax.random.split(train_rng)
  sample_rng, timestep_bias_rng, new_train_rng = jax.random.split(gen_dummy_rng, 3)

  noise_scheduler = pipeline.scheduler
  noise_scheduler_state = params["scheduler"]

  state_params = {"unet": unet_state.params}

  def compute_loss(state_params):
    if config.dataset_type == "tf" and config.cache_latents_text_encoder_outputs:
      latents = batch["pixel_values"]
      prompt_embeds = batch["prompt_embeds"]
      text_embeds = batch["text_embeds"]
    else:
      latents = maxdiffusion_utils.vae_apply(
          images=batch["pixel_values"],
          sample_rng=sample_rng,
          vae=pipeline.vae,
          vae_params=vae_state.params,
      )
      prompt_embeds, text_embeds = maxdiffusion_utils.encode_xl(
          input_ids=batch["input_ids"],
          text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2],
          text_encoder_params=[text_encoder_state.params, text_encoder_2_state.params],
      )

    # Sample noise that we'll add to the latents
    noise_rng, timestep_rng = jax.random.split(sample_rng)
    noise = jax.random.normal(noise_rng, latents.shape)
    # Sample a random timestep for each image
    bsz = latents.shape[0]
    if config.timestep_bias["strategy"] == "none":
      timesteps = jax.random.randint(
          timestep_rng,
          (bsz,),
          0,
          noise_scheduler.config.num_train_timesteps,
      )
    else:
      weights = generate_timestep_weights(config, noise_scheduler.config.num_train_timesteps)
      timesteps = jax.random.categorical(timestep_bias_rng, logits=jnp.log(weights), shape=(bsz,))

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

    # TODO : @jfacevedo - support cropping.
    add_time_ids = maxdiffusion_utils.get_add_time_ids(
        (config.resolution, config.resolution),
        (0, 0),
        (config.resolution, config.resolution),
        prompt_embeds.shape[0],
        dtype=prompt_embeds.dtype,
    )
    unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": text_embeds}

    # Predict the noise residual and compute loss
    model_pred = unet_state.apply_fn(
        {"params": state_params["unet"]},
        noisy_latents,
        timesteps,
        prompt_embeds,
        added_cond_kwargs=unet_added_conditions,
        train=True,
    ).sample

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
      target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
      target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
    else:
      raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    loss = (target - model_pred) ** 2

    # snr
    if config.snr_gamma > 0:
      snr = jnp.array(compute_snr(timesteps, noise_scheduler_state))
      snr_loss_weights = jnp.where(snr < config.snr_gamma, snr, jnp.ones_like(snr) * config.snr_gamma)
      if noise_scheduler.config.prediction_type == "epsilon":
        snr_loss_weights = snr_loss_weights / snr
      elif noise_scheduler.config.prediction_type == "v_prediction":
        snr_loss_weights = snr_loss_weights / (snr + 1)
      loss = loss * snr_loss_weights[:, None, None, None]

    loss = loss.mean()

    return loss

  grad_fn = jax.value_and_grad(compute_loss)
  loss, grad = grad_fn(state_params)

  new_state = unet_state.apply_gradients(grads=grad["unet"])

  metrics = {"scalar": {"learning/loss": loss}, "scalars": {}}

  return new_state, metrics, new_train_rng
