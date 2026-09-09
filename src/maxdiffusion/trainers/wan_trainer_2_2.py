"""
Copyright 2025 Google LLC

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

import functools
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P
import jaxopt

from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
from maxdiffusion.trainers.base_wan_trainer import BaseWanTrainer, TrainState, _to_array, print_ssim
from maxdiffusion import max_logging, max_utils, train_utils
from maxdiffusion.train_utils import (_metrics_queue, _tensorboard_writer_worker, load_next_batch)
from maxdiffusion.generate_wan import inference_generate_video
from maxdiffusion.generate_wan import run as generate_wan
from maxdiffusion.pipelines.wan.wan_pipeline_2_2 import WanPipeline2_2


class WanTrainer2_2(BaseWanTrainer):

  def __init__(self, config):
    super().__init__(config)
    if not (0.0 < self.config.boundary_ratio < 1.0):
      raise ValueError(f"boundary_ratio must be strictly between 0 and 1, got {self.config.boundary_ratio}")

  def _get_checkpointer(self):
    return WanCheckpointer2_2(config=self.config)

  def get_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    data_sharding = {"latents": data_sharding, "encoder_hidden_states": data_sharding}
    return data_sharding

  def get_eval_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    timesteps_axis = self.config.data_sharding[0] if self.config.data_sharding else None
    timesteps_sharding = jax.sharding.NamedSharding(mesh, P(timesteps_axis))
    return {"latents": data_sharding, "encoder_hidden_states": data_sharding, "timesteps": timesteps_sharding}

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    import tensorflow as tf
    from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator

    config = self.config
    if config.dataset_type == "synthetic":
      return make_data_iterator(
          config,
          jax.process_index(),
          jax.process_count(),
          mesh,
          config.global_batch_size_to_load,
          pipeline=pipeline,
          is_training=is_training,
      )

    if config.dataset_type != "tfrecord" or not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "Wan 2.2 training only supports config.dataset_type set to tfrecords and config.cache_latents_text_encoder_outputs set to True"
      )
    feature_description = {
        "latents": tf.io.FixedLenFeature([], tf.string),
        "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
    }

    if not is_training:
      feature_description["timesteps"] = tf.io.FixedLenFeature([], tf.int64)

    def prepare_sample_train(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
      return {"latents": latents, "encoder_hidden_states": encoder_hidden_states}

    def prepare_sample_eval(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
      timesteps = features["timesteps"]
      return {"latents": latents, "encoder_hidden_states": encoder_hidden_states, "timesteps": timesteps}

    data_iterator = make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        config.global_batch_size_to_load,
        feature_description=feature_description,
        prepare_sample_fn=prepare_sample_train if is_training else prepare_sample_eval,
        is_training=is_training,
    )
    return data_iterator

  def calculate_tflops(self, pipeline):
    maxdiffusion_config = pipeline.config
    height = pipeline.config.height
    width = pipeline.config.width
    num_frames = pipeline.config.num_frames

    transformer_config = pipeline.low_noise_transformer.config
    num_layers = transformer_config.num_layers
    heads = transformer_config.num_attention_heads
    head_dim = transformer_config.attention_head_dim
    hidden_dim = heads * head_dim
    ffn_dim = transformer_config.ffn_dim
    text_dim = getattr(transformer_config, "text_dim", 4096)
    text_seq_len = 512
    seq_len = int(((height / 8) * (width / 8) * ((num_frames - 1) // pipeline.vae_scale_factor_temporal + 1)) / 4)

    # Self-attention FLOPs
    self_attn_qkv_proj_flops = 3 * (2 * seq_len * hidden_dim**2)
    self_attn_qk_v_flops = 2 * (2 * seq_len**2 * hidden_dim)
    self_attn_output_proj_flops = 1 * (2 * seq_len * hidden_dim**2)

    # Cross-attention FLOPs
    cross_attn_q_proj_flops = 1 * (2 * seq_len * hidden_dim**2)
    cross_attn_kv_proj_flops = 2 * (2 * text_seq_len * text_dim * hidden_dim)
    cross_attention_qk_v_flops = 2 * (2 * seq_len * text_seq_len * hidden_dim)
    cross_attn_output_proj_flops = 1 * (2 * seq_len * hidden_dim**2)

    total_attn_flops = (
        self_attn_qkv_proj_flops
        + self_attn_qk_v_flops
        + self_attn_output_proj_flops
        + cross_attn_q_proj_flops
        + cross_attn_kv_proj_flops
        + cross_attention_qk_v_flops
        + cross_attn_output_proj_flops
    )

    # SwiGLU FFN FLOPs (3 projections: gate, up, down)
    ffn_flops = 3 * (2 * seq_len * hidden_dim * ffn_dim)
    flops_per_block = total_attn_flops + ffn_flops
    total_transformer_flops = flops_per_block * num_layers
    tflops = maxdiffusion_config.per_device_batch_size * total_transformer_flops / 1e12
    train_tflops = 3 * tflops

    max_logging.log(f"Calculated TFLOPs per pass: {train_tflops:.4f}")
    return train_tflops, total_attn_flops, seq_len

  def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
    return jax.jit(
        functools.partial(train_step_2_2, scheduler=pipeline.scheduler, config=self.config),
        in_shardings=(state_shardings["low_noise"], state_shardings["high_noise"], data_shardings, None, None),
        out_shardings=(state_shardings["low_noise"], state_shardings["high_noise"], None, None, None),
    )

  def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
    return jax.jit(
        functools.partial(eval_step_2_2, scheduler=pipeline.scheduler, config=self.config),
        in_shardings=(state_shardings["low_noise"], state_shardings["high_noise"], eval_data_shardings, None, None),
        out_shardings=(None, None),
    )

  def generate_sample(self, config, pipeline, filename_prefix):
    if not hasattr(pipeline, "vae"):
      wan_vae, vae_cache = WanPipeline2_2.load_vae(
          pipeline.mesh.devices, pipeline.mesh, nnx.Rngs(jax.random.key(config.seed)), config
      )
      pipeline.vae = wan_vae
      pipeline.vae_cache = vae_cache
    return generate_wan(config, pipeline, filename_prefix)

  def start_training(self):
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      pipeline, opt_state_dict, step = self.checkpointer.load_checkpoint()

    # Checkpoint restoration semantics:
    # - True resume requires both opt_state_dict and step, restoring optimizer states and resuming from step.
    # - Weights-only or pretrained model checkpoints (opt_state_dict is None) perform a warm start at step 0
    #   with fresh optimizer states and full warmup schedule, matching BaseWanTrainer.
    restore_args = {}
    if opt_state_dict is not None and step is not None:
      restore_args = {"opt_state": opt_state_dict, "step": step}
      del opt_state_dict

    if self.config.enable_ssim:
      pretrained_video_path = self.generate_sample(self.config, pipeline, filename_prefix="pre-training-")

    if self.config.eval_every == -1 or (not self.config.enable_generate_video_for_eval):
      if hasattr(pipeline, "vae"):
        del pipeline.vae
      if hasattr(pipeline, "vae_cache"):
        del pipeline.vae_cache

    mesh = pipeline.mesh
    train_data_iterator = self.load_dataset(mesh, pipeline=pipeline, is_training=True)

    scheduler, scheduler_state = self.create_scheduler()
    pipeline.scheduler = scheduler
    pipeline.scheduler_state = scheduler_state

    optimizer_low, learning_rate_scheduler_low = self.checkpointer._create_optimizer(
        pipeline.low_noise_transformer, self.config, self.config.learning_rate, scale_factor=self.config.boundary_ratio
    )
    optimizer_high, learning_rate_scheduler_high = self.checkpointer._create_optimizer(
        pipeline.high_noise_transformer,
        self.config,
        self.config.learning_rate,
        scale_factor=(1.0 - self.config.boundary_ratio),
    )

    pipeline = self.training_loop_2_2(
        pipeline,
        optimizer_low,
        optimizer_high,
        learning_rate_scheduler_low,
        learning_rate_scheduler_high,
        train_data_iterator,
        restore_args,
    )

    if self.config.enable_ssim:
      posttrained_video_path = self.generate_sample(self.config, pipeline, filename_prefix="post-training-")
      print_ssim(pretrained_video_path, posttrained_video_path)

  def training_loop_2_2(
      self,
      pipeline,
      optimizer_low,
      optimizer_high,
      learning_rate_scheduler_low,
      learning_rate_scheduler_high,
      train_data_iterator,
      restore_args: dict | None = None,
  ):
    if restore_args is None:
      restore_args = {}
    mesh = pipeline.mesh
    graphdef_low, params_low, rest_of_state_low = nnx.split(pipeline.low_noise_transformer, nnx.Param, ...)
    graphdef_high, params_high, rest_of_state_high = nnx.split(pipeline.high_noise_transformer, nnx.Param, ...)

    with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      state_low = TrainState.create(
          apply_fn=graphdef_low.apply,
          params=params_low,
          tx=optimizer_low,
          graphdef=graphdef_low,
          rest_of_state=rest_of_state_low,
      )
      state_high = TrainState.create(
          apply_fn=graphdef_high.apply,
          params=params_high,
          tx=optimizer_high,
          graphdef=graphdef_high,
          rest_of_state=rest_of_state_high,
      )

      if restore_args and restore_args.get("opt_state") is not None:
        step = restore_args.get("step", 0)
        max_logging.log(f"Restoring optimizer and resuming from step {step}")
        opt_state_dict = restore_args.get("opt_state", {})
        if isinstance(opt_state_dict, dict):
          if opt_state_dict.get("low_noise_transformer") is not None:
            state_low = state_low.replace(opt_state=opt_state_dict["low_noise_transformer"])
          if opt_state_dict.get("high_noise_transformer") is not None:
            state_high = state_high.replace(opt_state=opt_state_dict["high_noise_transformer"])
          if opt_state_dict.get("low_noise_step") is not None:
            state_low = state_low.replace(step=opt_state_dict["low_noise_step"])
          if opt_state_dict.get("high_noise_step") is not None:
            state_high = state_high.replace(step=opt_state_dict["high_noise_step"])

      state_low = jax.tree.map(_to_array, state_low)
      state_high = jax.tree.map(_to_array, state_high)

      state_spec_low = nnx.get_partition_spec(state_low)
      state_spec_high = nnx.get_partition_spec(state_high)

      state_low = jax.lax.with_sharding_constraint(state_low, state_spec_low)
      state_high = jax.lax.with_sharding_constraint(state_high, state_spec_high)

      state_shardings_low = nnx.get_named_sharding(state_low, mesh)
      state_shardings_high = nnx.get_named_sharding(state_high, mesh)

      state_shardings = {"low_noise": state_shardings_low, "high_noise": state_shardings_high}

    if self.config.hardware != "gpu":
      max_utils.delete_pytree(params_low)
      max_utils.delete_pytree(params_high)

    data_shardings = self.get_data_shardings(mesh)
    eval_data_shardings = self.get_eval_data_shardings(mesh)

    writer = max_utils.initialize_summary_writer(self.config)
    writer_thread = threading.Thread(target=_tensorboard_writer_worker, args=(writer, self.config), daemon=True)
    writer_thread.start()

    num_model_parameters = max_utils.calculate_num_params_from_pytree(
        state_low.params
    ) + max_utils.calculate_num_params_from_pytree(state_high.params)
    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
    max_utils.add_config_to_summary_writer(self.config, writer)

    if jax.process_index() == 0:
      max_logging.log("***** Running training *****")
      max_logging.log(f"  Total optimization steps = {self.config.max_train_steps}")

    p_train_step = self.get_train_step(pipeline, mesh, state_shardings, data_shardings)
    p_eval_step = self.get_eval_step(pipeline, mesh, state_shardings, eval_data_shardings)

    base_rng = jax.random.key(self.config.seed)
    base_rng, eval_rng_key = jax.random.split(base_rng)
    start_step = restore_args.get("step", 0)
    host_step_low = 0
    host_step_high = 0
    if restore_args and restore_args.get("opt_state") is not None:
      opt_state_dict = restore_args.get("opt_state", {})
      if isinstance(opt_state_dict, dict):
        if opt_state_dict.get("low_noise_step") is not None:
          host_step_low = int(opt_state_dict["low_noise_step"])
        if opt_state_dict.get("high_noise_step") is not None:
          host_step_high = int(opt_state_dict["high_noise_step"])

    last_step_completion = datetime.datetime.now()
    local_metrics_file = open(self.config.metrics_file, "a", encoding="utf8") if self.config.metrics_file else None
    running_gcs_metrics = [] if self.config.gcs_metrics else None

    per_device_tflops, _, _ = self.calculate_tflops(pipeline)
    scheduler_state = pipeline.scheduler_state
    example_batch = load_next_batch(train_data_iterator, None, self.config)

    first_profiling_step = self.config.skip_first_n_steps_for_profiler
    if max_utils.profiler_enabled(self.config) and first_profiling_step >= self.config.max_train_steps:
      max_logging.log(
          "Warning: Profiling requested for steps %d to %d, but max_train_steps is set to %d. Profiling will not occur."
          % (
              first_profiling_step + self.config.profiler_steps - 1,
              first_profiling_step,
              self.config.max_train_steps - 1,
          )
      )
    last_profiling_step = np.min([first_profiling_step + self.config.profiler_steps - 1, self.config.max_train_steps - 1])

    with ThreadPoolExecutor(max_workers=1) as executor:
      for step in np.arange(start_step, self.config.max_train_steps):
        step_int = int(step)
        if max_utils.profiler_enabled(self.config) and step == first_profiling_step:
          self._profiler = max_utils.Profiler(self.config)
          self._profiler.start()
        step_rng = jax.random.fold_in(base_rng, step_int)
        start_step_time = datetime.datetime.now()
        next_batch_future = executor.submit(load_next_batch, train_data_iterator, example_batch, self.config)

        # Track active expert on host without device-to-host sync stall
        _, _, _, _, cond_rng, _ = jax.random.split(step_rng, num=6)
        is_high_noise = bool(jax.random.uniform(cond_rng) > self.config.boundary_ratio)
        if is_high_noise:
          active_lr = float(learning_rate_scheduler_high(host_step_high))
          host_step_high += 1
        else:
          active_lr = float(learning_rate_scheduler_low(host_step_low))
          host_step_low += 1

        with (
            jax.profiler.StepTraceAnnotation("train", step_num=step),
            pipeline.mesh,
            nn_partitioning.axis_rules(self.config.logical_axis_rules),
        ):
          state_low, state_high, scheduler_state, train_metric, _ = p_train_step(
              state_low, state_high, example_batch, step_rng, scheduler_state
          )
          train_metric["scalar"]["learning/loss"].block_until_ready()
        last_step_completion = datetime.datetime.now()

        lr_low = float(learning_rate_scheduler_low(host_step_low))
        lr_high = float(learning_rate_scheduler_high(host_step_high))
        train_utils.record_scalar_metrics(train_metric, last_step_completion - start_step_time, per_device_tflops, active_lr)

        if self.config.write_metrics:
          train_metric["scalar"]["learning/current_learning_rate_low"] = lr_low
          train_metric["scalar"]["learning/current_learning_rate_high"] = lr_high
          train_metric["steps"] = {
              "learning/loss_low": host_step_low,
              "learning/loss_low_fine": host_step_low,
              "learning/loss_low_mid": host_step_low,
              "learning/loss_low_coarse": host_step_low,
              "learning/current_learning_rate_low": host_step_low,
              "learning/max_grad_norm_low": host_step_low,
              "learning/max_abs_grad_low": host_step_low,
              "learning/loss_high": host_step_high,
              "learning/current_learning_rate_high": host_step_high,
              "learning/max_grad_norm_high": host_step_high,
              "learning/max_abs_grad_high": host_step_high,
          }
          train_utils.write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config)

        if max_utils.profiler_enabled(self.config) and step == last_profiling_step:
          if self._profiler:
            self._profiler.stop()

        if self.config.eval_every > 0 and (step + 1) % self.config.eval_every == 0:
          if self.config.enable_generate_video_for_eval:
            pipeline.low_noise_transformer = nnx.merge(state_low.graphdef, state_low.params, state_low.rest_of_state)
            pipeline.high_noise_transformer = nnx.merge(state_high.graphdef, state_high.params, state_high.rest_of_state)
            inference_generate_video(self.config, pipeline, filename_prefix=f"{step+1}-train_steps-")
          self.eval_2_2(mesh, eval_rng_key, step, p_eval_step, state_low, state_high, scheduler_state, writer)

        example_batch = next_batch_future.result()
        if self.config.checkpoint_every > 0 and (step + 1) % self.config.checkpoint_every == 0:
          save_step = int(step + 1)
          max_logging.log(f"Saving checkpoint for step {save_step}")
          train_states = {
              "low_noise_transformer": state_low if self.config.save_optimizer else state_low.params,
              "high_noise_transformer": state_high if self.config.save_optimizer else state_high.params,
          }
          self.checkpointer.save_checkpoint(save_step, pipeline, train_states)

      _metrics_queue.put(None)
      writer_thread.join()
      if writer:
        writer.flush()
      if self.config.save_final_checkpoint:
        save_step = int(self.config.max_train_steps)
        max_logging.log(f"Saving final checkpoint for step {save_step}")
        train_states = {
            "low_noise_transformer": state_low if self.config.save_optimizer else state_low.params,
            "high_noise_transformer": state_high if self.config.save_optimizer else state_high.params,
        }
        self.checkpointer.save_checkpoint(save_step, pipeline, train_states)
        self.checkpointer.checkpoint_manager.wait_until_finished()

      pipeline.low_noise_transformer = nnx.merge(state_low.graphdef, state_low.params, state_low.rest_of_state)
      pipeline.high_noise_transformer = nnx.merge(state_high.graphdef, state_high.params, state_high.rest_of_state)
      return pipeline

  def eval_2_2(self, mesh, eval_rng_key, step, p_eval_step, state_low, state_high, scheduler_state, writer):
    eval_data_iterator = self.load_dataset(mesh, is_training=False)
    eval_rng = jax.random.fold_in(eval_rng_key, int(step))
    eval_losses_by_timestep = {}
    while True:
      try:
        eval_start_time = datetime.datetime.now()
        eval_batch = load_next_batch(eval_data_iterator, None, self.config)
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
          metrics, eval_rng = p_eval_step(state_low, state_high, eval_batch, eval_rng, scheduler_state)
          metrics["scalar"]["learning/eval_loss"].block_until_ready()

        losses = jax.device_get(metrics["scalar"]["learning/eval_loss"])
        timesteps = jax.device_get(metrics["scalar"]["learning/eval_timesteps"])

        if jax.process_index() == 0:
          for t, l in zip(timesteps.flatten(), losses.flatten()):
            timestep = int(t)
            if timestep not in eval_losses_by_timestep:
              eval_losses_by_timestep[timestep] = []
            eval_losses_by_timestep[timestep].append(float(l))
          eval_end_time = datetime.datetime.now()
          eval_duration = eval_end_time - eval_start_time
          max_logging.log(f"Eval time: {eval_duration.total_seconds():.2f} seconds.")
      except StopIteration:
        break

    if eval_losses_by_timestep and jax.process_index() == 0:
      mean_per_timestep = []
      for timestep, losses in sorted(eval_losses_by_timestep.items()):
        losses = jnp.array(losses)
        losses = losses[: min(self.config.eval_max_number_of_samples_in_bucket, len(losses))]
        mean_loss = jnp.mean(losses)
        mean_per_timestep.append(mean_loss)
      final_eval_loss = jnp.mean(jnp.array(mean_per_timestep))
      max_logging.log(f"Step {step}, Final Average Eval loss: {final_eval_loss:.4f}")
      if writer:
        writer.add_scalar("learning/eval_loss", final_eval_loss, step)


def train_step_2_2(state_low, state_high, data, rng, scheduler_state, scheduler, config):
  """Wan 2.2 joint dual-expert training step.

  Expert Routing:
    Routing is performed at the batch level (one expert per step) based on `boundary_ratio`.
    While both transformer states remain resident in TPU memory, batch-level routing ensures
    that only one ~27B transformer executes forward and backward passes per step, avoiding
    dual-model activation memory concurrency and reducing peak dynamic HBM usage.

  Timestep Distributions:
    - High-noise expert (coarse structure): samples timesteps uniformly from [boundary, num_train_timesteps),
      maintaining an unbiased flow-matching vector field estimator for coarse generation.
    - Low-noise expert (fine details): samples timesteps uniformly from [0, boundary),
      maintaining an unbiased flow-matching vector field estimator for fine details.
  """
  _, new_rng, timestep_rng, dropout_rng, cond_rng, noise_rng = jax.random.split(rng, num=6)

  data = {k: v[: config.global_batch_size_to_train_on] for k, v in data.items()}

  bsz = data["latents"].shape[0]
  num_train_timesteps = scheduler.config.num_train_timesteps
  boundary = int(config.boundary_ratio * num_train_timesteps)

  is_high_noise = jax.random.uniform(cond_rng) > config.boundary_ratio

  def compute_loss_high(high_params, s_high):
    timesteps = jax.random.randint(timestep_rng, shape=(bsz,), minval=boundary, maxval=num_train_timesteps)
    model = nnx.merge(s_high.graphdef, high_params, s_high.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)
    noise = jax.random.normal(key=noise_rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, training_weight = scheduler.apply_flow_match(noise, latents, timesteps)

    model_pred = model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        deterministic=False,
        rngs=nnx.Rngs(dropout=dropout_rng),
    )
    loss = (training_target - model_pred) ** 2
    if not getattr(config, "disable_training_weights", False):
      training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
      loss = loss * training_weight
    return jnp.mean(loss)

  def compute_loss_low(low_params, s_low):
    timesteps = jax.random.randint(timestep_rng, shape=(bsz,), minval=0, maxval=boundary)
    model = nnx.merge(s_low.graphdef, low_params, s_low.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)
    noise = jax.random.normal(key=noise_rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, training_weight = scheduler.apply_flow_match(noise, latents, timesteps)

    model_pred = model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        deterministic=False,
        rngs=nnx.Rngs(dropout=dropout_rng),
    )
    loss = (training_target - model_pred) ** 2
    if not getattr(config, "disable_training_weights", False):
      training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
      loss = loss * training_weight

    loss_mean = jnp.mean(loss)
    loss_per_example = loss.reshape(loss.shape[0], -1).mean(axis=1)
    fine_mask = timesteps < 200
    mid_mask = (timesteps >= 200) & (timesteps < 500)
    coarse_mask = timesteps >= 500

    loss_fine = jnp.where(
        jnp.any(fine_mask),
        jnp.sum(jnp.where(fine_mask, loss_per_example, 0.0)) / jnp.maximum(1, jnp.sum(fine_mask)),
        jnp.nan,
    )
    loss_mid = jnp.where(
        jnp.any(mid_mask), jnp.sum(jnp.where(mid_mask, loss_per_example, 0.0)) / jnp.maximum(1, jnp.sum(mid_mask)), jnp.nan
    )
    loss_coarse = jnp.where(
        jnp.any(coarse_mask),
        jnp.sum(jnp.where(coarse_mask, loss_per_example, 0.0)) / jnp.maximum(1, jnp.sum(coarse_mask)),
        jnp.nan,
    )

    return loss_mean, (loss_fine, loss_mid, loss_coarse)

  def true_fn(states):
    s_high, s_low = states
    loss, high_grads = nnx.value_and_grad(functools.partial(compute_loss_high, s_high=s_high))(s_high.params)
    nan_val = jnp.array(jnp.nan, dtype=loss.dtype)

    max_grad_norm_high = jaxopt.tree_util.tree_l2_norm(high_grads)
    max_abs_grad_high = jax.tree_util.tree_reduce(
        lambda max_val, arr: jnp.maximum(max_val, jnp.max(jnp.abs(arr))), high_grads, initializer=-1.0
    )

    new_s_high = s_high.apply_gradients(grads=high_grads)

    branch_metrics = {
        "loss_high": loss,
        "loss_low": nan_val,
        "loss_fine": nan_val,
        "loss_mid": nan_val,
        "loss_coarse": nan_val,
        "max_grad_norm_low": nan_val,
        "max_grad_norm_high": max_grad_norm_high,
        "max_abs_grad_low": nan_val,
        "max_abs_grad_high": max_abs_grad_high,
    }
    return branch_metrics, new_s_high, s_low

  def false_fn(states):
    s_high, s_low = states
    (loss, (loss_fine, loss_mid, loss_coarse)), low_grads = nnx.value_and_grad(
        functools.partial(compute_loss_low, s_low=s_low), has_aux=True
    )(s_low.params)
    nan_val = jnp.array(jnp.nan, dtype=loss.dtype)

    max_grad_norm_low = jaxopt.tree_util.tree_l2_norm(low_grads)
    max_abs_grad_low = jax.tree_util.tree_reduce(
        lambda max_val, arr: jnp.maximum(max_val, jnp.max(jnp.abs(arr))), low_grads, initializer=-1.0
    )

    new_s_low = s_low.apply_gradients(grads=low_grads)

    branch_metrics = {
        "loss_high": nan_val,
        "loss_low": loss,
        "loss_fine": loss_fine,
        "loss_mid": loss_mid,
        "loss_coarse": loss_coarse,
        "max_grad_norm_low": max_grad_norm_low,
        "max_grad_norm_high": nan_val,
        "max_abs_grad_low": max_abs_grad_low,
        "max_abs_grad_high": nan_val,
    }
    return branch_metrics, s_high, new_s_low

  branch_metrics, new_state_high, new_state_low = jax.lax.cond(
      is_high_noise,
      true_fn,
      false_fn,
      operand=(state_high, state_low),
  )

  metrics = {
      "scalar": {
          "learning/loss": jnp.nan_to_num(branch_metrics["loss_high"], nan=0.0)
          + jnp.nan_to_num(branch_metrics["loss_low"], nan=0.0),
          "learning/loss_low": branch_metrics["loss_low"],
          "learning/loss_high": branch_metrics["loss_high"],
          "learning/loss_low_fine": branch_metrics["loss_fine"],
          "learning/loss_low_mid": branch_metrics["loss_mid"],
          "learning/loss_low_coarse": branch_metrics["loss_coarse"],
          "learning/max_grad_norm_low": branch_metrics["max_grad_norm_low"],
          "learning/max_grad_norm_high": branch_metrics["max_grad_norm_high"],
          "learning/max_abs_grad_low": branch_metrics["max_abs_grad_low"],
          "learning/max_abs_grad_high": branch_metrics["max_abs_grad_high"],
      },
      "scalars": {},
  }

  return new_state_low, new_state_high, scheduler_state, metrics, new_rng


def eval_step_2_2(state_low, state_high, data, rng, scheduler_state, scheduler, config):
  """Wan 2.2 dual-expert evaluation step with batched conditional execution and per-sample routing.

  Evaluation uses batched conditional execution:
    - Homogeneous low batch: skips high-noise expert entirely (0 FLOPs for high expert).
    - Homogeneous high batch: skips low-noise expert entirely (0 FLOPs for low expert).
    - Mixed-timestep batch: executes at most one batched call per expert on the batch and
      selects per-sample outputs via `jnp.where(is_high_mask, pred_high, pred_low)`.
    This avoids unrolling $B$ single-sample model calls while guaranteeing exact per-sample
    expert routing semantics and constant HLO graph complexity.
  """
  num_train_timesteps = scheduler.config.num_train_timesteps
  boundary = int(config.boundary_ratio * num_train_timesteps)

  latents = data["latents"].astype(config.weights_dtype)
  encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)
  timesteps = data["timesteps"].astype("int32")
  rng, new_rng = jax.random.split(rng, num=2)

  noise = jax.random.normal(key=new_rng, shape=latents.shape, dtype=latents.dtype)
  noisy_latents, training_target, training_weight = scheduler.apply_flow_match(noise, latents, timesteps)

  is_high = timesteps >= boundary

  def _eval_high(operands):
    l, t, e = operands
    model = nnx.merge(state_high.graphdef, state_high.params, state_high.rest_of_state)
    return model(hidden_states=l, timestep=t, encoder_hidden_states=e, deterministic=True)

  def _skip_high(operands):
    l, _, _ = operands
    return jnp.zeros_like(l)

  def _eval_low(operands):
    l, t, e = operands
    model = nnx.merge(state_low.graphdef, state_low.params, state_low.rest_of_state)
    return model(hidden_states=l, timestep=t, encoder_hidden_states=e, deterministic=True)

  def _skip_low(operands):
    l, _, _ = operands
    return jnp.zeros_like(l)

  pred_high = jax.lax.cond(
      jnp.any(is_high),
      _eval_high,
      _skip_high,
      operand=(noisy_latents, timesteps, encoder_hidden_states),
  )

  pred_low = jax.lax.cond(
      jnp.any(~is_high),
      _eval_low,
      _skip_low,
      operand=(noisy_latents, timesteps, encoder_hidden_states),
  )

  is_high_mask = jnp.reshape(is_high, (is_high.shape[0],) + (1,) * (pred_high.ndim - 1))
  model_pred = jnp.where(is_high_mask, pred_high, pred_low)

  loss = (training_target - model_pred) ** 2
  if not getattr(config, "disable_training_weights", False):
    training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
    loss = loss * training_weight
  losses = loss.reshape(loss.shape[0], -1).mean(axis=1)

  metrics = {
      "scalar": {
          "learning/eval_loss": losses,
          "learning/eval_timesteps": timesteps,
      }
  }
  return metrics, new_rng
