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

import os
import datetime
import functools
from pprint import pprint
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P
from flax import nnx
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from flax.linen import partitioning as nn_partitioning
from maxdiffusion import max_utils, max_logging, train_utils
from maxdiffusion.checkpointing.wan_checkpointer import (WanCheckpointer, WAN_CHECKPOINT)
from maxdiffusion.input_pipeline.input_pipeline_interface import (make_data_iterator)
from maxdiffusion.generate_wan import run as generate_wan
from maxdiffusion.generate_wan import inference_generate_video
from maxdiffusion.train_utils import (_tensorboard_writer_worker, load_next_batch, _metrics_queue)
from maxdiffusion.video_processor import VideoProcessor
from maxdiffusion.utils import load_video
from skimage.metrics import structural_similarity as ssim
from flax.training import train_state
from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline
from jax.experimental import multihost_utils


class TrainState(train_state.TrainState):
  graphdef: nnx.GraphDef
  rest_of_state: nnx.State


def _to_array(x):
  if not isinstance(x, jax.Array):
    x = jnp.asarray(x)
  return x


def generate_sample(config, pipeline, filename_prefix):
  """
  Generates a video to validate training did not corrupt the model
  """
  if not hasattr(pipeline, "vae"):
    wan_vae, vae_cache = WanPipeline.load_vae(
        pipeline.mesh.devices, pipeline.mesh, nnx.Rngs(jax.random.key(config.seed)), config
    )
    pipeline.vae = wan_vae
    pipeline.vae_cache = vae_cache
  return generate_wan(config, pipeline, filename_prefix)


def print_ssim(pretrained_video_path, posttrained_video_path):
  video_processor = VideoProcessor()
  pretrained_video = load_video(pretrained_video_path[0])
  pretrained_video = video_processor.preprocess_video(pretrained_video)
  pretrained_video = np.array(pretrained_video)
  pretrained_video = np.transpose(pretrained_video, (0, 2, 3, 4, 1))
  pretrained_video = np.uint8((pretrained_video + 1) * 255 / 2)

  posttrained_video = load_video(posttrained_video_path[0])
  posttrained_video = video_processor.preprocess_video(posttrained_video)
  posttrained_video = np.array(posttrained_video)
  posttrained_video = np.transpose(posttrained_video, (0, 2, 3, 4, 1))
  posttrained_video = np.uint8((posttrained_video + 1) * 255 / 2)

  ssim_compare = ssim(pretrained_video[0], posttrained_video[0], multichannel=True, channel_axis=-1, data_range=255)

  max_logging.log(f"SSIM score after training is {ssim_compare}")


class WanTrainer(WanCheckpointer):

  def __init__(self, config):
    WanCheckpointer.__init__(self, config, WAN_CHECKPOINT)
    if config.train_text_encoder:
      raise ValueError("this script currently doesn't support training text_encoders")

  def post_training_steps(self, pipeline, params, train_states, msg=""):
    pass

  def create_scheduler(self):
    """Creates and initializes the Flow Match scheduler for training."""
    noise_scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(noise_scheduler_state, num_inference_steps=1000, training=True)
    return noise_scheduler, noise_scheduler_state

  @staticmethod
  def calculate_tflops(pipeline):

    maxdiffusion_config = pipeline.config
    # Model configuration
    height = pipeline.config.height
    width = pipeline.config.width
    num_frames = pipeline.config.num_frames

    # Transformer dimensions
    transformer_config = pipeline.transformer.config
    num_layers = transformer_config.num_layers
    heads = pipeline.transformer.config.num_attention_heads
    head_dim = pipeline.transformer.config.attention_head_dim
    ffn_dim = transformer_config.ffn_dim
    seq_len = int(((height / 8) * (width / 8) * ((num_frames - 1) // pipeline.vae_scale_factor_temporal + 1)) / 4)
    text_encoder_dim = 512
    # Attention FLOPS
    # Self
    self_attn_qkv_proj_flops = 3 * (2 * seq_len * (heads * head_dim) ** 2)
    self_attn_qk_v_flops = 2 * (2 * seq_len**2 * (heads * head_dim))
    # Cross
    cross_attn_kv_proj_flops = 3 * (2 * text_encoder_dim * (heads * head_dim) ** 2)
    cross_attn_q_proj_flops = 1 * (2 * seq_len * (heads * head_dim) ** 2)
    cross_attention_qk_v_flops = 2 * (2 * seq_len * text_encoder_dim * (heads * head_dim))

    # Output_projection from attention
    attn_output_proj_flops = 2 * (2 * seq_len * (heads * head_dim) ** 2)

    total_attn_flops = (
        self_attn_qkv_proj_flops
        + self_attn_qk_v_flops
        + cross_attn_kv_proj_flops
        + cross_attn_q_proj_flops
        + cross_attention_qk_v_flops
        + attn_output_proj_flops
    )

    # FFN
    ffn_flops = 2 * (2 * seq_len * (heads * head_dim) * ffn_dim)

    flops_per_block = total_attn_flops + ffn_flops

    total_transformer_flops = flops_per_block * num_layers

    tflops = maxdiffusion_config.per_device_batch_size * total_transformer_flops / 1e12
    train_tflops = 3 * tflops

    max_logging.log(f"Calculated TFLOPs per pass: {train_tflops:.4f}")
    return train_tflops, total_attn_flops, seq_len

  def get_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    data_sharding = {"latents": data_sharding, "encoder_hidden_states": data_sharding}
    return data_sharding

  def get_eval_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    data_sharding = {"latents": data_sharding, "encoder_hidden_states": data_sharding, "timesteps": data_sharding}
    return data_sharding

  def load_dataset(self, mesh, is_training=True):
    # Stages of training as described in the Wan 2.1 paper - https://arxiv.org/pdf/2503.20314
    # Image pre-training - txt2img 256px
    # Image-video joint training - stage 1. 256 px images and 192px 5 sec videos at fps=16
    # Image-video joint training - stage 2. 480px images and 480px 5 sec videos at fps=16
    # Image-video joint training - stage final. 720px images and 720px 5 sec videos at fps=16
    # prompt embeds shape: (1, 512, 4096)
    # For now, we will pass the same latents over and over
    # TODO - create a dataset
    config = self.config
    if config.dataset_type != "tfrecord" and not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "Wan 2.1 training only supports config.dataset_type set to tfrecords and config.cache_latents_text_encoder_outputs set to True"
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

  def start_training(self):

    pipeline, opt_state, step = self.load_checkpoint()
    restore_args = {}
    if opt_state and step:
      restore_args = {"opt_state": opt_state, "step": step}
      del opt_state
    if self.config.enable_ssim:
      # Generate a sample before training to compare against generated sample after training.
      pretrained_video_path = generate_sample(self.config, pipeline, filename_prefix="pre-training-")

    if self.config.eval_every == -1 or (not self.config.enable_generate_video_for_eval):
      # save some memory.
      del pipeline.vae
      del pipeline.vae_cache

    mesh = pipeline.mesh
    train_data_iterator = self.load_dataset(mesh, is_training=True)

    # Load FlowMatch scheduler
    scheduler, scheduler_state = self.create_scheduler()
    pipeline.scheduler = scheduler
    pipeline.scheduler_state = scheduler_state
    optimizer, learning_rate_scheduler = self._create_optimizer(pipeline.transformer, self.config, 1e-5)
    # Returns pipeline with trained transformer state
    pipeline = self.training_loop(pipeline, optimizer, learning_rate_scheduler, train_data_iterator, restore_args)

    if self.config.enable_ssim:
      posttrained_video_path = generate_sample(self.config, pipeline, filename_prefix="post-training-")
      print_ssim(pretrained_video_path, posttrained_video_path)

  def eval(self, mesh, eval_rng_key, step, p_eval_step, state, scheduler_state, writer):
    eval_data_iterator = self.load_dataset(mesh, is_training=False)
    eval_rng = eval_rng_key
    eval_losses_by_timestep = {}
    # Loop indefinitely until the iterator is exhausted
    while True:
      try:
        eval_start_time = datetime.datetime.now()
        eval_batch = load_next_batch(eval_data_iterator, None, self.config)
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
          metrics, eval_rng = p_eval_step(state, eval_batch, eval_rng, scheduler_state)
          metrics["scalar"]["learning/eval_loss"].block_until_ready()
        losses = metrics["scalar"]["learning/eval_loss"]
        timesteps = eval_batch["timesteps"]
        gathered_losses = multihost_utils.process_allgather(losses, tiled=True)
        gathered_losses = jax.device_get(gathered_losses)
        gathered_timesteps = multihost_utils.process_allgather(timesteps, tiled=True)
        gathered_timesteps = jax.device_get(gathered_timesteps)
        if jax.process_index() == 0:
          for t, l in zip(gathered_timesteps.flatten(), gathered_losses.flatten()):
            timestep = int(t)
            if timestep not in eval_losses_by_timestep:
              eval_losses_by_timestep[timestep] = []
            eval_losses_by_timestep[timestep].append(l)
          eval_end_time = datetime.datetime.now()
          eval_duration = eval_end_time - eval_start_time
          max_logging.log(f"Eval time: {eval_duration.total_seconds():.2f} seconds.")
      except StopIteration:
        # This block is executed when the iterator has no more data
        break
    # Check if any evaluation was actually performed
    if eval_losses_by_timestep and jax.process_index() == 0:
      mean_per_timestep = []
      if jax.process_index() == 0:
        max_logging.log(f"Step {step}, calculating mean loss per timestep...")
      for timestep, losses in sorted(eval_losses_by_timestep.items()):
        losses = jnp.array(losses)
        losses = losses[: min(self.config.eval_max_number_of_samples_in_bucket, len(losses))]
        mean_loss = jnp.mean(losses)
        max_logging.log(f"  Mean eval loss for timestep {timestep}: {mean_loss:.4f}")
        mean_per_timestep.append(mean_loss)
      final_eval_loss = jnp.mean(jnp.array(mean_per_timestep))
      max_logging.log(f"Step {step}, Final Average Eval loss: {final_eval_loss:.4f}")
      if writer:
        writer.add_scalar("learning/eval_loss", final_eval_loss, step)

  def training_loop(self, pipeline, optimizer, learning_rate_scheduler, train_data_iterator, restore_args: dict = {}):
    mesh = pipeline.mesh
    graphdef, params, rest_of_state = nnx.split(pipeline.transformer, nnx.Param, ...)

    with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      state = TrainState.create(
          apply_fn=graphdef.apply, params=params, tx=optimizer, graphdef=graphdef, rest_of_state=rest_of_state
      )
      if restore_args:
        step = restore_args.get("step", 0)
        max_logging.log(f"Restoring optimizer and resuming from step {step}")
        state.replace(opt_state=restore_args.get("opt_state"), step=restore_args.get("step", 0))
        del restore_args["opt_state"]
        del optimizer
      state = jax.tree.map(_to_array, state)
      state_spec = nnx.get_partition_spec(state)
      state = jax.lax.with_sharding_constraint(state, state_spec)
      state_shardings = nnx.get_named_sharding(state, mesh)
      if jax.process_index() == 0 and restore_args:
        max_logging.log("--- Optimizer State Sharding Spec (opt_state) ---")
        pretty_string = pprint.pformat(state_spec.opt_state, indent=4, width=60)
        max_logging.log(pretty_string)
        max_logging.log("------------------------------------------------")
    max_utils.delete_pytree(params)
    data_shardings = self.get_data_shardings(mesh)
    eval_data_shardings = self.get_eval_data_shardings(mesh)

    writer = max_utils.initialize_summary_writer(self.config)
    writer_thread = threading.Thread(target=_tensorboard_writer_worker, args=(writer, self.config), daemon=True)
    writer_thread.start()

    num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ.get("LIBTPU_INIT_ARGS", ""), writer)
    max_utils.add_config_to_summary_writer(self.config, writer)

    if jax.process_index() == 0:
      max_logging.log("***** Running training *****")
      max_logging.log(f"  Instantaneous batch size per device = {self.config.per_device_batch_size}")
      max_logging.log(f"  Total train batch size (w. parallel & distributed) = {self.config.global_batch_size_to_train_on}")
      max_logging.log(f"  Total optimization steps = {self.config.max_train_steps}")

    p_train_step = jax.jit(
        functools.partial(train_step, scheduler=pipeline.scheduler, config=self.config),
        in_shardings=(state_shardings, data_shardings, None, None),
        out_shardings=(state_shardings, None, None, None),
        donate_argnums=(0,),
    )
    p_eval_step = jax.jit(
        functools.partial(eval_step, scheduler=pipeline.scheduler, config=self.config),
        in_shardings=(state_shardings, eval_data_shardings, None, None),
        out_shardings=(None, None),
    )

    rng = jax.random.key(self.config.seed)
    rng, eval_rng_key = jax.random.split(rng)
    start_step = 0
    last_step_completion = datetime.datetime.now()
    local_metrics_file = open(self.config.metrics_file, "a", encoding="utf8") if self.config.metrics_file else None
    running_gcs_metrics = [] if self.config.gcs_metrics else None
    first_profiling_step = self.config.skip_first_n_steps_for_profiler
    if self.config.enable_profiler and first_profiling_step >= self.config.max_train_steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
    last_profiling_step = np.clip(
        first_profiling_step + self.config.profiler_steps - 1, first_profiling_step, self.config.max_train_steps - 1
    )
    if restore_args.get("step", 0):
      max_logging.log(f"Resuming training from step {step}")
    start_step = restore_args.get("step", 0)
    per_device_tflops, _, _ = WanTrainer.calculate_tflops(pipeline)
    scheduler_state = pipeline.scheduler_state
    example_batch = load_next_batch(train_data_iterator, None, self.config)

    with ThreadPoolExecutor(max_workers=1) as executor:
      for step in np.arange(start_step, self.config.max_train_steps):
        if self.config.enable_profiler and step == first_profiling_step:
          max_utils.activate_profiler(self.config)
        start_step_time = datetime.datetime.now()
        next_batch_future = executor.submit(load_next_batch, train_data_iterator, example_batch, self.config)
        with jax.profiler.StepTraceAnnotation("train", step_num=step), pipeline.mesh, nn_partitioning.axis_rules(
            self.config.logical_axis_rules
        ):
          state, scheduler_state, train_metric, rng = p_train_step(state, example_batch, rng, scheduler_state)
          train_metric["scalar"]["learning/loss"].block_until_ready()
        last_step_completion = datetime.datetime.now()

        if self.config.enable_profiler and step == last_profiling_step:
          max_utils.deactivate_profiler(self.config)

        train_utils.record_scalar_metrics(
            train_metric, last_step_completion - start_step_time, per_device_tflops, learning_rate_scheduler(step)
        )
        if self.config.write_metrics:
          train_utils.write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config)

        if self.config.eval_every > 0 and (step + 1) % self.config.eval_every == 0:
          if self.config.enable_generate_video_for_eval:
            pipeline.transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
            inference_generate_video(self.config, pipeline, filename_prefix=f"{step+1}-train_steps-")
          # Re-create the iterator each time you start evaluation to reset it
          # This assumes your data loading logic can be called to get a fresh iterator.
          self.eval(mesh, eval_rng_key, step, p_eval_step, state, scheduler_state, writer)

        example_batch = next_batch_future.result()
        if step != 0 and self.config.checkpoint_every != -1 and step % self.config.checkpoint_every == 0:
          max_logging.log(f"Saving checkpoint for step {step}")
          if self.config.save_optimizer:
            self.save_checkpoint(step, pipeline, state)
          else:
            self.save_checkpoint(step, pipeline, state.params)

      _metrics_queue.put(None)
      writer_thread.join()
      if writer:
        writer.flush()
      if self.config.save_final_checkpoint:
        max_logging.log(f"Saving final checkpoint for step {step}")
        self.save_checkpoint(self.config.max_train_steps - 1, pipeline, state.params)
        self.checkpoint_manager.wait_until_finished()
      # load new state for trained tranformer
      pipeline.transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
      return pipeline


def train_step(state, data, rng, scheduler_state, scheduler, config):
  return step_optimizer(state, data, rng, scheduler_state, scheduler, config)


def step_optimizer(state, data, rng, scheduler_state, scheduler, config):
  _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

  for k, v in data.items():
    data[k] = v[: config.global_batch_size_to_train_on, :]

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)
    bsz = latents.shape[0]
    timesteps = jax.random.randint(
        timestep_rng,
        (bsz,),
        0,
        scheduler.config.num_train_timesteps,
    )
    noise = jax.random.normal(key=new_rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents = scheduler.add_noise(scheduler_state, latents, noise, timesteps)

    model_pred = model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        deterministic=False,
        rngs=nnx.Rngs(dropout_rng),
    )

    training_target = scheduler.training_target(latents, noise, timesteps)
    training_weight = jnp.expand_dims(scheduler.training_weight(scheduler_state, timesteps), axis=(1, 2, 3, 4))
    loss = (training_target - model_pred) ** 2
    loss = loss * training_weight
    loss = jnp.mean(loss)

    return loss

  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)
  metrics = {"scalar": {"learning/loss": loss}, "scalars": {}}
  return new_state, scheduler_state, metrics, new_rng


def eval_step(state, data, rng, scheduler_state, scheduler, config):
  """
  Computes the evaluation loss for a single batch without updating model weights.
  """

  # The loss function logic is identical to training. We are evaluating the model's
  # ability to perform its core training objective (e.g., denoising).
  @jax.jit
  def loss_fn(params, latents, encoder_hidden_states, timesteps, rng):
    # Reconstruct the model from its definition and parameters
    model = nnx.merge(state.graphdef, params, state.rest_of_state)

    noise = jax.random.normal(key=rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents = scheduler.add_noise(scheduler_state, latents, noise, timesteps)

    # Get the model's prediction
    model_pred = model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        deterministic=True,
    )

    # Calculate the loss against the target
    training_target = scheduler.training_target(latents, noise, timesteps)
    training_weight = jnp.expand_dims(scheduler.training_weight(scheduler_state, timesteps), axis=(1, 2, 3, 4))
    loss = (training_target - model_pred) ** 2
    loss = loss * training_weight
    # Calculate the mean loss per sample across all non-batch dimensions.
    loss = loss.reshape(loss.shape[0], -1).mean(axis=1)

    return loss

  # --- Key Difference from train_step ---
  # Directly compute the loss without calculating gradients.
  # The model's state.params are used but not updated.
  # TODO(coolkp): Explore optimizing the creation of PRNGs in a vmap or statically outside of the loop
  bs = len(data["latents"])
  single_batch_size = config.global_batch_size_to_train_on
  losses = jnp.zeros(bs)
  for i in range(0, bs, single_batch_size):
    start = i
    end = min(i + single_batch_size, bs)
    latents = data["latents"][start:end, :].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"][start:end, :].astype(config.weights_dtype)
    timesteps = data["timesteps"][start:end].astype("int64")
    _, new_rng = jax.random.split(rng, num=2)
    loss = loss_fn(state.params, latents, encoder_hidden_states, timesteps, new_rng)
    losses = losses.at[start:end].set(loss)

  # Structure the metrics for logging and aggregation
  metrics = {"scalar": {"learning/eval_loss": losses}}

  # Return the computed metrics and the new RNG key for the next eval step
  return metrics, new_rng
