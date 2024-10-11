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

import datetime
import time
import logging
import os
from typing import Sequence
from functools import partial

import numpy as np
import shutil

import jax
import jax.numpy as jnp
import optax
import transformers
from absl import app
from maxdiffusion import (
    FlaxDDPMScheduler,
    FlaxStableDiffusionPipeline,
    max_logging,
    max_utils,
    pyconfig,
    mllog_utils,
    generate,
    eval,
)

from flax.linen import partitioning as nn_partitioning
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P, PositionalSharding
from transformers import set_seed
from maxdiffusion.input_pipeline.input_pipeline_interface import (
  make_pokemon_train_iterator,
  make_laion400m_train_iterator,
  get_shaped_batch
)
from maxdiffusion.models.vae_flax import FlaxDiagonalGaussianDistribution
from maxdiffusion.maxdiffusion_utils import calculate_unet_tflops

TOTAL_TRAIN_SAMPLES = 6513144
TOTAL_EVAL_SAMPLES = 30000

def compute_snr(scheduler_state, timesteps):
    alphas_cumprod = scheduler_state.common.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    alpha = sqrt_alphas_cumprod[timesteps]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def eval_at_checkpoint(
    config,
    checkpoint_number,
    unet_state,
    unet_state_mesh_shardings,
    vae_state,
    vae_state_mesh_shardings,
    pipeline,
    params,
    metrics,
    checkpoint_name):
    training_scheduler = pipeline.scheduler
    training_scheduler_state = params["scheduler"]
    images_directory = os.path.join(config.images_directory, "output")
    os.makedirs(images_directory, exist_ok=True)

    generate.run(config,  
         images_directory,        
         unet_state,
         unet_state_mesh_shardings,
         vae_state,
         vae_state_mesh_shardings,
         pipeline,
         params)

    clip, fid = eval.eval_scores(config, images_directory, checkpoint_name)
    print("clip score is :" + str(clip))
    print("fid score is : " + str(fid))
    metrics['scalar'].update({'eval/FID': fid})
    metrics['scalar'].update({'eval/CLIP': clip})
    if config.upload_images:
        max_utils.walk_and_upload_gen_images(config, images_directory, checkpoint_number)
    pipeline.scheduler = training_scheduler
    params["scheduler"] = training_scheduler_state

def get_first_step(state):
  with jax.spmd_mode('allow_all'):
    return int(state.step)

def load_next_batch(train_iter, example_batch, config):
    """Loads the next batch. Can keep reusing the same batch for performance reasons """
    if config.reuse_example_batch and example_batch is not None:
        return example_batch
    else:
        return train_iter()

def validate_train_config(config):
  """ Validates the configuration is set correctly for train.py"""

  def _validate_gcs_bucket_name(bucket_name, config_var):
    assert bucket_name, f"Please set {config_var}."
    assert len(bucket_name) > 5 and bucket_name[0:5]=='gs://', f"Erroring out, {config_var} should start with 'gs://' "

  assert config.run_name, "Erroring out, need a real run_name"
  _validate_gcs_bucket_name(config.base_output_directory, "base_output_directory")

  assert config.max_train_steps > 0 or config.num_train_epochs > 0, "You must set steps or learning_rate_schedule_steps to a positive interger."

def record_scalar_metrics(metrics, total_seconds, num_step, per_device_tflops, lr):
  """Records scalar metrics to be written to tensorboard"""
  metrics['scalar'].update({
      'perf/period_time_seconds': total_seconds
  })
  metrics['scalar'].update({
      'perf/period_step_count': num_step
  })
  metrics['scalar'].update({
      'perf/step_time_seconds': total_seconds / num_step
  })
  metrics['scalar'].update({
      'perf/per_device_tflops' : per_device_tflops
  })
  metrics['scalar'].update({
      'perf/per_device_tflops_per_sec':
          (num_step * per_device_tflops) /
          total_seconds
  })
  metrics['scalar'].update({'learning/current_learning_rate': lr })

_buffered_step = None
_buffered_metrics = None
def write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config):
  """Entry point for all metrics writing in Train's Main.
     TODO: would be better as a Class in the future (that initialized all state!)

     To avoid introducing an unnecessary dependency, we "double buffer" -- we hold
     onto the last metrics and step and only publish when we receive a new metrics and step.
     The logic is that this ensures that Jax is able to queues train_steps and we
     don't block when turning "lazy" Jax arrays into real Python numbers.
  """
  global _buffered_step, _buffered_metrics

  if _buffered_metrics is not None:
    if _buffered_step is None:
      raise ValueError(f"When writing metrics, {_buffered_step=} was none")
    write_metrics_to_tensorboard(writer, _buffered_metrics, _buffered_step, config)

    if config.metrics_file:
      max_utils.write_metrics_locally(_buffered_metrics, _buffered_step, config, local_metrics_file)

    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(_buffered_metrics, _buffered_step, config, running_gcs_metrics)

  _buffered_step = step
  _buffered_metrics = metrics

def write_metrics_to_tensorboard(writer, metrics, step, config):
  """Writes metrics to tensorboard"""
  with jax.spmd_mode('allow_all'):
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar",[]):
        writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars",[]):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    full_log = step % config.log_period == 0
    if jax.process_index() == 0:
        max_logging.log(f"step: {step}, "
            f"avg time for the period of {metrics['scalar']['perf/period_step_count']} steps: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
            f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
            f"loss: {metrics['scalar']['learning/loss']:.3f}")

    if full_log and jax.process_index() == 0:
      max_logging.log(
          f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'"
      )
      writer.flush()

def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  return total_parameters

def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x, params))

def train(config):
    rng = jax.random.PRNGKey(config.seed)

    writer = max_utils.initialize_summary_writer(config)
    if config.dataset_name is None and config.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # Setup Mesh
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if jax.process_index() == 0:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    data_sharding = jax.sharding.NamedSharding(mesh,P(*config.data_sharding))

    total_train_batch_size = config.per_device_batch_size * jax.device_count()
    if config.checkpoint_every % total_train_batch_size != 0:
        max_logging.log(f"Checkpoint at {config.checkpoint_every} samples is not evenly divisible by"
                        f" global batch size of {total_train_batch_size}. Checkpointing might not"
                        " work correctly.")

    weight_dtype = max_utils.get_dtype(config)
    flash_block_sizes = max_utils.get_flash_block_sizes(config)
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision,
        dtype=weight_dtype,
        safety_checker=None,
        feature_extractor=None,
        from_pt=config.from_pt,
        split_head_dim=config.split_head_dim,
        norm_num_groups=config.norm_num_groups,
        attention_kernel=config.attention,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
    )

    # TODO - add unit test to verify scheduler changes.
    noise_scheduler, noise_scheduler_state = max_utils.create_scheduler(config.training_scheduler, pipeline.scheduler.config, config)
    pipeline.scheduler = noise_scheduler
    params["scheduler"] = noise_scheduler_state

    sharding = PositionalSharding(devices_array).replicate()
    partial_device_put_replicated = partial(max_utils.device_put_replicated, sharding=sharding)
    params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])

    # Optimization
    if config.scale_lr:
        config.learning_rate = config.learning_rate * total_train_batch_size

    learning_rate_scheduler = max_utils.create_learning_rate_schedule(config)

    adamw = optax.adamw(
        learning_rate=learning_rate_scheduler,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
    )

    (unet_state,
    unet_state_mesh_shardings,
    vae_state, vae_state_mesh_shardings) = max_utils.get_states(mesh,
                                                                adamw, rng, config,
                                                                pipeline, params["unet"],
                                                                params["vae"], training=True)

    per_device_tflops = calculate_unet_tflops(config, pipeline, rng, train=True)
    max_logging.log(f"Per train step, estimated total TFLOPs will be {per_device_tflops:.2f}")

    params["unet"] = None
    params["vae"] = None

    if config.dataset_name == "lambdalabs/pokemon-blip-captions":
        data_iterator = make_pokemon_train_iterator(
           config,
           mesh,
           total_train_batch_size,
           pipeline,
           params,
           rng
        )
    else:
        data_iterator = make_laion400m_train_iterator(
           config, mesh, total_train_batch_size
        )

    if not config.eval_at_checkpoint:
        vae_state = None
        vae_state_mesh_shardings = None
        params["vae"] = None

    # Initialize our training
    _, train_rngs = jax.random.split(rng)

    def train_step(unet_state, batch, train_rng, cache_latents_text_encoder_outputs):
        _, gen_dummy_rng = jax.random.split(train_rng)
        sample_rng, new_train_rng = jax.random.split(gen_dummy_rng)

        def compute_loss(unet_params):
            
            if cache_latents_text_encoder_outputs:
               raise Exception(f"caching latents and text encoder outputs is not supported")
            else:
                # Convert moments to latent space
                latents = FlaxDiagonalGaussianDistribution(batch["moments"]).sample(sample_rng)
                # (NHWC) -> (NCHW)
                latents = jnp.transpose(latents, (0, 3, 1, 2))
                latents = latents * pipeline.vae.config.scaling_factor
                # Get the text embedding for conditioning
                encoder_hidden_states = batch["clip_embeddings"]

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise_rng, offset_rng, peturbation_rng = jax.random.split(noise_rng, num=3)
            noise = jax.random.normal(noise_rng, latents.shape)

            # noise offset
            if config.noise_offset > 0:
                noise += config.noise_offset * jax.random.normal(offset_rng, (latents.shape[0], latents.shape[1], 1, 1))

            # input perturbation
            if config.input_peturbation > 0:
                new_noise = noise + config.input_peturbation * jax.random.normal(peturbation_rng, noise.shape)

            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if config.input_peturbation > 0:
                noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, new_noise, timesteps)
            else:
               noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

            # Predict the noise residual and compute loss
            model_pred = pipeline.unet.apply(
                {"params": unet_params}, noisy_latents, timesteps, encoder_hidden_states, train=True
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            elif noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            # TODO  (Optional): L = lambda * (target - model_pred) ** 2 from
            # https://arxiv.org/pdf/2305.08891 3.2 (12)
            loss = (target - model_pred) ** 2

            # snr
            if config.snr_gamma > 0:
                snr = jnp.array(compute_snr(noise_scheduler_state, timesteps))
                snr_loss_weights = jnp.where(snr < config.snr_gamma, snr, jnp.ones_like(snr) * config.snr_gamma)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    snr_loss_weights = snr_loss_weights / (snr + 1)
                elif noise_scheduler.config.prediction_type == "epsilon":
                    snr_loss_weights = snr_loss_weights / snr
                loss = loss * snr_loss_weights[:, None, None, None]
                
            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, raw_grad = grad_fn(unet_state.params)

        if config.max_grad_norm > 0:
            grad, _ = optax.clip_by_global_norm(config.max_grad_norm).update(raw_grad, unet_state, None)
        else:
            grad = raw_grad

        new_state = unet_state.apply_gradients(grads=grad)
        #metrics = {'scalar' : {'learning/loss' : loss, 'learning/grad_norm' : max_utils.l2norm_pytree(grad)}, 'scalars': {}}
        #metrics = {'scalar' : {'learning/loss' : loss}, 'scalars': {}}
        return new_state, loss, new_train_rng

    num_model_parameters = calculate_num_params_from_pytree(unet_state.params)
    max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")

    my_data_sharding = {'clip_embeddings': data_sharding, 'moments': data_sharding}
    dummy_batch = get_shaped_batch(config, pipeline)
    if config.pre_compile:
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            p_train_step_lower = jax.jit(
                partial(train_step, cache_latents_text_encoder_outputs=config.cache_latents_text_encoder_outputs),
                in_shardings=(unet_state_mesh_shardings, my_data_sharding, None),
                out_shardings=(unet_state_mesh_shardings, None, None),
                donate_argnums=(0,)
            ).lower(unet_state, dummy_batch, train_rngs)
        p_train_step = p_train_step_lower.compile()
        #host_id = jax.process_index()
        #all_host_ids = jax.experimental.multihost_utils.process_allgather(host_id)

        p_learning_rate_scheduler = jax.jit(learning_rate_scheduler).lower(0)
        p_learning_rate_scheduler = p_learning_rate_scheduler.compile()

        host_id = jax.process_index()
        all_host_ids = jax.experimental.multihost_utils.process_allgather(host_id)     
    else:
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            p_train_step = jax.jit(
                partial(train_step, cache_latents_text_encoder_outputs=config.cache_latents_text_encoder_outputs),
                in_shardings=(unet_state_mesh_shardings, my_data_sharding, None),
                out_shardings=(unet_state_mesh_shardings, None, None),
                donate_argnums=(0,)
            )
    # Train!
    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
    max_utils.add_config_to_summary_writer(config, writer)

    if jax.process_index() == 0:
        max_logging.log("***** Running training *****")
        max_logging.log(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
        max_logging.log(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
        max_logging.log(f"  Total optimization steps = {config.max_train_steps}")
        max_logging.log(f"  Scheduler config = {pipeline.scheduler.config}")
        if config.noise_offset > 0:
            max_logging.log(f"  Noise offset = {config.noise_offset}")
        if config.input_peturbation > 0:
            max_logging.log(f"  Input Peturbation = {config.input_peturbation}")
        if config.snr_gamma > 0:
            max_logging.log(f"  SNR Gamma = {config.snr_gamma}")
    last_step_completion = datetime.datetime.now()

    local_metrics_file = open(config.metrics_file, 'a', encoding="utf8") if config.metrics_file else None
    running_gcs_metrics = [] if config.gcs_metrics else None
    example_batch = None

    first_profiling_step = config.skip_first_n_steps_for_profiler
    if config.enable_profiler and first_profiling_step >= config.max_train_steps:
       raise ValueError("Profiling requested but initial profiling step set past training final step")
    last_profiling_step = np.clip(first_profiling_step + config.profiler_steps -1, first_profiling_step, config.max_train_steps - 1)
    # ======================== Training ================================
    # train

    start_step = get_first_step(unet_state)
    mllog_utils.train_init_print(config)
    mllog_utils.train_init_stop(config)
    mllog_utils.train_run_start(config)
    mllog_utils.train_step_start(config, start_step, samples_count=0)
    # for checkpointing
    eval_checkpoints = []
    start_time = time.time()
    last_log_step_num = 0
    for step in np.arange(start_step, config.max_train_steps):
        example_batch = load_next_batch(data_iterator, example_batch, config)
        unet_state, loss, train_rngs = p_train_step(unet_state,
                                                            example_batch,
                                                            train_rngs)
        
        step_num = step + 1
        samples_count = total_train_batch_size * step_num

        if step_num % config.metrics_period == 0 or step_num == config.max_train_steps:

            if config.write_metrics:
                train_metric = {'scalar' : {'learning/loss' : loss}, 'scalars': {}}
    
                new_time = datetime.datetime.now()
                step_time_delta = float((new_time - last_step_completion).total_seconds())
                step_num_delta = step_num - last_log_step_num
                # record metrics of current period
                record_scalar_metrics(train_metric, step_time_delta, step_num_delta, per_device_tflops, p_learning_rate_scheduler(step))
                # print metrics of previous period
                # using global vars _buffered_step, _buffered_metrics
                if _buffered_step is None:
                    step_num_delta = step_num
                    _buffered_step_num = step_num
                else:
                    step_num_delta = step - _buffered_step
                    _buffered_step_num = _buffered_step + 1
                _buffered_sample_count = total_train_batch_size * _buffered_step_num
                if jax.process_index() == 0:
                    max_logging.log(f"At step {step}, log metrics of step {_buffered_step}")
                    # convert the jax array to a numpy array for mllog JSON encoding
                    loss = np.asarray(_buffered_metrics['scalar']['learning/loss'])
                    lr = np.asarray(_buffered_metrics['scalar']['learning/current_learning_rate'])

                    mllog_utils.maybe_train_step_log(config, start_step, _buffered_step_num, _buffered_sample_count, loss, lr)
                write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, config)
                
                last_step_completion = new_time
                last_log_step_num = step_num        
            else:
                if jax.process_index() == 0:
                    # convert the jax array to a numpy array for mllog JSON encoding
                    loss = np.asarray(loss)
                    lr = np.asarray(p_learning_rate_scheduler(step))
                    mllog_utils.maybe_train_step_log(config, start_step, step_num, samples_count, loss, lr)

        if step != 0 and samples_count % config.checkpoint_every == 0:
            checkpoint_name = f"{step_num=}-{samples_count=}"
            if config.eval_at_checkpoint:
                eval_at_checkpoint(config,
                    f"{str(step * total_train_batch_size)}",
                    unet_state, unet_state_mesh_shardings,
                    vae_state,
                    vae_state_mesh_shardings,
                    pipeline, params, train_metric,
                    checkpoint_name)
            if samples_count >= config.start_step_to_checkpoint:
                eval_checkpoints.append(max_utils.save_checkpoint(pipeline,
                                        unet_state,
                                        config,
                                        os.path.join(config.checkpoint_dir, checkpoint_name)))
                mllog_utils.train_checkpoint_step_log(step_num)
        # Start profiling at end of first step to avoid compilation.
        # Move before for loop to include.
        if step == first_profiling_step:
            max_utils.activate_profiler(config)
        if step == last_profiling_step:
            max_utils.deactivate_profiler(config)

    if config.write_metrics:
        # log the last metrics_period
        if jax.process_index() == 0:
            max_logging.log(f"Training loop finished, log metrics of step {_buffered_step}")
            # convert the jax array to a numpy array for mllog JSON encoding
            loss = np.asarray(_buffered_metrics['scalar']['learning/loss'])
            lr = np.asarray(_buffered_metrics['scalar']['learning/current_learning_rate'])

        mllog_utils.maybe_train_step_log(config, start_step, config.max_train_steps, config.max_train_steps*total_train_batch_size, loss, lr)
        write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, config.max_train_steps - config.metrics_period, config)
    
    totaltime = time.time() - start_time
    steptime = totaltime / (config.max_train_steps - start_step) * 1000
    max_logging.log(f"Total time for {config.max_train_steps} steps: {totaltime} s. Avg step time: {steptime} ms")
    del pipeline
    del params
    del unet_state
    del vae_state

    config.train_new_unet = False
    for checkpoint in eval_checkpoints:
        config.unet_checkpoint = checkpoint
        checkpoint_name = checkpoint.split("/")[-1]
        images_directory = os.path.join(config.images_directory, "output")
        if os.path.isdir(images_directory):
            shutil.rmtree(images_directory)
            max_logging.logging(f"remove image folders under exist {images_directory}")
        os.makedirs(images_directory, exist_ok=True)

        generate.run(config, images_directory)

        clip, fid = eval.eval_scores(config, images_directory, checkpoint_name)
        if config.write_metrics:
            if jax.process_index() == 0:
                num_samples = checkpoint_name.split("samples_count=")[-1]
                writer.add_scalar('eval/FID', np.array(fid), int(num_samples))
                writer.add_scalar('eval/CLIP', np.array(clip), int(num_samples))

                if clip >= 0.15 and fid <= 90:
                    writer.add_scalar('mlperf_convergence_samples', int(num_samples))
        shutil.rmtree(images_directory)
    max_utils.close_summary_writer(writer)

def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    mllog_utils.train_init_start(config)
    max_logging.log(f"Found {jax.device_count()} devices.")
    cc.initialize_cache(os.path.expanduser("~/jax_cache"))
    validate_train_config(config)
    train(config)
if __name__ == "__main__":
    app.run(main)
