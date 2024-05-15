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
import logging
import os
from typing import Sequence
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import optax
import transformers
from absl import app
from maxdiffusion import (
    FlaxDDPMScheduler,
    FlaxStableDiffusionXLPipeline,
    FlaxAutoencoderKL,
    max_logging,
    max_utils,
    pyconfig,
    mllog_utils,
)

from transformers import FlaxCLIPTextModel, FlaxCLIPTextModelWithProjection

from maxdiffusion.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from flax.linen import partitioning as nn_partitioning
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P, PositionalSharding
from transformers import CLIPImageProcessor, set_seed

from maxdiffusion.input_pipeline.input_pipeline_interface import (
  make_pokemon_train_iterator,
  make_laion400m_train_iterator
)

from maxdiffusion.maxdiffusion_utils import (
  vae_apply,
  transform_images,
  get_add_time_ids
)

def encode_xl(input_ids, text_encoders, text_encoder_params):
  te_1_inputs = input_ids[:, 0, :]
  te_2_inputs = input_ids[:, 1, :]

  prompt_embeds = text_encoders[0](
    te_1_inputs, params=text_encoder_params[0], output_hidden_states=True
  )
  prompt_embeds = prompt_embeds["hidden_states"][-2]

  prompt_embeds_2_out = text_encoders[1](
    te_2_inputs, params=text_encoder_params[1], output_hidden_states=True   
  )
  prompt_embeds_2 = prompt_embeds_2_out["hidden_states"][-2]
  text_embeds = prompt_embeds_2_out["text_embeds"]
  prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)

  return prompt_embeds, text_embeds


def tokenize_captions_xl(examples, caption_column, tokenizers, p_encode=None):
  inputs = []
  captions = list(examples[caption_column])
  for _tokenizer in tokenizers:
     text_inputs = _tokenizer(
        captions,
        padding="max_length",
        max_length=_tokenizer.model_max_length,
        truncation=True,
        return_tensors="np"
     )
     inputs.append(text_inputs.input_ids)
  inputs = np.stack(inputs,axis=1)

  if p_encode:
    prompt_embeds, text_embeds = p_encode(inputs) 
    examples["prompt_embeds"] = prompt_embeds
    examples["text_embeds"] = text_embeds
  examples["input_ids"] = inputs

  return examples

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

def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr):
  """Records scalar metrics to be written to tensorboard"""
  metrics['scalar'].update({
      'perf/step_time_seconds': step_time_delta.total_seconds()
  })
  metrics['scalar'].update({
      'perf/per_device_tflops' : per_device_tflops
  })
  metrics['scalar'].update({
      'perf/per_device_tflops_per_sec':
          per_device_tflops /
          step_time_delta.total_seconds()
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
        max_logging.log(f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
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

    weight_dtype = max_utils.get_dtype(config)
    flash_block_sizes = max_utils.get_flash_block_sizes(config)
    pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
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
    params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)

    noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(config.pretrained_model_name_or_path,
        revision=config.revision, subfolder="scheduler", dtype=jnp.float32)

    pipeline.scheduler = noise_scheduler
    params["scheduler"] = noise_scheduler_state

    sharding = PositionalSharding(devices_array).replicate()
    partial_device_put_replicated = partial(max_utils.device_put_replicated, sharding=sharding)
    params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])
    params["text_encoder_2"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder_2"])
    # Optimization
    if config.scale_lr:
        config.learning_rate = config.learning_rate * total_train_batch_size

    learning_rate_scheduler = max_utils.create_learning_rate_schedule(config)

    tx = optax.adamw(
        learning_rate=learning_rate_scheduler,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
    )

    (unet_state,
    unet_state_mesh_shardings,
    vae_state, vae_state_mesh_shardings) = max_utils.get_states(mesh,
                                                                tx, rng, config,
                                                                pipeline, params["unet"],
                                                                params["vae"], training=True)

    per_device_tflops = max_utils.calculate_training_tflops(pipeline, unet_state.params, config)
    max_logging.log(f"Per train step, estimated total TFLOPs will be {per_device_tflops:.2f}")

    if config.dataset_name == "diffusers/pokemon-gpt4-captions":
        p_encode = None
        p_vae_apply = None
        if config.cache_latents_text_encoder_outputs:
          p_encode = jax.jit(partial(encode_xl,
                              text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2],
                              text_encoder_params=[params["text_encoder"], params["text_encoder_2"]]))
          p_vae_apply = jax.jit(partial(vae_apply, vae=pipeline.vae, vae_params=params["vae"]))
        
        tokenize_fn = partial(tokenize_captions_xl,
                          caption_column=config.caption_column,
                          tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2],
                          p_encode=p_encode)
        image_transforms_fn = partial(transform_images,
                                  image_column=config.image_column,
                                  image_resolution=config.resolution,
                                  rng=rng,
                                  global_batch_size=total_train_batch_size,
                                  p_vae_apply=p_vae_apply)
        
        data_iterator = make_pokemon_train_iterator(
           config,
           mesh,
           total_train_batch_size,
           tokenize_fn,
           image_transforms_fn
        )
    else:
        data_iterator = make_laion400m_train_iterator(
           config, mesh, total_train_batch_size
        )

    if config.cache_latents_text_encoder_outputs:
       vae_state = None
       vae_state_mesh_shardings = None
       pipeline.vae = None
       params["vae"] = None
       pipeline.text_encoder = None
       params["text_encoder"] = None
       pipeline.text_encoder_2 = None
       params["text_encoder_2"] = None

    # Initialize our training
    _, train_rngs = jax.random.split(rng)

    def train_step(unet_state, vae_state, batch, train_rng, cache_latents_text_encoder_outputs):
        _, gen_dummy_rng = jax.random.split(train_rng)
        sample_rng, new_train_rng = jax.random.split(gen_dummy_rng)
        def compute_loss(unet_params):

            if cache_latents_text_encoder_outputs:
                latents = batch["pixel_values"]
                prompt_embeds = batch["prompt_embeds"]
                text_embeds = batch["text_embeds"]
            else:
                raise ValueError(f"cache_latents_text_encoder_outputs = False currently not supported!")

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
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
            noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)
            
            # TODO : @jfacevedo - support cropping.
            add_time_ids = get_add_time_ids(
              (config.resolution, config.resolution),
              (0, 0),
              (config.resolution, config.resolution),
              prompt_embeds.shape[0],
              dtype=prompt_embeds.dtype
            )
            unet_added_conditions = {"time_ids" : add_time_ids, "text_embeds" : text_embeds}

            # Predict the noise residual and compute loss
            model_pred = pipeline.unet.apply(
                {"params": unet_params},
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                train=True
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            loss = (target - model_pred) ** 2
            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(unet_state.params)

        new_state = unet_state.apply_gradients(grads=grad)
        metrics = {'scalar' : {'learning/loss' : loss}, 'scalars': {}}

        return new_state, metrics, new_train_rng

    num_model_parameters = calculate_num_params_from_pytree(unet_state.params)
    max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")

    my_data_sharding = {'input_ids': data_sharding,
                        'pixel_values': data_sharding,
                        'prompt_embeds' : data_sharding,
                        'text_embeds' : data_sharding}
    if (not config.enable_profiler):
        with jax.transfer_guard("disallow"):
            p_train_step = jax.jit(
                partial(train_step, cache_latents_text_encoder_outputs=config.cache_latents_text_encoder_outputs),
                in_shardings=(unet_state_mesh_shardings, vae_state_mesh_shardings, my_data_sharding, None),
                out_shardings=(unet_state_mesh_shardings, None, None),
                donate_argnums=(0,)
            )
    else:
        p_train_step = jax.jit(
            partial(train_step, cache_latents_text_encoder_outputs=config.cache_latents_text_encoder_outputs),
            in_shardings=(unet_state_mesh_shardings, vae_state_mesh_shardings, my_data_sharding, None),
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
    mllog_utils.train_step_start(config, start_step)
    for step in np.arange(start_step, config.max_train_steps):
        example_batch = load_next_batch(data_iterator, example_batch, config)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            unet_state, train_metric, train_rngs = p_train_step(unet_state,
                                                                vae_state,
                                                                example_batch,
                                                                train_rngs)
        new_time = datetime.datetime.now()
        record_scalar_metrics(train_metric, new_time - last_step_completion, per_device_tflops, learning_rate_scheduler(step))
        write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, config)
        last_step_completion = new_time
        # Start profiling at end of first step to avoid compilation.
        # Move before for loop to include.
        if step == first_profiling_step:
            max_utils.activate_profiler(config)
        if step == last_profiling_step:
            max_utils.deactivate_profiler(config)

        mllog_utils.maybe_train_step_log(config, start_step, step, train_metric)

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        # Restore vae and text encoder if we cached latents and encoder outputs.
        if config.cache_latents_text_encoder_outputs:
            text_encoder = FlaxCLIPTextModel.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="text_encoder", dtype=weight_dtype, from_pt=config.from_pt
            )
            text_encoder_2 = FlaxCLIPTextModelWithProjection.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="text_encoder_2", dtype=weight_dtype, from_pt=config.from_pt
            )
            vae, vae_params = FlaxAutoencoderKL.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="vae", dtype=weight_dtype, from_pt=config.from_pt
            )
            pipeline.vae = vae
            pipeline.text_encoder = text_encoder
            pipeline.text_encoder_2 = text_encoder_2
            params["text_encoder"] = text_encoder.params
            params["text_encoder_2"] = text_encoder_2.params
            params["vae"] = vae_params

        pipeline = FlaxStableDiffusionXLPipeline(
            text_encoder=pipeline.text_encoder,
            text_encoder_2=pipeline.text_encoder_2,
            vae=pipeline.vae,
            unet=pipeline.unet,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            scheduler=noise_scheduler,
        )

        pipeline.save_pretrained(
            config.output_dir,
            params={
                "text_encoder": get_params_to_save(params["text_encoder"]),
                "text_encoder_2" : get_params_to_save(params["text_encoder_2"]),
                "vae": get_params_to_save(params["vae"]),
                "unet": get_params_to_save(unet_state.params),
            },
        )
    max_utils.close_summary_writer(writer)

def main(argv: Sequence[str]) -> None:
    
    max_logging.log(f"Found {jax.device_count()} devices.")
    cc.set_cache_dir(os.path.expanduser("~/jax_cache"))
    pyconfig.initialize(argv)
    config = pyconfig.config
    mllog_utils.train_init_start(config)
    validate_train_config(config)
    train(config)
if __name__ == "__main__":
    app.run(main)