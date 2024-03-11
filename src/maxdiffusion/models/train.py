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
    FlaxStableDiffusionPipeline,
    FlaxAutoencoderKL,
    max_logging,
    max_utils,
    pyconfig
)

from transformers import FlaxCLIPTextModel

from maxdiffusion.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from flax.linen import partitioning as nn_partitioning
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P, PositionalSharding
from tensorboardX import SummaryWriter
from transformers import CLIPImageProcessor, set_seed

from maxdiffusion.input_pipeline.input_pipeline_interface import make_pokemon_train_iterator
from maxdiffusion.models import quantizations

dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

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

def record_scalar_metrics(metrics, step_time_delta, lr):
  """Records scalar metrics to be written to tensorboard"""
  metrics['scalar'].update({
      'perf/step_time_seconds': step_time_delta.total_seconds()
  })
  metrics['scalar'].update({'learning/current_learning_rate': lr })

def write_metrics(writer, metrics, step, config):
  """Writes metrics to tensorboard"""
  with jax.spmd_mode('allow_all'):
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar",[]):
        writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars",[]):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    full_log = step % config.log_period == 0

    max_logging.log(f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
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

    writer = SummaryWriter(config.tensorboard_dir)
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
    quant = quantizations.configure_quantization(config)
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
        safety_checker=None, feature_extractor=None, from_pt=config.from_pt,
        split_head_dim=config.split_head_dim, attention_kernel=config.attention, mesh=mesh,
        quant=quant
    )

    noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(config.pretrained_model_name_or_path,
        revision=config.revision, subfolder="scheduler", dtype=jnp.float32)

    pipeline.scheduler = noise_scheduler
    params["scheduler"] = noise_scheduler_state

    sharding = PositionalSharding(devices_array).replicate()
    partial_device_put_replicated = partial(max_utils.device_put_replicated, sharding=sharding)
    params["text_encoder"] = jax.tree_map(partial_device_put_replicated, params["text_encoder"])

    # Optimization
    if config.scale_lr:
        config.learning_rate = config.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(config.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
       adamw,
    )

    (unet_state,
    unet_state_mesh_shardings,
    vae_state, vae_state_mesh_shardings) = max_utils.get_states(mesh,
                                                                tx, rng, config,
                                                                pipeline, params["unet"],
                                                                params["vae"], training=True)

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
        raise Exception(f"dataset {config.dataset_name} is not currently supported.")

    if config.cache_latents_text_encoder_outputs:
       vae_state = None
       vae_state_mesh_shardings = None
       pipeline.vae = None
       params["vae"] = None
       pipeline.text_encoder = None
       params["text_encoder"] = None

    # Initialize our training
    _, train_rngs = jax.random.split(rng)

    def train_step(unet_state, vae_state, batch, train_rng, cache_latents_text_encoder_outputs):
        _, gen_dummy_rng = jax.random.split(train_rng)
        sample_rng, new_train_rng = jax.random.split(gen_dummy_rng)
        def compute_loss(unet_params):

            if cache_latents_text_encoder_outputs:
               latents = batch["pixel_values"]
               encoder_hidden_states = batch["input_ids"]
            else:
                # Convert images to latent space
                vae_outputs = pipeline.vae.apply(
                {"params": vae_state.params}, batch["pixel_values"], deterministic=True, method=pipeline.vae.encode
                )
                latents = vae_outputs.latent_dist.sample(sample_rng)
                # (NHWC) -> (NCHW)
                latents = jnp.transpose(latents, (0, 3, 1, 2))
                latents = latents * pipeline.vae.config.scaling_factor

                # Get the text embedding for conditioning
                encoder_hidden_states = pipeline.text_encoder(
                    batch["input_ids"],
                    params=params["text_encoder"],
                train=False,
                )[0]

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

            # Predict the noise residual and compute loss
            model_pred, intermediate_outputs = pipeline.unet.apply(
                {"params": unet_params},
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                train=True,
                rngs={"params" : new_train_rng, "dropout" : new_train_rng},
                mutable='intermediates'
            )
            model_pred = model_pred.sample

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

    my_data_sharding = {'input_ids': data_sharding, 'pixel_values': data_sharding}
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

    if jax.process_index() == 0:
        max_logging.log("***** Running training *****")
        max_logging.log(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
        max_logging.log(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
        max_logging.log(f"  Total optimization steps = {config.max_train_steps}")

    global_step = 0

    last_step_completion = datetime.datetime.now()

    local_metrics_file = open(config.metrics_file, 'a', encoding="utf8") if config.metrics_file else None
    running_gcs_metrics = [] if config.gcs_metrics else None
    example_batch = None
    # ======================== Training ================================
    # train
    for _ in np.arange(get_first_step(unet_state), config.max_train_steps):
        example_batch = load_next_batch(data_iterator, example_batch, config)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            unet_state, train_metric, train_rngs = p_train_step(unet_state,
                                                                vae_state,
                                                                example_batch,
                                                                train_rngs)
        new_time = datetime.datetime.now()
        record_scalar_metrics(train_metric, new_time - last_step_completion, config.learning_rate)
        write_metrics(writer, train_metric, global_step, config)
        last_step_completion = new_time

        if config.metrics_file:
            max_utils.write_metrics_locally(train_metric, global_step, config, local_metrics_file)

        if config.gcs_metrics and jax.process_index() == 0:
            running_gcs_metrics = max_utils.write_metrics_for_gcs(train_metric, global_step, config, running_gcs_metrics)

        # Start profiling at end of first step to avoid compilation.
        # Move before for loop to include.
        if global_step == 0:
            max_utils.activate_profiler(config)
        if global_step == 20:
            max_utils.deactivate_profiler(config)

        global_step += 1

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker", from_pt=True
        )
        # Restore vae and text encoder if we cached latents and encoder outputs.
        if config.cache_latents_text_encoder_outputs:
            text_encoder = FlaxCLIPTextModel.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="text_encoder", dtype=weight_dtype, from_pt=config.from_pt
            )
            vae, vae_params = FlaxAutoencoderKL.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, subfolder="vae", dtype=weight_dtype, from_pt=config.from_pt
            )
            pipeline.vae = vae
            pipeline.text_encoder = text_encoder
            params["text_encoder"] = text_encoder.params
            params["vae"] = vae_params

        pipeline = FlaxStableDiffusionPipeline(
            text_encoder=pipeline.text_encoder,
            vae=pipeline.vae,
            unet=pipeline.unet,
            tokenizer=pipeline.tokenizer,
            scheduler=noise_scheduler,
            safety_checker=safety_checker,
            feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        )

        pipeline.save_pretrained(
            config.output_dir,
            params={
                "text_encoder": get_params_to_save(params["text_encoder"]),
                "vae": get_params_to_save(params["vae"]),
                "unet": get_params_to_save(unet_state.params),
                "safety_checker": safety_checker.params,
            },
        )
    writer.close()

def main(argv: Sequence[str]) -> None:
    print(f"Found {jax.device_count()} devices.")
    cc.initialize_cache(os.path.expanduser("~/jax_cache"))
    pyconfig.initialize(argv)
    config = pyconfig.config
    validate_train_config(config)
    train(config)
if __name__ == "__main__":
    app.run(main)
