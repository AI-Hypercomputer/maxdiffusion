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
from pathlib import Path
import time
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
    FlaxUNet2DConditionModel,
    max_logging,
    max_utils,
    pyconfig,
    mllog_utils,
)
from maxdiffusion.maxdiffusion_utils import (
    calculate_unet_tflops,
    encode,
)

from maxdiffusion.train_utils import (
    get_first_step,
    load_next_batch,
    validate_train_config,
    record_scalar_metrics,
    write_metrics,
    get_params_to_save,
    generate_timestep_weights,
    save_checkpoint
)

from maxdiffusion.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P, PositionalSharding
from transformers import CLIPImageProcessor, set_seed
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from huggingface_hub.utils import insecure_hashlib

from maxdiffusion.input_pipeline.input_pipeline_interface import make_dreambooth_train_iterator

from dreambooth_constants import (
    INSTANCE_IMAGE_LATENTS,
    INSTANCE_PROMPT_INPUT_IDS,
    CLASS_IMAGE_LATENTS,
    CLASS_PROMPT_INPUT_IDS
)

def get_shaped_batch(config, pipeline):
  """Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078.
  This function works with sd1.x and 2.x.
  """
  vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
  total_train_batch_size = config.per_device_batch_size * jax.device_count()
  batch_image_shape = (total_train_batch_size, 4,
            config.resolution // vae_scale_factor,
            config.resolution // vae_scale_factor)
  batch_ids_shape = (total_train_batch_size, pipeline.text_encoder.config.max_position_embeddings)
  shaped_batch = (
      {
          INSTANCE_IMAGE_LATENTS : jax.ShapeDtypeStruct(batch_image_shape, jnp.float32),
          INSTANCE_PROMPT_INPUT_IDS : jax.ShapeDtypeStruct(batch_ids_shape, jnp.int32)
      },{
          CLASS_IMAGE_LATENTS : jax.ShapeDtypeStruct(batch_image_shape, jnp.float32),
          CLASS_PROMPT_INPUT_IDS : jax.ShapeDtypeStruct(batch_ids_shape, jnp.int32)
      }
  )
  return shaped_batch

class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def prepare_w_prior_preservation(rng, config):
    class_images_dir = Path(config.class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    # just use pmap here
    if cur_class_images < config.num_class_images:
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            safety_checker=None,
            revision=config.revision,
            split_head_dim=config.split_head_dim
        )
        pipeline.set_progress_bar_config(disable=True)
        num_new_images = config.num_class_images - cur_class_images
        max_logging.log(f"Number of class images to sample: {num_new_images}.")
        sample_dataset = PromptDataset(config.class_prompt, num_new_images)
        total_sample_batch_size = config.per_device_batch_size * jax.local_device_count()
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=total_sample_batch_size)
        for example in tqdm(
            sample_dataloader, desc="Generating class images",
            disable=not jax.process_index() == 0
        ):
            prompt_ids = pipeline.prepare_inputs(example["prompt"])
            prompt_ids = shard(prompt_ids)
            p_params = jax_utils.replicate(params)
            rng = jax.random.split(rng)[0]
            sample_rng = jax.random.split(rng, jax.device_count())
            images = pipeline(prompt_ids, p_params, sample_rng, jit=True).images
            images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
            images = pipeline.numpy_to_pil(np.array(images))

            for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)
            
        max_utils.delete_pytree(params)
        del pipeline

def train(config):
    rng = jax.random.PRNGKey(config.seed)

    writer = max_utils.initialize_summary_writer(config)
    if config.dataset_name is None and config.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if config.with_prior_preservation:
        prepare_w_prior_preservation(rng, config)

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
    params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, dtype=jnp.float32
    )
    noise_scheduler_state = noise_scheduler.create_state()

    pipeline.scheduler = noise_scheduler
    params["scheduler"] = noise_scheduler_state

    sharding = PositionalSharding(devices_array).replicate()
    partial_device_put_replicated = partial(max_utils.device_put_replicated, sharding=sharding)
    params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])

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
    _, _) = max_utils.get_states(mesh, tx, rng, config,
                                 pipeline, params["unet"],
                                 None, training=True)
    text_encoder_state = train_state.TrainState.create(
        apply_fn=pipeline.text_encoder.__call__, params=params["text_encoder"], tx=tx
    )
    per_device_tflops = calculate_unet_tflops(config, pipeline, rng, train=True)
    max_logging.log(f"Per train step, estimated total TFLOPs will be {per_device_tflops:.2f}")

    data_iterator = make_dreambooth_train_iterator(
        config,
        mesh,
        total_train_batch_size,
        pipeline.tokenizer,
        pipeline.vae,
        params["vae"]
    )

    # Initialize our training
    _, train_rngs = jax.random.split(rng)

    def train_step(unet_state, text_encoder_state, batch, train_rng):
        
        _, gen_dummy_rng = jax.random.split(train_rng)
        sample_rng, timestep_bias_rng, new_train_rng = jax.random.split(gen_dummy_rng, 3)
        instance_batch = batch[0]
        class_batch = batch[1]

        instance_latents = instance_batch[INSTANCE_IMAGE_LATENTS]
        instance_input_ids = instance_batch[INSTANCE_PROMPT_INPUT_IDS]
        class_latents = class_batch[CLASS_IMAGE_LATENTS]
        class_input_ids = class_batch[CLASS_PROMPT_INPUT_IDS]

        latents = jnp.concatenate((instance_latents, class_latents), axis=0)
        input_ids = jnp.concatenate((instance_input_ids, class_input_ids), axis=0)
        params = {"text_encoder" : text_encoder_state.params, "unet" : unet_state.params}
        
        def compute_loss(params):
            encoder_hidden_states = encode(input_ids, pipeline.text_encoder, params["text_encoder"])

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

            # Predict the noise residual and compute loss
            model_pred = pipeline.unet.apply(
                {"params": params["unet"]}, noisy_latents, timesteps, encoder_hidden_states, train=True
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            if config.with_prior_preservation:
                # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = jnp.split(model_pred, 2, axis=0)
                target, target_prior = jnp.split(target, 2, axis=0)

                # Compute instance loss
                loss = (target - model_pred) ** 2
                loss = loss.mean()

                # Compute prior loss
                prior_loss = (target_prior - model_pred_prior) ** 2
                prior_loss = prior_loss.mean()

                # Add the prior loss to the instance loss.
                loss = loss + config.prior_loss_weight * prior_loss
            else:
                loss = (target - model_pred) ** 2
                loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(params)

        new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
        new_text_encoder_state = text_encoder_state.apply_gradients(grads=grad["text_encoder"])
        metrics = {'scalar' : {'learning/loss' : loss}, 'scalars': {}}

        return new_unet_state, new_text_encoder_state, metrics, new_train_rng

    num_model_parameters = max_utils.calculate_num_params_from_pytree(unet_state.params)
    max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")

    my_data_sharding = (
        {INSTANCE_IMAGE_LATENTS : data_sharding,
        INSTANCE_PROMPT_INPUT_IDS : data_sharding},
        {CLASS_IMAGE_LATENTS : data_sharding,
        CLASS_PROMPT_INPUT_IDS : data_sharding}
    )
    # my_data_sharding = (
    #     {INSTANCE_IMAGE_LATENTS : data_sharding,
    #     INSTANCE_PROMPT_INPUT_IDS : data_sharding,
    #     CLASS_IMAGE_LATENTS : data_sharding,
    #     CLASS_PROMPT_INPUT_IDS : data_sharding
    # )

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        p_train_step = jax.jit(
            train_step,
            in_shardings=(unet_state_mesh_shardings, None, my_data_sharding, None),
            out_shardings=(unet_state_mesh_shardings, None, None, None),
            donate_argnums=(0,)
        )
        max_logging.log("Precompiling...")
        s = time.time()
        dummy_batch = get_shaped_batch(config, pipeline)
        p_train_step = p_train_step.lower(unet_state,
                                          text_encoder_state,
                                          dummy_batch,
                                          train_rngs)
        p_train_step = p_train_step.compile()
        max_logging.log(f"Compile time: {(time.time() - s )}")

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
    s = time.time()
    for step in np.arange(start_step, config.max_train_steps):
        example_batch = load_next_batch(data_iterator, example_batch, config)
        unet_state, text_encoder_state, train_metric, train_rngs = p_train_step(unet_state,
                                                            text_encoder_state,
                                                            example_batch,
                                                            train_rngs)
        samples_count = total_train_batch_size * (step + 1)
        new_time = datetime.datetime.now()
        record_scalar_metrics(train_metric, new_time - last_step_completion, per_device_tflops, learning_rate_scheduler(step))
        if config.write_metrics:
            write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, config)
        last_step_completion = new_time
        # Start profiling at end of first step to avoid compilation.
        # Move before for loop to include.
        if step == first_profiling_step:
            max_utils.activate_profiler(config)
        if step == last_profiling_step:
            max_utils.deactivate_profiler(config)

        mllog_utils.maybe_train_step_log(config, start_step, step, train_metric)
        if step != 0 and config.checkpoint_every != -1 and samples_count % config.checkpoint_every == 0:
            checkpoint_name = f"UNET-samples-{samples_count}"
            save_checkpoint(pipeline.unet.save_pretrained,
                            get_params_to_save(unet_state.params),
                            config,
                            os.path.join(config.checkpoint_dir, checkpoint_name))

    if config.write_metrics:
        write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, config)

    # save the last checkpoint
    if jax.process_index() == 0:
        checkpoint_name = "final"
        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker", from_pt=True
        )

        pipeline = FlaxStableDiffusionPipeline(
            text_encoder=pipeline.text_encoder,
            vae=pipeline.vae,
            unet=pipeline.unet,
            tokenizer=pipeline.tokenizer,
            scheduler=noise_scheduler,
            safety_checker=safety_checker,
            feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        )

        params = {
            "text_encoder": get_params_to_save(text_encoder_state.params),
            "vae": get_params_to_save(params["vae"]),
            "unet": get_params_to_save(unet_state.params),
            "safety_checker": safety_checker.params,
        }
        save_checkpoint(pipeline.save_pretrained, params, config, os.path.join(config.checkpoint_dir, checkpoint_name))

    max_utils.close_summary_writer(writer)
    max_logging.log(f"training time: {(time.time() - s)}")

def main(argv: Sequence[str]) -> None:
    max_logging.log(f"Found {jax.device_count()} devices.")
    pyconfig.initialize(argv)
    config = pyconfig.config
    if len(config.cache_dir) > 0:
        jax.config.update("jax_compilation_cache_dir", config.cache_dir)
    mllog_utils.train_init_start(config)
    validate_train_config(config)
    train(config)
if __name__ == "__main__":
    app.run(main)
