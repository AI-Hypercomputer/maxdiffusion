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

import functools
import os
import time
from typing import Sequence
import csv
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from maxdiffusion.max_utils import (
  create_device_mesh,
  get_dtype,
  get_states,
  device_put_replicated,
  get_flash_block_sizes,
  override_scheduler_config,
  create_scheduler
)
from maxdiffusion import pyconfig
from maxdiffusion import multihost_dataloading
from absl import app
from maxdiffusion import (
  FlaxStableDiffusionPipeline,
)
from flax.linen import partitioning as nn_partitioning
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.sharding import Mesh, PositionalSharding
from maxdiffusion.image_processor import VaeImageProcessor 
import tensorflow as tf
import pandas as pd

cc.initialize_cache(os.path.expanduser("~/jax_cache"))

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = jnp.std(noise_pred_text, axis=list(range(1, jnp.ndim(noise_pred_text))), keepdims=True)
    std_cfg = jnp.std(noise_cfg, axis=list(range(1, jnp.ndim(noise_cfg))), keepdims=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def loop_body(step, args, model, pipeline, prompt_embeds, guidance_scale):
    latents, scheduler_state, state = args
    latents_input = jnp.concatenate([latents] * 2)

    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    timestep = jnp.broadcast_to(t, latents_input.shape[0])

    latents_input = pipeline.scheduler.scale_model_input(scheduler_state, latents_input, t)

    noise_pred = model.apply(
        {"params" : state.params},
        jnp.array(latents_input),
        jnp.array(timestep, dtype=jnp.int32),
        encoder_hidden_states=prompt_embeds
    ).sample

    noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
    noise_pred = rescale_noise_cfg(noise_pred, noise_prediction_text, guidance_rescale=0.7)

    latents, scheduler_state = pipeline.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()

    return latents, scheduler_state, state

def tokenize(prompt, tokenizer):
    """Tokenizes prompt."""
    return tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np"
    ).input_ids

def get_unet_inputs(rng, config, batch_size, pipeline, params, prompt_ids, negative_prompt_ids):
    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    guidance_scale = config.guidance_scale
    num_inference_steps = config.num_inference_steps
    prompt_embeds = pipeline.text_encoder(prompt_ids, params=params["text_encoder"])[0]
    negative_prompt_embeds = pipeline.text_encoder(negative_prompt_ids, params=params["text_encoder"])[0]
    context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])
    guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)

    batch_size = prompt_ids.shape[0]
    latents_shape = (
        batch_size,
        pipeline.unet.config.in_channels,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor,
    )
    latents = jax.random.normal(rng, shape=latents_shape, dtype=jnp.float32)

    scheduler_state = pipeline.scheduler.set_timesteps(
        params["scheduler"],
        num_inference_steps=num_inference_steps,
        shape=latents.shape,
        timestep_spacing=config.timestep_spacing
    )
    latents = latents * params["scheduler"].init_noise_sigma

    return latents, context, guidance_scale, scheduler_state

def vae_decode(latents, state, pipeline):
    latents = 1 / pipeline.vae.config.scaling_factor * latents
    image = pipeline.vae.apply(
        {"params" : state.params},
        latents,
        method=pipeline.vae.decode
    ).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image

def save_process(images, images_directory, img_ids, mask=None):
    images = VaeImageProcessor.numpy_to_pil(images)
    if mask is not None:
        for i, (image, valid) in enumerate(zip(images, mask)):
            if valid:
                img_save_path = os.path.join(images_directory, f"image_{img_ids[i]}.png")
                image.save(img_save_path)
    else:
         for i, image, in enumerate(images):
                img_save_path = os.path.join(images_directory, f"image_{img_ids[i]}.png")
                image.save(img_save_path)       

def run_inference(unet_state, vae_state, params, prompt_ids, negative_prompt_ids, rng, config, batch_size, pipeline, mesh):                
    (latents,
    context,
    guidance_scale,
    scheduler_state) = get_unet_inputs(rng, config, batch_size, pipeline, params, prompt_ids, negative_prompt_ids)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        loop_body_p = functools.partial(loop_body, model=pipeline.unet,
                                        pipeline=pipeline,
                                        prompt_embeds=context,
                                        guidance_scale=guidance_scale)

        vae_decode_p = functools.partial(vae_decode, pipeline=pipeline)
        latents, _, _ = jax.lax.fori_loop(0, config.num_inference_steps,
                                        loop_body_p, (latents, scheduler_state, unet_state))
        images = vae_decode_p(latents, vae_state)
        return images

def run(config,
        images_directory = None,
         unet_state = None,
         unet_state_mesh_shardings = None,
         vae_state = None,
         vae_state_mesh_shardings = None,
         pipeline = None, params = None):
    
    if images_directory is None:
        images_directory = config.images_directory
    rng = jax.random.PRNGKey(config.seed)
    # Setup Mesh
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    batch_size = jax.local_device_count() * config.per_device_batch_size
    weight_dtype = get_dtype(config)
    flash_block_sizes = get_flash_block_sizes(config)
    if pipeline is None or params is None:
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
            safety_checker=None,
            feature_extractor=None,
            split_head_dim=config.split_head_dim,
            norm_num_groups=config.norm_num_groups,
            from_pt=config.from_pt,
            attention_kernel=config.attention,
            flash_block_sizes=flash_block_sizes,
            mesh=mesh
        )
        params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)
         # Text encoder params
        sharding = PositionalSharding(mesh.devices).replicate()
        partial_device_put_replicated = functools.partial(device_put_replicated, sharding=sharding)
        params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])
        (unet_state,
        unet_state_mesh_shardings,
        vae_state,
        vae_state_mesh_shardings) = get_states(mesh, None, rng, config, pipeline, params["unet"], params["vae"], training=False)
        del params["vae"]
        del params["unet"]
    
    scheduler, scheduler_state = create_scheduler(config.inference_scheduler, pipeline.scheduler.config, config)
    pipeline.scheduler = scheduler
    params["scheduler"] = scheduler_state

    p_run_inference = jax.jit(
        functools.partial(run_inference, rng=rng, config=config, batch_size=batch_size, pipeline=pipeline, mesh=mesh),
        in_shardings=(unet_state_mesh_shardings, vae_state_mesh_shardings, None, None, None),
        out_shardings=None,
    )

    def parse_tsv_line(line):
    # Customize this function to parse your TSV file based on your specific format
    # For example, you can use tf.strings.split to split the line into columns
        columns = tf.strings.split([line], sep='\t')
        return columns

    def get_list_prompt_shards_from_file(file_path, batch_size_per_process):
      # Create a dataset using tf.data
      dataset = tf.data.TextLineDataset(file_path)
      dataset = dataset.map(parse_tsv_line, num_parallel_calls=tf.data.AUTOTUNE)
      dataset = dataset.batch(batch_size_per_process)
      dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())

      # Create an iterator to iterate through the batches
      iterator = iter(dataset)
      batch_number = 1
      row_shards = []
      for batch in iterator:
        rows_batch = []
        for row in batch:
            row_tensor = row[0]
            rows_batch.append([row_tensor[0], row_tensor[1], row_tensor[2]])
        row_shards.append(rows_batch)
          
        batch_number += 1
      return row_shards

    PerHostBatchSize = jax.local_device_count() * config.per_device_batch_size
    shards = get_list_prompt_shards_from_file(config.caption_coco_file, PerHostBatchSize)

    negative_prompt_ids = tokenize([""] * PerHostBatchSize, pipeline.tokenizer)

    os.makedirs(config.images_directory, exist_ok=True)

    for i, shard_i in enumerate(shards):
        df = pd.DataFrame(shard_i[:], columns=["image_id", "id", "prompt"])
        batches = [df[i:i + PerHostBatchSize] for i in range(0, len(df), PerHostBatchSize)]

        batch = batches[0]
        prompt_tensors = batch["prompt"].tolist()
        prompt = [t.numpy().decode('utf-8') for t in prompt_tensors]

        prompt_ids = tokenize(prompt, pipeline.tokenizer)

        image_ids_tensor = batch["image_id"]
        img_ids = [t.numpy().decode('utf-8') for t in image_ids_tensor]

        prompt_ids_sharded = multihost_dataloading.get_data_sharded(prompt_ids, mesh)
        negative_prompt_ids_sharded = multihost_dataloading.get_data_sharded(negative_prompt_ids, mesh)

        images = p_run_inference(unet_state, vae_state, params, prompt_ids_sharded, negative_prompt_ids_sharded)
        images = [s.data for s in images.addressable_shards]
        
        ids = batch["id"].tolist()
        msk = [ id_item!='0' for id_item in ids]

        numpy_images = np.array(images)
        numpy_images = np.reshape(numpy_images, (numpy_images.shape[0] * numpy_images.shape[1], numpy_images.shape[2],numpy_images.shape[3], numpy_images.shape[4]))
        
        ids = batch["id"].tolist()
        msk = [ id_item!='0' for id_item in ids]

        save_process(numpy_images, images_directory, img_ids, msk)

def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    run(pyconfig.config)

if __name__ == "__main__":
    app.run(main)
