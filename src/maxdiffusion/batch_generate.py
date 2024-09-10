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
import functools
import time
from typing import Sequence

import pandas as pd
import numpy as np
import tensorflow as tf

import jax
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from absl import app
from maxdiffusion import (
  pyconfig,
  FlaxDDIMScheduler,
  max_utils,
  maxdiffusion_utils
)

from maxdiffusion.maxdiffusion_utils import rescale_noise_cfg
from flax.linen import partitioning as nn_partitioning
from maxdiffusion.image_processor import VaeImageProcessor
from maxdiffusion import multihost_dataloading

from maxdiffusion.checkpointing.checkpointing_utils import load_params_from_path
from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import (
    STABLE_DIFFUSION_CHECKPOINT
)


def loop_body(step, args, model, pipeline, prompt_embeds, guidance_scale, guidance_rescale):
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
    noise_pred = rescale_noise_cfg(noise_pred, noise_prediction_text, guidance_rescale=guidance_rescale)

    latents, scheduler_state = pipeline.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
    return latents, scheduler_state, state

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

def tokenize(prompt, tokenizer):
    """Tokenizes prompt."""
    return tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np"
    ).input_ids

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

def get_unet_inputs(pipeline, params, states, config, rng, mesh, batch_size, prompt_ids, negative_prompt_ids):

    data_sharding = jax.sharding.NamedSharding(mesh,P(*config.data_sharding))

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    guidance_scale = config.guidance_scale
    guidance_rescale = config.guidance_rescale
    num_inference_steps = config.num_inference_steps

    prompt_embeds = pipeline.text_encoder(prompt_ids, params=states["text_encoder_state"].params)[0]
    negative_prompt_embeds = pipeline.text_encoder(negative_prompt_ids, params=states["text_encoder_state"].params)[0]
    context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])
    guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)
    guidance_rescale = jnp.array([guidance_rescale], dtype=jnp.float32)

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
        shape=latents.shape
    )
    latents = latents * params["scheduler"].init_noise_sigma

    latents = jax.device_put(latents, data_sharding)
    context = jax.device_put(context, data_sharding)

    return latents, context, guidance_scale, guidance_rescale, scheduler_state

def vae_decode(latents, state, pipeline):
    latents = 1 / pipeline.vae.config.scaling_factor * latents
    image = pipeline.vae.apply(
        {"params" : state.params},
        latents,
        method=pipeline.vae.decode
    ).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image

def run_inference(states, prompt_ids_sharded, negative_prompt_ids_sharded, pipeline, params, config, rng, mesh, batch_size):

    unet_state = states["unet_state"]
    vae_state = states["vae_state"]

    (latents,
    context,
    guidance_scale,
    guidance_rescale,
    scheduler_state) = get_unet_inputs(pipeline, params, states, config, rng, mesh, batch_size, prompt_ids_sharded, negative_prompt_ids_sharded)

    loop_body_p = functools.partial(loop_body, model=pipeline.unet,
                                    pipeline=pipeline,
                                    prompt_embeds=context,
                                    guidance_scale=guidance_scale,
                                    guidance_rescale=guidance_rescale)

    vae_decode_p = functools.partial(vae_decode, pipeline=pipeline)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        latents, _, _ = jax.lax.fori_loop(0, config.num_inference_steps,
                                        loop_body_p, (latents, scheduler_state, unet_state))
        image = vae_decode_p(latents, vae_state)
        return image

def run(config):

    from maxdiffusion.trainers.stable_diffusion_trainer import (
    StableDiffusionTrainer
    )
    from maxdiffusion import eval

    # Just re-use the trainer since it has helper functions to generate states.
    class GenerateSD(StableDiffusionTrainer):
        def __init__(self, config, checkpoint_type):
            StableDiffusionTrainer.__init__(self, config, checkpoint_type)

        def post_training_steps(self, pipeline, params, train_states):
            return super().post_training_steps(pipeline, params, train_states)

    checkpoint_loader = GenerateSD(config, STABLE_DIFFUSION_CHECKPOINT)
    mesh = checkpoint_loader.mesh
    
    # loads all saved checkpoints
    checkpoint_steps = checkpoint_loader.checkpoint_manager.all_steps()
    
    # we only do this once as it loads the configs, but not the actual states.
    pipeline, params = checkpoint_loader.load_checkpoint()

    vae_state, vae_state_shardings = checkpoint_loader.create_vae_state(
        pipeline,
        params,
        checkpoint_item_name="vae_state",
        is_training=False
    )

    weights_init_fn = functools.partial(
        pipeline.text_encoder.init_weights,
        rng=checkpoint_loader.rng,
        input_shape=(checkpoint_loader.total_train_batch_size, pipeline.tokenizer.model_max_length))
    unboxed_abstract_state, _, _ = max_utils.get_abstract_state(
      pipeline.text_encoder,
      None,
      config,
      checkpoint_loader.mesh,
      weights_init_fn,
      False
    )
    text_encoder_params = load_params_from_path(
      config,
      checkpoint_loader.checkpoint_manager,
      unboxed_abstract_state.params,
      "text_encoder_state"
    )
    if text_encoder_params:
        params["text_encoder"] = text_encoder_params

    text_encoder_state, text_encoder_state_shardings = max_utils.setup_initial_state(
      model=pipeline.text_encoder,
      tx=None,
      config=config,
      mesh=checkpoint_loader.mesh,
      weights_init_fn=weights_init_fn,
      model_params=params.get("text_encoder", None),
      training=False
    )

    states = {}
    state_shardings = {}

    state_shardings["vae_state"] = vae_state_shardings
    state_shardings["text_encoder_state"] = text_encoder_state_shardings

    states["vae_state"] = vae_state
    states["text_encoder_state"] = text_encoder_state

    # TODO - switch scheduler to the one from mlperf_4 branch
    scheduler, scheduler_state = FlaxDDIMScheduler.from_pretrained(
        config.pretrained_model_name_or_path, revision=config.revision, subfolder="scheduler", dtype=jnp.float32
    )
    scheduler, scheduler_state = maxdiffusion_utils.create_scheduler(scheduler.config, config)
    pipeline.scheduler = scheduler
    params["scheduler"] = scheduler_state

    per_host_batch_size = jax.local_device_count() * config.per_device_batch_size
    shards = get_list_prompt_shards_from_file(config.caption_coco_file, per_host_batch_size)

    negative_prompt_ids = tokenize([""] * per_host_batch_size, pipeline.tokenizer)

    for checkpoint_step in checkpoint_steps:
        # load unet
        weights_init_fn = functools.partial(pipeline.unet.init_weights, rng=checkpoint_loader.rng)
        unboxed_abstract_state, _, _ = max_utils.get_abstract_state(pipeline.unet, None, config, checkpoint_loader.mesh, weights_init_fn, False)
        unet_params = load_params_from_path(
            config,
            checkpoint_loader.checkpoint_manager,
            unboxed_abstract_state.params,
            "unet_state",
            step=checkpoint_step
        )
        # Here just to get the state
        unet_state, unet_state_shardings = max_utils.setup_initial_state(
        model=pipeline.unet,
        tx=None,
        config=config,
        mesh=checkpoint_loader.mesh,
        weights_init_fn=weights_init_fn,
        model_params=unet_params,
        training=False
        )

        states["unet_state"] = unet_state
        state_shardings["unet_state"] = unet_state_shardings

        p_run_inference = jax.jit(
            functools.partial(run_inference,
                            pipeline=pipeline,
                            params=params,
                            config=config,
                            rng=checkpoint_loader.rng,
                            mesh=checkpoint_loader.mesh,
                            batch_size=checkpoint_loader.total_train_batch_size),
            in_shardings=(state_shardings, None, None),
            out_shardings=None
        )

        os.makedirs(config.images_directory, exist_ok=True)

        for i, shard_i in enumerate(shards):
            df = pd.DataFrame(shard_i[:], columns=["image_id", "id", "prompt"])
            batches = [df[i:i + per_host_batch_size] for i in range(0, len(df), per_host_batch_size)]

            batch = batches[0]
            prompt_tensors = batch["prompt"].tolist()
            prompt = [t.numpy().decode('utf-8') for t in prompt_tensors]

            prompt_ids = tokenize(prompt, pipeline.tokenizer)

            image_ids_tensor = batch["image_id"]
            img_ids = [t.numpy().decode('utf-8') for t in image_ids_tensor]

            prompt_ids_sharded = multihost_dataloading.get_data_sharded(prompt_ids, mesh)
            negative_prompt_ids_sharded = multihost_dataloading.get_data_sharded(negative_prompt_ids, mesh)

            images = p_run_inference(states, prompt_ids_sharded, negative_prompt_ids_sharded)
            images = [s.data for s in images.addressable_shards]
            
            numpy_images = np.array(images)
            numpy_images = np.reshape(numpy_images, (numpy_images.shape[0] * numpy_images.shape[1], numpy_images.shape[2],numpy_images.shape[3], numpy_images.shape[4]))
            
            ids = batch["id"].tolist()
            msk = [ id_item!='0' for id_item in ids]
            save_process(numpy_images, config.images_directory, img_ids, msk)
        checkpoint_name = f"step_num={str(checkpoint_step+1)}-samples_count={(checkpoint_step * config.per_device_batch_size * jax.device_count())}"
        _, _ = eval.eval_scores(config, config.images_directory, checkpoint_name)

    return images

def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    run(pyconfig.config)

if __name__ == "__main__":
    app.run(main)
