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
import time
from typing import Sequence

import numpy as np

import jax
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from maxdiffusion import pyconfig
from absl import app
from maxdiffusion import (
  FlaxDDIMScheduler,
)

from maxdiffusion.maxdiffusion_utils import rescale_noise_cfg
from flax.linen import partitioning as nn_partitioning
from maxdiffusion.image_processor import VaeImageProcessor
from maxdiffusion.checkpointing.base_stable_diffusion_checkpointer import (
    BaseStableDiffusionCheckpointer,
    STABLE_DIFFUSION_CHECKPOINT
)

class CheckpointLoader(BaseStableDiffusionCheckpointer):
    def __init__(self, config, checkpoint_type):
        BaseStableDiffusionCheckpointer.__init__(self, config, checkpoint_type)

    def post_create_states_and_shard(self):
        return super().post_create_states_and_shard()


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

def tokenize(prompt, tokenizer):
    """Tokenizes prompt."""
    return tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np"
    ).input_ids

def get_unet_inputs(checkpoint_loader):
    rng = checkpoint_loader.rng
    batch_size = checkpoint_loader.total_train_batch_size
    params = checkpoint_loader.params
    states = checkpoint_loader.train_states
    config = checkpoint_loader.config
    pipeline = checkpoint_loader.pipeline
    mesh = checkpoint_loader.mesh

    data_sharding = jax.sharding.NamedSharding(mesh,P(*config.data_sharding))

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    prompt_ids = [config.prompt] * batch_size
    prompt_ids = tokenize(prompt_ids, pipeline.tokenizer)
    negative_prompt_ids = [config.negative_prompt] * batch_size
    negative_prompt_ids = tokenize(negative_prompt_ids, pipeline.tokenizer)
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

def run_inference(unet_state, vae_state, checkpoint_loader):

    pipeline = checkpoint_loader.pipeline
    config = checkpoint_loader.config
    mesh = checkpoint_loader.mesh

    (latents,
    context,
    guidance_scale,
    guidance_rescale,
    scheduler_state) = get_unet_inputs(checkpoint_loader)

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

    checkpoint_loader = CheckpointLoader(config, STABLE_DIFFUSION_CHECKPOINT)
    checkpoint_loader.load_checkpoint()
    state_shardings = checkpoint_loader.state_shardings
    states = checkpoint_loader.train_states

    scheduler, scheduler_state = FlaxDDIMScheduler.from_pretrained(
        config.pretrained_model_name_or_path, revision=config.revision, subfolder="scheduler", dtype=jnp.float32
    )
    checkpoint_loader.pipeline.scheduler = scheduler
    checkpoint_loader.params["scheduler"] = scheduler_state

    p_run_inference = jax.jit(
        functools.partial(run_inference, checkpoint_loader=checkpoint_loader),
        in_shardings=(state_shardings["unet_state_shardings"], state_shardings["vae_state_shardings"]),
        out_shardings=None
    )

    s = time.time()
    p_run_inference(states["unet_state"], states["vae_state"]).block_until_ready()
    print("compile time: ", (time.time() - s))

    s = time.time()
    images = p_run_inference(states["unet_state"], states["vae_state"]).block_until_ready()
    print("inference time: ",(time.time() - s))
    numpy_images = np.array(images)
    images = VaeImageProcessor.numpy_to_pil(numpy_images)
    for i, image in enumerate(images):
        image.save(f"image_{i}.png")

    return images

def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    if len(config.cache_dir) > 0:
        jax.config.update("jax_compilation_cache_dir", config.cache_dir)
    run(pyconfig.config)

if __name__ == "__main__":
    app.run(main)
