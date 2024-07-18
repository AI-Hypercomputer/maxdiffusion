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
from absl import app
from typing import Sequence
import time
from maxdiffusion.models.unet_2d_condition_flax import FlaxUNet2DConditionModel

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache as cc
from flax.linen import partitioning as nn_partitioning
from jax.sharding import PositionalSharding
from aqt.jax.v2.flax import aqt_flax
import optax

from maxdiffusion import (
    FlaxStableDiffusionXLPipeline,
    FlaxEulerDiscreteScheduler,
    FlaxDDPMScheduler
)


from maxdiffusion import pyconfig
from maxdiffusion.image_processor import VaeImageProcessor
from maxdiffusion.max_utils import (
  InferenceState,
  create_device_mesh,
  get_dtype,
  get_states,
  activate_profiler,
  deactivate_profiler,
  device_put_replicated,
  get_flash_block_sizes,
  get_abstract_state,
  setup_initial_state,
  create_learning_rate_schedule
)
from maxdiffusion.maxdiffusion_utils import (
  load_sdxllightning_unet,
  get_add_time_ids,
  rescale_noise_cfg
)
from maxdiffusion.models import quantizations
from jax.tree_util import tree_flatten_with_path, tree_unflatten


cc.set_cache_dir(os.path.expanduser("~/jax_cache"))

def _get_aqt_key_paths(aqt_vars):
  """Generate a list of paths which have aqt state"""
  aqt_tree_flat, _ = jax.tree_util.tree_flatten_with_path(aqt_vars)
  aqt_key_paths = []
  for k, _ in aqt_tree_flat:
    pruned_keys = []
    for d in list(k):
      if "AqtDotGeneral" in d.key:
        pruned_keys.append(jax.tree_util.DictKey(key="kernel"))
        break
      else:
        assert "Aqt" not in d.key, f"Unexpected Aqt op {d.key} in {k}."
        pruned_keys.append(d)
    aqt_key_paths.append(tuple(pruned_keys))
  return aqt_key_paths

def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  aqt_paths = _get_aqt_key_paths(aqt_vars)
  tree_flat, tree_struct = tree_flatten_with_path(params)
  for i, (k, v) in enumerate(tree_flat):
    if k in aqt_paths:
      v = {}
    tree_flat[i] = v
  return tree_unflatten(tree_struct, tree_flat)


def get_quantized_unet_variables(config):

  # Setup Mesh
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  batch_size = config.per_device_batch_size * jax.device_count()

  weight_dtype = get_dtype(config)
  flash_block_sizes = get_flash_block_sizes(config)

  quant = quantizations.configure_quantization(config=config, lhs_quant_mode=aqt_flax.QuantMode.TRAIN, rhs_quant_mode=aqt_flax.QuantMode.CONVERT)
  pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    config.pretrained_model_name_or_path,
    revision=config.revision,
    dtype=weight_dtype,
    split_head_dim=config.split_head_dim,
    norm_num_groups=config.norm_num_groups,
    attention_kernel=config.attention,
    flash_block_sizes=flash_block_sizes,
    mesh=mesh,
    quant=quant,
    )

  k = jax.random.key(0)
  latents = jnp.ones((8, 4,128,128), dtype=jnp.float32)
  timesteps = jnp.ones((8,))
  encoder_hidden_states = jnp.ones((8, 77, 2048))

  added_cond_kwargs = {
                "text_embeds": jnp.zeros((8, 1280), dtype=jnp.float32),
                "time_ids": jnp.zeros((8, 6), dtype=jnp.float32),
            }
  noise_pred, quantized_unet_vars = pipeline.unet.apply(
    params["unet"] | {"aqt" : {}},
    latents,
    timesteps,
    encoder_hidden_states=encoder_hidden_states,
    added_cond_kwargs=added_cond_kwargs,
    rngs={"params": jax.random.PRNGKey(0)},
    mutable=True,
  )
  del pipeline
  del params
  
  return quantized_unet_vars

def loop_body(step, args, model, pipeline, added_cond_kwargs, prompt_embeds, guidance_scale, guidance_rescale):
  latents, scheduler_state, state = args
  latents_input = jnp.concatenate([latents] * 2)

  t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
  timestep = jnp.broadcast_to(t, latents_input.shape[0])

  latents_input = pipeline.scheduler.scale_model_input(scheduler_state, latents_input, t)
  # breakpoint()
  noise_pred = model.apply(
    state.params,
   # {"params" : state.params, "aqt": state.params["aqt"] },
    jnp.array(latents_input),
    jnp.array(timestep, dtype=jnp.int32),
    encoder_hidden_states=prompt_embeds,
    added_cond_kwargs=added_cond_kwargs,
    rngs={"params": jax.random.PRNGKey(0)}
  ).sample

  noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
  noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

  # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
  noise_pred = rescale_noise_cfg(noise_pred, noise_prediction_text, guidance_rescale=guidance_rescale)


  latents, scheduler_state = pipeline.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()

  return latents, scheduler_state, state

def loop_body_for_quantization(latents, scheduler_state, state, rng,  model, pipeline, added_cond_kwargs, prompt_embeds, guidance_scale, guidance_rescale):
  # latents, scheduler_state, state, rng = args
  latents_input = jnp.concatenate([latents] * 2)

  t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[0]
  timestep = jnp.broadcast_to(t, latents_input.shape[0])

  latents_input = pipeline.scheduler.scale_model_input(scheduler_state, latents_input, t)
  noise_pred, quantized_unet_vars = model.apply(
    state.params | {"aqt" : {}},
    jnp.array(latents_input),
    jnp.array(timestep, dtype=jnp.int32),
    encoder_hidden_states=prompt_embeds,
    added_cond_kwargs=added_cond_kwargs,
    rngs={"params": rng},
    mutable=True,
  )
  return quantized_unet_vars


def get_embeddings(prompt_ids, pipeline, params):
  te_1_inputs = prompt_ids[:, 0, :]
  te_2_inputs = prompt_ids[:, 1, :]

  prompt_embeds = pipeline.text_encoder(
    te_1_inputs, params=params["text_encoder"], output_hidden_states=True
  )
  prompt_embeds = prompt_embeds["hidden_states"][-2]
  prompt_embeds_2_out = pipeline.text_encoder_2(
    te_2_inputs, params=params["text_encoder_2"], output_hidden_states=True
  )
  prompt_embeds_2 = prompt_embeds_2_out["hidden_states"][-2]
  text_embeds = prompt_embeds_2_out["text_embeds"]
  prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)
  return prompt_embeds, text_embeds

def tokenize(prompt, pipeline):
  inputs = []
  for _tokenizer in [pipeline.tokenizer, pipeline.tokenizer_2]:
    text_inputs = _tokenizer(
      prompt,
      padding="max_length",
      max_length=_tokenizer.model_max_length,
      truncation=True,
      return_tensors="np"
    )
    inputs.append(text_inputs.input_ids)
  inputs = jnp.stack(inputs,axis=1)
  return inputs

def run(config, q_v):
  rng = jax.random.PRNGKey(config.seed)

  # Setup Mesh
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  batch_size = config.per_device_batch_size * jax.device_count()

  weight_dtype = get_dtype(config)
  flash_block_sizes = get_flash_block_sizes(config)

  quant = quantizations.configure_quantization(config=config, lhs_quant_mode=aqt_flax.QuantMode.TRAIN, rhs_quant_mode=aqt_flax.QuantMode.SERVE, weights_quant_mode=aqt_flax.QuantMode.SERVE)
  pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "output_trained_working",
    revision=config.revision,
    dtype=weight_dtype,
    split_head_dim=config.split_head_dim,
    norm_num_groups=config.norm_num_groups,
    attention_kernel=config.attention,
    flash_block_sizes=flash_block_sizes,
    mesh=mesh,
    quant=quant,
  )
  breakpoint()

  # if this checkpoint was trained with maxdiffusion
  # the training scheduler was saved with it, switch it
  # to a Euler scheduler
  if isinstance(pipeline.scheduler, FlaxDDPMScheduler):
    noise_scheduler, noise_scheduler_state = FlaxEulerDiscreteScheduler.from_pretrained(
      config.pretrained_model_name_or_path,
      revision=config.revision, subfolder="scheduler", dtype=jnp.float32
    )
    pipeline.scheduler = noise_scheduler
    params["scheduler"] = noise_scheduler_state

  if config.lightning_repo:
    pipeline, params = load_sdxllightning_unet(config, pipeline, params)

  scheduler_state = params.pop("scheduler")
  old_params = params
  params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), old_params)
  params["scheduler"] = scheduler_state

  data_sharding = jax.sharding.NamedSharding(mesh,P(*config.data_sharding))

  sharding = PositionalSharding(devices_array).replicate()
  partial_device_put_replicated = functools.partial(device_put_replicated, sharding=sharding)
  params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])
  params["text_encoder_2"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder_2"])

  #p1 = {}
  #import pdb
  #pdb.set_trace()
  #p1.update(params['unet']['params'])
  #p1['aqt'] = params['unet']['aqt']
  #del params['unet']
  
  unet_state, unet_state_mesh_shardings, vae_state, vae_state_mesh_shardings  = get_states(mesh, None, rng, config, pipeline, params["unet"], params["vae"], training=False, q_v=params["unet"])
  

  del params["vae"]
  # unet_state.params = q_v
  # params["unet"] = jax.tree_util.tree_map(partial_device_put_replicated, params["unet"])
  # unet_state = InferenceState(pipeline.unet.apply, params=params["unet"])

  def get_unet_inputs(rng, config, batch_size, pipeline, params):
    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    prompt_ids = [config.prompt] * batch_size
    prompt_ids = tokenize(prompt_ids, pipeline)
    negative_prompt_ids = [config.negative_prompt] * batch_size
    negative_prompt_ids = tokenize(negative_prompt_ids, pipeline)
    guidance_scale = config.guidance_scale
    guidance_rescale = config.guidance_rescale
    num_inference_steps = config.num_inference_steps
    height = config.resolution
    width = config.resolution
    prompt_embeds, pooled_embeds = get_embeddings(prompt_ids, pipeline, params)
    batch_size = prompt_embeds.shape[0]
    negative_prompt_embeds, negative_pooled_embeds = get_embeddings(negative_prompt_ids, pipeline, params)
    add_time_ids = get_add_time_ids(
      (height, width), (0, 0), (height, width), prompt_embeds.shape[0], dtype=prompt_embeds.dtype
    )

    prompt_embeds = jnp.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)
    add_text_embeds = jnp.concatenate([negative_pooled_embeds, pooled_embeds], axis=0)
    add_time_ids = jnp.concatenate([add_time_ids, add_time_ids], axis=0)
    # Ensure model output will be `float32` before going into the scheduler
    guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)
    guidance_rescale = jnp.array([guidance_rescale], dtype=jnp.float32)

    latents_shape = (
      batch_size,
      pipeline.unet.config.in_channels,
      height // vae_scale_factor,
      width // vae_scale_factor,
    )

    latents = jax.random.normal(rng, shape=latents_shape, dtype=jnp.float32)

    scheduler_state = pipeline.scheduler.set_timesteps(
      params["scheduler"],
      num_inference_steps=num_inference_steps,
      shape=latents.shape
    )

    latents = latents * scheduler_state.init_noise_sigma

    added_cond_kwargs = {"text_embeds" : add_text_embeds, "time_ids" : add_time_ids}
    latents = jax.device_put(latents, data_sharding)
    prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
    #guidance_scale = jax.device_put(guidance_scale, PositionalSharding(devices_array).replicate())
    added_cond_kwargs['text_embeds'] = jax.device_put(added_cond_kwargs['text_embeds'], data_sharding)
    added_cond_kwargs['time_ids'] = jax.device_put(added_cond_kwargs['time_ids'], data_sharding)

    return latents, prompt_embeds, added_cond_kwargs, guidance_scale, guidance_rescale, scheduler_state

  def vae_decode(latents, state, pipeline):
    latents = 1 / pipeline.vae.config.scaling_factor * latents
    image = pipeline.vae.apply(
      {"params" : state.params},
      latents,
      method=pipeline.vae.decode
    ).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image
  
  def get_quantized_unet_vars(unet_state, params, rng, config, batch_size, pipeline):

    (latents,
    prompt_embeds,
    added_cond_kwargs,
    guidance_scale,
    guidance_rescale,
    scheduler_state) = get_unet_inputs(rng, config, batch_size, pipeline, params)

    loop_body_quant_p = jax.jit(functools.partial(loop_body_for_quantization, 
                                                  model=pipeline.unet,
                                                  pipeline=pipeline,
                                                  added_cond_kwargs=added_cond_kwargs,
                                                  prompt_embeds=prompt_embeds,
                                                  guidance_scale=guidance_scale,
                                                  guidance_rescale=guidance_rescale))
    # with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    quantized_unet_vars  = loop_body_quant_p(latents=latents, scheduler_state=scheduler_state, state=unet_state,rng=rng)


    return quantized_unet_vars

  def run_inference(unet_state, vae_state, params, rng, config, batch_size, pipeline):

    (latents,
    prompt_embeds,
    added_cond_kwargs,
    guidance_scale,
    guidance_rescale,
    scheduler_state) = get_unet_inputs(rng, config, batch_size, pipeline, params)

    loop_body_p = functools.partial(loop_body, model=pipeline.unet,
                        pipeline=pipeline,
                        added_cond_kwargs=added_cond_kwargs,
                        prompt_embeds=prompt_embeds,
                        guidance_scale=guidance_scale,
                        guidance_rescale=guidance_rescale)
    vae_decode_p = functools.partial(vae_decode, pipeline=pipeline)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      latents, _, _ = jax.lax.fori_loop(0, config.num_inference_steps,
                                        loop_body_p, (latents, scheduler_state, unet_state))
      image = vae_decode_p(latents, vae_state)
      return image
  
  #quantized_unet_vars = get_quantized_unet_vars(unet_state, params, rng, config, batch_size, pipeline)
  
  #del params
  #del pipeline
  #del unet_state
  #quant = quantizations.configure_quantization(config=config, lhs_quant_mode=aqt_flax.QuantMode.TRAIN, rhs_quant_mode=aqt_flax.QuantMode.SERVE)
  
  # pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
  #   config.pretrained_model_name_or_path,
  #   revision=config.revision,
  #   dtype=weight_dtype,
  #   split_head_dim=config.split_head_dim,
  #   norm_num_groups=config.norm_num_groups,
  #   attention_kernel=config.attention,
  #   flash_block_sizes=flash_block_sizes,
  #   mesh=mesh,
  #   quant=quant,
  # )

  # scheduler_state = params.pop("scheduler")
  # old_params = params
  # params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), old_params)
  # params["scheduler"] = scheduler_state

  # data_sharding = jax.sharding.NamedSharding(mesh,P(*config.data_sharding))

  # sharding = PositionalSharding(devices_array).replicate()
  # partial_device_put_replicated = functools.partial(device_put_replicated, sharding=sharding)
  # params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])
  # params["text_encoder_2"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder_2"])

  # unet_state = InferenceState(pipeline.unet.apply, params=quantized_unet_vars)
  # unet_state, unet_state_mesh_shardings, vae_state, vae_state_mesh_shardings  = get_states(mesh, None, rng, config, pipeline, quantized_unet_vars, params["vae"], training=False)
  #del params["vae"]
  #del params["unet"]
  
  
  p_run_inference = jax.jit(
    functools.partial(run_inference, rng=rng, config=config, batch_size=batch_size, pipeline=pipeline),
    in_shardings=(unet_state_mesh_shardings, vae_state_mesh_shardings, None),
    out_shardings=None
  )

  s = time.time()
  p_run_inference(unet_state, vae_state, params).block_until_ready()
  print("compile time: ", (time.time() - s))
  s = time.time()
  images = p_run_inference(unet_state, vae_state, params).block_until_ready()
  images.block_until_ready()
  print("inference time: ",(time.time() - s))
  s = time.time()
  images = p_run_inference(unet_state, vae_state, params).block_until_ready() #run_inference(unet_state, vae_state, latents, scheduler_state)
  images.block_until_ready()
  print("inference time: ",(time.time() - s))
  s = time.time()
  images = p_run_inference(unet_state, vae_state, params).block_until_ready() # run_inference(unet_state, vae_state, latents, scheduler_state)
  images.block_until_ready()
  print("inference time: ",(time.time() - s))
  s = time.time()
  activate_profiler(config)
  images = p_run_inference(unet_state, vae_state, params).block_until_ready()
  deactivate_profiler(config)
  images.block_until_ready()
  print("inference time: ",(time.time() - s))
  images = jax.experimental.multihost_utils.process_allgather(images)
  numpy_images = np.array(images)
  images = VaeImageProcessor.numpy_to_pil(numpy_images)
  for i, image in enumerate(images):
    image.save(f"image_sdxl_{i}.png")

  return images

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  # q_v = get_quantized_unet_variables(pyconfig.config)
  # breakpoint()
  # del q_v['params']
  # print(q_v.keys())
  # addedkw_args...., params, aqt
  run(pyconfig.config, q_v=None)

if __name__ == "__main__":
  app.run(main)
