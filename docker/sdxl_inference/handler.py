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
import io
import base64
from PIL import Image

from fastapi import FastAPI, Request

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache as cc
from flax.linen import partitioning as nn_partitioning
from jax.sharding import PositionalSharding

from maxdiffusion import (
    FlaxStableDiffusionXLPipeline
)


from maxdiffusion import pyconfig
from maxdiffusion.image_processor import VaeImageProcessor
from maxdiffusion.max_utils import (
  create_device_mesh,
  get_dtype,
  get_states,
  device_put_replicated
)

cc.initialize_cache(os.path.expanduser("~/jax_cache"))

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
pyconfig.initialize([None,os.path.join(THIS_DIR,'src/maxdiffusion','configs','base_xl.yml'),
      f"pretrained_model_name_or_path={os.environ['MODEL_ID']}",
      f"revision={os.environ['REVISION']}",f"dtype={os.environ['MODEL_ID']}",f"resolution={os.environ['RESOLUTION']}",
      "prompt=A magical castle in the middle of a forest, artistic drawing",
      "negative_prompt=purple, red","guidance_scale=9",
      "num_inference_steps=20","seed=47","per_device_batch_size=1",
      "run_name=sdxl-inference-test","split_head_dim=True"])

config = pyconfig.config

rng = jax.random.PRNGKey(config.seed)
devices_array = create_device_mesh(config)
mesh = Mesh(devices_array, config.mesh_axes)

batch_size = config.per_device_batch_size * jax.device_count()
_latents = np.load(f"{THIS_DIR}/latents.npy")
_latents = jnp.array([_latents[0]] * batch_size)
weight_dtype= get_dtype(config)

pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
  config.pretrained_model_name_or_path,
  revision=config.revision,
  dtype=weight_dtype,
  split_head_dim=config.split_head_dim
)

scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(weight_dtype), params)
params["scheduler"] = scheduler_state

data_sharding = jax.sharding.NamedSharding(mesh,P(*config.data_sharding))

sharding = PositionalSharding(devices_array).replicate()
partial_device_put_replicated = functools.partial(device_put_replicated, sharding=sharding)
params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder"])
params["text_encoder_2"] = jax.tree_util.tree_map(partial_device_put_replicated, params["text_encoder_2"])

unet_state, unet_state_mesh_shardings, vae_state, vae_state_mesh_shardings  = get_states(mesh, None, rng, config, pipeline, params["unet"], params["vae"], training=False)
del params["vae"]
del params["unet"]

def image_to_base64(image: Image.Image) -> str:
  """Convert a PIL image to a base64 string."""
  buffer = io.BytesIO()
  image.save(buffer, format="JPEG")
  image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
  return image_str

def loop_body(step, args, model, pipeline, added_cond_kwargs, prompt_embeds, guidance_scale):
  latents, scheduler_state, state = args
  latents_input = jnp.concatenate([latents] * 2)

  t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
  timestep = jnp.broadcast_to(t, latents_input.shape[0])

  latents_input = pipeline.scheduler.scale_model_input(scheduler_state, latents_input, t)
  noise_pred = model.apply(
    {"params" : state.params},
    jnp.array(latents_input),
    jnp.array(timestep, dtype=jnp.int32),
    encoder_hidden_states=prompt_embeds,
    added_cond_kwargs=added_cond_kwargs
  ).sample

  noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
  noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

  latents, scheduler_state = pipeline.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()

  return latents, scheduler_state, state

def get_add_time_ids(original_size, crops_coords_top_left, target_size, bs, dtype):
  add_time_ids = list(original_size + crops_coords_top_left + target_size)
  add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
  return add_time_ids

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

def get_prompt_ids(prompts, batch_size):
  if len(prompts) != batch_size:
    prompts += [prompts[0]] * (batch_size - len(prompts))
  prompt_ids = tokenize(prompts, pipeline)
  return prompt_ids

def get_unet_inputs(rng, config, batch_size, pipeline, params, prompt_ids):
  # pad with first element if it doesn't fill batch_size
  # if len(prompts) != batch_size:
  #   prompts = [prompts[0]] * (batch_size - len(prompts))
  # prompt_ids = tokenize(prompt_ids, pipeline)
  negative_prompt_ids = ["normal quality, low quality, worst quality, low res, blurry, nsfw, nude"] * batch_size
  negative_prompt_ids = tokenize(negative_prompt_ids, pipeline)
  guidance_scale = config.guidance_scale
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

  scheduler_state = pipeline.scheduler.set_timesteps(
    params["scheduler"],
    num_inference_steps=num_inference_steps,
    shape=_latents.shape
  )

  latents = _latents * scheduler_state.init_noise_sigma

  added_cond_kwargs = {"text_embeds" : add_text_embeds, "time_ids" : add_time_ids}
  latents = jax.device_put(latents, data_sharding)
  prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
  guidance_scale = jax.device_put(guidance_scale, PositionalSharding(devices_array).replicate())
  added_cond_kwargs['text_embeds'] = jax.device_put(added_cond_kwargs['text_embeds'], data_sharding)
  added_cond_kwargs['time_ids'] = jax.device_put(added_cond_kwargs['time_ids'], data_sharding)

  return latents, prompt_embeds, added_cond_kwargs, guidance_scale, scheduler_state


def vae_decode(latents, state, pipeline):
  latents = 1 / pipeline.vae.config.scaling_factor * latents
  image = pipeline.vae.apply(
    {"params" : state.params},
    latents,
    method=pipeline.vae.decode
  ).sample
  image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
  return image

def run_inference(unet_state, vae_state, params, prompt_ids, rng, config, batch_size, pipeline):

  (latents,
  prompt_embeds,
  added_cond_kwargs,
  guidance_scale,
  scheduler_state) = get_unet_inputs(rng, config, batch_size, pipeline, params, prompt_ids)

  loop_body_p = functools.partial(loop_body, model=pipeline.unet,
                      pipeline=pipeline,
                      added_cond_kwargs=added_cond_kwargs,
                      prompt_embeds=prompt_embeds,
                      guidance_scale=guidance_scale)
  vae_decode_p = functools.partial(vae_decode, pipeline=pipeline)

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    latents, _, _ = jax.lax.fori_loop(0, config.num_inference_steps,
                                      loop_body_p, (latents, scheduler_state, unet_state))
    images = vae_decode_p(latents, vae_state)
    return images

p_run_inference = jax.jit(
  functools.partial(run_inference, rng=rng, config=config, batch_size=batch_size, pipeline=pipeline),
  in_shardings=(unet_state_mesh_shardings, vae_state_mesh_shardings, None, None),
  out_shardings=None
)

prompt_ids = get_prompt_ids([config.prompt], batch_size)
images = p_run_inference(unet_state, vae_state, params, prompt_ids)

app = FastAPI()

@app.get("/health", status_code=200)
def health():
    return {}

@app.post("/predict")
async def predict(request: Request):
  body = await request.json()
  instances = body["instances"]
  retval = []
  for instance in instances:
    prompt = instance["prompt"] # list
    prompt_ids = get_prompt_ids(prompt, batch_size)
    images = p_run_inference(unet_state, vae_state, params, prompt_ids)
    images = VaeImageProcessor.numpy_to_pil(np.array(images))

    retval_images = []
    for image in images:
      retval_images.append(image_to_base64(image))

    retval.append({"instance" : instance, "images" : retval_images})
  return {"predictions" : retval}
