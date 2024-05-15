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

import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax


def load_sdxllightning_unet(config, pipeline, params):
  """Load lightning """
  if not config.lightning_from_pt:
    raise ValueError("Only loading lightning models from Pytorch is currently supported.")
  unet_lightning_state_dict = load_file(hf_hub_download(config.lightning_repo, config.lightning_ckpt), device="cpu")
  flax_unet_dict = convert_pytorch_state_dict_to_flax(unet_lightning_state_dict, pipeline.unet)
  params["unet"] = flax_unet_dict
  return pipeline, params

def vae_apply(images, sample_rng, vae, vae_params):
  """Apply vae encoder to images."""
  vae_outputs = vae.apply(
    {"params" : vae_params}, images,
      deterministic=True, method=vae.encode
  )
  latents = vae_outputs.latent_dist.sample(sample_rng)
  latents = jnp.transpose(latents, (0, 3, 1, 2))
  latents = latents * vae.config.scaling_factor

  return latents

def transform_images(examples, image_column, image_resolution, rng, global_batch_size, p_vae_apply = None):
    """Preprocess images to latents."""
    images = list(examples[image_column])
    images = [np.asarray(image) for image in images]
    tensor_list = []
    for image in images:
        image = tf.image.resize(image, [image_resolution, image_resolution], method="bilinear", antialias=True)
        image = image / 255.0
        image = (image - 0.5) / 0.5
        image = tf.transpose(image, perm=[2,0,1])
        tensor_list.append(image)
    if p_vae_apply:
        tensor_list = np.stack(tensor_list)
        ds_length = tensor_list.shape[0]
        iters = ds_length // global_batch_size
        latents_list = []
        for i in range(0, iters * global_batch_size, global_batch_size):
            sample_rng, rng = jax.random.split(rng)
            latents = p_vae_apply(tensor_list[i:i+global_batch_size], sample_rng)
            latents_list.append(latents)

        latents_list = np.stack(latents_list)
        b1, b2, c, l1, l2 = latents_list.shape
        latents_list = np.reshape(latents_list, (b1*b2,c, l1, l2))

        # TODO (Juan Acevedo): do last iteration, its required for the Pyarrow dataset
        # to not break due to items being fewer than expected. Is there a better way?
        sample_rng, rng = jax.random.split(rng)
        latents = p_vae_apply(tensor_list[i+global_batch_size:], sample_rng)

        examples["pixel_values"] = np.append(latents_list, latents, axis=0)
    else:
        examples["pixel_values"] = tf.stack(tensor_list)

    return examples

def get_add_time_ids(original_size, crops_coords_top_left, target_size, bs, dtype):
  add_time_ids = list(original_size + crops_coords_top_left + target_size)
  add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
  return add_time_ids