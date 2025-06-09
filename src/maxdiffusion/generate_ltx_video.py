from absl import app
from typing import Sequence
import jax
import json
from maxdiffusion.models.ltx_video.transformers.transformer3d import Transformer3DModel
import os
import functools
import jax.numpy as jnp
from maxdiffusion import FlaxAutoencoderKL, pyconfig, max_logging
from maxdiffusion.max_utils import (
    create_device_mesh,
    setup_initial_state,
)
from jax.sharding import Mesh, PositionalSharding, PartitionSpec as P


def validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond):
  print("prompts_embeds.shape: ", prompt_embeds.shape, prompt_embeds.dtype)
  print("fractional_coords.shape: ", fractional_coords.shape, fractional_coords.dtype)
  print("latents.shape: ", latents.shape, latents.dtype)
  print("noise_cond.shape: ", noise_cond.shape, noise_cond.dtype)

def run(config):
  key = jax.random.PRNGKey(0)

  devices_array = create_device_mesh(config) 
  mesh = Mesh(devices_array, config.mesh_axes)
  
  batch_size, text_tokens, num_tokens, features = 4, 256, 2048, 128
  base_dir = os.path.dirname(__file__)

  ##load in model config
  config_path = os.path.join(base_dir, "models/ltx_video/xora_v1.2-13B-balanced-128.json")
  with open(config_path, "r") as f:
    model_config = json.load(f)

  
  transformer = Transformer3DModel(**model_config, dtype=jnp.bfloat16, gradient_checkpointing="matmul_without_batch")
  transformer_param_shapes = transformer.init_weights(key, batch_size, text_tokens, num_tokens, features, eval_only = False) 
  
  key, split_key = jax.random.split(key)
  weights_init_fn = functools.partial(
      transformer.init_weights, 
      split_key, 
      batch_size, 
      text_tokens, 
      num_tokens, 
      features, 
      eval_only = False
  )

  transformer_state, transformer_state_shardings = setup_initial_state(  
    model=transformer,
    tx=None,
    config=config,
    mesh=mesh,
    weights_init_fn=weights_init_fn,
    model_params=None,
    training=False,
  )
  


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)



