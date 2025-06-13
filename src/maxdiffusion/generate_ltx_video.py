from json import encoder
from absl import app
from typing import Sequence
import jax
from flax import linen as nn
import json
from flax.linen import partitioning as nn_partitioning
from maxdiffusion.models.ltx_video.transformers.transformer3d import Transformer3DModel
import os
import functools
import jax.numpy as jnp
from maxdiffusion import pyconfig
from maxdiffusion.max_utils import (
    create_device_mesh,
    setup_initial_state,
    get_memory_allocations,
)
from jax.sharding import Mesh, PartitionSpec as P
import orbax.checkpoint as ocp


def validate_transformer_inputs(prompt_embeds, fractional_coords, latents, noise_cond, segment_ids, encoder_attention_segment_ids):
  print("prompts_embeds.shape: ", prompt_embeds.shape, prompt_embeds.dtype)
  print("fractional_coords.shape: ", fractional_coords.shape, fractional_coords.dtype)
  print("latents.shape: ", latents.shape, latents.dtype)
  print("noise_cond.shape: ", noise_cond.shape, noise_cond.dtype)
  print("noise_cond.shape: ", noise_cond.shape, noise_cond.dtype)
  print("segment_ids.shape: ", segment_ids.shape, segment_ids.dtype)
  print("encoder_attention_segment_ids.shape: ", encoder_attention_segment_ids.shape, encoder_attention_segment_ids.dtype)

def run(config):
  key = jax.random.PRNGKey(0)

  devices_array = create_device_mesh(config) 
  mesh = Mesh(devices_array, config.mesh_axes)
  
  base_dir = os.path.dirname(__file__)

  ##load in model config
  config_path = os.path.join(base_dir, "models/ltx_video/xora_v1.2-13B-balanced-128.json")
  with open(config_path, "r") as f:
    model_config = json.load(f)
  relative_ckpt_path = model_config["ckpt_path"]

  ignored_keys = ["_class_name", "_diffusers_version", "_name_or_path", "causal_temporal_positioning", "in_channels", "ckpt_path"]
  in_channels = model_config["in_channels"]
  for name in ignored_keys:
    if name in model_config:
      del model_config[name]
  
 
  transformer = Transformer3DModel(**model_config, dtype=jnp.float32, gradient_checkpointing="matmul_without_batch", sharding_mesh=mesh)
  transformer_param_shapes = transformer.init_weights(in_channels, model_config['caption_channels'], eval_only = True)
  
  weights_init_fn = functools.partial(
      transformer.init_weights, 
      in_channels, 
      model_config['caption_channels'],
      eval_only = True
  )

  absolute_ckpt_path = os.path.abspath(relative_ckpt_path)

  checkpoint_manager = ocp.CheckpointManager(absolute_ckpt_path)
  transformer_state, transformer_state_shardings = setup_initial_state(
    model=transformer,
    tx=None,
    config=config,
    mesh=mesh,
    weights_init_fn=weights_init_fn,
    checkpoint_manager=checkpoint_manager,
    checkpoint_item=" ",
    model_params=None,
    training=False,
  )

  



def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)





