
from absl import app
from typing import Sequence
from maxdiffusion.pipelines.ltx_video.ltx_video_pipeline import LTXVideoPipeline
from maxdiffusion import pyconfig
import jax.numpy as jnp
import os
import json

def run(config):
  pipeline = LTXVideoPipeline.from_pretrained(config)
  base_dir = os.path.dirname(__file__)

  ##load in model config
  config_path = os.path.join(base_dir, "models/ltx_video/xora_v1.2-13B-balanced-128.json")
  with open(config_path, "r") as f:
    model_config = json.load(f)
  example_inputs = {}
  batch_size, num_tokens = 4, 256
  input_shapes = {
    "latents": (batch_size, num_tokens, model_config["in_channels"]),
    "fractional_coords": (batch_size, 3, num_tokens),
    "prompt_embeds": (batch_size, 128, model_config["caption_channels"]),
    "timestep": (batch_size, 256), 
    "segment_ids": (batch_size, 256),
    "encoder_attention_segment_ids": (batch_size, 128),
  }
  for name, shape in input_shapes.items():
    example_inputs[name] = jnp.ones(
      shape, dtype=jnp.float32 if name not in ["attention_mask", "encoder_attention_mask"] else jnp.bool
    )
  noise_pred = pipeline(example_inputs)
  print(noise_pred)
  
  


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)