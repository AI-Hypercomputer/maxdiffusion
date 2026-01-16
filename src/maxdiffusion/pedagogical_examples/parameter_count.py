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
from typing import Sequence
from absl import app
import jax
from diffusers import FlaxStableDiffusionXLPipeline
from diffusers import pyconfig


def run(config):

  pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
      config.pretrained_model_name_or_path,
      revision=config.revision,
      dtype=config.activations_dtype,
      split_head_dim=True,
      cache_dir="/data",
  )
  del params["scheduler"]
  unet_count = sum(x.size for x in jax.tree_util.tree_leaves(params["unet"]))
  vae_count = sum(x.size for x in jax.tree_util.tree_leaves(params["vae"]))
  text_encoder_count = sum(x.size for x in jax.tree_util.tree_leaves(params["text_encoder"]))
  text_encoder_2_count = sum(x.size for x in jax.tree_util.tree_leaves(params["text_encoder_2"]))
  print("Parameters count")
  print("unet: ", unet_count)
  print("vae: ", vae_count)
  print("text_encoder: ", text_encoder_count)
  print("text_encoder_2: ", text_encoder_2_count)
  print("total count: ", (unet_count + vae_count + text_encoder_count + text_encoder_2_count))


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
