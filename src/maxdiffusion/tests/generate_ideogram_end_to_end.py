# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
from flax import nnx
import numpy as np
from PIL import Image

from maxdiffusion.models.ideogram import Ideogram4Transformer, Ideogram4Config, AutoEncoder, AutoEncoderParams
from maxdiffusion.pipelines.ideogram.ideogram_pipeline import IdeogramPipeline


def test_end_to_end():
  print("Initializing components...")
  rngs = nnx.Rngs(0)

  # Use small dimensions for fast testing
  config = Ideogram4Config(emb_dim=128, num_heads=2, in_channels=128, llm_features_dim=128, adanln_dim=128, num_layers=2)
  transformer = Ideogram4Transformer(rngs, config)

  params = AutoEncoderParams(resolution=256, in_channels=3, ch=32, out_ch=3, ch_mult=(1, 2), num_res_blocks=1, z_channels=32)
  autoencoder = AutoEncoder(rngs, params)

  # Dummy text encoder returning zero features
  def mock_text_encoder(token_ids, _attention_mask, _pos_2d):
    batch_size, seq_len = token_ids.shape
    return jnp.zeros((batch_size, seq_len, config.llm_features_dim), dtype=jnp.float32)

  pipeline = IdeogramPipeline(
      conditional_transformer=transformer,
      unconditional_transformer=transformer,
      autoencoder=autoencoder,
      text_encoder=mock_text_encoder,
      tokenizer=None,
  )

  print("Running generate...")
  # Generate image with dummy prompt
  # Use 256x256 image and 2 steps to make it fast
  images = pipeline.generate(prompts=["a cute dog"], height=256, width=256, num_steps=2, guidance_scale=7.0, seed=42)

  print("Generation complete! Output shape:", images.shape)

  # Save the output image
  image_np = np.array(images[0])
  image_np = (image_np * 255).astype(np.uint8)
  img = Image.fromarray(image_np)
  out_path = "ideogram_end_to_end_test.png"
  img.save(out_path)
  print(f"Saved test image to {out_path}")


if __name__ == "__main__":
  test_end_to_end()
