"""
 Copyright 2025 Google LLC

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
import unittest
from absl.testing import absltest

import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp

from maxdiffusion.transformers import CLIPTokenizer, FlaxCLIPTextModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TextEncoderTest(unittest.TestCase):
  """Test text encoders"""

  def setUp(self):
    TextEncoderTest.dummy_data = {}
  
  def test_flux_text_encoders(self):

    def get_clip_prompt_embeds(
      prompt,
      num_images_per_prompt,
      tokenizer,
      text_encoder
    ):
      prompt = [prompt] if isinstance(prompt, str) else prompt
      batch_size = len(prompt)

      text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="np"
      )

      text_input_ids = text_inputs.input_ids
  
      prompt_embeds = text_encoder(text_input_ids, params=text_encoder.params, train=False)
      prompt_embeds = prompt_embeds.pooler_output
      prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=-1)
      prompt_embeds = np.reshape(prompt_embeds, (batch_size * num_images_per_prompt, -1))
      return prompt_embeds

    text_encoder = FlaxCLIPTextModel.from_pretrained(
      "black-forest-labs/FLUX.1-dev",
      subfolder="text_encoder",
      from_pt=True,
      dtype="bfloat16"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
      "black-forest-labs/FLUX.1-dev",
      subfolder="tokenizer",
      dtype="bfloat16"
    )
    embeds = get_clip_prompt_embeds("A cat riding a skateboard", 2, tokenizer, text_encoder)
    assert embeds.shape == (2, 768)


