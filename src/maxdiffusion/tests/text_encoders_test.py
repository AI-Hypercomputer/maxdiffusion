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

from transformers import CLIPTokenizer, FlaxCLIPTextModel
from transformers import T5TokenizerFast, T5EncoderModel

from ..generate_flux import get_clip_prompt_embeds, get_t5_prompt_embeds

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TextEncoderTest(unittest.TestCase):
  """Test text encoders"""

  def setUp(self):
    TextEncoderTest.dummy_data = {}
  
  def test_flux_t5_text_encoder(self):

    text_encoder_2_pt = T5EncoderModel.from_pretrained(
      "black-forest-labs/FLUX.1-dev",
      subfolder="text_encoder_2",
    )

    tokenizer_2 = T5TokenizerFast.from_pretrained(
      "black-forest-labs/FLUX.1-dev",
      subfolder="tokenizer_2",
    )

    embeds = get_t5_prompt_embeds("A dog on a skateboard", 2, tokenizer_2, text_encoder_2_pt)

    assert embeds.shape == (2, 512, 4096)

  def test_flux_clip_text_encoder(self):

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


