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
import pytest
from absl.testing import absltest

import jax.numpy as jnp
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import T5TokenizerFast, T5EncoderModel

from ..generate_flux import get_clip_prompt_embeds, get_t5_prompt_embeds
from ..models.flux.text_encoders.torchax_text_encoders import TorchaxCLIPTextEncoder, TorchaxT5TextEncoder

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

FLUX_DEV = "black-forest-labs/FLUX.1-dev"


class TextEncoderTest(unittest.TestCase):
  """Test text encoders"""

  def setUp(self):
    TextEncoderTest.dummy_data = {}

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_flux_t5_text_encoder(self):
    text_encoder = TorchaxT5TextEncoder.from_torch(
        T5EncoderModel.from_pretrained(FLUX_DEV, subfolder="text_encoder_2", dtype=torch.bfloat16),
        jnp.bfloat16,
    )

    tokenizer_2 = T5TokenizerFast.from_pretrained(FLUX_DEV, subfolder="tokenizer_2")

    embeds = get_t5_prompt_embeds("A dog on a skateboard", 2, tokenizer_2, text_encoder)

    assert embeds.shape == (2, 512, 4096)

  @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run smoke tests on Github Actions")
  def test_flux_clip_text_encoder(self):
    text_encoder = TorchaxCLIPTextEncoder.from_torch(
        CLIPTextModel.from_pretrained(FLUX_DEV, subfolder="text_encoder", dtype=torch.bfloat16),
        jnp.bfloat16,
    )
    tokenizer = CLIPTokenizer.from_pretrained(FLUX_DEV, subfolder="tokenizer")
    embeds = get_clip_prompt_embeds("A cat riding a skateboard", 2, tokenizer, text_encoder)
    assert embeds.shape == (2, 768)


if __name__ == "__main__":
  absltest.main()
