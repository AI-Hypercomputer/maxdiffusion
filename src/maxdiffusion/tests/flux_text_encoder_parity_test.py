"""
Copyright 2026 Google LLC

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

# Parity between eager PyTorch and the Torchax-traced FLUX text encoders. The
# models here are small randomly-initialized stand-ins, so the test runs on CPU
# without touching the hub: what is under test is the Torchax wrapper, not the
# CLIP or T5 weights.

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel

from ..models.flux.text_encoders.torchax_text_encoders import TorchaxCLIPTextEncoder, TorchaxT5TextEncoder

SEQ_LEN = 16
BATCH_SIZE = 2
VOCAB_SIZE = 99


def _input_ids():
  ids = np.random.default_rng(0).integers(3, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN), dtype=np.int32)
  # CLIP pools at the EOS position, so every sequence needs one.
  ids[:, -1] = 2
  return ids


def _tiny_clip_text_model():
  torch.manual_seed(0)
  config = CLIPTextConfig(
      vocab_size=VOCAB_SIZE,
      hidden_size=32,
      intermediate_size=37,
      num_hidden_layers=2,
      num_attention_heads=4,
      max_position_embeddings=SEQ_LEN,
      bos_token_id=0,
      pad_token_id=1,
      eos_token_id=2,
      attn_implementation="eager",
  )
  return CLIPTextModel(config).eval()


def _tiny_t5_encoder_model():
  torch.manual_seed(0)
  config = T5Config(
      vocab_size=VOCAB_SIZE,
      d_model=32,
      d_ff=37,
      d_kv=8,
      num_layers=2,
      num_heads=4,
      is_encoder_decoder=False,
      attn_implementation="eager",
  )
  return T5EncoderModel(config).eval()


class FluxTextEncoderParityTest(unittest.TestCase):
  """Compares Torchax outputs against the eager PyTorch reference."""

  def test_clip_pooled_output_matches_eager(self):
    input_ids = _input_ids()
    model = _tiny_clip_text_model()
    with torch.no_grad():
      expected = model(input_ids=torch.from_numpy(input_ids).long()).pooler_output.numpy()

    encoder = TorchaxCLIPTextEncoder.from_torch(model, jnp.float32)
    actual = encoder(jnp.asarray(input_ids, dtype=jnp.int32))

    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-4, rtol=1e-4)

  def test_t5_last_hidden_state_matches_eager(self):
    input_ids = _input_ids()
    model = _tiny_t5_encoder_model()
    with torch.no_grad():
      expected = model(input_ids=torch.from_numpy(input_ids).long()).last_hidden_state.numpy()

    encoder = TorchaxT5TextEncoder.from_torch(model, jnp.float32)
    actual = encoder(jnp.asarray(input_ids, dtype=jnp.int32))

    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-4, rtol=1e-4)

  def test_place_and_offload_params_preserve_outputs(self):
    input_ids = jnp.asarray(_input_ids(), dtype=jnp.int32)
    encoder = TorchaxT5TextEncoder.from_torch(_tiny_t5_encoder_model(), jnp.float32)
    before = np.asarray(encoder(input_ids))

    encoder.place_params(jax.devices()[0])
    np.testing.assert_allclose(np.asarray(encoder(input_ids)), before, atol=1e-6, rtol=1e-6)

    encoder.offload_params()
    np.testing.assert_allclose(np.asarray(encoder(input_ids)), before, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
