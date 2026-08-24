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

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxdiffusion.models.ideogram.transformer_ideogram import (
    Ideogram4Transformer,
    Ideogram4Config,
    _splash_attention,
    _PAD_SEGMENT,
    _VALID_SEGMENT,
)
from maxdiffusion.models.ideogram.autoencoder_ideogram import AutoEncoder, AutoEncoderParams
from maxdiffusion.models.modeling_flax_pytorch_utils import rename_key_and_reshape_tensor

TOY = dict(emb_dim=128, num_heads=2, in_channels=64, llm_features_dim=128, adanln_dim=128, num_layers=2)


def _toy_inputs(seq_len=1024, text_len=256, pad_len=64, llm_dim=128, in_channels=64):
  key = jax.random.PRNGKey(0)
  k1, k2 = jax.random.split(key)
  llm = jax.random.normal(k1, (1, text_len, llm_dim), jnp.float32)
  x = jax.random.normal(k2, (1, seq_len, in_channels), jnp.float32)
  t = jnp.full((1,), 0.5, jnp.float32)
  position_ids = jnp.zeros((1, seq_len, 3), jnp.int32)
  # Ideogram left-pads text with SEQUENCE_PADDING_INDICATOR (-1).
  segment_ids = jnp.concatenate([jnp.full((1, pad_len), -1, jnp.int32), jnp.ones((1, seq_len - pad_len), jnp.int32)], 1)
  indicator = jnp.concatenate(
      [
          jnp.zeros((1, pad_len), jnp.int32),
          jnp.full((1, text_len - pad_len), 3, jnp.int32),  # LLM_TOKEN_INDICATOR
          jnp.full((1, seq_len - text_len), 2, jnp.int32),  # OUTPUT_IMAGE_INDICATOR
      ],
      1,
  )
  return llm, x, t, position_ids, segment_ids, indicator


class TestIdeogram(unittest.TestCase):

  def test_instantiate_transformer(self):
    model = Ideogram4Transformer(nnx.Rngs(0), Ideogram4Config(**TOY))
    self.assertIsNotNone(model)

  def test_instantiate_autoencoder(self):
    params = AutoEncoderParams(resolution=32, in_channels=3, ch=32, out_ch=3, ch_mult=(1, 2), num_res_blocks=1, z_channels=8)
    ae = AutoEncoder(nnx.Rngs(0), params)
    self.assertIsNotNone(ae)

  def test_splash_matches_dense_mask(self):
    """The splash segment-id mask must equal the dense segment_ids equality mask."""
    b, h, seq_len, d = 1, 4, 1024, 256
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(k1, (b, h, seq_len, d), jnp.float32) * 0.3
    k = jax.random.normal(k2, (b, h, seq_len, d), jnp.float32) * 0.3
    v = jax.random.normal(k3, (b, h, seq_len, d), jnp.float32) * 0.3

    for pad_len in (0, 137):
      seg = np.full((b, seq_len), 1, np.int32)
      seg[:, :pad_len] = -1
      seg = jnp.asarray(seg)

      mask = (seg[:, :, None] == seg[:, None, :])[:, None]
      scale = 1.0 / np.sqrt(d)
      w = jnp.where(mask, jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale, -1e10)
      ref = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(w, -1), v)

      splash_seg = jnp.where(seg > 0, _VALID_SEGMENT, _PAD_SEGMENT).astype(jnp.int32)
      got = _splash_attention(q * scale, k, v, splash_seg, 512, 512)

      valid = seg[0] > 0
      err = float(jnp.abs(ref - got)[:, :, valid, :].max())
      self.assertLess(err, 2e-3, f"splash != dense for pad_len={pad_len} (max abs err {err})")

  def test_attention_backends_agree_end_to_end(self):
    dense = Ideogram4Transformer(nnx.Rngs(0), Ideogram4Config(attention="dot_product", **TOY))
    flash = Ideogram4Transformer(nnx.Rngs(0), Ideogram4Config(attention="flash", **TOY))
    args = _toy_inputs()
    out_dense = dense(*args)
    out_flash = flash(*args)
    valid = args[4][0] > 0
    err = float(jnp.abs(out_dense - out_flash)[:, valid].max())
    scale = float(jnp.abs(out_dense).max())
    self.assertLess(err / scale, 5e-2, f"flash and dot_product disagree (rel err {err / scale})")

  def test_config_dtypes_reach_the_parameters(self):
    """weights_dtype/activations_dtype in base_ideogram.yml must not be inert."""
    model = Ideogram4Transformer(
        nnx.Rngs(0), Ideogram4Config(weights_dtype=jnp.bfloat16, activations_dtype=jnp.bfloat16, **TOY)
    )
    self.assertEqual(model.layers[0].attention.qkv.kernel.value.dtype, jnp.bfloat16)
    self.assertEqual(model.layers[0].feed_forward.w1.kernel.value.dtype, jnp.bfloat16)

  def test_config_is_hashable(self):
    """The config lives on the module and therefore inside the nnx graphdef,
    which is a static argument to the jitted denoise step."""
    hash(Ideogram4Config(**TOY))

  def test_logical_axis_rules_resolve_to_a_partition_spec(self):
    """Without axis names on the modules, the logical_axis_rules block in
    base_ideogram.yml is inert and every param silently replicates."""
    model = Ideogram4Transformer(nnx.Rngs(0), Ideogram4Config(**TOY))
    _, state, _ = nnx.split(model, nnx.Param, ...)
    rules = (("embed", "fsdp"), ("heads", "tensor"), ("mlp", "tensor"))
    state = jax.tree.map(
        lambda vs: (vs.set_metadata(sharding_rules=rules), vs)[1],
        state,
        is_leaf=lambda x: isinstance(x, nnx.VariableState),
    )
    specs = {".".join(str(p) for p in k): v for k, v in nnx.get_partition_spec(state).flat_state()}
    self.assertEqual(specs["layers.0.attention.qkv.kernel"].value, jax.sharding.PartitionSpec("fsdp", "tensor"))
    self.assertEqual(specs["layers.0.feed_forward.w1.kernel"].value, jax.sharding.PartitionSpec("fsdp", "tensor"))

  def test_direct_match_is_opt_in(self):
    """The 'direct match' short-circuit skips the linear-layer transpose, so it
    must not fire for models that did not ask for it."""
    flax_state = {("layer", "weight"): jnp.zeros((4, 8))}
    pt_key = ("layer", "weight")
    tensor = np.zeros((4, 8), np.float32)

    key_off, out_off = rename_key_and_reshape_tensor(pt_key, tensor, flax_state)
    self.assertEqual((key_off, out_off.shape), (("layer", "kernel"), (8, 4)), "default path must still transpose")

    key_on, out_on = rename_key_and_reshape_tensor(pt_key, tensor, flax_state, allow_direct_match=True)
    self.assertEqual((key_on, out_on.shape), (pt_key, (4, 8)))

  def test_embedding_branch_returns_the_embedding_key(self):
    """Regression test for the shared-file fix: the embedding branch used to
    return `renamed_pt_tuple_key`, which was still (..., 'scale')."""
    flax_state = {("emb", "embedding"): jnp.zeros((4, 8))}
    key, _ = rename_key_and_reshape_tensor(("emb", "weight"), np.zeros((4, 8), np.float32), flax_state)
    self.assertEqual(key, ("emb", "embedding"))


if __name__ == "__main__":
  unittest.main()
