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

import os
import unittest
from importlib import resources

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh

jax.config.update("jax_platforms", "cpu")

from diffusers.models.attention import FeedForward as HFFeedForward
from diffusers.models.transformers.transformer_wan import (
    WanImageEmbedding as HFWanImageEmbedding,
    WanRotaryPosEmbed as HFWanRotaryPosEmbed,
    WanTimeTextImageEmbedding as HFWanTimeTextImageEmbedding,
    WanTransformer3DModel as HFWanTransformer3DModel,
    WanTransformerBlock as HFWanTransformerBlock,
)

from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.embeddings_flax import NNXWanImageEmbedding
from maxdiffusion.models.modeling_flax_pytorch_utils import rename_key
from maxdiffusion.models.wan.transformers.transformer_wan import (
    WanFeedForward,
    WanModel,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)
from maxdiffusion.models.wan.wan_utils import get_key_and_value

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def to_numpy(array):
  if isinstance(array, torch.Tensor):
    if array.dtype == torch.bfloat16:
      array = array.float()
    return array.detach().cpu().numpy()
  return np.asarray(array)


def assert_allclose(test_case, actual, expected, *, atol=1e-6, rtol=1e-6):
  test_case.assertEqual(to_numpy(actual).shape, to_numpy(expected).shape)
  np.testing.assert_allclose(to_numpy(actual), to_numpy(expected), atol=atol, rtol=rtol)


def copy_linear_params(max_module, hf_module):
  max_module.kernel[...] = jnp.asarray(to_numpy(hf_module.weight).T)
  if getattr(max_module, "bias", None) is not None and hf_module.bias is not None:
    max_module.bias[...] = jnp.asarray(to_numpy(hf_module.bias))


def copy_fp32_layer_norm_params(max_module, hf_module):
  max_module.layer_norm.scale[...] = jnp.asarray(to_numpy(hf_module.weight))
  max_module.layer_norm.bias[...] = jnp.asarray(to_numpy(hf_module.bias))


def copy_wan_image_embedding_params(max_module, hf_module):
  copy_fp32_layer_norm_params(max_module.norm1, hf_module.norm1)
  max_module.ff.net_0.kernel[...] = jnp.asarray(to_numpy(hf_module.ff.net[0].proj.weight).T)
  max_module.ff.net_0.bias[...] = jnp.asarray(to_numpy(hf_module.ff.net[0].proj.bias))
  max_module.ff.net_2.kernel[...] = jnp.asarray(to_numpy(hf_module.ff.net[2].weight).T)
  max_module.ff.net_2.bias[...] = jnp.asarray(to_numpy(hf_module.ff.net[2].bias))
  copy_fp32_layer_norm_params(max_module.norm2, hf_module.norm2)
  if max_module.pos_embed is not None and hf_module.pos_embed is not None:
    max_module.pos_embed[...] = jnp.asarray(to_numpy(hf_module.pos_embed))


def copy_wan_time_text_image_embedding_params(max_module, hf_module):
  copy_linear_params(max_module.time_embedder.linear_1, hf_module.time_embedder.linear_1)
  copy_linear_params(max_module.time_embedder.linear_2, hf_module.time_embedder.linear_2)
  copy_linear_params(max_module.time_proj, hf_module.time_proj)
  copy_linear_params(max_module.text_embedder.linear_1, hf_module.text_embedder.linear_1)
  copy_linear_params(max_module.text_embedder.linear_2, hf_module.text_embedder.linear_2)
  if max_module.image_embedder is not None and hf_module.image_embedder is not None:
    copy_wan_image_embedding_params(max_module.image_embedder, hf_module.image_embedder)


def copy_wan_feed_forward_params(max_module, hf_module):
  copy_linear_params(max_module.act_fn.proj, hf_module.net[0].proj)
  copy_linear_params(max_module.proj_out, hf_module.net[2])


def copy_wan_attention_params(max_module, hf_module):
  copy_linear_params(max_module.query, hf_module.to_q)
  copy_linear_params(max_module.key, hf_module.to_k)
  copy_linear_params(max_module.value, hf_module.to_v)
  copy_linear_params(max_module.proj_attn, hf_module.to_out[0])
  max_module.norm_q.scale[...] = jnp.asarray(to_numpy(hf_module.norm_q.weight))
  max_module.norm_k.scale[...] = jnp.asarray(to_numpy(hf_module.norm_k.weight))


def copy_wan_transformer_block_params(max_module, hf_module):
  max_module.adaln_scale_shift_table[...] = jnp.asarray(to_numpy(hf_module.scale_shift_table))
  copy_wan_attention_params(max_module.attn1, hf_module.attn1)
  copy_wan_attention_params(max_module.attn2, hf_module.attn2)
  copy_fp32_layer_norm_params(max_module.norm2, hf_module.norm2)
  copy_wan_feed_forward_params(max_module.ffn, hf_module.ffn)


def map_hf_wan_state_to_local(max_model, hf_model, num_layers):
  state = nnx.state(max_model)
  flat_vars = dict(nnx.to_flat_state(state))
  random_flax_state_dict = {
      tuple(str(item) for item in key): value for key, value in flatten_dict(state.to_pure_dict()).items()
  }
  flax_state_dict = {}

  for pt_key, tensor in hf_model.state_dict().items():
    if "norm_added_q" in pt_key:
      continue

    renamed_pt_key = rename_key(pt_key)

    if "condition_embedder" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("time_embedding_0", "time_embedder.linear_1")
      renamed_pt_key = renamed_pt_key.replace("time_embedding_2", "time_embedder.linear_2")
      renamed_pt_key = renamed_pt_key.replace("time_projection_1", "time_proj")
      renamed_pt_key = renamed_pt_key.replace("text_embedding_0", "text_embedder.linear_1")
      renamed_pt_key = renamed_pt_key.replace("text_embedding_2", "text_embedder.linear_2")

    if "image_embedder" in renamed_pt_key:
      if "net.0.proj" in renamed_pt_key:
        renamed_pt_key = renamed_pt_key.replace("net.0.proj", "net_0")
      elif "net_0.proj" in renamed_pt_key:
        renamed_pt_key = renamed_pt_key.replace("net_0.proj", "net_0")
      if "net.2" in renamed_pt_key:
        renamed_pt_key = renamed_pt_key.replace("net.2", "net_2")
      renamed_pt_key = renamed_pt_key.replace("norm1", "norm1.layer_norm")
      if "norm1" in renamed_pt_key or "norm2" in renamed_pt_key:
        renamed_pt_key = renamed_pt_key.replace("weight", "scale")
        renamed_pt_key = renamed_pt_key.replace("kernel", "scale")

    renamed_pt_key = renamed_pt_key.replace("blocks_", "blocks.")
    renamed_pt_key = renamed_pt_key.replace(".scale_shift_table", ".adaln_scale_shift_table")
    renamed_pt_key = renamed_pt_key.replace("to_out_0", "proj_attn")
    renamed_pt_key = renamed_pt_key.replace("ffn.net_2", "ffn.proj_out")
    renamed_pt_key = renamed_pt_key.replace("ffn.net_0", "ffn.act_fn")
    renamed_pt_key = renamed_pt_key.replace("norm2", "norm2.layer_norm")

    pt_tuple_key = tuple(renamed_pt_key.split("."))
    flax_key, flax_tensor = get_key_and_value(
        pt_tuple_key,
        to_numpy(tensor),
        flax_state_dict,
        random_flax_state_dict,
        False,
        num_layers,
    )
    flax_state_dict[flax_key] = jnp.asarray(flax_tensor)

  missing_keys = [key for key in flax_state_dict if key not in flat_vars]
  for key, value in flax_state_dict.items():
    if key in flat_vars:
      flat_vars[key][...] = value

  return missing_keys, flax_state_dict


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Don't run WAN parity tests on Github Actions")
class WanCommonModuleParityTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    with resources.as_file(resources.files("maxdiffusion.configs").joinpath("base_wan_14b.yml")) as config_path:
      pyconfig.initialize([None, os.fspath(config_path)], unittest=True)
    config = pyconfig.config
    cls.logical_axis_rules = config.logical_axis_rules
    cls.mesh = Mesh(create_device_mesh(config), config.mesh_axes)

  def setUp(self):
    torch.manual_seed(0)
    self.rngs = nnx.Rngs(jax.random.key(0))

  def test_wan_rotary_pos_embed_parity(self):
    hf_module = HFWanRotaryPosEmbed(attention_head_dim=12, patch_size=(1, 2, 2), max_seq_len=32)
    max_module = WanRotaryPosEmbed(attention_head_dim=12, patch_size=(1, 2, 2), max_seq_len=32)

    hidden_states = torch.randn(1, 12, 3, 4, 4)
    freqs_cos, freqs_sin = hf_module(hidden_states)
    hf_complex = to_numpy(freqs_cos)[..., 0::2] + 1j * to_numpy(freqs_sin)[..., 1::2]
    expected = np.transpose(hf_complex, (0, 2, 1, 3))

    actual = max_module(jnp.asarray(np.transpose(to_numpy(hidden_states), (0, 2, 3, 4, 1))))

    assert_allclose(self, actual, expected, atol=0.0, rtol=0.0)

  def test_wan_image_embedding_parity(self):
    hf_module = HFWanImageEmbedding(4, 8, pos_embed_seq_len=None).eval()
    max_module = NNXWanImageEmbedding(
        rngs=self.rngs,
        in_features=4,
        out_features=8,
        pos_embed_seq_len=None,
        dtype=jnp.float32,
        weights_dtype=jnp.float32,
        precision=None,
        flash_min_seq_length=4096,
    )
    copy_wan_image_embedding_params(max_module, hf_module)

    encoder_hidden_states_image = torch.randn(2, 3, 4)
    expected = hf_module(encoder_hidden_states_image)
    actual, attention_mask = max_module(jnp.asarray(to_numpy(encoder_hidden_states_image)))

    self.assertIsNone(attention_mask)
    assert_allclose(self, actual, expected, atol=3e-4, rtol=3e-4)

  def test_wan_time_text_image_embedding_parity(self):
    hf_module = HFWanTimeTextImageEmbedding(
        dim=8,
        time_freq_dim=8,
        time_proj_dim=48,
        text_embed_dim=6,
        image_embed_dim=4,
        pos_embed_seq_len=None,
    ).eval()
    max_module = WanTimeTextImageEmbedding(
        rngs=self.rngs,
        dim=8,
        time_freq_dim=8,
        time_proj_dim=48,
        text_embed_dim=6,
        image_embed_dim=4,
        pos_embed_seq_len=None,
        flash_min_seq_length=4096,
    )
    copy_wan_time_text_image_embedding_params(max_module, hf_module)

    timestep = torch.tensor([3, 7], dtype=torch.long)
    encoder_hidden_states = torch.randn(2, 5, 6)
    encoder_hidden_states_image = torch.randn(2, 3, 4)

    expected_temb, expected_tproj, expected_text, expected_image = hf_module(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    actual_temb, actual_tproj, actual_text, actual_image, actual_mask = max_module(
        jnp.asarray(to_numpy(timestep)),
        jnp.asarray(to_numpy(encoder_hidden_states)),
        jnp.asarray(to_numpy(encoder_hidden_states_image)),
    )

    self.assertIsNone(actual_mask)
    assert_allclose(self, actual_temb, expected_temb, atol=1e-7, rtol=1e-7)
    assert_allclose(self, actual_tproj, expected_tproj, atol=1e-7, rtol=1e-7)
    assert_allclose(self, actual_text, expected_text, atol=2e-7, rtol=1e-6)
    assert_allclose(self, actual_image, expected_image, atol=2e-4, rtol=3e-4)

  def test_wan_feed_forward_parity(self):
    hf_module = HFFeedForward(8, inner_dim=16, activation_fn="gelu-approximate").eval()
    max_module = WanFeedForward(rngs=self.rngs, dim=8, inner_dim=16, activation_fn="gelu-approximate")
    copy_wan_feed_forward_params(max_module, hf_module)

    hidden_states = torch.randn(2, 5, 8)
    expected = hf_module(hidden_states)
    actual = max_module(jnp.asarray(to_numpy(hidden_states)))

    assert_allclose(self, actual, expected, atol=1e-7, rtol=1e-6)

  def test_wan_transformer_block_parity(self):
    hf_module = HFWanTransformerBlock(
        dim=8,
        ffn_dim=16,
        num_heads=2,
        qk_norm="rms_norm_across_heads",
        cross_attn_norm=True,
        eps=1e-6,
    ).eval()

    with self.mesh, nn_partitioning.axis_rules(self.logical_axis_rules):
      max_module = WanTransformerBlock(
          rngs=self.rngs,
          dim=8,
          ffn_dim=16,
          num_heads=2,
          qk_norm="rms_norm_across_heads",
          cross_attn_norm=True,
          eps=1e-6,
          attention="dot_product",
          flash_min_seq_length=4096,
          mesh=self.mesh,
      )
      copy_wan_transformer_block_params(max_module, hf_module)

      rope_hf = HFWanRotaryPosEmbed(attention_head_dim=4, patch_size=(1, 2, 2), max_seq_len=32)
      rope_max = WanRotaryPosEmbed(attention_head_dim=4, patch_size=(1, 2, 2), max_seq_len=32)
      hidden_5d = torch.randn(1, 8, 3, 4, 4)
      freqs_cos, freqs_sin = rope_hf(hidden_5d)
      rotary_emb = rope_max(jnp.asarray(np.transpose(to_numpy(hidden_5d), (0, 2, 3, 4, 1))))

      hidden_states = torch.randn(1, 12, 8)
      encoder_hidden_states = torch.randn(1, 5, 8)
      temb = torch.randn(1, 6, 8)

      expected = hf_module(hidden_states, encoder_hidden_states, temb, (freqs_cos, freqs_sin))
      actual = max_module(
          jnp.asarray(to_numpy(hidden_states)),
          jnp.asarray(to_numpy(encoder_hidden_states)),
          jnp.asarray(to_numpy(temb)),
          rotary_emb,
      )

    assert_allclose(self, actual, expected, atol=3e-7, rtol=1e-6)

  def test_wan_model_weight_mapping_covers_all_local_params(self):
    cfg = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 8,
        "freq_dim": 8,
        "ffn_dim": 16,
        "num_layers": 1,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "image_dim": 4,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 32,
        "pos_embed_seq_len": None,
    }
    hf_model = HFWanTransformer3DModel(**cfg).eval()

    with self.mesh, nn_partitioning.axis_rules(self.logical_axis_rules):
      max_model = WanModel(
          rngs=self.rngs,
          scan_layers=False,
          mesh=self.mesh,
          attention="dot_product",
          flash_min_seq_length=4096,
          **cfg,
      )
      missing_keys, flax_state_dict = map_hf_wan_state_to_local(max_model, hf_model, num_layers=cfg["num_layers"])

    self.assertFalse(missing_keys, msg=f"Unmapped WAN parameters: {missing_keys}")
    self.assertGreater(len(flax_state_dict), 0)

  def test_wan_model_forward_parity(self):
    cfg = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 8,
        "freq_dim": 8,
        "ffn_dim": 16,
        "num_layers": 1,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "image_dim": 4,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 32,
        "pos_embed_seq_len": None,
    }
    hf_model = HFWanTransformer3DModel(**cfg).eval()

    with self.mesh, nn_partitioning.axis_rules(self.logical_axis_rules):
      max_model = WanModel(
          rngs=self.rngs,
          scan_layers=False,
          mesh=self.mesh,
          attention="dot_product",
          flash_min_seq_length=4096,
          **cfg,
      )
      missing_keys, _ = map_hf_wan_state_to_local(max_model, hf_model, num_layers=cfg["num_layers"])
      self.assertFalse(missing_keys, msg=f"Unmapped WAN parameters: {missing_keys}")

      hidden_states = torch.randn(1, 4, 3, 4, 4)
      timestep = torch.tensor([7], dtype=torch.long)
      encoder_hidden_states = torch.randn(1, 5, 8)
      encoder_hidden_states_image = torch.randn(1, 3, 4)

      with torch.no_grad():
        expected = hf_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
        ).sample

      actual = max_model(
          hidden_states=jnp.asarray(to_numpy(hidden_states)),
          timestep=jnp.asarray(to_numpy(timestep)),
          encoder_hidden_states=jnp.asarray(to_numpy(encoder_hidden_states)),
          encoder_hidden_states_image=jnp.asarray(to_numpy(encoder_hidden_states_image)),
      )

    assert_allclose(self, actual, expected, atol=2e-5, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
