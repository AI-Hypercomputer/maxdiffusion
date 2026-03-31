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
import sys
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
DIFFUSERS_SRC = os.path.join(REPO_ROOT, "diffusers", "src")
if DIFFUSERS_SRC not in sys.path:
  sys.path.insert(0, DIFFUSERS_SRC)

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh

from diffusers.models.transformers.transformer_wan_animate import (
    FusedLeakyReLU as HFFusedLeakyReLU,
    MotionConv2d as HFMotionConv2d,
    MotionEncoderResBlock as HFMotionEncoderResBlock,
    MotionLinear as HFMotionLinear,
    WanAnimateFaceBlockCrossAttention as HFWanAnimateFaceBlockCrossAttention,
    WanAnimateFaceEncoder as HFWanAnimateFaceEncoder,
    WanAnimateMotionEncoder as HFWanAnimateMotionEncoder,
    WanAnimateTransformer3DModel as HFWanAnimateTransformer3DModel,
)

from .. import pyconfig
from ..max_utils import create_device_mesh
from ..models.modeling_flax_pytorch_utils import rename_key
from ..models.wan.transformers.transformer_wan_animate import (
    FlaxFusedLeakyReLU,
    FlaxMotionConv2d,
    FlaxMotionEncoderResBlock,
    FlaxMotionLinear,
    FlaxWanAnimateFaceBlockCrossAttention,
    FlaxWanAnimateFaceEncoder,
    FlaxWanAnimateMotionEncoder,
    NNXWanAnimateTransformer3DModel,
)
from ..models.wan.wan_utils import (
    _is_motion_encoder_custom_weight,
    _normalize_animate_list_key,
    _tuple_str_to_int,
    get_key_and_value,
)


def to_numpy(array):
  if isinstance(array, torch.Tensor):
    if array.dtype == torch.bfloat16:
      array = array.float()
    return array.detach().cpu().numpy()
  return np.asarray(array)


def assert_allclose(test_case, actual, expected, *, atol=1e-6, rtol=1e-6):
  test_case.assertEqual(to_numpy(actual).shape, to_numpy(expected).shape)
  np.testing.assert_allclose(to_numpy(actual), to_numpy(expected), atol=atol, rtol=rtol)


def copy_fused_leaky_relu_params(max_module, hf_module):
  if max_module.bias is not None:
    max_module.bias[...] = jnp.asarray(to_numpy(hf_module.bias))


def copy_motion_conv2d_params(max_module, hf_module):
  max_module.weight[...] = jnp.asarray(to_numpy(hf_module.weight))
  if max_module.bias is not None and hf_module.bias is not None:
    max_module.bias[...] = jnp.asarray(to_numpy(hf_module.bias))
  if max_module.act_fn is not None and hf_module.act_fn is not None:
    copy_fused_leaky_relu_params(max_module.act_fn, hf_module.act_fn)


def copy_motion_linear_params(max_module, hf_module):
  max_module.weight[...] = jnp.asarray(to_numpy(hf_module.weight))
  if max_module.bias is not None and hf_module.bias is not None:
    max_module.bias[...] = jnp.asarray(to_numpy(hf_module.bias))
  if max_module.act_fn is not None and hf_module.act_fn is not None:
    copy_fused_leaky_relu_params(max_module.act_fn, hf_module.act_fn)


def copy_motion_encoder_resblock_params(max_module, hf_module):
  copy_motion_conv2d_params(max_module.conv1, hf_module.conv1)
  copy_motion_conv2d_params(max_module.conv2, hf_module.conv2)
  copy_motion_conv2d_params(max_module.conv_skip, hf_module.conv_skip)


def copy_motion_encoder_params(max_module, hf_module):
  copy_motion_conv2d_params(max_module.conv_in, hf_module.conv_in)
  copy_motion_conv2d_params(max_module.conv_out, hf_module.conv_out)
  for max_block, hf_block in zip(max_module.res_blocks, hf_module.res_blocks):
    copy_motion_encoder_resblock_params(max_block, hf_block)
  for max_linear, hf_linear in zip(max_module.motion_network, hf_module.motion_network):
    copy_motion_linear_params(max_linear, hf_linear)
  max_module.motion_synthesis_weight[...] = jnp.asarray(to_numpy(hf_module.motion_synthesis_weight))


def copy_face_encoder_params(max_module, hf_module):
  max_module.conv1_local.kernel[...] = jnp.asarray(np.transpose(to_numpy(hf_module.conv1_local.weight), (2, 1, 0)))
  max_module.conv1_local.bias[...] = jnp.asarray(to_numpy(hf_module.conv1_local.bias))
  max_module.conv2.kernel[...] = jnp.asarray(np.transpose(to_numpy(hf_module.conv2.weight), (2, 1, 0)))
  max_module.conv2.bias[...] = jnp.asarray(to_numpy(hf_module.conv2.bias))
  max_module.conv3.kernel[...] = jnp.asarray(np.transpose(to_numpy(hf_module.conv3.weight), (2, 1, 0)))
  max_module.conv3.bias[...] = jnp.asarray(to_numpy(hf_module.conv3.bias))
  max_module.out_proj.kernel[...] = jnp.asarray(to_numpy(hf_module.out_proj.weight).T)
  max_module.out_proj.bias[...] = jnp.asarray(to_numpy(hf_module.out_proj.bias))
  max_module.padding_tokens[...] = jnp.asarray(to_numpy(hf_module.padding_tokens))


def copy_face_block_cross_attention_params(max_module, hf_module):
  max_module.to_q.kernel[...] = jnp.asarray(to_numpy(hf_module.to_q.weight).T)
  max_module.to_q.bias[...] = jnp.asarray(to_numpy(hf_module.to_q.bias))
  max_module.to_k.kernel[...] = jnp.asarray(to_numpy(hf_module.to_k.weight).T)
  max_module.to_k.bias[...] = jnp.asarray(to_numpy(hf_module.to_k.bias))
  max_module.to_v.kernel[...] = jnp.asarray(to_numpy(hf_module.to_v.weight).T)
  max_module.to_v.bias[...] = jnp.asarray(to_numpy(hf_module.to_v.bias))
  max_module.to_out.kernel[...] = jnp.asarray(to_numpy(hf_module.to_out.weight).T)
  max_module.to_out.bias[...] = jnp.asarray(to_numpy(hf_module.to_out.bias))
  max_module.norm_q.scale[...] = jnp.asarray(to_numpy(hf_module.norm_q.weight))
  max_module.norm_k.scale[...] = jnp.asarray(to_numpy(hf_module.norm_k.weight))


def map_hf_wan_animate_state_to_local(max_model, hf_model, num_layers):
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
    is_motion_custom_weight = _is_motion_encoder_custom_weight(pt_key)

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
    renamed_pt_key = renamed_pt_key.replace(".activation.bias", ".act_fn.bias")

    if is_motion_custom_weight and renamed_pt_key.endswith(".kernel"):
      renamed_pt_key = renamed_pt_key[:-7] + ".weight"

    pt_tuple_key = tuple(renamed_pt_key.split("."))

    if is_motion_custom_weight:
      flax_key = _normalize_animate_list_key(_tuple_str_to_int(pt_tuple_key))
      flax_tensor = jnp.asarray(to_numpy(tensor))
    else:
      flax_key, flax_tensor = get_key_and_value(
          pt_tuple_key,
          to_numpy(tensor),
          flax_state_dict,
          random_flax_state_dict,
          False,
          num_layers,
      )
      flax_key = _normalize_animate_list_key(flax_key)

    flax_state_dict[flax_key] = jnp.asarray(flax_tensor)

  missing_keys = [key for key in flax_state_dict if key not in flat_vars]
  for key, value in flax_state_dict.items():
    if key in flat_vars:
      flat_vars[key][...] = value

  return missing_keys, flax_state_dict


class WanAnimateModuleParityTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    pyconfig.initialize(
        [
            None,
            os.path.join(REPO_ROOT, "src", "maxdiffusion", "configs", "base_wan_14b.yml"),
        ],
        unittest=True,
    )
    config = pyconfig.config
    cls.logical_axis_rules = config.logical_axis_rules
    cls.mesh = Mesh(create_device_mesh(config), config.mesh_axes)

  def setUp(self):
    torch.manual_seed(0)
    self.rngs = nnx.Rngs(jax.random.key(0))

  def test_fused_leaky_relu_parity(self):
    hf_module = HFFusedLeakyReLU(bias_channels=3).eval()
    max_module = FlaxFusedLeakyReLU(rngs=self.rngs, bias_channels=3)
    copy_fused_leaky_relu_params(max_module, hf_module)

    inputs = torch.randn(2, 3, 4, 5)
    expected = hf_module(inputs)
    actual = max_module(jnp.asarray(to_numpy(inputs)))

    assert_allclose(self, actual, expected, atol=0.0, rtol=0.0)

  def test_motion_conv2d_parity(self):
    hf_module = HFMotionConv2d(3, 5, kernel_size=3, stride=2, padding=0, blur_kernel=(1, 3, 3, 1)).eval()
    max_module = FlaxMotionConv2d(
        rngs=self.rngs,
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        stride=2,
        padding=0,
        blur_kernel=(1, 3, 3, 1),
    )
    copy_motion_conv2d_params(max_module, hf_module)

    inputs = torch.randn(2, 3, 8, 8)
    expected = hf_module(inputs)
    actual = max_module(jnp.asarray(to_numpy(inputs)))

    assert_allclose(self, actual, expected, atol=2e-7, rtol=2e-6)

  def test_motion_linear_parity(self):
    hf_module = HFMotionLinear(7, 5, use_activation=True).eval()
    max_module = FlaxMotionLinear(rngs=self.rngs, in_dim=7, out_dim=5, use_activation=True)
    copy_motion_linear_params(max_module, hf_module)

    inputs = torch.randn(4, 7)
    expected = hf_module(inputs)
    actual = max_module(jnp.asarray(to_numpy(inputs)))

    assert_allclose(self, actual, expected, atol=1e-7, rtol=1e-7)

  def test_motion_encoder_resblock_parity(self):
    hf_module = HFMotionEncoderResBlock(8, 10).eval()
    max_module = FlaxMotionEncoderResBlock(rngs=self.rngs, in_channels=8, out_channels=10)
    copy_motion_encoder_resblock_params(max_module, hf_module)

    inputs = torch.randn(2, 8, 8, 8)
    expected = hf_module(inputs)
    actual = max_module(jnp.asarray(to_numpy(inputs)))

    assert_allclose(self, actual, expected, atol=2e-7, rtol=1e-6)

  def test_motion_encoder_parity(self):
    cfg = {
        "size": 4,
        "style_dim": 8,
        "motion_dim": 4,
        "out_dim": 8,
        "motion_blocks": 3,
        "channels": {"4": 8, "8": 8, "16": 8},
    }
    hf_module = HFWanAnimateMotionEncoder(**cfg).eval()
    max_module = FlaxWanAnimateMotionEncoder(rngs=self.rngs, **cfg)
    copy_motion_encoder_params(max_module, hf_module)

    inputs = torch.randn(3, 3, 4, 4)
    expected = hf_module(inputs)
    actual = max_module(jnp.asarray(to_numpy(inputs)))

    assert_allclose(self, actual, expected, atol=5e-7, rtol=1e-6)

  def test_face_encoder_parity(self):
    hf_module = HFWanAnimateFaceEncoder(in_dim=8, out_dim=12, hidden_dim=16, num_heads=2).eval()
    max_module = FlaxWanAnimateFaceEncoder(rngs=self.rngs, in_dim=8, out_dim=12, hidden_dim=16, num_heads=2)
    copy_face_encoder_params(max_module, hf_module)

    inputs = torch.randn(2, 7, 8)
    expected = hf_module(inputs)
    actual = max_module(jnp.asarray(to_numpy(inputs)))

    assert_allclose(self, actual, expected, atol=5e-7, rtol=1e-6)

  def test_face_block_cross_attention_parity(self):
    hf_module = HFWanAnimateFaceBlockCrossAttention(dim=12, heads=3, dim_head=4, cross_attention_dim_head=4).eval()
    max_module = FlaxWanAnimateFaceBlockCrossAttention(
        rngs=self.rngs,
        dim=12,
        heads=3,
        dim_head=4,
        cross_attention_dim_head=4,
    )
    copy_face_block_cross_attention_params(max_module, hf_module)

    hidden_states = torch.randn(2, 8, 12)
    encoder_hidden_states = torch.randn(2, 2, 3, 12)
    expected = hf_module(hidden_states, encoder_hidden_states)
    actual = max_module(jnp.asarray(to_numpy(hidden_states)), jnp.asarray(to_numpy(encoder_hidden_states)))

    assert_allclose(self, actual, expected, atol=2e-7, rtol=1e-6)

  def test_wan_animate_transformer_weight_mapping_covers_all_local_params(self):
    cfg = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "in_channels": 12,
        "latent_channels": 4,
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
        "motion_encoder_channel_sizes": {"4": 8, "8": 8, "16": 8},
        "motion_encoder_size": 4,
        "motion_style_dim": 8,
        "motion_dim": 4,
        "motion_encoder_dim": 8,
        "face_encoder_hidden_dim": 8,
        "face_encoder_num_heads": 2,
        "inject_face_latents_blocks": 1,
        "motion_encoder_batch_size": 2,
    }
    hf_model = HFWanAnimateTransformer3DModel(**cfg).eval()

    with self.mesh, nn_partitioning.axis_rules(self.logical_axis_rules):
      max_model = NNXWanAnimateTransformer3DModel(rngs=self.rngs, scan_layers=False, mesh=self.mesh, **cfg)
      missing_keys, flax_state_dict = map_hf_wan_animate_state_to_local(max_model, hf_model, num_layers=cfg["num_layers"])

    self.assertFalse(missing_keys, msg=f"Unmapped animate parameters: {missing_keys}")
    self.assertGreater(len(flax_state_dict), 0)

  def test_wan_animate_transformer_forward_parity(self):
    cfg = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "in_channels": 12,
        "latent_channels": 4,
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
        "motion_encoder_channel_sizes": {"4": 8, "8": 8, "16": 8},
        "motion_encoder_size": 4,
        "motion_style_dim": 8,
        "motion_dim": 4,
        "motion_encoder_dim": 8,
        "face_encoder_hidden_dim": 8,
        "face_encoder_num_heads": 2,
        "inject_face_latents_blocks": 1,
        "motion_encoder_batch_size": 2,
    }
    hf_model = HFWanAnimateTransformer3DModel(**cfg).eval()

    with self.mesh, nn_partitioning.axis_rules(self.logical_axis_rules):
      max_model = NNXWanAnimateTransformer3DModel(rngs=self.rngs, scan_layers=False, mesh=self.mesh, **cfg)
      missing_keys, _ = map_hf_wan_animate_state_to_local(max_model, hf_model, num_layers=cfg["num_layers"])
      self.assertFalse(missing_keys, msg=f"Unmapped animate parameters: {missing_keys}")

      hidden_states = torch.randn(1, 12, 3, 4, 4)
      pose_hidden_states = torch.randn(1, 4, 2, 4, 4)
      encoder_hidden_states = torch.randn(1, 5, 8)
      encoder_hidden_states_image = torch.randn(1, 3, 4)
      face_pixel_values = torch.randn(1, 3, 2, 4, 4)
      timestep = torch.tensor([7], dtype=torch.long)

      with torch.no_grad():
        expected = hf_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            pose_hidden_states=pose_hidden_states,
            face_pixel_values=face_pixel_values,
        ).sample

      actual = max_model(
          hidden_states=jnp.asarray(to_numpy(hidden_states)),
          timestep=jnp.asarray(to_numpy(timestep)),
          encoder_hidden_states=jnp.asarray(to_numpy(encoder_hidden_states)),
          encoder_hidden_states_image=jnp.asarray(to_numpy(encoder_hidden_states_image)),
          pose_hidden_states=jnp.asarray(to_numpy(pose_hidden_states)),
          face_pixel_values=jnp.asarray(to_numpy(face_pixel_values)),
      )["sample"]

    assert_allclose(self, actual, expected, atol=5e-5, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
