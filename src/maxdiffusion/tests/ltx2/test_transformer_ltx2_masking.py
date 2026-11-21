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
import types
import unittest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
import flax.linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
  import chex  # noqa: F401
except ImportError:
  _chex_stub = types.ModuleType("chex")
  _chex_stub.Array = jax.Array
  _chex_stub.PRNGKey = jax.Array
  sys.modules["chex"] = _chex_stub

from maxdiffusion.common_types import (
    SEQUENCE_PARALLEL_AXIS_RULES,
    ULYSSES_ATTENTION_AXIS_RULES,
)
from maxdiffusion.models.attention_flax import _apply_attention
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel


def _build_axis_rules(attention_kernel: str):
  attn_rules = ULYSSES_ATTENTION_AXIS_RULES if attention_kernel == "ulysses" else SEQUENCE_PARALLEL_AXIS_RULES
  base_rules = [
      ("batch", ("data", "fsdp")),
      ("activation_batch", ("data", "fsdp")),
      ("activation_length", "context"),
      ("activation_kv_length", "context"),
      ("activation_embed", None),
      ("activation_heads", "tensor"),
      ("activation_kv", None),
      ("mlp", "tensor"),
      ("embed", ("context", "fsdp")),
      ("heads", "tensor"),
      ("norm", "tensor"),
      ("layers", None),
  ]
  return tuple(tuple(r) for r in attn_rules) + tuple(base_rules)


def _randomize_zero_initialized_params(model: LTX2VideoTransformer3DModel, key: jax.Array) -> None:
  """Populates zero-initialized biases/AdaLN kernels so bias-drift bugs cannot hide."""
  graphdef, state = nnx.split(model)
  leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
  new_leaves = []
  for idx, (path, val) in enumerate(leaves):
    if isinstance(val, jax.Array) and jnp.all(val == 0):
      subkey = jax.random.fold_in(key, idx)
      noise = jax.random.normal(subkey, val.shape, dtype=val.dtype) * 0.15
      if hasattr(val, "sharding") and val.sharding is not None:
        noise = jax.device_put(noise, val.sharding)
      new_leaves.append(noise)
    else:
      new_leaves.append(val)
  new_state = jax.tree_util.tree_unflatten(treedef, new_leaves)
  nnx.update(model, new_state)


def _shard_model_state(model: LTX2VideoTransformer3DModel, mesh: Mesh, rules) -> None:
  graphdef, state = nnx.split(model)
  logical_spec = nnx.get_partition_spec(state)
  mesh_sharding = nn.logical_to_mesh_sharding(logical_spec, mesh, rules)
  sharded_state = jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s), state, mesh_sharding)
  nnx.update(model, sharded_state)


class LTX2TransformerMaskingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.default_matmul_precision("float32"))
    devices = np.array(jax.devices())
    if devices.size >= 8:
      self.mesh = Mesh(devices[:8].reshape(1, 2, 4, 1), ("data", "fsdp", "context", "tensor"))
    else:
      self.mesh = Mesh(devices.reshape(1, 1, devices.size, 1), ("data", "fsdp", "context", "tensor"))
    self.block_sizes = splash_attention_kernel.BlockSizes(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
        use_fused_bwd_kernel=False,
    )

  def _make_model(
      self,
      *,
      attention_kernel: str = "dot_product",
      scan_layers: bool = True,
      remat_policy: str = "NONE",
      cross_attn_mod: bool = False,
      gated_attn: bool = False,
      use_svg_attention: bool = False,
      seed: int = 0,
  ) -> LTX2VideoTransformer3DModel:
    rules = _build_axis_rules(attention_kernel)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
      model = LTX2VideoTransformer3DModel(
          rngs=nnx.Rngs(jax.random.key(seed)),
          in_channels=16,
          out_channels=16,
          patch_size=1,
          patch_size_t=1,
          num_attention_heads=4,
          attention_head_dim=128,
          cross_attention_dim=512,
          caption_channels=64,
          audio_in_channels=8,
          audio_out_channels=8,
          audio_num_attention_heads=4,
          audio_attention_head_dim=128,
          audio_cross_attention_dim=512,
          num_layers=2,
          dtype=jnp.float32,
          weights_dtype=jnp.float32,
          mesh=self.mesh,
          scan_layers=scan_layers,
          remat_policy=remat_policy,
          attention_kernel=attention_kernel,
          a2v_attention_kernel=attention_kernel,
          v2a_attention_kernel=attention_kernel,
          flash_block_sizes=self.block_sizes,
          flash_min_seq_length=0,
          cross_attn_mod=cross_attn_mod,
          gated_attn=gated_attn,
          attention_config={"use_svg_attention": use_svg_attention} if use_svg_attention else None,
      )
      _randomize_zero_initialized_params(model, jax.random.key(seed + 99))
      _shard_model_state(model, self.mesh, rules)
    return model

  def test_unpadded_vs_padded_fwd_and_param_grad_invariance(self):
    """Proves unpadded vs right-padded fwd + nnx.value_and_grad invariance and 8-device FSDP+CP sharding."""
    batch_size = 2
    f_lat, h_lat, w_lat = 2, 8, 8
    s_vid = f_lat * h_lat * w_lat  # 128, divisible by context=4 and block=128
    s_aud = 128
    s_txt = 128
    b_vid, b_aud, b_txt = 256, 256, 256
    self.assertEqual(b_vid % self.mesh.shape["context"], 0)

    k = jax.random.key(42)
    keys = jax.random.split(k, 12)
    vid_unpad = jax.random.normal(keys[0], (batch_size, s_vid, 16), dtype=jnp.float32)
    aud_unpad = jax.random.normal(keys[1], (batch_size, s_aud, 8), dtype=jnp.float32)
    txt_unpad = jax.random.normal(keys[2], (batch_size, s_txt, 64), dtype=jnp.float32)
    atxt_unpad = jax.random.normal(keys[3], (batch_size, s_txt, 64), dtype=jnp.float32)
    target_vid_unpad = jax.random.normal(keys[4], (batch_size, s_vid, 16), dtype=jnp.float32)
    target_aud_unpad = jax.random.normal(keys[5], (batch_size, s_aud, 8), dtype=jnp.float32)

    # Large non-zero garbage in all right-pad slots to catch any leakage.
    vid_pad = jnp.concatenate(
        [vid_unpad, 5.0 * jax.random.normal(keys[6], (batch_size, b_vid - s_vid, 16), dtype=jnp.float32)], axis=1
    )
    aud_pad = jnp.concatenate(
        [aud_unpad, 5.0 * jax.random.normal(keys[7], (batch_size, b_aud - s_aud, 8), dtype=jnp.float32)], axis=1
    )
    txt_pad = jnp.concatenate(
        [txt_unpad, 5.0 * jax.random.normal(keys[8], (batch_size, b_txt - s_txt, 64), dtype=jnp.float32)], axis=1
    )
    atxt_pad = jnp.concatenate(
        [atxt_unpad, 5.0 * jax.random.normal(keys[9], (batch_size, b_txt - s_txt, 64), dtype=jnp.float32)], axis=1
    )
    target_vid_pad = jnp.concatenate(
        [target_vid_unpad, 9.0 * jax.random.normal(keys[10], (batch_size, b_vid - s_vid, 16), dtype=jnp.float32)],
        axis=1,
    )
    target_aud_pad = jnp.concatenate(
        [target_aud_unpad, 9.0 * jax.random.normal(keys[11], (batch_size, b_aud - s_aud, 8), dtype=jnp.float32)],
        axis=1,
    )

    vid_mask = jnp.zeros((batch_size, b_vid), dtype=jnp.bool_).at[:, :s_vid].set(True)
    aud_mask = jnp.zeros((batch_size, b_aud), dtype=jnp.bool_).at[:, :s_aud].set(True)
    txt_mask = jnp.zeros((batch_size, b_txt), dtype=jnp.bool_).at[:, :s_txt].set(True)

    timestep = jnp.array([350.0, 720.0], dtype=jnp.float32)
    sigma = jnp.array([0.35, 0.72], dtype=jnp.float32)
    audio_sigma = jnp.array([0.40, 0.68], dtype=jnp.float32)

    configs = [
        ("dot_product", True, "FULL", True),
        ("dot_product", False, "NONE", False),
        ("flash", True, "FULL", True),
        ("ulysses", True, "FULL", True),
    ]

    for kern, scan_layers, remat_policy, cross_attn_mod in configs:
      with self.subTest(kernel=kern, scan_layers=scan_layers, cross_attn_mod=cross_attn_mod):
        rules = _build_axis_rules(kern)
        model = self._make_model(
            attention_kernel=kern,
            scan_layers=scan_layers,
            remat_policy=remat_policy,
            cross_attn_mod=cross_attn_mod,
            gated_attn=True,
            seed=7,
        )

        # Build explicit coordinates so padded prefix bit-matches unpadded coords (Note N1).
        vid_coords_unpad = model.rope.prepare_video_coords(batch_size, f_lat, h_lat, w_lat, fps=24.0)
        aud_coords_unpad = model.audio_rope.prepare_audio_coords(batch_size, s_aud)
        vid_coords_pad = jnp.pad(vid_coords_unpad, ((0, 0), (0, 0), (0, b_vid - s_vid), (0, 0)), constant_values=0.0)
        aud_coords_pad = jnp.pad(aud_coords_unpad, ((0, 0), (0, 0), (0, b_aud - s_aud), (0, 0)), constant_values=0.0)
        self.assertTrue(bool(jnp.array_equal(vid_coords_pad[:, :, :s_vid, :], vid_coords_unpad)))
        self.assertTrue(bool(jnp.array_equal(aud_coords_pad[:, :, :s_aud, :], aud_coords_unpad)))

        # Shard inputs across FSDP + context axes to prove true distributed execution (Note N2).
        sh_vid = NamedSharding(self.mesh, P(("data", "fsdp"), "context", None))
        sh_aud = NamedSharding(self.mesh, P(("data", "fsdp"), None, None))
        sh_txt = NamedSharding(self.mesh, P(("data", "fsdp"), "context", None))
        sh_b = NamedSharding(self.mesh, P(("data", "fsdp")))
        sh_mask_vid = NamedSharding(self.mesh, P(("data", "fsdp"), "context"))
        sh_mask_aud = NamedSharding(self.mesh, P(("data", "fsdp"), None))
        sh_coords_vid = NamedSharding(self.mesh, P(("data", "fsdp"), None, "context", None))
        sh_coords_aud = NamedSharding(self.mesh, P(("data", "fsdp"), None, None, None))

        graphdef, state = nnx.split(model)

        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):

          @jax.jit
          def run_unpadded(state_in, v_in, a_in, t_in, at_in, vc_in, ac_in, ts_in, sig_in, asig_in, tv_in, ta_in):
            def loss_fn(st):
              m = nnx.merge(graphdef, st)
              out = m(
                  hidden_states=v_in,
                  audio_hidden_states=a_in,
                  encoder_hidden_states=t_in,
                  audio_encoder_hidden_states=at_in,
                  timestep=ts_in,
                  sigma=sig_in if cross_attn_mod else None,
                  audio_sigma=asig_in if cross_attn_mod else None,
                  video_coords=vc_in,
                  audio_coords=ac_in,
                  return_dict=True,
              )
              v_loss = jnp.sum((out["sample"] - tv_in) ** 2) / (batch_size * s_vid * 16)
              a_loss = jnp.sum((out["audio_sample"] - ta_in) ** 2) / (batch_size * s_aud * 8)
              return v_loss + a_loss, out

            (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(state_in)
            return loss, out, grads

          @jax.jit
          def run_padded(
              state_in, v_in, a_in, t_in, at_in, vc_in, ac_in, ts_in, sig_in, asig_in, tv_in, ta_in, vm_in, am_in, tm_in
          ):
            def loss_fn(st):
              m = nnx.merge(graphdef, st)
              out = m(
                  hidden_states=v_in,
                  audio_hidden_states=a_in,
                  encoder_hidden_states=t_in,
                  audio_encoder_hidden_states=at_in,
                  timestep=ts_in,
                  sigma=sig_in if cross_attn_mod else None,
                  audio_sigma=asig_in if cross_attn_mod else None,
                  encoder_attention_mask=tm_in,
                  audio_encoder_attention_mask=tm_in,
                  video_coords=vc_in,
                  audio_coords=ac_in,
                  video_attention_mask=vm_in,
                  audio_attention_mask=am_in,
                  return_dict=True,
              )
              v_err = jnp.where(vm_in[:, :, None], (out["sample"] - tv_in) ** 2, 0.0)
              a_err = jnp.where(am_in[:, :, None], (out["audio_sample"] - ta_in) ** 2, 0.0)
              v_loss = jnp.sum(v_err) / (jnp.sum(vm_in) * 16)
              a_loss = jnp.sum(a_err) / (jnp.sum(am_in) * 8)
              return v_loss + a_loss, out

            (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(state_in)
            return loss, out, grads

          p_args = (
              state,
              jax.device_put(vid_pad, sh_vid),
              jax.device_put(aud_pad, sh_aud),
              jax.device_put(txt_pad, sh_txt),
              jax.device_put(atxt_pad, sh_aud),
              jax.device_put(vid_coords_pad, sh_coords_vid),
              jax.device_put(aud_coords_pad, sh_coords_aud),
              jax.device_put(timestep, sh_b),
              jax.device_put(sigma, sh_b),
              jax.device_put(audio_sigma, sh_b),
              jax.device_put(target_vid_pad, sh_vid),
              jax.device_put(target_aud_pad, sh_aud),
              jax.device_put(vid_mask, sh_mask_vid),
              jax.device_put(aud_mask, sh_mask_aud),
              jax.device_put(txt_mask, sh_mask_aud),
          )
          compiled_pad = run_padded.lower(*p_args).compile()
          hlo_text = compiled_pad.as_text()
          if self.mesh.shape["fsdp"] > 1:
            self.assertIn("all-gather", hlo_text)
          if kern == "ulysses" and self.mesh.shape["context"] > 1:
            self.assertIn("all-to-all", hlo_text)

          loss_p, out_p, grads_p = compiled_pad(*p_args)
          self.assertTrue(out_p["sample"].sharding.is_equivalent_to(sh_vid, out_p["sample"].ndim))

          loss_u, out_u, grads_u = run_unpadded(
              state,
              jax.device_put(vid_unpad, sh_vid),
              jax.device_put(aud_unpad, sh_aud),
              jax.device_put(txt_unpad, sh_txt),
              jax.device_put(atxt_unpad, sh_aud),
              jax.device_put(vid_coords_unpad, sh_coords_vid),
              jax.device_put(aud_coords_unpad, sh_coords_aud),
              jax.device_put(timestep, sh_b),
              jax.device_put(sigma, sh_b),
              jax.device_put(audio_sigma, sh_b),
              jax.device_put(target_vid_unpad, sh_vid),
              jax.device_put(target_aud_unpad, sh_aud),
          )

        # Padded tail must be exact 0.0.
        self.assertEqual(float(jnp.max(jnp.abs(out_p["sample"][:, s_vid:, :]))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(out_p["audio_sample"][:, s_aud:, :]))), 0.0)

        # Valid prefix forward outputs & scalar loss must match.
        vid_diff = float(jnp.max(jnp.abs(out_u["sample"] - out_p["sample"][:, :s_vid, :])))
        aud_diff = float(jnp.max(jnp.abs(out_u["audio_sample"] - out_p["audio_sample"][:, :s_aud, :])))
        self.assertLess(vid_diff, 1e-4, f"Video fwd diff too high for {kern}: {vid_diff}")
        self.assertLess(aud_diff, 1e-4, f"Audio fwd diff too high for {kern}: {aud_diff}")
        self.assertAlmostEqual(float(loss_u), float(loss_p), places=5)

        # Every nnx.Param leaf gradient must be finite and match unpadded!
        u_leaves = jax.tree_util.tree_leaves_with_path(grads_u)
        p_leaves = jax.tree_util.tree_leaves_with_path(grads_p)
        self.assertEqual(len(u_leaves), len(p_leaves))
        prompt_mod_grad_norms = []
        for (path, gu), (_, gp) in zip(u_leaves, p_leaves):
          name = "/".join(str(k) for k in path)
          self.assertTrue(bool(jnp.isfinite(gu).all()), f"Non-finite unpadded grad in {name}")
          self.assertTrue(bool(jnp.isfinite(gp).all()), f"Non-finite padded grad in {name}")
          max_diff = float(jnp.max(jnp.abs(gu - gp)))
          self.assertLess(max_diff, 2e-4, f"Grad mismatch in {name} ({kern}): {max_diff}")
          if "prompt_adaln" in name or "prompt_scale_shift_table" in name:
            prompt_mod_grad_norms.append(float(jnp.linalg.norm(gp)))

        if cross_attn_mod:
          self.assertGreater(len(prompt_mod_grad_norms), 0)
          self.assertTrue(
              any(n > 0.0 for n in prompt_mod_grad_norms),
              f"Expected non-zero prompt_adaln / prompt_scale_shift_table grads when sigma is passed: {prompt_mod_grad_norms}",
          )

  def test_a2v_v2a_mask_defaults_and_explicit_intersection(self):
    """Verifies a2v/v2a masks default to audio/video masks and intersect with explicit masks."""
    model = self._make_model(attention_kernel="dot_product", scan_layers=True, seed=11)
    rules = _build_axis_rules("dot_product")
    batch_size, b_vid, b_aud, b_txt = 2, 16, 12, 8
    vid = jax.random.normal(jax.random.key(1), (batch_size, b_vid, 16))
    aud = jax.random.normal(jax.random.key(2), (batch_size, b_aud, 8))
    txt = jax.random.normal(jax.random.key(3), (batch_size, b_txt, 64))
    vid_coords = jnp.zeros((batch_size, 3, b_vid, 2), dtype=jnp.float32)
    aud_coords = jnp.zeros((batch_size, 1, b_aud, 2), dtype=jnp.float32)
    vid_mask = jnp.zeros((batch_size, b_vid), dtype=jnp.bool_).at[:, :10].set(True)
    aud_mask = jnp.zeros((batch_size, b_aud), dtype=jnp.bool_).at[:, :6].set(True)
    ts = jnp.array([500.0, 250.0], dtype=jnp.float32)

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
      out_default = model(
          hidden_states=vid,
          audio_hidden_states=aud,
          encoder_hidden_states=txt,
          audio_encoder_hidden_states=txt,
          timestep=ts,
          video_coords=vid_coords,
          audio_coords=aud_coords,
          video_attention_mask=vid_mask,
          audio_attention_mask=aud_mask,
      )
      # Passing over-permissive all-True explicit cross-masks must still intersect (&) with vid_mask / aud_mask.
      out_intersect = model(
          hidden_states=vid,
          audio_hidden_states=aud,
          encoder_hidden_states=txt,
          audio_encoder_hidden_states=txt,
          timestep=ts,
          video_coords=vid_coords,
          audio_coords=aud_coords,
          video_attention_mask=vid_mask,
          audio_attention_mask=aud_mask,
          attention_kwargs={
              "a2v_cross_attention_mask": jnp.ones_like(aud_mask),
              "v2a_cross_attention_mask": jnp.ones_like(vid_mask),
          },
      )
      self.assertTrue(bool(jnp.array_equal(out_default["sample"], out_intersect["sample"])))
      self.assertTrue(bool(jnp.array_equal(out_default["audio_sample"], out_intersect["audio_sample"])))

  def test_all_zero_modality_and_empty_kv_row_guards(self):
    """Verifies zero NaNs, zero bias drift on video-only samples, and STG isolated-audio parity."""
    batch_size = 2
    s_vid, s_aud, s_txt = 128, 128, 128
    vid = jax.random.normal(jax.random.key(20), (batch_size, s_vid, 16))
    aud = jax.random.normal(jax.random.key(21), (batch_size, s_aud, 8))
    txt = jax.random.normal(jax.random.key(22), (batch_size, s_txt, 64))
    vid_coords = jnp.zeros((batch_size, 3, s_vid, 2), dtype=jnp.float32)
    aud_coords = jnp.zeros((batch_size, 1, s_aud, 2), dtype=jnp.float32)
    ts = jnp.array([400.0, 600.0], dtype=jnp.float32)

    vid_mask = jnp.ones((batch_size, s_vid), dtype=jnp.bool_)
    # Sample 0 has valid audio; sample 1 is video-only (all-False audio_attention_mask).
    aud_mask_mixed = jnp.zeros((batch_size, s_aud), dtype=jnp.bool_).at[0, :64].set(True)
    aud_mask_all_zero = jnp.zeros((batch_size, s_aud), dtype=jnp.bool_)

    for kern in ("dot_product", "flash", "ulysses"):
      with self.subTest(kernel=kern):
        rules = _build_axis_rules(kern)
        model = self._make_model(attention_kernel=kern, scan_layers=True, seed=31)
        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
          # 1. Overlong mask slice safety in _apply_attention for non-dot_product kernels (Note M2).
          if kern != "dot_product":
            q_test = jax.random.normal(jax.random.key(90), (batch_size, 128, 512))
            overlong_mask = jnp.zeros((batch_size, 160), dtype=jnp.bool_).at[:, 128:].set(True)
            out_overlong = _apply_attention(
                query=q_test,
                key=q_test,
                value=q_test,
                heads=4,
                dim_head=128,
                split_head_dim=True,
                float32_qk_product=True,
                attention_kernel=kern,
                flash_min_seq_length=0,
                use_memory_efficient_attention=False,
                scale=128**-0.5,
                dtype=jnp.float32,
                mesh=self.mesh,
                axis_names_q=(
                    "activation_batch",
                    "activation_self_attn_heads",
                    "activation_self_attn_q_length",
                    "activation_kv",
                ),
                axis_names_kv=(
                    "activation_batch",
                    "activation_self_attn_heads",
                    "activation_self_attn_kv_length",
                    "activation_kv",
                ),
                flash_block_sizes=self.block_sizes,
                dpa_layer=None,
                attention_mask=overlong_mask,
            )
            self.assertEqual(float(jnp.max(jnp.abs(out_overlong))), 0.0)

          # 2. Mixed batch: sample 1 has all-False audio mask; changing aud[1] must not change sample[1] even when modality_mask=None!
          out_a = model(
              hidden_states=vid,
              audio_hidden_states=aud,
              encoder_hidden_states=txt,
              audio_encoder_hidden_states=txt,
              timestep=ts,
              video_coords=vid_coords,
              audio_coords=aud_coords,
              video_attention_mask=vid_mask,
              audio_attention_mask=aud_mask_mixed,
              modality_mask=None,
          )
          out_b = model(
              hidden_states=vid,
              audio_hidden_states=aud.at[1].set(99.0),
              encoder_hidden_states=txt,
              audio_encoder_hidden_states=txt,
              timestep=ts,
              video_coords=vid_coords,
              audio_coords=aud_coords,
              video_attention_mask=vid_mask,
              audio_attention_mask=aud_mask_mixed,
              modality_mask=jnp.array([1.0, 0.0]),
          )
          self.assertTrue(bool(jnp.isfinite(out_a["sample"]).all()))
          self.assertTrue(bool(jnp.isfinite(out_a["audio_sample"]).all()))
          self.assertEqual(float(jnp.max(jnp.abs(out_a["audio_sample"][1]))), 0.0)
          # Zero to_out.bias drift: modality_mask=None with all-False audio_attention_mask[1] must match modality_mask=0!
          self.assertLess(float(jnp.max(jnp.abs(out_a["sample"][1] - out_b["sample"][1]))), 1e-5)

          # 3. All-video batch (aud_mask_all_zero + modality_mask=0): zero gradients into all audio & a2v/v2a params!
          def video_only_loss(m):
            out = m(
                hidden_states=vid,
                audio_hidden_states=aud,
                encoder_hidden_states=txt,
                audio_encoder_hidden_states=txt,
                timestep=ts,
                video_coords=vid_coords,
                audio_coords=aud_coords,
                video_attention_mask=vid_mask,
                audio_attention_mask=aud_mask_all_zero,
                modality_mask=jnp.zeros((batch_size,), dtype=jnp.float32),
            )
            return jnp.mean(out["sample"] ** 2)

          loss_vo, grads_vo = nnx.value_and_grad(video_only_loss)(model)
          self.assertTrue(bool(jnp.isfinite(loss_vo)))
          for path, g in jax.tree_util.tree_leaves_with_path(nnx.state(grads_vo, nnx.Param)):
            pname = "/".join(str(p) for p in path)
            self.assertTrue(bool(jnp.isfinite(g).all()), f"Non-finite grad in {pname}")
            if any(
                tag in pname
                for tag in (
                    "audio_proj_in",
                    "audio_proj_out",
                    "audio_time_embed",
                    "audio_attn1",
                    "audio_attn2",
                    "audio_ff",
                    "audio_to_video_attn",
                    "video_to_audio_attn",
                    "av_cross_attn",
                )
            ):
              self.assertEqual(float(jnp.max(jnp.abs(g))), 0.0, f"Expected 0 grad for {pname} on video-only batch")

          # 4. STG/CFG isolated modality check: modality_mask=0 with valid audio still denoises audio_sample!
          out_stg = model(
              hidden_states=vid,
              audio_hidden_states=aud,
              encoder_hidden_states=txt,
              audio_encoder_hidden_states=txt,
              timestep=ts,
              video_coords=vid_coords,
              audio_coords=aud_coords,
              modality_mask=jnp.zeros((batch_size, 1, 1), dtype=jnp.float32),
          )
          self.assertGreater(float(jnp.linalg.norm(out_stg["audio_sample"])), 0.0)

  def test_svg_guard_and_positional_backwards_compatibility(self):
    """Verifies explicit SVG + video_attention_mask guard (Note N3) and 13-arg positional compatibility."""
    svg_model = self._make_model(attention_kernel="ulysses", use_svg_attention=True, seed=55)
    rules = _build_axis_rules("ulysses")
    b, s_vid, s_aud, s_txt = 2, 128, 128, 128
    vid = jnp.ones((b, s_vid, 16), dtype=jnp.float32)
    aud = jnp.ones((b, s_aud, 8), dtype=jnp.float32)
    txt = jnp.ones((b, s_txt, 64), dtype=jnp.float32)
    ts = jnp.ones((b,), dtype=jnp.float32)
    vmask = jnp.ones((b, s_vid), dtype=jnp.bool_)

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
      with self.assertRaisesRegex(ValueError, "SVG attention"):
        svg_model(
            hidden_states=vid,
            audio_hidden_states=aud,
            encoder_hidden_states=txt,
            audio_encoder_hidden_states=txt,
            timestep=ts,
            num_frames=2,
            height=8,
            width=8,
            audio_num_frames=s_aud,
            video_attention_mask=vmask,
        )

    # 13-positional-arg compatibility (matching maxdiffusion_utils.get_dummy_ltx2_inputs layout).
    dense_model = self._make_model(attention_kernel="dot_product", seed=56)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(_build_axis_rules("dot_product")):
      positional_args = (
          vid,
          aud,
          txt,
          txt,
          ts,
          None,
          ts,
          ts,
          jnp.ones((b, s_txt), dtype=jnp.bool_),
          jnp.ones((b, s_txt), dtype=jnp.bool_),
          2,
          8,
          8,
      )
      out_pos = dense_model(*positional_args, audio_num_frames=s_aud)
      self.assertEqual(out_pos["sample"].shape, (b, s_vid, 16))
      self.assertEqual(out_pos["audio_sample"].shape, (b, s_aud, 8))


if __name__ == "__main__":
  absltest.main()
