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
import tempfile
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

from maxdiffusion.common_types import SEQUENCE_PARALLEL_AXIS_RULES
from maxdiffusion.data_preprocessing.ltx2_bucketing import (
    LTX2BucketSpec,
    LTX2BucketedBatchIterator,
    LTX2Sample,
    LTX2SyntheticBucketedIterator,
    align_bucket_length,
    collate_ltx2_bucket_batch,
    compute_audio_coords_1sample,
    compute_video_coords_1sample,
    denormalize_audio_latents,
    denormalize_video_latents,
    encode_with_ltx2_connectors,
    get_ltx2_batch_shardings,
    iter_preencoded_npz_samples,
    normalize_audio_latents,
    normalize_video_latents,
    patchify_audio_latents,
    patchify_video_latents,
    select_bucket,
    shard_ltx2_batch_for_mesh,
    unpatchify_audio_latents,
    unpatchify_video_latents,
)
from maxdiffusion.models.ltx2.attention_ltx2 import LTX2RotaryPosEmbed
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2_audio import FlaxLTX2AudioAudioPatchifier
from maxdiffusion.models.ltx2.text_encoders.text_encoders_ltx2 import LTX2AudioVideoGemmaTextEncoder
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel


def _axis_rules():
  base_rules = [
      ("batch", ("data", "fsdp")),
      ("activation_batch", ("data", "fsdp")),
      ("activation_length", "context"),
      ("activation_kv_length", "context"),
      ("activation_embed", None),
      ("activation_heads", "tensor"),
      ("activation_kv", None),
      ("mlp", "tensor"),
      ("embed", ("fsdp", "context")),
      ("heads", "tensor"),
      ("norm", "tensor"),
      ("layers", None),
  ]
  return tuple(tuple(r) for r in SEQUENCE_PARALLEL_AXIS_RULES) + tuple(base_rules)


def _randomize_zero_params(model: LTX2VideoTransformer3DModel, key: jax.Array) -> None:
  _, state = nnx.split(model)
  leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
  out_leaves = []
  for idx, (_, val) in enumerate(leaves):
    if isinstance(val, jax.Array) and jnp.all(val == 0):
      noise = jax.random.normal(jax.random.fold_in(key, idx), val.shape, dtype=val.dtype) * 0.15
      out_leaves.append(noise)
    else:
      out_leaves.append(val)
  nnx.update(model, jax.tree_util.tree_unflatten(treedef, out_leaves))


def _place_state_on_mesh(state, mesh: Mesh, rules):
  logical_spec = nnx.get_partition_spec(state)
  mesh_sharding = nn.logical_to_mesh_sharding(logical_spec, mesh, rules)
  host_state = jax.device_get(state)
  return jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s), host_state, mesh_sharding)


class LTX2BucketingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    devices = np.array(jax.devices())
    self.devices = devices
    if devices.size >= 8:
      self.mesh_8 = Mesh(devices[:8].reshape(1, 2, 4, 1), ("data", "fsdp", "context", "tensor"))
      self.submesh_fsdp1 = Mesh(devices[:4].reshape(1, 1, 4, 1), ("data", "fsdp", "context", "tensor"))
    elif devices.size >= 4:
      self.mesh_8 = Mesh(devices[:4].reshape(1, 2, 2, 1), ("data", "fsdp", "context", "tensor"))
      self.submesh_fsdp1 = Mesh(devices[:2].reshape(1, 1, 2, 1), ("data", "fsdp", "context", "tensor"))
    else:
      raise unittest.SkipTest(f"Requires >= 4 host devices, found {devices.size}")

  def test_video_and_audio_normalization_and_patchify_parity(self):
    """Verifies asymmetric video/audio normalization + patchify/unpatchify round-trip & pipeline parity."""
    rng = np.random.default_rng(0)
    # 1. Video normalization + patchify/unpatchify (channel-first & channel-last).
    vid_cf = rng.standard_normal((2, 16, 4, 6, 8)).astype(np.float32)
    vid_cl = np.transpose(vid_cf, (0, 2, 3, 4, 1))
    v_mean = rng.standard_normal((16,)).astype(np.float32)
    v_std = np.abs(rng.standard_normal((16,)).astype(np.float32)) + 0.5
    scale = 0.85

    # Reference formula from LTX2Pipeline._normalize_latents / _pack_latents (ltx2_pipeline.py:L1464-L1530).
    ref_norm = (vid_cf - v_mean.reshape(1, -1, 1, 1, 1)) * scale / v_std.reshape(1, -1, 1, 1, 1)
    got_norm_cf = normalize_video_latents(vid_cf, v_mean, v_std, scaling_factor=scale, channel_last=False)
    got_norm_cl = normalize_video_latents(vid_cl, v_mean, v_std, scaling_factor=scale, channel_last=True)
    np.testing.assert_allclose(got_norm_cf, ref_norm, atol=1e-6)
    np.testing.assert_allclose(np.transpose(got_norm_cl, (0, 4, 1, 2, 3)), ref_norm, atol=1e-6)
    np.testing.assert_allclose(denormalize_video_latents(got_norm_cf, v_mean, v_std, scale), vid_cf, atol=1e-6)

    for p_t, p in [(1, 1), (2, 2)]:
      ref_packed = (
          ref_norm.reshape(2, 16, 4 // p_t, p_t, 6 // p, p, 8 // p, p)
          .transpose(0, 2, 4, 6, 1, 3, 5, 7)
          .reshape(2, (4 // p_t) * (6 // p) * (8 // p), 16 * p_t * p * p)
      )
      packed_cf = patchify_video_latents(got_norm_cf, patch_size=p, patch_size_t=p_t, channel_last=False)
      packed_cl = patchify_video_latents(got_norm_cl, patch_size=p, patch_size_t=p_t, channel_last=True)
      np.testing.assert_array_equal(packed_cf, ref_packed)
      np.testing.assert_array_equal(packed_cl, ref_packed)
      unpacked = unpatchify_video_latents(packed_cf, 4, 6, 8, patch_size=p, patch_size_t=p_t, channel_last=False)
      np.testing.assert_array_equal(unpacked, got_norm_cf)

    # 2. Audio patchify (BCFM & BTFC) + post-patchify normalization parity with FlaxLTX2AudioAudioPatchifier.
    aud_btfc = rng.standard_normal((2, 12, 16, 8)).astype(np.float32)  # (B, T, F_mel, C)
    aud_bcfm = np.transpose(aud_btfc, (0, 3, 1, 2))  # (B, C, T, F_mel)
    patchifier = FlaxLTX2AudioAudioPatchifier(sample_rate=16000, hop_length=160, audio_latent_downsample_factor=4)
    ref_aud_packed = np.asarray(patchifier.patchify(jnp.asarray(aud_btfc)))

    got_aud_btfc = patchify_audio_latents(aud_btfc, layout="BTFC")
    got_aud_bcfm = patchify_audio_latents(aud_bcfm, layout="BCFM")
    np.testing.assert_array_equal(got_aud_btfc, ref_aud_packed)
    np.testing.assert_array_equal(got_aud_bcfm, ref_aud_packed)
    np.testing.assert_array_equal(unpatchify_audio_latents(got_aud_btfc, 12, 16, 8, layout="BTFC"), aud_btfc)
    np.testing.assert_array_equal(unpatchify_audio_latents(got_aud_bcfm, 12, 16, 8, layout="BCFM"), aud_bcfm)

    a_mean = rng.standard_normal((128,)).astype(np.float32)
    a_std = np.abs(rng.standard_normal((128,)).astype(np.float32)) + 0.5
    ref_aud_norm = (ref_aud_packed - a_mean) / a_std
    got_aud_norm = normalize_audio_latents(got_aud_btfc, a_mean, a_std)
    np.testing.assert_allclose(got_aud_norm, ref_aud_norm, atol=1e-6)
    np.testing.assert_allclose(denormalize_audio_latents(got_aud_norm, a_mean, a_std), ref_aud_packed, atol=1e-6)

    # 3. Collate unnormalized raw VAE sample (`is_normalized=False`): video normalizes BEFORE pack; audio packs BEFORE normalize!
    raw_sample = LTX2Sample(
        video_latents=vid_cl[0],
        audio_latents=aud_btfc[0],
        encoder_hidden_states=rng.standard_normal((64, 32)).astype(np.float32),
        is_normalized=False,
        video_channel_last=True,
        audio_layout="BTFC",
    )
    bucket = LTX2BucketSpec(b_vid=256, b_aud=32, b_txt=64)
    collated = collate_ltx2_bucket_batch(
        [raw_sample],
        bucket,
        video_latents_mean=v_mean,
        video_latents_std=v_std,
        video_scaling_factor=scale,
        audio_latents_mean=a_mean,
        audio_latents_std=a_std,
    )
    ref_vid_1 = patchify_video_latents(ref_norm[0], patch_size=1, patch_size_t=1)
    np.testing.assert_allclose(collated["hidden_states"][0, : ref_vid_1.shape[0]], ref_vid_1, atol=1e-6)
    np.testing.assert_allclose(collated["audio_hidden_states"][0, :12], ref_aud_norm[0], atol=1e-6)

  def test_post_connector_register_parity_and_right_pad_guard(self):
    """Verifies post-connector register-filled conditioning parity and guards against right-pad skew."""
    connectors = LTX2AudioVideoGemmaTextEncoder(
        caption_channels=64,
        text_proj_in_factor=2,
        video_connector_attention_head_dim=32,
        video_connector_num_attention_heads=2,
        video_connector_num_layers=1,
        video_connector_num_learnable_registers=8,
        audio_connector_attention_head_dim=32,
        audio_connector_num_attention_heads=2,
        audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=8,
        attention_kernel="dot_product",
        rngs=nnx.Rngs(0),
    )
    # Left-padded tokenizer mask of fixed length t=16 (divisible by num_learnable_registers=8).
    gemma_states = [jax.random.normal(jax.random.key(i), (2, 16, 64)) for i in range(2)]
    left_pad_mask = jnp.zeros((2, 16), dtype=jnp.bool_).at[0, 10:].set(True).at[1, 6:].set(True)

    video_embeds, audio_embeds, new_mask = encode_with_ltx2_connectors(connectors, gemma_states, left_pad_mask)
    self.assertEqual(video_embeds.shape, (2, 16, 64))
    self.assertEqual(audio_embeds.shape, (2, 16, 64))
    self.assertTrue(bool(jnp.all(new_mask)))

    # Also verify standalone Embeddings1DConnector._replace_padded_with_learnable_registers left-shift + register fill.
    conn_1d = connectors.video_embeddings_connector
    raw_feat = jax.random.normal(jax.random.key(9), (2, 16, 64))
    reg_out, reg_mask = conn_1d._replace_padded_with_learnable_registers(raw_feat, left_pad_mask)
    self.assertTrue(bool(jnp.all(reg_mask)))
    np.testing.assert_allclose(np.asarray(reg_out[0, :6]), np.asarray(raw_feat[0, 10:]), atol=1e-6)

    samples = [
        LTX2Sample(
            video_latents=np.ones((16, 2, 2, 2), dtype=np.float32),
            audio_latents=np.ones((8, 8), dtype=np.float32),
            encoder_hidden_states=np.asarray(video_embeds[i]),
            audio_encoder_hidden_states=np.asarray(audio_embeds[i]),
            encoder_attention_mask=np.asarray(new_mask[i]),
        )
        for i in range(2)
    ]
    bucket = LTX2BucketSpec(b_vid=16, b_aud=16, b_txt=16)
    batch = collate_ltx2_bucket_batch(samples, bucket, audio_in_channels=8, text_padding_mode="post_connector")
    np.testing.assert_allclose(batch["encoder_hidden_states"], np.asarray(video_embeds), atol=1e-6)
    np.testing.assert_allclose(batch["audio_encoder_hidden_states"], np.asarray(audio_embeds), atol=1e-6)
    self.assertTrue(bool(np.all(batch["encoder_attention_mask"])))
    self.assertTrue(bool(np.all(batch["audio_encoder_attention_mask"])))

    # Mismatched L_txt under post_connector must raise ValueError.
    bad_sample = LTX2Sample(
        video_latents=np.ones((16, 2, 2, 2), dtype=np.float32),
        encoder_hidden_states=np.ones((10, 64), dtype=np.float32),
        has_audio=False,
    )
    with self.assertRaisesRegex(ValueError, "post_connector"):
      collate_ltx2_bucket_batch([bad_sample], bucket, audio_in_channels=8, text_padding_mode="post_connector")

    # Opting into right_pad_masked with num_learnable_registers > 0 must raise ValueError.
    with self.assertRaisesRegex(ValueError, "num_learnable_registers"):
      collate_ltx2_bucket_batch(
          [bad_sample],
          bucket,
          audio_in_channels=8,
          text_padding_mode="right_pad_masked",
          connectors=connectors,
      )
    with self.assertRaisesRegex(ValueError, "num_learnable_registers"):
      collate_ltx2_bucket_batch(
          [bad_sample],
          bucket,
          audio_in_channels=8,
          text_padding_mode="right_pad_masked",
          connector_num_learnable_registers=128,
      )

    # Succeeds when num_learnable_registers == 0.
    masked_batch = collate_ltx2_bucket_batch(
        [bad_sample],
        bucket,
        audio_in_channels=8,
        text_padding_mode="right_pad_masked",
        connector_num_learnable_registers=0,
    )
    self.assertEqual(int(np.sum(masked_batch["encoder_attention_mask"])), 10)

  def test_coords_parity_and_heterogeneous_batch_fwd_and_grad_invariance(self):
    """Verifies per-sample RoPE coords parity & heterogeneous B=2 vs unpadded B=1 fwd + Param grad equality."""
    # 1. Standalone coordinate parity against LTX2RotaryPosEmbed.
    vid_rope = LTX2RotaryPosEmbed(dim=512, patch_size=2, patch_size_t=1, scale_factors=(8, 32, 32), modality="video")
    aud_rope = LTX2RotaryPosEmbed(
        dim=512, patch_size=1, patch_size_t=2, sampling_rate=16000, hop_length=160, scale_factors=(4,), modality="audio"
    )
    for f, h, w, fps in [(3, 6, 10, 24.0), (5, 8, 4, 16.0)]:
      ref_vc = np.asarray(vid_rope.prepare_video_coords(1, f, h, w, fps=fps)[0])
      got_vc = compute_video_coords_1sample(f, h, w, fps=fps, patch_size=2, patch_size_t=1)
      np.testing.assert_allclose(got_vc, ref_vc, atol=1e-6)

    for t_aud in [10, 26]:
      ref_ac = np.asarray(aud_rope.prepare_audio_coords(1, t_aud)[0])
      got_ac = compute_audio_coords_1sample(t_aud, patch_size_t=2)
      np.testing.assert_allclose(got_ac, ref_ac, atol=1e-6)

    # 2. Heterogeneous B=2 bucket batch vs unpadded B=1 references using IDENTICAL model state.
    rules = _axis_rules()
    block_sizes = splash_attention_kernel.BlockSizes(
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
    with jax.set_mesh(self.mesh_8), nn_partitioning.axis_rules(rules):
      model = LTX2VideoTransformer3DModel(
          rngs=nnx.Rngs(jax.random.key(12)),
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
          mesh=self.mesh_8,
          scan_layers=True,
          remat_policy="FULL",
          attention_kernel="dot_product",
          a2v_attention_kernel="dot_product",
          v2a_attention_kernel="dot_product",
          flash_block_sizes=block_sizes,
          flash_min_seq_length=0,
          cross_attn_mod=True,
          gated_attn=True,
      )
      _randomize_zero_params(model, jax.random.key(77))

    graphdef, raw_state = nnx.split(model)
    state_8 = _place_state_on_mesh(raw_state, self.mesh_8, rules)
    state_sub = _place_state_on_mesh(raw_state, self.submesh_fsdp1, rules)

    rng = np.random.default_rng(123)
    # Sample 0: landscape (F=2, H=6, W=10) -> S_vid_0=120, fps=24.0, S_aud_0=48
    # Sample 1: portrait  (F=3, H=8, W=4)  -> S_vid_1=96,  fps=16.0, S_aud_1=28 (or 0 in mixed-modality subtest)
    vid_0 = rng.standard_normal((16, 2, 6, 10)).astype(np.float32)
    vid_1 = rng.standard_normal((16, 3, 8, 4)).astype(np.float32)
    aud_0 = rng.standard_normal((48, 8)).astype(np.float32)
    aud_1 = rng.standard_normal((28, 8)).astype(np.float32)
    txt_0 = rng.standard_normal((64, 64)).astype(np.float32)
    txt_1 = rng.standard_normal((64, 64)).astype(np.float32)
    atxt_0 = rng.standard_normal((64, 64)).astype(np.float32)
    atxt_1 = rng.standard_normal((64, 64)).astype(np.float32)
    tv_0 = rng.standard_normal((120, 16)).astype(np.float32)
    tv_1 = rng.standard_normal((96, 16)).astype(np.float32)
    ta_0 = rng.standard_normal((48, 8)).astype(np.float32)
    ta_1 = rng.standard_normal((28, 8)).astype(np.float32)

    bucket = LTX2BucketSpec(b_vid=128, b_aud=64, b_txt=64)
    bucket.validate_for_mesh(context_parallelism=4)

    timesteps = jnp.array([320.0, 710.0], dtype=jnp.float32)
    sigmas = jnp.array([0.32, 0.71], dtype=jnp.float32)

    for sample1_has_audio in (True, False):
      with self.subTest(sample1_has_audio=sample1_has_audio):
        s0 = LTX2Sample(
            video_latents=vid_0,
            audio_latents=aud_0,
            encoder_hidden_states=txt_0,
            audio_encoder_hidden_states=atxt_0,
            fps=24.0,
            has_audio=True,
        )
        s1 = LTX2Sample(
            video_latents=vid_1,
            audio_latents=aud_1 if sample1_has_audio else None,
            encoder_hidden_states=txt_1,
            audio_encoder_hidden_states=atxt_1,
            fps=16.0,
            has_audio=sample1_has_audio,
        )
        batch_np = collate_ltx2_bucket_batch([s0, s1], bucket, audio_in_channels=8, text_padding_mode="post_connector")
        batch_sh = shard_ltx2_batch_for_mesh(batch_np, self.mesh_8)

        # Check temporal cross-attn slice `video_coords[:, 0:1, :]` bit-matches standalone per-sample coords!
        vc0_ref = compute_video_coords_1sample(2, 6, 10, fps=24.0)
        vc1_ref = compute_video_coords_1sample(3, 8, 4, fps=16.0)
        np.testing.assert_array_equal(batch_np["video_coords"][0, 0:1, :120, :], vc0_ref[0:1])
        np.testing.assert_array_equal(batch_np["video_coords"][1, 0:1, :96, :], vc1_ref[0:1])

        tv_pad = np.zeros((2, 128, 16), dtype=np.float32)
        tv_pad[0, :120] = tv_0
        tv_pad[1, :96] = tv_1
        ta_pad = np.zeros((2, 64, 8), dtype=np.float32)
        ta_pad[0, :48] = ta_0
        if sample1_has_audio:
          ta_pad[1, :28] = ta_1

        sh_vid = get_ltx2_batch_shardings(self.mesh_8, bucket)["hidden_states"]
        sh_aud = get_ltx2_batch_shardings(self.mesh_8, bucket)["audio_hidden_states"]
        sh_b = NamedSharding(self.mesh_8, P(("data", "fsdp")))
        tv_pad_sh = jax.device_put(tv_pad, sh_vid)
        ta_pad_sh = jax.device_put(ta_pad, sh_aud)
        ts_sh = jax.device_put(timesteps, sh_b)
        sig_sh = jax.device_put(sigmas, sh_b)

        with jax.set_mesh(self.mesh_8), nn_partitioning.axis_rules(rules):

          @jax.jit
          def run_batch(st, b_dict, tv, ta, ts, sig):
            def loss_fn(s):
              m = nnx.merge(graphdef, s)
              m.mesh = self.mesh_8
              out = m(
                  hidden_states=b_dict["hidden_states"],
                  audio_hidden_states=b_dict["audio_hidden_states"],
                  encoder_hidden_states=b_dict["encoder_hidden_states"],
                  audio_encoder_hidden_states=b_dict["audio_encoder_hidden_states"],
                  timestep=ts,
                  sigma=sig,
                  audio_sigma=sig,
                  encoder_attention_mask=b_dict["encoder_attention_mask"],
                  audio_encoder_attention_mask=b_dict["audio_encoder_attention_mask"],
                  video_coords=b_dict["video_coords"],
                  audio_coords=b_dict["audio_coords"],
                  modality_mask=b_dict["modality_mask"],
                  video_attention_mask=b_dict["video_attention_mask"],
                  audio_attention_mask=b_dict["audio_attention_mask"],
              )
              vm = b_dict["video_attention_mask"][:, :, None]
              am = b_dict["audio_attention_mask"][:, :, None]
              v_loss = jnp.sum(jnp.where(vm, (out["sample"] - tv) ** 2, 0.0)) / (jnp.sum(vm) * 16)
              a_denom = jnp.maximum(jnp.sum(am) * 8, 1)
              a_loss = jnp.sum(jnp.where(am, (out["audio_sample"] - ta) ** 2, 0.0)) / a_denom
              return v_loss + a_loss, out

            return nnx.value_and_grad(loss_fn, has_aux=True)(st)

          (loss_b, out_b), grads_b = run_batch(
              state_8,
              {k: v for k, v in batch_sh.items() if k != "bucket_key"},
              tv_pad_sh,
              ta_pad_sh,
              ts_sh,
              sig_sh,
          )

        # Single-sample B=1 unpadded references on fsdp=1 submesh with identical weights.
        s_vid_total = 120 + 96
        s_aud_1 = 28 if sample1_has_audio else 0
        s_aud_total = max(48 + s_aud_1, 1)

        def run_single_unpadded(v_np, a_np, t_np, at_np, f, h, w, fps, has_aud, tv_s, ta_s, ts_val, sig_val, w_v, w_a):
          v_p = jnp.asarray(patchify_video_latents(v_np)[None, ...])
          vc = jnp.asarray(compute_video_coords_1sample(f, h, w, fps=fps)[None, ...])
          if has_aud:
            a_p = jnp.asarray(a_np[None, ...])
            ac = jnp.asarray(compute_audio_coords_1sample(a_np.shape[0])[None, ...])
            am = jnp.ones((1, a_np.shape[0]), dtype=jnp.bool_)
            mm = jnp.ones((1, 1, 1), dtype=jnp.float32)
          else:
            a_p = jnp.zeros((1, 4, 8), dtype=jnp.float32)
            ac = jnp.zeros((1, 1, 4, 2), dtype=jnp.float32)
            am = jnp.zeros((1, 4), dtype=jnp.bool_)
            mm = jnp.zeros((1, 1, 1), dtype=jnp.float32)

          with jax.set_mesh(self.submesh_fsdp1), nn_partitioning.axis_rules(rules):

            def single_loss(s):
              m = nnx.merge(graphdef, s)
              m.mesh = self.submesh_fsdp1
              out = m(
                  hidden_states=v_p,
                  audio_hidden_states=a_p,
                  encoder_hidden_states=jnp.asarray(t_np[None, ...]),
                  audio_encoder_hidden_states=jnp.asarray(at_np[None, ...]),
                  timestep=jnp.array([ts_val], dtype=jnp.float32),
                  sigma=jnp.array([sig_val], dtype=jnp.float32),
                  audio_sigma=jnp.array([sig_val], dtype=jnp.float32),
                  encoder_attention_mask=jnp.ones((1, 64), dtype=jnp.bool_),
                  audio_encoder_attention_mask=jnp.ones((1, 64), dtype=jnp.bool_),
                  video_coords=vc,
                  audio_coords=ac,
                  modality_mask=mm,
                  audio_attention_mask=am if not has_aud else None,
              )
              lv = jnp.mean((out["sample"] - jnp.asarray(tv_s[None, ...])) ** 2)
              la = jnp.mean((out["audio_sample"] - jnp.asarray(ta_s[None, ...])) ** 2) if has_aud else 0.0
              return w_v * lv + w_a * la, out

            return nnx.value_and_grad(single_loss, has_aux=True)(state_sub)

        (loss_0, out_0), g_0 = run_single_unpadded(
            vid_0, aud_0, txt_0, atxt_0, 2, 6, 10, 24.0, True, tv_0, ta_0, 320.0, 0.32, 120 / s_vid_total, 48 / s_aud_total
        )
        (loss_1, out_1), g_1 = run_single_unpadded(
            vid_1,
            aud_1,
            txt_1,
            atxt_1,
            3,
            8,
            4,
            16.0,
            sample1_has_audio,
            tv_1,
            ta_1,
            710.0,
            0.71,
            96 / s_vid_total,
            s_aud_1 / s_aud_total,
        )

        # Forward valid-prefix parity + padded-tail exact zero.
        out_b_host = jax.device_get(out_b)
        out_0_host = jax.device_get(out_0)
        out_1_host = jax.device_get(out_1)
        np.testing.assert_allclose(out_b_host["sample"][0, :120], out_0_host["sample"][0], atol=1e-4)
        np.testing.assert_allclose(out_b_host["sample"][1, :96], out_1_host["sample"][0], atol=1e-4)
        self.assertEqual(float(np.max(np.abs(out_b_host["sample"][0, 120:]))), 0.0)
        self.assertEqual(float(np.max(np.abs(out_b_host["sample"][1, 96:]))), 0.0)
        np.testing.assert_allclose(out_b_host["audio_sample"][0, :48], out_0_host["audio_sample"][0], atol=1e-4)
        if sample1_has_audio:
          np.testing.assert_allclose(out_b_host["audio_sample"][1, :28], out_1_host["audio_sample"][0], atol=1e-4)
        else:
          self.assertEqual(float(np.max(np.abs(out_b_host["audio_sample"][1]))), 0.0)

        self.assertAlmostEqual(float(loss_b), float(loss_0 + loss_1), places=5)

        # Parameter gradient parity across EVERY Param leaf.
        gb_leaves = jax.tree_util.tree_leaves_with_path(jax.device_get(grads_b))
        g0_leaves = jax.tree_util.tree_leaves_with_path(jax.device_get(g_0))
        g1_leaves = jax.tree_util.tree_leaves_with_path(jax.device_get(g_1))
        max_leaf_diff = 0.0
        for (path, gb), (_, l0), (_, l1) in zip(gb_leaves, g0_leaves, g1_leaves):
          pname = "/".join(str(p) for p in path)
          diff = float(np.max(np.abs(gb - (l0 + l1))))
          max_leaf_diff = max(max_leaf_diff, diff)
          self.assertLess(diff, 2e-4, f"Heterogeneous grad mismatch on {pname}: {diff}")
        print(
            f"[Heterogeneous B=2 vs B=1 sample1_has_audio={sample1_has_audio}] max |Δgrad| across {len(gb_leaves)} leaves = {max_leaf_diff:.3e}"
        )

  def test_bucket_alignment_iterators_npz_and_mesh_shardings(self):
    """Verifies bucket alignment/validation, NPZ & FIFO bucket iterators, synthetic iterator, and sharding specs."""
    # 1. Alignment & mesh validation.
    self.assertEqual(align_bucket_length(130, context_parallelism=4, block_multiple=128, ring_shards=1), 256)
    self.assertEqual(align_bucket_length(130, context_parallelism=4, block_multiple=128, ring_shards=2), 256)
    self.assertEqual(align_bucket_length(130, context_parallelism=4, block_multiple=128, scale_block_by_context=True), 512)

    with self.assertRaisesRegex(ValueError, "b_txt"):
      LTX2BucketSpec(b_vid=128, b_aud=64, b_txt=62).validate_for_mesh(context_parallelism=4)
    with self.assertRaisesRegex(ValueError, "b_aud"):
      LTX2BucketSpec(b_vid=128, b_aud=62, b_txt=64).validate_for_mesh(context_parallelism=4)

    buckets = [LTX2BucketSpec(128, 64, 64), LTX2BucketSpec(256, 128, 64)]
    self.assertEqual(select_bucket(100, 32, 64, buckets), LTX2BucketSpec(128, 64, 64))
    self.assertEqual(select_bucket(140, 32, 64, buckets), LTX2BucketSpec(256, 128, 64))
    self.assertIsNone(select_bucket(300, 32, 64, buckets, policy="drop"))
    self.assertEqual(
        select_bucket(300, 32, 64, buckets, policy="pad_to_aligned", align_kwargs={"context_parallelism": 4}),
        LTX2BucketSpec(384, 32, 128),
    )

    # 2. Pre-encoded NPZ directory -> LTX2BucketedBatchIterator on 8-device mesh with global_batch_size=2 (< 8 devices!).
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmpdir:
      specs = [
          (2, 4, 8, 24),  # S_vid=64  -> bucket (128, 64, 64)
          (3, 8, 8, 80),  # S_vid=192 -> bucket (256, 128, 64)
          (2, 5, 10, 40),  # S_vid=100 -> bucket (128, 64, 64) -> triggers 1st batch emit for (128, 64, 64)!
          (2, 8, 12, 96),  # S_vid=192 -> bucket (256, 128, 64) -> triggers 2nd batch emit for (256, 128, 64)!
      ]
      for idx, (f, h, w, ta) in enumerate(specs):
        np.savez(
            os.path.join(tmpdir, f"sample_{idx:03d}.npz"),
            video_latents=rng.standard_normal((16, f, h, w)).astype(np.float32),
            audio_latents=rng.standard_normal((ta, 8)).astype(np.float32),
            encoder_hidden_states=rng.standard_normal((64, 32)).astype(np.float32),
            fps=np.array(24.0, dtype=np.float32),
            has_audio=np.array(True),
        )

      it = LTX2BucketedBatchIterator(
          iter_preencoded_npz_samples(tmpdir, repeat=False),
          buckets=buckets,
          batch_size=2,
          mesh=self.mesh_8,
          collate_kwargs={"audio_in_channels": 8, "text_padding_mode": "post_connector"},
      )
      emitted = list(it)
      self.assertEqual(len(emitted), 2)
      self.assertEqual(emitted[0]["bucket_key"], (128, 64, 64))
      self.assertEqual(emitted[1]["bucket_key"], (256, 128, 64))

      # Assert exact sharding specs on every tensor.
      b0 = emitted[0]
      self.assertEqual(b0["hidden_states"].sharding.spec, P(("data", "fsdp"), "context", None))
      self.assertEqual(b0["encoder_hidden_states"].sharding.spec, P(("data", "fsdp"), "context", None))
      self.assertEqual(b0["video_coords"].sharding.spec, P(("data", "fsdp"), None, "context", None))
      self.assertEqual(b0["audio_hidden_states"].sharding.spec, P(("data", "fsdp"), None, None))
      self.assertEqual(b0["audio_encoder_hidden_states"].sharding.spec, P(("data", "fsdp"), None, None))
      self.assertEqual(b0["audio_coords"].sharding.spec, P(("data", "fsdp"), None, None, None))
      self.assertEqual(b0["video_attention_mask"].sharding.spec, P(("data", "fsdp"), None))
      self.assertEqual(b0["audio_attention_mask"].sharding.spec, P(("data", "fsdp"), None))
      self.assertEqual(b0["encoder_attention_mask"].sharding.spec, P(("data", "fsdp"), None))
      self.assertEqual(b0["audio_encoder_attention_mask"].sharding.spec, P(("data", "fsdp"), None))
      self.assertEqual(b0["modality_mask"].sharding.spec, P(("data", "fsdp"), None, None))

      # Also test input_is_process_local=True explicit global_shape path.
      host_b0 = {k: (v if k == "bucket_key" else np.asarray(jax.device_get(v))) for k, v in b0.items()}
      local_sharded = shard_ltx2_batch_for_mesh(host_b0, self.mesh_8, input_is_process_local=True)
      self.assertEqual(local_sharded["hidden_states"].sharding.spec, P(("data", "fsdp"), "context", None))
      self.assertEqual(local_sharded["encoder_hidden_states"].sharding.spec, P(("data", "fsdp"), "context", None))

    # 3. Deterministic LTX2SyntheticBucketedIterator repeatability & video-only mix.
    synth = LTX2SyntheticBucketedIterator(
        buckets=buckets,
        global_batch_size=4,
        in_channels=16,
        audio_in_channels=8,
        caption_channels=32,
        video_only_prob=0.5,
        num_unique_batches_per_bucket=1,
        num_steps=4,
        seed=99,
        mesh=self.mesh_8,
    )
    synth_batches = list(synth)
    self.assertEqual(
        [b["bucket_key"] for b in synth_batches], [(128, 64, 64), (256, 128, 64), (128, 64, 64), (256, 128, 64)]
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(synth_batches[0]["hidden_states"])),
        np.asarray(jax.device_get(synth_batches[2]["hidden_states"])),
    )


if __name__ == "__main__":
  absltest.main()
