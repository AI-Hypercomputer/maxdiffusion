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
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

from absl.testing import absltest
from flax import nnx
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

try:
  import chex  # noqa: F401
except ImportError:
  _chex_stub = types.ModuleType("chex")
  _chex_stub.Array = jax.Array
  _chex_stub.PRNGKey = jax.Array
  sys.modules["chex"] = _chex_stub

from maxdiffusion import pyconfig
from maxdiffusion.data_preprocessing.ltx2_bucketing import (
    LTX2BucketSpec,
    LTX2Sample,
    collate_ltx2_bucket_batch,
    compute_audio_coords_1sample,
    compute_video_coords_1sample,
    get_ltx2_batch_shardings,
    patchify_video_latents,
    shard_ltx2_batch_for_mesh,
)
from maxdiffusion.models.ltx2.logical_sharding_ltx2 import get_sharding_specs
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.train_ltx2 import train as run_train_ltx2
from maxdiffusion.trainers.ltx2_trainer import (
    LTX2Trainer,
    compute_ltx2_microbatch_errors,
    extract_unmerged_base_params,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
CONFIG_PATH = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "configs", "ltx2_train.yml"))
UPSTREAM_BASELINE_SHA = "a560b4d834d551f77c7227989794d2ab8d66845a"

FLASH_BS_128 = splash_attention_kernel.BlockSizes(
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
FLASH_BS_128_JSON = (
    '{"block_q":128,"block_kv":128,"block_kv_compute":128,'
    '"block_q_dkv":128,"block_kv_dkv":128,"block_kv_dkv_compute":128,'
    '"block_q_dq":128,"block_kv_dq":128,"use_fused_bwd_kernel":false}'
)


def _init_bucketing_test_config(tmpdir: str, extra_overrides=()):
  argv = [
      None,
      CONFIG_PATH,
      f"output_dir={tmpdir}",
      "checkpoint_dir=",
      "run_name=ltx2_bucketing_gate_test",
      "init_transformer_from_config_only=True",
      "load_encoders_for_training=False",
      "weights_dtype=float32",
      "activations_dtype=float32",
      "precision=HIGHEST",
      "attention=dot_product",
      "a2v_attention_kernel=dot_product",
      "v2a_attention_kernel=dot_product",
      "transformer_in_channels=16",
      "transformer_audio_in_channels=8",
      "transformer_num_attention_heads=4",
      "transformer_attention_head_dim=32",
      "transformer_audio_num_attention_heads=4",
      "transformer_audio_attention_head_dim=32",
      "transformer_num_layers=2",
      "transformer_caption_channels=32",
      "transformer_cross_attn_mod=True",
      "transformer_gated_attn=True",
      "ici_data_parallelism=1",
      "ici_fsdp_parallelism=2",
      "ici_context_parallelism=4",
      "ici_tensor_parallelism=1",
      "per_device_batch_size=0.25",
      "ltx2_buckets=[[128,64,64],[256,128,64]]",
      "max_sequence_length=64",
      "write_metrics=False",
  ] + list(extra_overrides)
  pyconfig.initialize(argv, unittest=True)
  return pyconfig.config


def _randomize_zero_params(model: LTX2VideoTransformer3DModel, key: jax.Array) -> None:
  _, state = nnx.split(model)
  leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
  out_leaves = []
  for idx, (_, val) in enumerate(leaves):
    if isinstance(val, jax.Array) and jnp.all(val == 0):
      out_leaves.append(jax.random.normal(jax.random.fold_in(key, idx), val.shape, dtype=val.dtype) * 0.1)
    else:
      out_leaves.append(val)
  nnx.update(model, jax.tree_util.tree_unflatten(treedef, out_leaves))


def _encode_flat_path(path_tuple) -> str:
  return "__".join(f"{type(part).__name__}:{part}" for part in path_tuple)


def _dump_nnx_params_to_dict(params_state: nnx.State) -> dict[str, np.ndarray]:
  out = {}
  for path_tuple, var in nnx.to_flat_state(params_state):
    out[_encode_flat_path(path_tuple)] = np.asarray(jax.device_get(var.get_value()))
  return out


def _place_state_on_mesh(state: nnx.State, mesh: Mesh, rules) -> nnx.State:
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
    logical_spec = nnx.get_partition_spec(state)
    shardings = nn.logical_to_mesh_sharding(logical_spec, mesh, rules)
  return jax.tree_util.tree_map(lambda x, s: jax.device_put(jax.device_get(x), s), state, shardings)


def _max_tree_abs_diff(tree_a, tree_b) -> tuple[float, int]:
  leaves_a = jax.tree_util.tree_leaves(jax.device_get(tree_a))
  leaves_b = jax.tree_util.tree_leaves(jax.device_get(tree_b))
  assert len(leaves_a) == len(leaves_b), f"Leaf count mismatch: {len(leaves_a)} vs {len(leaves_b)}"
  max_diff = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(leaves_a, leaves_b))
  return max_diff, len(leaves_a)


class LTX2TrainerBucketingVerificationTest(unittest.TestCase):
  """Layer 4 Full Zero-Trust Verification Suite covering Gates 1a, 1b, 1c, 2, and 3."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    devices = jax.devices()
    if len(devices) < 8:
      raise unittest.SkipTest(f"Requires 8 host devices, found {len(devices)}")
    cls.devices = devices[:8]
    cls.mesh_8 = Mesh(np.array(cls.devices).reshape(1, 2, 4, 1), ("data", "fsdp", "context", "tensor"))
    cls.submesh_fsdp1 = Mesh(np.array(cls.devices[:4]).reshape(1, 1, 4, 1), ("data", "fsdp", "context", "tensor"))

  def setUp(self):
    super().setUp()
    self.enterContext(jax.default_matmul_precision("float32"))

  # ============================================================================
  # Gate 1a: Upstream a560b4d8 Inference Backwards-Compatibility +
  #          Homogeneous 3-Stream (F, H, W, T_aud, L_txt) Right-Padding Gate
  # ============================================================================
  def test_gate1a_upstream_a560b4d8_parity_and_homogeneous_3stream_padding_invariance(self):
    """Proves upstream a560b4d8 masks=None inference parity + 3-stream fwd/grad right-padding invariance."""
    # --- Sub-check (i): Read-only upstream a560b4d8 baseline parity ---
    if shutil.which("git") is None:
      self.skipTest("git executable unavailable for upstream baseline extraction")
    chk = subprocess.run(
        ["git", "-C", REPO_ROOT, "cat-file", "-e", f"{UPSTREAM_BASELINE_SHA}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if chk.returncode != 0:
      self.skipTest(f"Upstream commit {UPSTREAM_BASELINE_SHA} not present in local git object store")

    with tempfile.TemporaryDirectory() as tmpdir:
      up_dir = os.path.join(tmpdir, "upstream")
      os.makedirs(up_dir, exist_ok=True)
      archive_proc = subprocess.Popen(
          ["git", "-C", REPO_ROOT, "archive", UPSTREAM_BASELINE_SHA, "src"],
          stdout=subprocess.PIPE,
      )
      tar_proc = subprocess.Popen(["tar", "-x", "-C", up_dir], stdin=archive_proc.stdout)
      assert archive_proc.stdout is not None
      archive_proc.stdout.close()
      self.assertEqual(tar_proc.wait(), 0)
      self.assertEqual(archive_proc.wait(), 0)

      cfg = _init_bucketing_test_config(tmpdir)
      rules = cfg.logical_axis_rules
      dit_specs = get_sharding_specs("default", "ltx2_dit")

      def _build_head_model(scan_layers: bool, attention_kernel: str) -> LTX2VideoTransformer3DModel:
        with jax.set_mesh(self.mesh_8), nn_partitioning.axis_rules(rules):
          m = LTX2VideoTransformer3DModel(
              in_channels=16,
              out_channels=16,
              patch_size=1,
              patch_size_t=1,
              num_attention_heads=4,
              attention_head_dim=32,
              cross_attention_dim=128,
              audio_in_channels=8,
              audio_out_channels=8,
              audio_num_attention_heads=4,
              audio_attention_head_dim=32,
              audio_cross_attention_dim=128,
              num_layers=2,
              caption_channels=32,
              cross_attn_mod=True,
              gated_attn=True,
              perturbed_attn=True,
              spatio_temporal_guidance_blocks=(0,),
              scan_layers=scan_layers,
              attention_kernel=attention_kernel,
              a2v_attention_kernel=attention_kernel,
              v2a_attention_kernel=attention_kernel,
              flash_block_sizes=FLASH_BS_128,
              flash_min_seq_length=0,
              dtype=jnp.float32,
              weights_dtype=jnp.float32,
              mesh=self.mesh_8,
              sharding_specs=dit_specs,
              rngs=nnx.Rngs(jax.random.key(101 if scan_layers else 202)),
          )
          _randomize_zero_params(m, jax.random.key(303 if scan_layers else 404))
          gdef, p_st, r_st = nnx.split(m, nnx.Param, ...)
          p_st = _place_state_on_mesh(p_st, self.mesh_8, rules)
          return nnx.merge(gdef, p_st, r_st)

      rng_np = np.random.default_rng(11)
      # Standard B=2 inputs (S_vid=128, S_aud=128, L_txt=128) + B=4 CFG+STG 4-way split inputs.
      inputs_npz = {
          "vid_b2": rng_np.standard_normal((2, 128, 16)).astype(np.float32),
          "aud_b2": rng_np.standard_normal((2, 128, 8)).astype(np.float32),
          "txt_b2": rng_np.standard_normal((2, 128, 32)).astype(np.float32),
          "atxt_b2": rng_np.standard_normal((2, 128, 32)).astype(np.float32),
          "ts_b2": np.array([250.0, 750.0], dtype=np.float32),
          "sig_b2": np.array([250.0, 750.0], dtype=np.float32),
          "em_b2": np.array([[True] * 96 + [False] * 32, [True] * 80 + [False] * 48], dtype=np.bool_),
          "vid_b4": rng_np.standard_normal((4, 128, 16)).astype(np.float32),
          "aud_b4": rng_np.standard_normal((4, 128, 8)).astype(np.float32),
          "txt_b4": rng_np.standard_normal((4, 128, 32)).astype(np.float32),
          "atxt_b4": rng_np.standard_normal((4, 128, 32)).astype(np.float32),
          "ts_b4": np.array([420.0, 420.0, 420.0, 420.0], dtype=np.float32),
          "sig_b4": np.array([420.0, 420.0, 420.0, 420.0], dtype=np.float32),
          "mod_b4": np.array([[[1.0]], [[1.0]], [[1.0]], [[0.0]]], dtype=np.float32),
          "pert_b4": np.array([[[1.0]], [[1.0]], [[0.0]], [[1.0]]], dtype=np.float32),
      }

      head_outputs = {}
      weights_scan_true = None
      weights_scan_false = None

      for scan_flag in (True, False):
        for kern in ("dot_product", "flash"):
          m = _build_head_model(scan_layers=scan_flag, attention_kernel=kern)
          _, p_st, _ = nnx.split(m, nnx.Param, ...)
          if scan_flag and weights_scan_true is None:
            weights_scan_true = _dump_nnx_params_to_dict(p_st)
          if (not scan_flag) and weights_scan_false is None:
            weights_scan_false = _dump_nnx_params_to_dict(p_st)

          with jax.set_mesh(self.mesh_8), nn_partitioning.axis_rules(rules):
            s_out, a_out = m(
                hidden_states=jnp.asarray(inputs_npz["vid_b2"]),
                encoder_hidden_states=jnp.asarray(inputs_npz["txt_b2"]),
                timestep=jnp.asarray(inputs_npz["ts_b2"]),
                sigma=jnp.asarray(inputs_npz["sig_b2"]),
                audio_sigma=jnp.asarray(inputs_npz["sig_b2"]),
                encoder_attention_mask=jnp.asarray(inputs_npz["em_b2"]),
                modality_mask=None,
                num_frames=2,
                height=8,
                width=8,
                audio_hidden_states=jnp.asarray(inputs_npz["aud_b2"]),
                audio_encoder_hidden_states=jnp.asarray(inputs_npz["atxt_b2"]),
                audio_encoder_attention_mask=jnp.asarray(inputs_npz["em_b2"]),
                fps=24.0,
                audio_num_frames=128,
                return_dict=False,
                perturbation_mask=None,
                use_cross_timestep=False,
            )
            tag = f"scan{int(scan_flag)}_{kern}"
            head_outputs[f"{tag}_vid"] = np.asarray(jax.device_get(s_out))
            head_outputs[f"{tag}_aud"] = np.asarray(jax.device_get(a_out))

            if scan_flag and kern == "flash":
              s_stg, a_stg = m(
                  hidden_states=jnp.asarray(inputs_npz["vid_b4"]),
                  encoder_hidden_states=jnp.asarray(inputs_npz["txt_b4"]),
                  timestep=jnp.asarray(inputs_npz["ts_b4"]),
                  sigma=jnp.asarray(inputs_npz["sig_b4"]),
                  audio_sigma=jnp.asarray(inputs_npz["sig_b4"]),
                  encoder_attention_mask=None,
                  modality_mask=jnp.asarray(inputs_npz["mod_b4"]),
                  num_frames=2,
                  height=8,
                  width=8,
                  audio_hidden_states=jnp.asarray(inputs_npz["aud_b4"]),
                  audio_encoder_hidden_states=jnp.asarray(inputs_npz["atxt_b4"]),
                  audio_encoder_attention_mask=None,
                  fps=24.0,
                  audio_num_frames=128,
                  return_dict=False,
                  perturbation_mask=jnp.asarray(inputs_npz["pert_b4"]),
                  use_cross_timestep=False,
              )
              head_outputs["stg_b4_vid"] = np.asarray(jax.device_get(s_stg))
              head_outputs["stg_b4_aud"] = np.asarray(jax.device_get(a_stg))

      assert weights_scan_true is not None and weights_scan_false is not None
      w_true_path = os.path.join(tmpdir, "weights_scan_true.npz")
      w_false_path = os.path.join(tmpdir, "weights_scan_false.npz")
      in_path = os.path.join(tmpdir, "inputs.npz")
      out_up_path = os.path.join(tmpdir, "upstream_outputs.npz")
      np.savez(w_true_path, **weights_scan_true)
      np.savez(w_false_path, **weights_scan_false)
      np.savez(in_path, **inputs_npz)

      runner_path = os.path.join(tmpdir, "run_upstream_a560b4d8.py")
      with open(runner_path, "w", encoding="utf-8") as fh:
        fh.write(
            textwrap.dedent(
                """
            import os, sys, types
            os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
            import jax
            import jax.numpy as jnp
            import numpy as np
            from flax import nnx
            import flax.linen as nn
            from flax.linen import partitioning as nn_partitioning
            from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
            from jax.sharding import Mesh

            try:
              import chex
            except ImportError:
              st = types.ModuleType("chex"); st.Array = jax.Array; st.PRNGKey = jax.Array; sys.modules["chex"] = st

            from maxdiffusion.models.ltx2.logical_sharding_ltx2 import get_sharding_specs
            from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel

            w_true = np.load(sys.argv[1])
            w_false = np.load(sys.argv[2])
            inp = np.load(sys.argv[3])
            out_file = sys.argv[4]

            mesh = Mesh(np.array(jax.devices()[:8]).reshape(1, 2, 4, 1), ("data", "fsdp", "context", "tensor"))
            rules = (
                ("activation_self_attn_heads", None),
                ("activation_self_attn_q_length", "context"),
                ("activation_self_attn_kv_length", None),
                ("activation_cross_attn_heads", None),
                ("activation_cross_attn_q_length", "context"),
                ("activation_cross_attn_kv_length", None),
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
                ("conv_batch", ("data", "fsdp", "context")),
                ("out_channels", "tensor"),
                ("conv_out", "context"),
            )
            bs = splash_attention_kernel.BlockSizes(
                block_q=128, block_kv=128, block_kv_compute=128,
                block_q_dkv=128, block_kv_dkv=128, block_kv_dkv_compute=128,
                block_q_dq=128, block_kv_dq=128, use_fused_bwd_kernel=False,
            )
            dit_specs = get_sharding_specs("default", "ltx2_dit")

            def encode_path(pt):
              return "__".join(f"{type(p).__name__}:{p}" for p in pt)

            def build_and_load(scan_flag, kern, npz_weights):
              with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
                m = LTX2VideoTransformer3DModel(
                    in_channels=16, out_channels=16, patch_size=1, patch_size_t=1,
                    num_attention_heads=4, attention_head_dim=32, cross_attention_dim=128,
                    audio_in_channels=8, audio_out_channels=8,
                    audio_num_attention_heads=4, audio_attention_head_dim=32, audio_cross_attention_dim=128,
                    num_layers=2, caption_channels=32,
                    cross_attn_mod=True, gated_attn=True, perturbed_attn=True,
                    spatio_temporal_guidance_blocks=(0,),
                    scan_layers=scan_flag,
                    attention_kernel=kern, a2v_attention_kernel=kern, v2a_attention_kernel=kern,
                    flash_block_sizes=bs, flash_min_seq_length=0,
                    dtype=jnp.float32, weights_dtype=jnp.float32,
                    mesh=mesh, sharding_specs=dit_specs,
                    rngs=nnx.Rngs(jax.random.key(0)),
                )
                gdef, p_st, r_st = nnx.split(m, nnx.Param, ...)
                flat = nnx.to_flat_state(p_st)
                up_keys = {encode_path(pt) for pt, _ in flat}
                assert up_keys == set(npz_weights.files), f"Param key mismatch: {up_keys ^ set(npz_weights.files)}"
                for pt, var in flat:
                  var.set_value(jnp.asarray(npz_weights[encode_path(pt)]))
                p_st = nnx.from_flat_state(flat)
                lspec = nnx.get_partition_spec(p_st)
                sh = nn.logical_to_mesh_sharding(lspec, mesh, rules)
                p_st = jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s), p_st, sh)
                return nnx.merge(gdef, p_st, r_st)

            results = {}
            for sflag in (True, False):
              w_npz = w_true if sflag else w_false
              for kern in ("dot_product", "flash"):
                m = build_and_load(sflag, kern, w_npz)
                with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
                  sv, sa = m(
                      hidden_states=jnp.asarray(inp["vid_b2"]),
                      encoder_hidden_states=jnp.asarray(inp["txt_b2"]),
                      timestep=jnp.asarray(inp["ts_b2"]),
                      sigma=jnp.asarray(inp["sig_b2"]),
                      audio_sigma=jnp.asarray(inp["sig_b2"]),
                      encoder_attention_mask=jnp.asarray(inp["em_b2"]),
                      modality_mask=None,
                      num_frames=2, height=8, width=8,
                      audio_hidden_states=jnp.asarray(inp["aud_b2"]),
                      audio_encoder_hidden_states=jnp.asarray(inp["atxt_b2"]),
                      audio_encoder_attention_mask=jnp.asarray(inp["em_b2"]),
                      fps=24.0, audio_num_frames=128,
                      return_dict=False, perturbation_mask=None, use_cross_timestep=False,
                  )
                  tag = f"scan{int(sflag)}_{kern}"
                  results[f"{tag}_vid"] = np.asarray(jax.device_get(sv))
                  results[f"{tag}_aud"] = np.asarray(jax.device_get(sa))
                  if sflag and kern == "flash":
                    sv4, sa4 = m(
                        hidden_states=jnp.asarray(inp["vid_b4"]),
                        encoder_hidden_states=jnp.asarray(inp["txt_b4"]),
                        timestep=jnp.asarray(inp["ts_b4"]),
                        sigma=jnp.asarray(inp["sig_b4"]),
                        audio_sigma=jnp.asarray(inp["sig_b4"]),
                        encoder_attention_mask=None,
                        modality_mask=jnp.asarray(inp["mod_b4"]),
                        num_frames=2, height=8, width=8,
                        audio_hidden_states=jnp.asarray(inp["aud_b4"]),
                        audio_encoder_hidden_states=jnp.asarray(inp["atxt_b4"]),
                        audio_encoder_attention_mask=None,
                        fps=24.0, audio_num_frames=128,
                        return_dict=False,
                        perturbation_mask=jnp.asarray(inp["pert_b4"]),
                        use_cross_timestep=False,
                    )
                    results["stg_b4_vid"] = np.asarray(jax.device_get(sv4))
                    results["stg_b4_aud"] = np.asarray(jax.device_get(sa4))

            np.savez(out_file, **results)
        """
            )
        )

      env = dict(os.environ)
      env["PYTHONPATH"] = os.path.join(up_dir, "src")
      env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
      proc = subprocess.run(
          [sys.executable, runner_path, w_true_path, w_false_path, in_path, out_up_path],
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=f"Upstream baseline subprocess failed:\n{proc.stderr}")

      upstream_outputs = np.load(out_up_path)
      parity_summary = {}
      for key in sorted(head_outputs.keys()):
        h_arr = head_outputs[key]
        u_arr = upstream_outputs[key]
        max_delta = float(np.max(np.abs(h_arr - u_arr)))
        parity_summary[key] = max_delta
        if max_delta == 0.0:
          np.testing.assert_array_equal(h_arr, u_arr)
        else:
          np.testing.assert_allclose(h_arr, u_arr, rtol=1e-6, atol=1e-6)
      print(f"[Gate 1a Upstream {UPSTREAM_BASELINE_SHA[:8]} Parity] max|Δ|={parity_summary}")

      # --- Sub-check (ii): HEAD masks=None vs all-True + 3-stream right-padding invariance ---
      trainer = LTX2Trainer(cfg)
      pipeline, _ = trainer.load_or_init_pipeline()
      with jax.set_mesh(pipeline.mesh), nn_partitioning.axis_rules(rules):
        _randomize_zero_params(pipeline.transformer, jax.random.key(515))
      optimizer, _ = trainer.checkpointer._create_optimizer(cfg, float(cfg.learning_rate))
      state, _, _ = trainer.init_train_state(pipeline, optimizer)
      mesh = pipeline.mesh

      s0 = LTX2Sample(
          video_latents=rng_np.standard_normal((16, 2, 8, 8)).astype(np.float32),
          audio_latents=rng_np.standard_normal((48, 8)).astype(np.float32),
          encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          audio_encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          fps=24.0,
          has_audio=True,
      )
      s1 = LTX2Sample(
          video_latents=rng_np.standard_normal((16, 2, 8, 8)).astype(np.float32),
          audio_latents=rng_np.standard_normal((48, 8)).astype(np.float32),
          encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          audio_encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          fps=24.0,
          has_audio=True,
      )

      b_unpad = LTX2BucketSpec(128, 48, 64)
      b_pad = LTX2BucketSpec(256, 96, 128)
      unpad_np = collate_ltx2_bucket_batch(
          [s0, s1],
          b_unpad,
          audio_in_channels=8,
          text_padding_mode="right_pad_masked",
          connector_num_learnable_registers=0,
      )
      pad_np = collate_ltx2_bucket_batch(
          [s0, s1],
          b_pad,
          audio_in_channels=8,
          text_padding_mode="right_pad_masked",
          connector_num_learnable_registers=0,
      )
      unpad_sh = {k: v for k, v in shard_ltx2_batch_for_mesh(unpad_np, mesh).items() if k != "bucket_key"}
      pad_sh = {k: v for k, v in shard_ltx2_batch_for_mesh(pad_np, mesh).items() if k != "bucket_key"}

      sigmas = jnp.array([0.28, 0.73], dtype=jnp.float32)
      eps_vid_unpad = rng_np.standard_normal((2, 128, 16)).astype(np.float32)
      eps_aud_unpad = rng_np.standard_normal((2, 48, 8)).astype(np.float32)

      # Inject NaN and large garbage into right-padded noise slots (128:256, 48:96).
      eps_vid_pad = np.full((2, 256, 16), np.nan, dtype=np.float32)
      eps_vid_pad[:, :128, :] = eps_vid_unpad
      eps_vid_pad[1, 128:192, :] = 99.0
      eps_aud_pad = np.full((2, 96, 8), np.nan, dtype=np.float32)
      eps_aud_pad[:, :48, :] = eps_aud_unpad
      eps_aud_pad[0, 48:72, :] = -77.0

      sh_unpad = get_ltx2_batch_shardings(mesh, b_unpad)
      sh_pad = get_ltx2_batch_shardings(mesh, b_pad)
      sh_b = NamedSharding(mesh, P(("data", "fsdp")))
      sig_sh = jax.device_put(sigmas, sh_b)
      ev_unpad_sh = jax.device_put(eps_vid_unpad, sh_unpad["hidden_states"])
      ea_unpad_sh = jax.device_put(eps_aud_unpad, sh_unpad["audio_hidden_states"])
      ev_pad_sh = jax.device_put(eps_vid_pad, sh_pad["hidden_states"])
      ea_pad_sh = jax.device_put(eps_aud_pad, sh_pad["audio_hidden_states"])

      def _eval_loss_and_grads(p_in, r_in, batch_dict, ev, ea):
        def _loss_fn(p_train, p_rest):
          e_vid, e_aud, aux = compute_ltx2_microbatch_errors(
              state.graphdef,
              p_train,
              p_rest,
              batch_dict,
              jax.random.key(0),
              scheduler=pipeline.scheduler,
              sigmas_override=sig_sh,
              video_noise_override=ev,
              audio_noise_override=ea,
          )
          loss = e_vid / aux["valid_vid_elems"] + e_aud / aux["valid_aud_elems"]
          return loss, aux

        return nnx.value_and_grad(_loss_fn, argnums=0, has_aux=True)(p_in, r_in)

      jit_eval = jax.jit(_eval_loss_and_grads)
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
        # Masks=None vs explicit all-True masks on HEAD.
        unpad_none_masks = dict(unpad_sh)
        unpad_none_masks["video_attention_mask"] = None
        unpad_none_masks["audio_attention_mask"] = None
        unpad_none_masks["encoder_attention_mask"] = None
        unpad_none_masks["audio_encoder_attention_mask"] = None

        (loss_none, _), grads_none = jit_eval(state.params, state.rest_of_state, unpad_none_masks, ev_unpad_sh, ea_unpad_sh)
        (loss_unpad, aux_unpad), grads_unpad = jit_eval(
            state.params, state.rest_of_state, unpad_sh, ev_unpad_sh, ea_unpad_sh
        )
        (loss_pad, aux_pad), grads_pad = jit_eval(state.params, state.rest_of_state, pad_sh, ev_pad_sh, ea_pad_sh)

      none_vs_true_grad_diff, n_leaves = _max_tree_abs_diff(grads_none, grads_unpad)
      np.testing.assert_allclose(float(loss_none), float(loss_unpad), rtol=1e-6, atol=1e-6)
      self.assertLess(none_vs_true_grad_diff, 1e-5)

      sample_unpad = np.asarray(jax.device_get(aux_unpad["sample"]))
      sample_pad = np.asarray(jax.device_get(aux_pad["sample"]))
      aud_unpad = np.asarray(jax.device_get(aux_unpad["audio_sample"]))
      aud_pad = np.asarray(jax.device_get(aux_pad["audio_sample"]))

      np.testing.assert_allclose(sample_pad[:, :128], sample_unpad, atol=1e-4)
      np.testing.assert_allclose(aud_pad[:, :48], aud_unpad, atol=1e-4)
      self.assertEqual(float(np.max(np.abs(sample_pad[:, 128:]))), 0.0)
      self.assertEqual(float(np.max(np.abs(aud_pad[:, 48:]))), 0.0)

      np.testing.assert_allclose(float(loss_pad), float(loss_unpad), rtol=1e-5, atol=1e-5)
      pad_grad_diff, _ = _max_tree_abs_diff(grads_pad, grads_unpad)
      self.assertLess(pad_grad_diff, 1e-4)
      print(
          f"[Gate 1a 3-Stream Right-Pad Invariance] leaves={n_leaves} "
          f"loss_unpad={float(loss_unpad):.6f} loss_pad={float(loss_pad):.6f} "
          f"max|Δgrad|={pad_grad_diff:.3e} masks_none_vs_true_max|Δgrad|={none_vs_true_grad_diff:.3e}"
      )

  # ============================================================================
  # Gate 1b: Heterogeneous 3-Modality Batch Invariance Across dot_product,
  #          flash (Splash), and ulysses (All-to-All + Splash)
  # ============================================================================
  def test_gate1b_heterogeneous_batch_invariance_across_dot_flash_and_ulysses(self):
    """Proves heterogeneous B=2 fwd+grad invariance across dot_product, flash, and ulysses on all 6 attn sites.

    Note: Also exercises Splash's internal `_pad_data_for_flash` path on unpadded B=1
    `S_vid=384` across `cp=4` (`384 // 4 = 96 < 128` block size) with `R=1`, verifying
    `_validate_ring_sequence_divisibility` is not triggered.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
      rules_by_kernel = {
          kern: _init_bucketing_test_config(tmpdir, extra_overrides=[f"attention={kern}"]).logical_axis_rules
          for kern in ("dot_product", "flash", "ulysses")
      }
      cfg = _init_bucketing_test_config(tmpdir)
      ulysses_rule_tuples = [tuple(r) for r in rules_by_kernel["ulysses"]]
      self.assertIn(("activation_self_attn_kv_length", "context"), ulysses_rule_tuples)
      self.assertIn(("activation_cross_attn_kv_length", "context"), ulysses_rule_tuples)
      dit_specs = get_sharding_specs("default", "ltx2_dit")

      def _make_model(mesh: Mesh, kern: str, axis_rules, raw_weights_dict=None) -> LTX2VideoTransformer3DModel:
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(axis_rules):
          m = LTX2VideoTransformer3DModel(
              in_channels=16,
              out_channels=16,
              patch_size=1,
              patch_size_t=1,
              num_attention_heads=4,
              attention_head_dim=32,
              cross_attention_dim=128,
              audio_in_channels=8,
              audio_out_channels=8,
              audio_num_attention_heads=4,
              audio_attention_head_dim=32,
              audio_cross_attention_dim=128,
              num_layers=2,
              caption_channels=32,
              cross_attn_mod=True,
              gated_attn=True,
              scan_layers=True,
              attention_kernel=kern,
              a2v_attention_kernel=kern,
              v2a_attention_kernel=kern,
              flash_block_sizes=FLASH_BS_128,
              flash_min_seq_length=0,
              dtype=jnp.float32,
              weights_dtype=jnp.float32,
              mesh=mesh,
              sharding_specs=dit_specs,
              rngs=nnx.Rngs(jax.random.key(777)),
          )
          if raw_weights_dict is None:
            _randomize_zero_params(m, jax.random.key(888))
          gdef, p_st, r_st = nnx.split(m, nnx.Param, ...)
          if raw_weights_dict is not None:
            flat = nnx.to_flat_state(p_st)
            for pt, var in flat:
              var.set_value(jnp.asarray(raw_weights_dict[_encode_flat_path(pt)]))
            p_st = nnx.from_flat_state(flat)
          p_st = _place_state_on_mesh(p_st, mesh, axis_rules)
          return nnx.merge(gdef, p_st, r_st)

      dot_rules = rules_by_kernel["dot_product"]
      base_m = _make_model(self.mesh_8, "dot_product", dot_rules)
      _, base_p_st, _ = nnx.split(base_m, nnx.Param, ...)
      shared_weights = _dump_nnx_params_to_dict(base_p_st)

      trainer = LTX2Trainer(cfg)
      scheduler, _ = trainer.create_scheduler()

      rng_np = np.random.default_rng(23)
      # Sample 0: (F=2, H=16, W=12 -> S_vid=384, S_aud=96, L_txt=96, fps=24.0)
      # Sample 1: (F=3, H=12, W=8  -> S_vid=288, S_aud=60, L_txt=64, fps=16.0)
      vid_0 = rng_np.standard_normal((16, 2, 16, 12)).astype(np.float32)
      aud_0 = rng_np.standard_normal((96, 8)).astype(np.float32)
      txt_0 = rng_np.standard_normal((96, 32)).astype(np.float32)
      atxt_0 = rng_np.standard_normal((96, 32)).astype(np.float32)
      ev_0 = rng_np.standard_normal((1, 384, 16)).astype(np.float32)
      ea_0 = rng_np.standard_normal((1, 96, 8)).astype(np.float32)

      vid_1 = rng_np.standard_normal((16, 3, 12, 8)).astype(np.float32)
      aud_1 = rng_np.standard_normal((60, 8)).astype(np.float32)
      txt_1 = rng_np.standard_normal((64, 32)).astype(np.float32)
      atxt_1 = rng_np.standard_normal((64, 32)).astype(np.float32)
      ev_1 = rng_np.standard_normal((1, 288, 16)).astype(np.float32)
      ea_1 = rng_np.standard_normal((1, 60, 8)).astype(np.float32)

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
          audio_latents=aud_1,
          encoder_hidden_states=txt_1,
          audio_encoder_hidden_states=atxt_1,
          fps=16.0,
          has_audio=True,
      )

      bucket = LTX2BucketSpec(512, 128, 128)
      batch_np = collate_ltx2_bucket_batch(
          [s0, s1],
          bucket,
          audio_in_channels=8,
          text_padding_mode="right_pad_masked",
          connector_num_learnable_registers=0,
      )
      batch_sh = {k: v for k, v in shard_ltx2_batch_for_mesh(batch_np, self.mesh_8).items() if k != "bucket_key"}

      ev_pad = np.full((2, 512, 16), np.nan, dtype=np.float32)
      ev_pad[0, :384] = ev_0[0]
      ev_pad[1, :288] = ev_1[0]
      ea_pad = np.full((2, 128, 8), np.nan, dtype=np.float32)
      ea_pad[0, :96] = ea_0[0]
      ea_pad[1, :60] = ea_1[0]

      sh_b8 = get_ltx2_batch_shardings(self.mesh_8, bucket)
      ev_pad_sh = jax.device_put(ev_pad, sh_b8["hidden_states"])
      ea_pad_sh = jax.device_put(ea_pad, sh_b8["audio_hidden_states"])
      sig_b2_sh = jax.device_put(jnp.array([0.31, 0.68], dtype=jnp.float32), NamedSharding(self.mesh_8, P(("data", "fsdp"))))

      total_vid_elems = float((384 + 288) * 16)
      total_aud_elems = float((96 + 60) * 8)

      # Unpadded B=1 exact dot_product single-VJP references on submesh_fsdp1 (1, 1, 4, 1).
      ref_m_sub = _make_model(self.submesh_fsdp1, "dot_product", dot_rules, raw_weights_dict=shared_weights)
      ref_gdef, ref_p_sub, ref_r_sub = nnx.split(ref_m_sub, nnx.Param, ...)

      def _run_unpadded_b1(v_raw, a_raw, t_raw, at_raw, f_lat, h_lat, w_lat, fps, ev_s, ea_s, sig_val):
        b_1 = {
            "hidden_states": jnp.asarray(patchify_video_latents(v_raw)[None, ...]),
            "audio_hidden_states": jnp.asarray(a_raw[None, ...]),
            "encoder_hidden_states": jnp.asarray(t_raw[None, ...]),
            "audio_encoder_hidden_states": jnp.asarray(at_raw[None, ...]),
            "video_coords": jnp.asarray(compute_video_coords_1sample(f_lat, h_lat, w_lat, fps=fps)[None, ...]),
            "audio_coords": jnp.asarray(compute_audio_coords_1sample(a_raw.shape[0])[None, ...]),
            "modality_mask": jnp.ones((1, 1, 1), dtype=jnp.float32),
        }

        def _single_loss(p_t, p_r):
          e_v, e_a, aux = compute_ltx2_microbatch_errors(
              ref_gdef,
              p_t,
              p_r,
              b_1,
              jax.random.key(0),
              scheduler=scheduler,
              sigmas_override=jnp.array([sig_val], dtype=jnp.float32),
              video_noise_override=jnp.asarray(ev_s),
              audio_noise_override=jnp.asarray(ea_s),
          )
          return e_v / total_vid_elems + e_a / total_aud_elems, aux

        with jax.set_mesh(self.submesh_fsdp1), nn_partitioning.axis_rules(dot_rules):
          return nnx.value_and_grad(_single_loss, argnums=0, has_aux=True)(ref_p_sub, ref_r_sub)

      (l_ref_0, aux_ref_0), g_ref_0 = _run_unpadded_b1(vid_0, aud_0, txt_0, atxt_0, 2, 16, 12, 24.0, ev_0, ea_0, 0.31)
      (l_ref_1, aux_ref_1), g_ref_1 = _run_unpadded_b1(vid_1, aud_1, txt_1, atxt_1, 3, 12, 8, 16.0, ev_1, ea_1, 0.68)
      loss_ref = float(l_ref_0) + float(l_ref_1)
      grads_ref = jax.tree.map(lambda a, b: jax.device_get(a) + jax.device_get(b), g_ref_0, g_ref_1)
      s0_ref = np.asarray(jax.device_get(aux_ref_0["sample"]))[0]
      s1_ref = np.asarray(jax.device_get(aux_ref_1["sample"]))[0]
      a0_ref = np.asarray(jax.device_get(aux_ref_0["audio_sample"]))[0]
      a1_ref = np.asarray(jax.device_get(aux_ref_1["audio_sample"]))[0]

      kernel_grad_diffs = {}
      ulysses_a2a_count = 0
      ulysses_a2a_seq_shards = set()
      for kern in ("dot_product", "flash", "ulysses"):
        with self.subTest(attention_kernel=kern):
          rules_k = rules_by_kernel[kern]
          m_k = _make_model(self.mesh_8, kern, rules_k, raw_weights_dict=shared_weights)
          k_gdef, k_params, k_rest = nnx.split(m_k, nnx.Param, ...)

          def _bucket_loss(p_t, p_r):
            e_v, e_a, aux = compute_ltx2_microbatch_errors(
                k_gdef,
                p_t,
                p_r,
                batch_sh,
                jax.random.key(0),
                scheduler=scheduler,
                sigmas_override=sig_b2_sh,
                video_noise_override=ev_pad_sh,
                audio_noise_override=ea_pad_sh,
            )
            return e_v / aux["valid_vid_elems"] + e_a / aux["valid_aud_elems"], aux

          jit_fn = jax.jit(lambda p, r: nnx.value_and_grad(_bucket_loss, argnums=0, has_aux=True)(p, r))
          with jax.set_mesh(self.mesh_8), nn_partitioning.axis_rules(rules_k):
            if kern == "ulysses":
              hlo_u = jit_fn.lower(k_params, k_rest).compile().as_text()
              a2a_lines = [ln for ln in hlo_u.splitlines() if "all-to-all(" in ln and "kernel_ulysses" in ln]
              ulysses_a2a_count = len(a2a_lines)
              self.assertGreater(ulysses_a2a_count, 0, "Expected kernel_ulysses all-to-all collectives in compiled HLO")
              # Under ULYSSES_ATTENTION_AXIS_RULES on cp=4, per-shard Q/K/V sequence dims in kernel_ulysses
              # all-to-all must be S / 4 = {512 // 4 == 128, 128 // 4 == 32}, never replicated {512}.
              for ln in a2a_lines:
                for dims_str in re.findall(r"f32\[([0-9,]+)\]", ln):
                  dims = [int(x) for x in dims_str.split(",")]
                  if len(dims) == 4:
                    ulysses_a2a_seq_shards.add(dims[2])
              self.assertEqual(
                  ulysses_a2a_seq_shards,
                  {bucket.b_vid // 4, bucket.b_aud // 4},
                  f"Expected per-shard S/4={{128, 32}} in Ulysses all-to-all HLO, got {ulysses_a2a_seq_shards}",
              )
            (loss_k, aux_k), grads_k = jit_fn(k_params, k_rest)

          sk = np.asarray(jax.device_get(aux_k["sample"]))
          ak = np.asarray(jax.device_get(aux_k["audio_sample"]))
          np.testing.assert_allclose(sk[0, :384], s0_ref, atol=2e-4)
          np.testing.assert_allclose(sk[1, :288], s1_ref, atol=2e-4)
          np.testing.assert_allclose(ak[0, :96], a0_ref, atol=2e-4)
          np.testing.assert_allclose(ak[1, :60], a1_ref, atol=2e-4)
          self.assertEqual(float(np.max(np.abs(sk[0, 384:]))), 0.0)
          self.assertEqual(float(np.max(np.abs(sk[1, 288:]))), 0.0)
          self.assertEqual(float(np.max(np.abs(ak[0, 96:]))), 0.0)
          self.assertEqual(float(np.max(np.abs(ak[1, 60:]))), 0.0)

          np.testing.assert_allclose(float(loss_k), loss_ref, rtol=2e-4, atol=2e-4)
          gdiff, n_leaves = _max_tree_abs_diff(grads_k, grads_ref)
          kernel_grad_diffs[kern] = gdiff
          self.assertLess(gdiff, 5e-4, f"Kernel {kern} heterogeneous grad diff {gdiff} >= 5e-4")

      # Sub-check: Unpadded B=1 at S_vid=384 on cp=4 (384 // 4 = 96 < 128 block size)
      # exercises Splash's internal `_pad_data_for_flash` path (R=1, no ring divisibility error).
      sub_b1 = LTX2BucketSpec(384, 128, 128)
      s_padflash = LTX2Sample(
          video_latents=vid_0,
          audio_latents=rng_np.standard_normal((128, 8)).astype(np.float32),
          encoder_hidden_states=rng_np.standard_normal((128, 32)).astype(np.float32),
          audio_encoder_hidden_states=rng_np.standard_normal((128, 32)).astype(np.float32),
          fps=24.0,
          has_audio=True,
      )
      b1_np = collate_ltx2_bucket_batch([s_padflash], sub_b1, audio_in_channels=8, text_padding_mode="post_connector")
      b1_sh = {k: v for k, v in shard_ltx2_batch_for_mesh(b1_np, self.submesh_fsdp1).items() if k != "bucket_key"}
      ev_b1 = jax.device_put(ev_0, get_ltx2_batch_shardings(self.submesh_fsdp1, sub_b1)["hidden_states"])
      ea_b1 = jax.device_put(
          rng_np.standard_normal((1, 128, 8)).astype(np.float32),
          get_ltx2_batch_shardings(self.submesh_fsdp1, sub_b1)["audio_hidden_states"],
      )
      sig_b1 = jax.device_put(jnp.array([0.45], dtype=jnp.float32), NamedSharding(self.submesh_fsdp1, P(("data", "fsdp"))))

      b1_results = {}
      for kern in ("dot_product", "flash", "ulysses"):
        rules_k = rules_by_kernel[kern]
        m_sub_k = _make_model(self.submesh_fsdp1, kern, rules_k, raw_weights_dict=shared_weights)
        sgdef, sp_st, sr_st = nnx.split(m_sub_k, nnx.Param, ...)

        def _b1_loss(p_t, p_r):
          ev_err, ea_err, aux = compute_ltx2_microbatch_errors(
              sgdef,
              p_t,
              p_r,
              b1_sh,
              jax.random.key(0),
              scheduler=scheduler,
              sigmas_override=sig_b1,
              video_noise_override=ev_b1,
              audio_noise_override=ea_b1,
          )
          return ev_err / aux["valid_vid_elems"] + ea_err / aux["valid_aud_elems"]

        with jax.set_mesh(self.submesh_fsdp1), nn_partitioning.axis_rules(rules_k):
          b1_results[kern] = jax.jit(lambda p, r: nnx.value_and_grad(_b1_loss, argnums=0)(p, r))(sp_st, sr_st)

      for kern in ("flash", "ulysses"):
        np.testing.assert_allclose(float(b1_results[kern][0]), float(b1_results["dot_product"][0]), atol=2e-4)
        pf_diff, _ = _max_tree_abs_diff(b1_results[kern][1], b1_results["dot_product"][1])
        self.assertLess(pf_diff, 5e-4)

      print(
          f"[Gate 1b Heterogeneous 3-Stream All-6-Attn Kernels] leaves={n_leaves} "
          f"ulysses_all_to_all={ulysses_a2a_count} ulysses_a2a_seq_shards={sorted(ulysses_a2a_seq_shards)} "
          f"max|Δgrad|={kernel_grad_diffs}"
      )

  # ============================================================================
  # Gate 1c: Video-Only Batch Complete Audio & Cross-Modal Zero-Grad Audit
  # ============================================================================
  def test_gate1c_video_only_batch_complete_audio_zero_grad_audit(self):
    """Proves exact 0.0 gradients on all audio & cross-modal params (norm_elementwise_affine False & True)."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = _init_bucketing_test_config(tmpdir, extra_overrides=["transformer_cross_attn_mod=True"])
      trainer = LTX2Trainer(cfg)
      pipeline, _ = trainer.load_or_init_pipeline()
      with jax.set_mesh(pipeline.mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        _randomize_zero_params(pipeline.transformer, jax.random.key(612))
      optimizer, _ = trainer.checkpointer._create_optimizer(cfg, float(cfg.learning_rate))
      state, _, _ = trainer.init_train_state(pipeline, optimizer)
      mesh = pipeline.mesh

      rng_np = np.random.default_rng(33)
      bucket = LTX2BucketSpec(128, 64, 64)
      s0 = LTX2Sample(
          video_latents=rng_np.standard_normal((16, 2, 8, 8)).astype(np.float32),
          audio_latents=None,
          encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          has_audio=False,
      )
      s1 = LTX2Sample(
          video_latents=rng_np.standard_normal((16, 2, 8, 8)).astype(np.float32),
          audio_latents=None,
          encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          has_audio=False,
      )
      batch_np = collate_ltx2_bucket_batch([s0, s1], bucket, audio_in_channels=8)
      self.assertTrue(np.all(batch_np["modality_mask"] == 0.0))
      self.assertFalse(np.any(batch_np["audio_attention_mask"]))

      batch_sh = shard_ltx2_batch_for_mesh(batch_np, mesh)
      inv_vid, inv_aud = trainer._jit_window_denoms(
          (batch_sh["video_attention_mask"],),
          (batch_sh["audio_attention_mask"],),
          16,
          8,
          1.0,
      )
      self.assertGreater(float(inv_vid), 0.0)
      self.assertEqual(float(inv_aud), 0.0)

      # --- Check A: Default norm_elementwise_affine=False via LTX2Trainer ---
      paramless_norm_names = (
          "audio_norm_out",
          "audio_norm1",
          "audio_norm2",
          "audio_to_video_norm",
          "video_to_audio_norm",
          "audio_norm3",
      )
      for st_name, st_tree in (("params", state.params), ("rest_of_state", state.rest_of_state)):
        for path_tuple, var in nnx.to_flat_state(st_tree):
          str_parts = {str(p) for p in path_tuple}
          for norm_name in paramless_norm_names:
            if norm_name in str_parts:
              self.assertNotIsInstance(
                  var, nnx.Param, f"Expected {norm_name} to have 0 nnx.Param leaves in {st_name}, found {path_tuple}"
              )

      scalar_sh = NamedSharding(mesh, P())
      step_rng = jax.device_put(jax.random.key(99), scalar_sh)
      tensor_batch = {k: v for k, v in batch_sh.items() if k != "bucket_key"}
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        grad_fn = trainer.get_or_compile_bucket_grad_step(bucket, mesh, pipeline.scheduler, state.graphdef)
        (m_loss, m_vid_loss, m_aud_loss), grads = grad_fn(
            state.params, state.rest_of_state, tensor_batch, step_rng, inv_vid, inv_aud
        )

      self.assertTrue(np.isfinite(float(m_loss)))
      self.assertGreater(float(m_vid_loss), 0.0)
      self.assertEqual(float(m_aud_loss), 0.0)

      flat_grads = [
          (tuple(str(p) for p in pt), np.asarray(jax.device_get(var.get_value()))) for pt, var in nnx.to_flat_state(grads)
      ]

      # Verify .norm_q and .norm_k inside all 4 audio / cross-modal attention modules are non-empty and exact 0.0.
      for attn_mod in ("audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"):
        for qk_norm in ("norm_q", "norm_k"):
          matched = [arr for pt, arr in flat_grads if attn_mod in pt and qk_norm in pt]
          self.assertGreater(len(matched), 0, f"Missing {attn_mod}.{qk_norm} in grads")
          for arr in matched:
            self.assertEqual(float(np.max(np.abs(arr))), 0.0, f"Non-zero grad on {attn_mod}.{qk_norm}")

      # Verify every top-level & block-level audio / cross-modal Param group is non-empty and exact 0.0.
      zero_grad_groups = (
          ("audio_proj_in",),
          ("audio_proj_out",),
          ("audio_caption_projection",),
          ("audio_prompt_adaln",),
          ("audio_time_embed",),
          ("av_cross_attn_video_scale_shift",),
          ("av_cross_attn_audio_scale_shift",),
          ("av_cross_attn_video_a2v_gate",),
          ("av_cross_attn_audio_v2a_gate",),
          ("audio_scale_shift_table",),
          ("transformer_blocks", "audio_attn1"),
          ("transformer_blocks", "audio_attn2"),
          ("transformer_blocks", "audio_ff"),
          ("transformer_blocks", "audio_scale_shift_table"),
          ("transformer_blocks", "video_a2v_cross_attn_scale_shift_table"),
          ("transformer_blocks", "audio_a2v_cross_attn_scale_shift_table"),
          ("transformer_blocks", "audio_prompt_scale_shift_table"),
          ("transformer_blocks", "audio_to_video_attn"),
          ("transformer_blocks", "video_to_audio_attn"),
      )
      for req_tokens in zero_grad_groups:
        if req_tokens == ("audio_scale_shift_table",):
          matched = [arr for pt, arr in flat_grads if pt[0] == "audio_scale_shift_table"]
        else:
          matched = [arr for pt, arr in flat_grads if all(tok in pt for tok in req_tokens)]
        self.assertGreater(len(matched), 0, f"Expected non-empty Param group for {req_tokens}")
        for arr in matched:
          self.assertEqual(float(np.max(np.abs(arr))), 0.0, f"Non-zero grad in video-only group {req_tokens}")

      # Pure video stream parameters must receive strictly positive finite gradients.
      for vid_tok in ("proj_in", "proj_out", "attn1", "attn2", "ff"):
        matched_v = [
            arr for pt, arr in flat_grads if vid_tok in pt and not any("audio" in p or "a2v" in p or "v2a" in p for p in pt)
        ]
        self.assertGreater(len(matched_v), 0, f"Missing video group {vid_tok}")
        max_v = max(float(np.max(np.abs(arr))) for arr in matched_v)
        self.assertTrue(np.isfinite(max_v) and max_v > 0.0, f"Expected positive finite video grad for {vid_tok}")

      # --- Check B: norm_elementwise_affine=True + cross_attn_mod=True direct check ---
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        m_aff = LTX2VideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=4,
            attention_head_dim=32,
            cross_attention_dim=128,
            audio_in_channels=8,
            audio_out_channels=8,
            audio_num_attention_heads=4,
            audio_attention_head_dim=32,
            audio_cross_attention_dim=128,
            num_layers=2,
            caption_channels=32,
            norm_elementwise_affine=True,
            cross_attn_mod=True,
            gated_attn=True,
            attention_kernel="dot_product",
            a2v_attention_kernel="dot_product",
            v2a_attention_kernel="dot_product",
            mesh=mesh,
            sharding_specs=get_sharding_specs("default", "ltx2_dit"),
            rngs=nnx.Rngs(jax.random.key(613)),
        )
        _randomize_zero_params(m_aff, jax.random.key(614))
        aff_gdef, aff_params, aff_rest = nnx.split(m_aff, nnx.Param, ...)
        aff_params = _place_state_on_mesh(aff_params, mesh, cfg.logical_axis_rules)

        def _aff_loss(p_t, p_r):
          ev_err, ea_err, _ = compute_ltx2_microbatch_errors(
              aff_gdef, p_t, p_r, tensor_batch, step_rng, scheduler=pipeline.scheduler
          )
          return ev_err * inv_vid + ea_err * jnp.float32(0.0)

        _, aff_grads = jax.jit(lambda p, r: nnx.value_and_grad(_aff_loss, argnums=0)(p, r))(aff_params, aff_rest)

      flat_aff = [
          (tuple(str(p) for p in pt), np.asarray(jax.device_get(var.get_value())))
          for pt, var in nnx.to_flat_state(aff_grads)
      ]
      for norm_name in ("audio_norm1", "audio_norm2", "audio_norm3", "audio_to_video_norm", "video_to_audio_norm"):
        matched_n = [arr for pt, arr in flat_aff if norm_name in pt]
        self.assertGreater(len(matched_n), 0, f"Expected {norm_name}.scale Param when norm_elementwise_affine=True")
        for arr in matched_n:
          self.assertEqual(float(np.max(np.abs(arr))), 0.0, f"Non-zero grad on {norm_name}.scale")

      print("[Gate 1c Video-Only Audio Zero-Grad Audit] Check A (affine=False) & Check B (affine=True) verified exact 0.0")

  # ============================================================================
  # Gate 2: 8-Device Compiled HLO Sharding, Collectives & Non-Colliding
  #         Sequence-Dim Involuntary-Remat Audit
  # ============================================================================
  def test_gate2_8device_compiled_hlo_sharding_collectives_and_remat_audit(self):
    """Verifies 8-device FSDP2+CP sharding, HLO collectives, and 0 sequence-dim involuntary remats."""
    # Non-colliding sequence buckets:
    # Global S = {640, 84, 76, 896, 108, 92}, per-cp=4 shard S/4 = {160, 21, 19, 224, 27, 23}.
    seq_audit_dims = {640, 160, 84, 21, 76, 19, 896, 224, 108, 27, 92, 23}
    model_dims = {1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768}
    self.assertEqual(seq_audit_dims & model_dims, set())

    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = _init_bucketing_test_config(
          tmpdir,
          extra_overrides=[
              "attention=flash",
              "a2v_attention_kernel=dot_product",
              "v2a_attention_kernel=dot_product",
              "flash_min_seq_length=128",
              f"flash_block_sizes={FLASH_BS_128_JSON}",
              "remat_policy=FULL",
              "scan_layers=True",
              "transformer_caption_channels=64",
              "transformer_cross_attn_mod=False",
              "transformer_gated_attn=False",
              "ltx2_buckets=[[640,84,76],[896,108,92]]",
              "text_padding_mode=right_pad_masked",
              "connector_num_learnable_registers=0",
              "gradient_accumulation_steps=2",
          ],
      )
      trainer = LTX2Trainer(cfg)
      pipeline, restore_args = trainer.load_or_init_pipeline()
      mesh = pipeline.mesh
      optimizer, _ = trainer.checkpointer._create_optimizer(cfg, float(cfg.learning_rate))
      state, state_shardings, _ = trainer.init_train_state(pipeline, optimizer, restore_args)

      def _extract_shardings(st):
        return jax.tree.map(lambda a: a.sharding, st)

      init_shardings = _extract_shardings(state)
      self.assertEqual(init_shardings, state_shardings)

      # Check 2: Spec membership ('fsdp', 'context') + 1/8 local shard size on 'embed'-tagged 2D/3D leaves.
      embed_checked = 0
      for pt, arr_var in nnx.to_flat_state(state.params):
        arr = arr_var.get_value()
        logical_axes = tuple(getattr(arr_var, "out_sharding", None) or ())
        if arr.ndim >= 2 and "embed" in logical_axes:
          embed_checked += 1
          phys_spec = tuple(arr.sharding.spec)
          self.assertIn(
              ("fsdp", "context"),
              phys_spec,
              f"Expected ('fsdp', 'context') in sharding spec of {pt}, got {phys_spec}",
          )
          local_elems = arr.addressable_shards[0].data.size
          self.assertEqual(
              local_elems,
              arr.size // 8,
              f"Expected 1/8 shard size ({arr.size // 8}) on {pt}, got {local_elems}",
          )
      self.assertGreater(embed_checked, 20)

      train_iter = trainer.load_dataset(mesh, pipeline.transformer)
      batches_by_key = {}
      while len(batches_by_key) < 2:
        b = next(train_iter)
        if b["bucket_key"] not in batches_by_key:
          batches_by_key[b["bucket_key"]] = b

      b1_spec = LTX2BucketSpec(640, 84, 76)
      b2_spec = LTX2BucketSpec(896, 108, 92)
      b1_batch = batches_by_key[(640, 84, 76)]
      b2_batch = batches_by_key[(896, 108, 92)]

      scalar_sh = NamedSharding(mesh, P())
      rng_0 = jax.device_put(jax.random.key(0), scalar_sh)
      one_sh = jax.device_put(jnp.float32(1.0), scalar_sh)

      hlo_texts = {}
      entry_param_prefixes = {}
      root_grad_shapes = {}
      n_trainable_leaves = len(jax.tree_util.tree_leaves(state.params))

      # Capture C++ XLA SPMD stderr (fd 2) across compilation AND execution of both buckets.
      with tempfile.TemporaryFile(mode="w+") as stderr_file:
        saved_fd = os.dup(2)
        os.dup2(stderr_file.fileno(), 2)
        try:
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
            for bspec, bdict in ((b1_spec, b1_batch), (b2_spec, b2_batch)):
              tb = {k: v for k, v in bdict.items() if k != "bucket_key"}
              fn = trainer.get_or_compile_bucket_grad_step(bspec, mesh, pipeline.scheduler, state.graphdef)
              comp = fn.lower(state.params, state.rest_of_state, tb, rng_0, one_sh, one_sh).compile()
              hlo = comp.as_text()
              hlo_texts[bspec.as_tuple()] = hlo

              entry_line = next(ln for ln in hlo.splitlines() if ln.strip().startswith("ENTRY "))
              entry_lhs, entry_rhs = entry_line.split("->", 1)
              # Compare the first N parameter entries corresponding to state.params + state.rest_of_state,
              # and verify ENTRY return ROOT gradient shard shapes match ENTRY trainable param shard shapes.
              n_state_leaves = len(jax.tree_util.tree_leaves((state.params, state.rest_of_state)))
              param_tokens = re.findall(r"(?:f32|bf16|s32|pred)\[[0-9,]*\]", entry_lhs)[:n_state_leaves]
              ret_tokens = re.findall(r"(?:f32|bf16|s32|pred)\[[0-9,]*\]", entry_rhs)
              entry_param_prefixes[bspec.as_tuple()] = param_tokens
              root_grad_shapes[bspec.as_tuple()] = ret_tokens

          # Execute single-bucket and cross-bucket GA steps to verify _cache_size() == 1 and sharding stability.
          state, _, rng_1 = trainer.run_optimizer_step(
              state, [b1_batch], jax.random.key(10), mesh=mesh, scheduler=pipeline.scheduler
          )
          self.assertEqual(_extract_shardings(state), init_shardings)

          state, _, rng_2 = trainer.run_optimizer_step(state, [b2_batch], rng_1, mesh=mesh, scheduler=pipeline.scheduler)
          self.assertEqual(_extract_shardings(state), init_shardings)

          state, _, _ = trainer.run_optimizer_step(
              state, [b1_batch, b2_batch], rng_2, mesh=mesh, scheduler=pipeline.scheduler
          )
          self.assertEqual(_extract_shardings(state), init_shardings)
        finally:
          os.dup2(saved_fd, 2)
          os.close(saved_fd)
        stderr_file.seek(0)
        spmd_stderr = stderr_file.read()

      # Check 1: Identical HLO ENTRY state parameter local shard shapes & exact _cache_size() == 1.
      self.assertEqual(entry_param_prefixes[(640, 84, 76)], entry_param_prefixes[(896, 108, 92)])
      self.assertEqual(len(trainer._jit_micro_grad_cache), 2)
      for bkey, fn in trainer._jit_micro_grad_cache.items():
        self.assertEqual(fn._cache_size(), 1, f"Expected _cache_size()==1 on bucket {bkey}")
      self.assertEqual(trainer._jit_apply_grads._cache_size(), 1)

      # Check 3: Expected FSDP all-gather + backward scatter-reduction collectives & matching 1/8 ROOT grad shard shapes.
      collective_counts = {}
      for bkey, hlo in hlo_texts.items():
        ag = hlo.count("all-gather(")
        rs = hlo.count("reduce-scatter(")
        ar = hlo.count("all-reduce(")
        ds = hlo.count("dynamic-slice(")
        collective_counts[bkey] = {"all_gather": ag, "reduce_scatter": rs, "all_reduce": ar, "dynamic_slice": ds}
        self.assertGreater(ag, 0, f"Missing FSDP all-gather in HLO for bucket {bkey}")
        self.assertTrue(
            rs > 0 or (ar > 0 and ds > 0),
            f"Missing backward gradient scatter-reduction in HLO for bucket {bkey}: rs={rs}, ar={ar}, ds={ds}",
        )
        self.assertEqual(root_grad_shapes[bkey][:3], ["f32[]", "f32[]", "f32[]"])
        self.assertEqual(
            root_grad_shapes[bkey][3:],
            entry_param_prefixes[bkey][:n_trainable_leaves],
            f"HLO ROOT gradient shard shapes do not match ENTRY state.params shard shapes on bucket {bkey}",
        )

      # Check 4: Involuntary rematerialization audit with non-colliding sequence dimensions.
      inv_lines = [ln for ln in spmd_stderr.splitlines() if "Involuntary full rematerialization" in ln]
      where_offenders = [ln for ln in inv_lines if "_where" in ln]
      param_offenders = [ln for ln in inv_lines if "parameter(" in ln]
      seq_offenders = []
      shape_re = re.compile(r"(?:f32|bf16|f16|s32|pred)\[([0-9,]+)\]")
      for ln in inv_lines:
        dims_in_line = set()
        for match in shape_re.findall(ln):
          for tok in match.split(","):
            if tok:
              dims_in_line.add(int(tok))
        if dims_in_line & seq_audit_dims:
          seq_offenders.append(ln)

      self.assertEqual(where_offenders, [], "Involuntary remat on _where:\n" + "\n".join(where_offenders))
      self.assertEqual(param_offenders, [], "Involuntary remat on parameter(...):\n" + "\n".join(param_offenders))
      self.assertEqual(
          seq_offenders,
          [],
          f"Involuntary remat on sequence-bearing tensor matching {sorted(seq_audit_dims)}:\n" + "\n".join(seq_offenders),
      )
      print(
          f"[Gate 2 8-Device HLO & Remat Audit] embed_leaves_checked={embed_checked} "
          f"collectives={collective_counts} involuntary_total={len(inv_lines)} "
          f"where_offenders=0 param_offenders=0 seq_offenders=0"
      )

  # ============================================================================
  # Gate 3: Multi-Bucket Convergence & Cross-Bucket GA Gate (Full-Param + LoRA)
  # ============================================================================
  def test_gate3_multibucket_cross_bucket_ga_convergence_full_and_lora(self):
    """Runs 18 cross-bucket GA steps (36 micro-steps) in full-param & NNX LoRA modes + train_ltx2 E2E.

    Monotonicity criterion: Requires `losses[-1] < 0.5 * losses[0]` (full-param) and
    `losses[-1] < 0.75 * losses[0]` (LoRA), strict monotonic decrease across all 6 non-overlapping
    3-step window averages (`avg3[i+1] < avg3[i]`), AND at most 1 single-step non-decrease
    (`sum(losses[t+1] >= losses[t]) <= 1`).
    """
    rng_np = np.random.default_rng(109)
    bucket_1 = LTX2BucketSpec(128, 64, 64)
    bucket_2 = LTX2BucketSpec(256, 128, 64)

    # Bucket 1 mixes audio + video-only (has_audio=False); Bucket 2 has longer audio+video on both samples.
    b1_np = collate_ltx2_bucket_batch(
        [
            LTX2Sample(
                video_latents=rng_np.standard_normal((16, 2, 8, 8)).astype(np.float32),
                audio_latents=rng_np.standard_normal((44, 8)).astype(np.float32),
                encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
                has_audio=True,
            ),
            LTX2Sample(
                video_latents=rng_np.standard_normal((16, 2, 6, 8)).astype(np.float32),
                audio_latents=None,
                encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
                has_audio=False,
            ),
        ],
        bucket_1,
        audio_in_channels=8,
    )
    b2_np = collate_ltx2_bucket_batch(
        [
            LTX2Sample(
                video_latents=rng_np.standard_normal((16, 3, 8, 9)).astype(np.float32),
                audio_latents=rng_np.standard_normal((96, 8)).astype(np.float32),
                encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
                has_audio=True,
            ),
            LTX2Sample(
                video_latents=rng_np.standard_normal((16, 2, 10, 10)).astype(np.float32),
                audio_latents=rng_np.standard_normal((80, 8)).astype(np.float32),
                encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
                has_audio=True,
            ),
        ],
        bucket_2,
        audio_in_channels=8,
    )

    for enable_lora, lr, target_ratio in ((False, 3e-3, 0.5), (True, 5e-3, 0.75)):
      mode_label = "LoRA" if enable_lora else "FullParam"
      with self.subTest(mode=mode_label), tempfile.TemporaryDirectory() as tmpdir:
        cfg = _init_bucketing_test_config(
            tmpdir,
            extra_overrides=[
                f"enable_lora_training={enable_lora}",
                "lora_rank=8",
                "lora_alpha=16.0",
                f"learning_rate={lr}",
                "lr_schedule_type=constant",
                "warmup_steps_fraction=0.0",
                "gradient_accumulation_steps=2",
                "max_train_steps=18",
            ],
        )
        trainer = LTX2Trainer(cfg)
        pipeline, _ = trainer.load_or_init_pipeline()
        optimizer, _ = trainer.checkpointer._create_optimizer(cfg, float(cfg.learning_rate))
        state, _, _ = trainer.init_train_state(pipeline, optimizer)
        mesh = pipeline.mesh

        base_before = jax.device_get(extract_unmerged_base_params(state)) if enable_lora else None
        lora_before = jax.device_get(state.params) if enable_lora else None

        b1_sh = shard_ltx2_batch_for_mesh(b1_np, mesh)
        b2_sh = shard_ltx2_batch_for_mesh(b2_np, mesh)
        fixed_window_rng = jax.random.key(2026)

        losses = []
        grad_norms = []
        for _ in range(18):
          state, metrics, _ = trainer.run_optimizer_step(
              state,
              [b1_sh, b2_sh],
              fixed_window_rng,
              mesh=mesh,
              scheduler=pipeline.scheduler,
          )
          sc = metrics["scalar"]
          loss_val = float(sc["learning/loss"])
          vloss_val = float(sc["learning/video_loss"])
          aloss_val = float(sc["learning/audio_loss"])
          gnorm_val = float(sc["learning/grad_norm"])
          self.assertTrue(np.isfinite(loss_val) and np.isfinite(vloss_val) and np.isfinite(aloss_val))
          self.assertTrue(np.isfinite(gnorm_val) and gnorm_val > 0.0)
          losses.append(loss_val)
          grad_norms.append(gnorm_val)

        non_decreases = sum(1 for t in range(len(losses) - 1) if losses[t + 1] >= losses[t])
        avg3 = [float(np.mean(losses[3 * k : 3 * (k + 1)])) for k in range(6)]
        for k in range(5):
          self.assertLess(avg3[k + 1], avg3[k], f"{mode_label} 3-step window average failed to decrease: {avg3}")
        self.assertLessEqual(non_decreases, 1, f"{mode_label} had {non_decreases} > 1 single-step non-decreases: {losses}")
        self.assertLess(
            losses[-1],
            target_ratio * losses[0],
            f"{mode_label} final loss {losses[-1]:.5f} not < {target_ratio} * {losses[0]:.5f}",
        )

        if enable_lora:
          base_after = jax.device_get(extract_unmerged_base_params(state))
          base_diff, _ = _max_tree_abs_diff(base_after, base_before)
          lora_diff, _ = _max_tree_abs_diff(jax.device_get(state.params), lora_before)
          self.assertEqual(base_diff, 0.0)
          self.assertGreater(lora_diff, 0.0)

        print(
            f"[Gate 3 {mode_label} 18-Step Cross-Bucket GA] "
            f"loss[0]={losses[0]:.5f} -> loss[-1]={losses[-1]:.5f} "
            f"non_decreases={non_decreases} avg3={[round(x, 5) for x in avg3]}"
        )

    # Sub-check (e): End-to-end advancing-RNG `train_ltx2.train(cfg)` with Orbax checkpoint save.
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_e2e = _init_bucketing_test_config(
          tmpdir,
          extra_overrides=[
              "save_final_checkpoint=True",
              "enable_async_checkpointing=False",
              "max_train_steps=8",
              "gradient_accumulation_steps=2",
              "synthetic_num_unique_batches_per_bucket=1",
              "learning_rate=3e-3",
              "lr_schedule_type=constant",
              "warmup_steps_fraction=0.0",
          ],
      )

      # Measure deterministic pre-training multi-bucket validation loss on the exact initial model.
      pre_trainer = LTX2Trainer(cfg_e2e)
      pre_pipe, _ = pre_trainer.load_or_init_pipeline()
      pre_iter = pre_trainer.load_dataset(pre_pipe.mesh, pre_pipe.transformer)
      eb0 = {k: v for k, v in next(pre_iter).items() if k != "bucket_key"}
      eb1 = {k: v for k, v in next(pre_iter).items() if k != "bucket_key"}

      def _eval_pair(transformer_model):
        gdef, p_st, r_st = nnx.split(transformer_model, nnx.Param, ...)
        total = 0.0
        with jax.set_mesh(pre_pipe.mesh), nn_partitioning.axis_rules(cfg_e2e.logical_axis_rules):
          for idx, eb in enumerate((eb0, eb1)):
            ev_err, ea_err, aux = compute_ltx2_microbatch_errors(
                gdef,
                p_st,
                r_st,
                eb,
                jax.random.key(500 + idx),
                scheduler=pre_pipe.scheduler,
            )
            vid_t = float(ev_err / jnp.maximum(aux["valid_vid_elems"], 1.0))
            aud_t = float(jnp.where(aux["valid_aud_elems"] > 0, ea_err / jnp.maximum(aux["valid_aud_elems"], 1.0), 0.0))
            total += vid_t + aud_t
        return total

      pre_eval = _eval_pair(pre_pipe.transformer)
      post_pipe = run_train_ltx2(cfg_e2e)
      post_eval = _eval_pair(post_pipe.transformer)
      self.assertLess(post_eval, pre_eval)
      self.assertTrue(os.path.isdir(os.path.join(cfg_e2e.checkpoint_dir, "7")))
      print(f"[Gate 3 E2E train_ltx2.train] pre_eval={pre_eval:.5f} -> post_eval={post_eval:.5f} ckpt=7 OK")


if __name__ == "__main__":
  absltest.main()
