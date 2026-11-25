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

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
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
    shard_ltx2_batch_for_mesh,
)
from maxdiffusion.loaders.ltx2_lora_nnx_loader import LTX2NNXLoraLoader
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.train_ltx2 import _validate_ltx2_training_model_name, train as run_train_ltx2
from maxdiffusion.trainers.ltx2_trainer import (
    LTX2LoRALinear,
    LTX2Trainer,
    compute_ltx2_microbatch_errors,
    save_lora_safetensors,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "configs", "ltx2_train.yml"))


def _init_mini_config(tmpdir: str, extra_overrides=()):
  n_dev = jax.device_count()
  if n_dev < 4:
    raise unittest.SkipTest(f"Requires >= 4 host devices, found {n_dev}")
  cp = 4 if n_dev >= 8 else 2
  per_dev_bs = 0.25 if n_dev >= 8 else 0.5
  base_argv = [
      None,
      CONFIG_PATH,
      f"output_dir={tmpdir}",
      "run_name=ltx2_unit_test",
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
      f"ici_context_parallelism={cp}",
      "ici_tensor_parallelism=1",
      f"per_device_batch_size={per_dev_bs}",
      "ltx2_buckets=[[128,64,64],[256,128,64]]",
      "max_sequence_length=64",
      "synthetic_video_only_prob=0.5",
      "max_train_steps=4",
      "learning_rate=1e-3",
      "write_metrics=False",
  ] + list(extra_overrides)
  pyconfig.initialize(base_argv, unittest=True)
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


class LTX2TrainerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.default_matmul_precision("float32"))

  def test_config_entrypoint_jit_cache_and_sharding_invariance(self):
    """Verifies train_ltx2 validation, per-bucket JIT compile counts, and TrainState sharding invariance."""
    _validate_ltx2_training_model_name("ltx2_video")
    _validate_ltx2_training_model_name("ltx2.3")
    with self.assertRaisesRegex(ValueError, "train_ltx2.py"):
      _validate_ltx2_training_model_name("wan2.1")

    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = _init_mini_config(
          tmpdir,
          extra_overrides=[
              "gradient_accumulation_steps=2",
              "max_train_steps=3",
              "checkpoint_every=-1",
              "save_final_checkpoint=False",
          ],
      )
      trainer = LTX2Trainer(cfg)
      pipeline, restore_args = trainer.load_or_init_pipeline()
      optimizer, _ = trainer.checkpointer._create_optimizer(cfg, float(cfg.learning_rate))
      state, state_shardings, start_step = trainer.init_train_state(pipeline, optimizer, restore_args)
      self.assertEqual(start_step, 0)

      def _extract_shardings(st):
        return jax.tree.map(lambda a: a.sharding, st)

      init_shardings = _extract_shardings(state)
      self.assertEqual(init_shardings, state_shardings)

      train_iter = trainer.load_dataset(pipeline.mesh, pipeline.transformer)
      rng = jax.random.key(0)

      # Step 0: micro-batch 0 hits bucket (128,64,64), micro-batch 1 hits bucket (256,128,64).
      b0 = next(train_iter)
      b1 = next(train_iter)
      self.assertEqual(b0["bucket_key"], (128, 64, 64))
      self.assertEqual(b1["bucket_key"], (256, 128, 64))

      state, m0, rng = trainer.run_optimizer_step(state, [b0], rng, mesh=pipeline.mesh, scheduler=pipeline.scheduler)
      shardings_after_b0 = _extract_shardings(state)
      self.assertEqual(shardings_after_b0, init_shardings)

      state, m1, rng = trainer.run_optimizer_step(state, [b1], rng, mesh=pipeline.mesh, scheduler=pipeline.scheduler)
      shardings_after_b1 = _extract_shardings(state)
      self.assertEqual(shardings_after_b1, init_shardings)

      # Revisit both buckets inside a K=2 cross-bucket GA window -> 0 new compilations!
      b2 = next(train_iter)
      b3 = next(train_iter)
      state, m2, rng = trainer.run_optimizer_step(state, [b2, b3], rng, mesh=pipeline.mesh, scheduler=pipeline.scheduler)
      self.assertEqual(_extract_shardings(state), init_shardings)
      self.assertTrue(np.isfinite(float(m2["scalar"]["learning/loss"])))

      # Assert exact compile counts.
      self.assertEqual(len(trainer._jit_micro_grad_cache), 2)
      for bkey, fn in trainer._jit_micro_grad_cache.items():
        self.assertEqual(fn._cache_size(), 1, f"Expected 1 compile for bucket {bkey}, got {fn._cache_size()}")
      self.assertEqual(trainer._jit_apply_grads._cache_size(), 1)
      print(
          f"[JIT Cache Audit] buckets={list(trainer._jit_micro_grad_cache.keys())} "
          f"per_bucket_compiles={[fn._cache_size() for fn in trainer._jit_micro_grad_cache.values()]} "
          f"apply_grads_compiles={trainer._jit_apply_grads._cache_size()}"
      )

      # Also verify CLI train(cfg) entrypoint end-to-end.
      trained_pipe = run_train_ltx2(cfg)
      self.assertIsNotNone(trained_pipe.transformer)

  def test_cross_bucket_token_weighted_ga_exact_vjp_and_nan_pad_guard(self):
    """Verifies K=2 cross-bucket GA equals 1 combined VJP + LTX-2.3 use_cross_timestep=True + NaN pad guard."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = _init_mini_config(
          tmpdir,
          extra_overrides=[
              "use_cross_timestep=True",
              "audio_loss_weight=1.5",
              "gradient_accumulation_steps=2",
              "timestep_sampling=logit_normal",
          ],
      )
      trainer = LTX2Trainer(cfg)
      pipeline, _ = trainer.load_or_init_pipeline()
      with jax.set_mesh(pipeline.mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        _randomize_zero_params(pipeline.transformer, jax.random.key(42))

      optimizer, _ = trainer.checkpointer._create_optimizer(cfg, float(cfg.learning_rate))
      state, _, _ = trainer.init_train_state(pipeline, optimizer)
      mesh = pipeline.mesh

      rng_np = np.random.default_rng(7)
      bucket_a = LTX2BucketSpec(128, 64, 64)
      bucket_b = LTX2BucketSpec(256, 128, 64)

      # Bucket A (B=2): sample 0 has audio (40 tokens), sample 1 is video-only (0 audio tokens!).
      sa_0 = LTX2Sample(
          video_latents=rng_np.standard_normal((16, 2, 6, 8)).astype(np.float32),
          audio_latents=rng_np.standard_normal((40, 8)).astype(np.float32),
          encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          has_audio=True,
      )
      sa_1 = LTX2Sample(
          video_latents=rng_np.standard_normal((16, 2, 5, 8)).astype(np.float32),
          audio_latents=None,
          encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          has_audio=False,
      )
      # Bucket B (B=2): longer video + longer audio on both samples.
      sb_0 = LTX2Sample(
          video_latents=rng_np.standard_normal((16, 3, 8, 9)).astype(np.float32),
          audio_latents=rng_np.standard_normal((96, 8)).astype(np.float32),
          encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          has_audio=True,
      )
      sb_1 = LTX2Sample(
          video_latents=rng_np.standard_normal((16, 2, 9, 10)).astype(np.float32),
          audio_latents=rng_np.standard_normal((72, 8)).astype(np.float32),
          encoder_hidden_states=rng_np.standard_normal((64, 32)).astype(np.float32),
          has_audio=True,
      )

      ba_np = collate_ltx2_bucket_batch([sa_0, sa_1], bucket_a, audio_in_channels=8)
      bb_np = collate_ltx2_bucket_batch([sb_0, sb_1], bucket_b, audio_in_channels=8)
      ba_sh = shard_ltx2_batch_for_mesh(ba_np, mesh)
      bb_sh = shard_ltx2_batch_for_mesh(bb_np, mesh)

      # 1. Compare trainer's 2-micro-step cross-bucket GA vs a single combined VJP over the window.
      scalar_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      step_rng = jax.device_put(jax.random.key(314), scalar_sh)
      rngs = [jax.device_put(r, scalar_sh) for r in jax.random.split(step_rng, 3)]
      vmasks = (ba_sh["video_attention_mask"], bb_sh["video_attention_mask"])
      amasks = (ba_sh["audio_attention_mask"], bb_sh["audio_attention_mask"])
      inv_vid, inv_aud = trainer._jit_window_denoms(vmasks, amasks, 16, 8, 1.5)

      ta = {k: v for k, v in ba_sh.items() if k != "bucket_key"}
      tb = {k: v for k, v in bb_sh.items() if k != "bucket_key"}

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        fn_a = trainer.get_or_compile_bucket_grad_step(bucket_a, mesh, pipeline.scheduler, state.graphdef)
        fn_b = trainer.get_or_compile_bucket_grad_step(bucket_b, mesh, pipeline.scheduler, state.graphdef)
        (la, _, _), ga = fn_a(state.params, state.rest_of_state, ta, rngs[1], inv_vid, inv_aud)
        (lb, _, _), gb = fn_b(state.params, state.rest_of_state, tb, rngs[2], inv_vid, inv_aud)
        ga_accum = trainer._jit_add_grads(ga, gb)
        loss_accum = float(la + lb)

        # Single-VJP combined window reference!
        @jax.jit
        def combined_ref(p, r):
          def window_loss(p_in, r_in):
            ev_a, ea_a, _ = compute_ltx2_microbatch_errors(
                state.graphdef,
                p_in,
                r_in,
                ta,
                rngs[1],
                scheduler=pipeline.scheduler,
                use_cross_timestep=True,
            )
            ev_b, ea_b, _ = compute_ltx2_microbatch_errors(
                state.graphdef,
                p_in,
                r_in,
                tb,
                rngs[2],
                scheduler=pipeline.scheduler,
                use_cross_timestep=True,
            )
            d_vid = (jnp.sum(ta["video_attention_mask"]) + jnp.sum(tb["video_attention_mask"])) * 16.0
            d_aud = (jnp.sum(ta["audio_attention_mask"]) + jnp.sum(tb["audio_attention_mask"])) * 8.0
            return (ev_a + ev_b) / d_vid + 1.5 * (ea_a + ea_b) / d_aud

          return nnx.value_and_grad(window_loss, argnums=0)(p, r)

        loss_ref, grad_ref = combined_ref(state.params, state.rest_of_state)

      np.testing.assert_allclose(loss_accum, float(loss_ref), rtol=1e-5, atol=1e-4)
      accum_leaves = jax.tree_util.tree_leaves_with_path(jax.device_get(ga_accum))
      ref_leaves = jax.tree_util.tree_leaves_with_path(jax.device_get(grad_ref))
      max_diff = 0.0
      for (path, g_got), (_, g_exp) in zip(accum_leaves, ref_leaves):
        d = float(np.max(np.abs(g_got - g_exp)))
        max_diff = max(max_diff, d)
        np.testing.assert_allclose(g_got, g_exp, rtol=5e-4, atol=1e-3, err_msg=f"Cross-bucket GA mismatch at {path}")
      print(
          f"[Cross-Bucket GA vs Combined Single-VJP] loss_accum={loss_accum:.6f} "
          f"loss_ref={float(loss_ref):.6f} max |Δgrad| across {len(accum_leaves)} leaves = {max_diff:.3e}"
      )

      # 2. NaN/Inf pad-slot guard: inject NaN/Inf into padded noise-override slots and verify finite grads!
      v_noise = np.ones((2, 128, 16), dtype=np.float32) * 0.25
      v_noise[0, 96:] = np.inf  # Sample 0 valid length is 2*6*8 = 96; 96:128 is padded!
      v_noise[1, 80:] = np.nan  # Sample 1 valid length is 2*5*8 = 80; 80:128 is padded!
      a_noise = np.ones((2, 64, 8), dtype=np.float32) * -0.25
      a_noise[0, 40:] = np.nan
      a_noise[1, :] = np.inf  # Sample 1 has_audio=False -> all 64 audio slots are padded!

      v_noise_sh = jax.device_put(v_noise, ta["hidden_states"].sharding)
      a_noise_sh = jax.device_put(a_noise, ta["audio_hidden_states"].sharding)

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):

        @jax.jit
        def nan_guard_check(p, r):
          def loss_fn(p_in, r_in):
            ev, ea, _ = compute_ltx2_microbatch_errors(
                state.graphdef,
                p_in,
                r_in,
                ta,
                jax.random.key(9),
                scheduler=pipeline.scheduler,
                use_cross_timestep=True,
                sigmas_override=jnp.array([0.4, 0.6], dtype=jnp.float32),
                video_noise_override=v_noise_sh,
                audio_noise_override=a_noise_sh,
            )
            return ev * inv_vid + ea * inv_aud

          return nnx.value_and_grad(loss_fn, argnums=0)(p, r)

        nan_loss, nan_grads = nan_guard_check(state.params, state.rest_of_state)

      self.assertTrue(np.isfinite(float(nan_loss)), f"Expected finite loss with NaN/Inf pad noise, got {nan_loss}")
      for path, leaf in jax.tree_util.tree_leaves_with_path(jax.device_get(nan_grads)):
        self.assertTrue(bool(np.all(np.isfinite(leaf))), f"Non-finite gradient at {path} under NaN/Inf pad noise!")

  def test_lora_dtype_zero_init_freeze_and_diffusers_loader_roundtrip(self):
    """Verifies bf16/fp32 LoRA dtype invariance, zero-init parity, frozen base, and LTX2NNXLoraLoader round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # 1a. Direct LTX2LoRALinear module bfloat16 dtype & bitwise zero-init check.
      base_lin_bf16 = nnx.Linear(32, 64, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(5))
      x_bf16 = jax.random.normal(jax.random.key(6), (2, 16, 32), dtype=jnp.bfloat16)
      ref_lin_out = base_lin_bf16(x_bf16)
      lora_lin_bf16 = LTX2LoRALinear(base_lin_bf16, rank=4, alpha=8.0, rngs=nnx.Rngs(7), param_dtype=jnp.float32)
      got_lin_out = lora_lin_bf16(x_bf16)
      self.assertEqual(got_lin_out.dtype, jnp.bfloat16)
      np.testing.assert_array_equal(np.asarray(got_lin_out), np.asarray(ref_lin_out))

      # 1b. Full scanned 8-device transformer bfloat16 activation dtype & zero-init check.
      cfg_bf16 = _init_mini_config(
          tmpdir,
          extra_overrides=[
              "weights_dtype=bfloat16",
              "activations_dtype=bfloat16",
              "enable_lora_training=True",
              "lora_rank=4",
              "lora_alpha=8.0",
          ],
      )
      trainer_bf16 = LTX2Trainer(cfg_bf16)
      pipe_bf16, _ = trainer_bf16.load_or_init_pipeline()
      mesh = pipe_bf16.mesh

      batch_it = trainer_bf16.load_dataset(mesh, pipe_bf16.transformer)
      b_sample = next(batch_it)
      ts = jax.device_put(
          jnp.array([400.0, 600.0], dtype=jnp.float32),
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp"))),
      )

      def _eval_model(m):
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_bf16.logical_axis_rules):
          return m(
              hidden_states=b_sample["hidden_states"].astype(m.dtype),
              audio_hidden_states=b_sample["audio_hidden_states"].astype(m.dtype),
              encoder_hidden_states=b_sample["encoder_hidden_states"].astype(m.dtype),
              audio_encoder_attention_mask=b_sample["audio_encoder_attention_mask"],
              audio_encoder_hidden_states=b_sample["audio_encoder_hidden_states"].astype(m.dtype),
              timestep=ts,
              audio_timestep=ts,
              sigma=ts,
              audio_sigma=ts,
              encoder_attention_mask=b_sample["encoder_attention_mask"],
              video_coords=b_sample["video_coords"],
              audio_coords=b_sample["audio_coords"],
              modality_mask=b_sample["modality_mask"],
              video_attention_mask=b_sample["video_attention_mask"],
              audio_attention_mask=b_sample["audio_attention_mask"],
          )

      ref_bf16_out = jax.device_get(_eval_model(pipe_bf16.transformer))
      self.assertEqual(ref_bf16_out["sample"].dtype, jnp.bfloat16)
      self.assertEqual(ref_bf16_out["audio_sample"].dtype, jnp.bfloat16)

      opt_bf16, _ = trainer_bf16.checkpointer._create_optimizer(cfg_bf16, float(cfg_bf16.learning_rate))
      st_bf16, _, _ = trainer_bf16.init_train_state(pipe_bf16, opt_bf16)
      wrapped_bf16 = nnx.merge(st_bf16.graphdef, st_bf16.params, st_bf16.rest_of_state)
      got_bf16_init = jax.device_get(_eval_model(wrapped_bf16))
      self.assertEqual(got_bf16_init["sample"].dtype, jnp.bfloat16)
      self.assertEqual(got_bf16_init["audio_sample"].dtype, jnp.bfloat16)
      np.testing.assert_allclose(
          got_bf16_init["sample"].astype(np.float32),
          ref_bf16_out["sample"].astype(np.float32),
          rtol=2e-2,
          atol=5e-2,
      )
      np.testing.assert_allclose(
          got_bf16_init["audio_sample"].astype(np.float32),
          ref_bf16_out["audio_sample"].astype(np.float32),
          rtol=2e-2,
          atol=5e-2,
      )

      # 2. Float32 LoRA training + base-weight freeze + LTX2NNXLoraLoader safetensors round-trip.
      cfg_fp32 = _init_mini_config(
          tmpdir,
          extra_overrides=[
              "weights_dtype=float32",
              "activations_dtype=float32",
              "enable_lora_training=True",
              "lora_rank=4",
              "lora_alpha=8.0",
              "learning_rate=5e-3",
          ],
      )
      trainer = LTX2Trainer(cfg_fp32)
      pipe, _ = trainer.load_or_init_pipeline()
      with jax.set_mesh(pipe.mesh), nn_partitioning.axis_rules(cfg_fp32.logical_axis_rules):
        _randomize_zero_params(pipe.transformer, jax.random.key(11))

      # Clone clean unwrapped transformer before LoRA wrapping so LTX2NNXLoraLoader can merge into it!
      g_clean, st_clean = nnx.split(pipe.transformer)
      clean_copy = nnx.merge(g_clean, jax.tree.map(lambda x: x.copy(), st_clean))

      opt, _ = trainer.checkpointer._create_optimizer(cfg_fp32, float(cfg_fp32.learning_rate))
      state, _, _ = trainer.init_train_state(pipe, opt)
      ref_fp32_out = jax.device_get(_eval_model(clean_copy))
      init_fp32_out = jax.device_get(_eval_model(nnx.merge(state.graphdef, state.params, state.rest_of_state)))
      np.testing.assert_allclose(init_fp32_out["sample"], ref_fp32_out["sample"], rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(init_fp32_out["audio_sample"], ref_fp32_out["audio_sample"], rtol=1e-4, atol=1e-4)
      base_before = jax.device_get(state.rest_of_state)
      lora_before = jax.device_get(state.params)

      it = trainer.load_dataset(pipe.mesh, pipe.transformer)
      rng = jax.random.key(55)
      for _ in range(3):
        state, _, rng = trainer.run_optimizer_step(state, [next(it)], rng, mesh=pipe.mesh, scheduler=pipe.scheduler)

      base_after = jax.device_get(state.rest_of_state)
      lora_after = jax.device_get(state.params)
      for (p, b0), (_, b1) in zip(
          jax.tree_util.tree_leaves_with_path(base_before),
          jax.tree_util.tree_leaves_with_path(base_after),
      ):
        if isinstance(b0, np.ndarray):
          self.assertEqual(float(np.max(np.abs(b1 - b0))), 0.0, f"Frozen base weight {p} mutated!")

      max_lora_delta = max(
          float(np.max(np.abs(l1 - l0)))
          for l0, l1 in zip(jax.tree_util.tree_leaves(lora_before), jax.tree_util.tree_leaves(lora_after))
      )
      self.assertGreater(max_lora_delta, 1e-5)

      # Export to safetensors and load through LTX2NNXLoraLoader!
      lora_file = os.path.join(tmpdir, "lora_export", "test_ltx2_lora.safetensors")
      exported_dict = save_lora_safetensors(lora_file, state.params, lora_alpha=8.0, scan_layers=True)
      self.assertGreater(len(exported_dict), 0)
      self.assertTrue(all(k.startswith("diffusion_model.") for k in exported_dict.keys()))

      dummy_pipe = types.SimpleNamespace(transformer=clean_copy)
      loader = LTX2NNXLoraLoader()
      with jax.set_mesh(pipe.mesh), nn_partitioning.axis_rules(cfg_fp32.logical_axis_rules):
        loader.load_lora_weights(
            dummy_pipe,
            os.path.dirname(lora_file),
            os.path.basename(lora_file),
            rank=4,
            scale=1.0,
            scan_layers=True,
            dtype="float32",
            local_files_only=True,
        )

      trained_lora_model = nnx.merge(state.graphdef, state.params, state.rest_of_state)
      out_wrapped = jax.device_get(_eval_model(trained_lora_model))
      out_loader = jax.device_get(_eval_model(dummy_pipe.transformer))
      np.testing.assert_allclose(out_loader["sample"], out_wrapped["sample"], rtol=5e-4, atol=5e-4)
      np.testing.assert_allclose(out_loader["audio_sample"], out_wrapped["audio_sample"], rtol=5e-4, atol=5e-4)
      print(
          f"[LoRA Export -> LTX2NNXLoraLoader] exported_keys={len(exported_dict)} "
          f"max |Δsample|={float(np.max(np.abs(out_loader['sample'] - out_wrapped['sample']))):.3e}"
      )

  def test_orbax_checkpointer_full_and_lora_roundtrip(self):
    """Verifies real Orbax LTX2Checkpointer save & restore for full-param, LoRA resume, and merged LoRA."""
    # 1. Full-param Orbax save + resume (with opt_state & step + 1).
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = _init_mini_config(
          tmpdir,
          extra_overrides=[
              "save_optimizer=True",
              "enable_async_checkpointing=False",
          ],
      )
      trainer1 = LTX2Trainer(cfg)
      pipe1, _ = trainer1.load_or_init_pipeline()
      opt1, _ = trainer1.checkpointer._create_optimizer(cfg, float(cfg.learning_rate))
      state1, _, _ = trainer1.init_train_state(pipe1, opt1)
      it1 = trainer1.load_dataset(pipe1.mesh, pipe1.transformer)
      state1, _, _ = trainer1.run_optimizer_step(
          state1, [next(it1)], jax.random.key(3), mesh=pipe1.mesh, scheduler=pipe1.scheduler
      )
      trainer1.save_training_checkpoint(0, pipe1, state1)
      trainer1.checkpointer.checkpoint_manager.wait_until_finished()

      trainer2 = LTX2Trainer(cfg)
      pipe2, restore_args2 = trainer2.load_or_init_pipeline()
      opt2, _ = trainer2.checkpointer._create_optimizer(cfg, float(cfg.learning_rate))
      state2, _, start_step2 = trainer2.init_train_state(pipe2, opt2, restore_args2)
      self.assertEqual(start_step2, 1)
      self.assertEqual(int(jax.device_get(state2.step)), 1)
      for l1, l2 in zip(
          jax.tree_util.tree_leaves(jax.device_get(state1.params)),
          jax.tree_util.tree_leaves(jax.device_get(state2.params)),
      ):
        np.testing.assert_array_equal(l1, l2)

    # 2. LoRA Orbax resume (`merge_lora_on_save=False`) AND merged inference export (`merge_lora_on_save=True`).
    with tempfile.TemporaryDirectory() as tmpdir_lora, tempfile.TemporaryDirectory() as tmpdir_merged:
      cfg_lora = _init_mini_config(
          tmpdir_lora,
          extra_overrides=[
              "enable_lora_training=True",
              "lora_rank=4",
              "lora_alpha=8.0",
              "save_optimizer=True",
              "enable_async_checkpointing=False",
          ],
      )
      t_lora = LTX2Trainer(cfg_lora)
      p_lora, _ = t_lora.load_or_init_pipeline()
      opt_l, _ = t_lora.checkpointer._create_optimizer(cfg_lora, float(cfg_lora.learning_rate))
      st_l, _, _ = t_lora.init_train_state(p_lora, opt_l)
      b_eval = next(t_lora.load_dataset(p_lora.mesh, p_lora.transformer))
      st_l, _, _ = t_lora.run_optimizer_step(st_l, [b_eval], jax.random.key(8), mesh=p_lora.mesh, scheduler=p_lora.scheduler)
      t_lora.save_training_checkpoint(2, p_lora, st_l, merge_lora_on_save=False)
      t_lora.checkpointer.checkpoint_manager.wait_until_finished()

      # Resume LoRA run at step 3!
      t_lora_res = LTX2Trainer(cfg_lora)
      p_res, r_args = t_lora_res.load_or_init_pipeline()
      opt_res, _ = t_lora_res.checkpointer._create_optimizer(cfg_lora, float(cfg_lora.learning_rate))
      st_res, _, start_res = t_lora_res.init_train_state(p_res, opt_res, r_args)
      self.assertEqual(start_res, 3)
      for l1, l2 in zip(
          jax.tree_util.tree_leaves(jax.device_get(st_l.params)),
          jax.tree_util.tree_leaves(jax.device_get(st_res.params)),
      ):
        np.testing.assert_array_equal(l1, l2)

      # Save merged inference checkpoint into tmpdir_merged and load into standard non-LoRA LTX2VideoTransformer3DModel!
      cfg_merged_save = _init_mini_config(
          tmpdir_merged,
          extra_overrides=[
              "enable_lora_training=True",
              "lora_rank=4",
              "lora_alpha=8.0",
              "save_optimizer=False",
              "enable_async_checkpointing=False",
          ],
      )
      t_msave = LTX2Trainer(cfg_merged_save)
      t_msave._base_transformer_config_dict = t_lora._base_transformer_config_dict
      t_msave.save_training_checkpoint(0, p_lora, st_l, merge_lora_on_save=True)
      t_msave.checkpointer.checkpoint_manager.wait_until_finished()

      cfg_merged_load = _init_mini_config(
          tmpdir_merged,
          extra_overrides=[
              "enable_lora_training=False",
              "enable_async_checkpointing=False",
          ],
      )
      t_mload = LTX2Trainer(cfg_merged_load)
      p_mload, _ = t_mload.load_or_init_pipeline()

      ts = jax.device_put(
          jnp.array([350.0, 650.0], dtype=jnp.float32),
          jax.sharding.NamedSharding(p_lora.mesh, jax.sharding.PartitionSpec(("data", "fsdp"))),
      )
      with jax.set_mesh(p_lora.mesh), nn_partitioning.axis_rules(cfg_lora.logical_axis_rules):
        m_lora_live = nnx.merge(st_l.graphdef, st_l.params, st_l.rest_of_state)
        kw = {
            "hidden_states": b_eval["hidden_states"],
            "audio_hidden_states": b_eval["audio_hidden_states"],
            "encoder_hidden_states": b_eval["encoder_hidden_states"],
            "audio_encoder_hidden_states": b_eval["audio_encoder_hidden_states"],
            "timestep": ts,
            "audio_timestep": ts,
            "sigma": ts,
            "audio_sigma": ts,
            "encoder_attention_mask": b_eval["encoder_attention_mask"],
            "audio_encoder_attention_mask": b_eval["audio_encoder_attention_mask"],
            "video_coords": b_eval["video_coords"],
            "audio_coords": b_eval["audio_coords"],
            "modality_mask": b_eval["modality_mask"],
            "video_attention_mask": b_eval["video_attention_mask"],
            "audio_attention_mask": b_eval["audio_attention_mask"],
        }
        out_live = jax.device_get(m_lora_live(**kw))
        out_merged_ckpt = jax.device_get(p_mload.transformer(**kw))

      np.testing.assert_allclose(out_merged_ckpt["sample"], out_live["sample"], rtol=5e-4, atol=5e-4)
      np.testing.assert_allclose(out_merged_ckpt["audio_sample"], out_live["audio_sample"], rtol=5e-4, atol=5e-4)


if __name__ == "__main__":
  absltest.main()
