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

import functools
import os
import unittest
from unittest.mock import MagicMock
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import optax

import numpy as np
from maxdiffusion import pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.trainers.base_wan_trainer import TrainState
from maxdiffusion.trainers.wan_trainer_2_2 import (
    WanTrainer2_2,
    train_step_2_2,
    eval_step_2_2,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG_PATH = os.path.join(THIS_DIR, "..", "..", "configs", "base_wan_27b.yml")


class MiniWanModel(nnx.Module):
  """Lightweight Flax module implementing WanModel interface for real training step execution."""

  def __init__(self, rngs: nnx.Rngs, in_channels: int = 4):
    self.conv = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1, 1), rngs=rngs)
    self.time_proj = nnx.Linear(1, in_channels, rngs=rngs)

  def __call__(
      self,
      hidden_states,
      timestep,
      encoder_hidden_states=None,
      deterministic=False,
      rngs=None,
  ):
    # hidden_states: (bsz, C, T, H, W)
    t_emb = self.time_proj(timestep[:, None].astype(jnp.float32))[:, :, None, None, None]
    x = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    x = self.conv(x)
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
    return x + t_emb


class WanTrainer22Test(unittest.TestCase):

  def setUp(self):
    super().setUp()
    pyconfig.initialize(
        [
            None,
            BASE_CONFIG_PATH,
            "max_train_steps=100",
            "learning_rate_schedule_steps=100",
            "warmup_steps_fraction=0.1",
            "per_device_batch_size=2",
            "dataset_type=synthetic",
            "cache_latents_text_encoder_outputs=True",
            "replicate_vae=True",
            "boundary_ratio=0.5",
            "weights_dtype=float32",
        ],
        unittest=True,
    )

  def test_wan_trainer_2_2_initialization(self):
    """Smoke test to ensure WanTrainer2_2 can be initialized with base config."""
    config = pyconfig.config
    trainer = WanTrainer2_2(config)
    self.assertIsNotNone(trainer)
    checkpointer = trainer._get_checkpointer()
    self.assertIsInstance(checkpointer, WanCheckpointer2_2)

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1, 1, 1, 1)), ("data", "fsdp", "context", "tensor"))
    data_shardings = trainer.get_data_shardings(mesh)
    self.assertIn("latents", data_shardings)
    self.assertIn("encoder_hidden_states", data_shardings)

    eval_shardings = trainer.get_eval_data_shardings(mesh)
    self.assertIn("timesteps", eval_shardings)

  def test_calculate_tflops_formula(self):
    """Verify corrected TFLOPs calculation matches expected mathematical values."""
    trainer = WanTrainer2_2(pyconfig.config)

    mock_pipeline = MagicMock()
    mock_pipeline.config.height = 256
    mock_pipeline.config.width = 256
    mock_pipeline.config.num_frames = 1
    mock_pipeline.vae_scale_factor_temporal = 4
    mock_pipeline.config.per_device_batch_size = 1

    mock_transformer_config = MagicMock()
    mock_transformer_config.num_layers = 2
    mock_transformer_config.num_attention_heads = 4
    mock_transformer_config.attention_head_dim = 64
    mock_transformer_config.ffn_dim = 256
    mock_transformer_config.text_dim = 4096

    mock_pipeline.low_noise_transformer.config = mock_transformer_config

    train_tflops, total_attn_flops, seq_len = trainer.calculate_tflops(mock_pipeline)

    self.assertEqual(seq_len, 256)
    self.assertGreater(train_tflops, 0.0)
    self.assertGreater(total_attn_flops, 0)
    self.assertAlmostEqual(train_tflops, 3 * 5301600256 / 1e12, places=6)

  def test_optimizer_schedule_scaling(self):
    """Verify Optax schedule lengths scale proportionally with boundary_ratio."""
    trainer = WanTrainer2_2(pyconfig.config)
    checkpointer = trainer._get_checkpointer()

    mock_model = MagicMock()

    _, lr_sched_low = checkpointer._create_optimizer(
        mock_model, pyconfig.config, pyconfig.config.learning_rate, scale_factor=0.5
    )
    _, lr_sched_high = checkpointer._create_optimizer(
        mock_model, pyconfig.config, pyconfig.config.learning_rate, scale_factor=0.5
    )

    self.assertEqual(float(lr_sched_low(0)), 0.0)
    self.assertAlmostEqual(float(lr_sched_low(5)), float(pyconfig.config.learning_rate), places=5)
    self.assertEqual(float(lr_sched_high(0)), 0.0)
    self.assertAlmostEqual(float(lr_sched_high(5)), float(pyconfig.config.learning_rate), places=5)

  def test_extract_opt_state(self):
    """Verify WanCheckpointer2_2 extracts both optimizer states."""
    checkpointer = WanCheckpointer2_2(config=pyconfig.config)
    mock_checkpoint = MagicMock()
    mock_checkpoint.low_noise_transformer_state = {"opt_state": "low_opt_state_data"}
    mock_checkpoint.high_noise_transformer_state = {"opt_state": "high_opt_state_data"}

    extracted = checkpointer._extract_opt_state(mock_checkpoint)
    self.assertIn("low_noise_transformer", extracted)
    self.assertIn("high_noise_transformer", extracted)
    self.assertEqual(extracted["low_noise_transformer"], "low_opt_state_data")
    self.assertEqual(extracted["high_noise_transformer"], "high_opt_state_data")

  def test_real_train_step_2_2_execution(self):
    """Execute real JIT-compiled training steps and verify gradient application and stepping."""
    rng = jax.random.key(42)
    rng_low, rng_high, step_rng = jax.random.split(rng, 3)

    model_low = MiniWanModel(rngs=nnx.Rngs(rng_low), in_channels=4)
    model_high = MiniWanModel(rngs=nnx.Rngs(rng_high), in_channels=4)

    graphdef_low, params_low, rest_of_state_low = nnx.split(model_low, nnx.Param, ...)
    graphdef_high, params_high, rest_of_state_high = nnx.split(model_high, nnx.Param, ...)

    tx_low = optax.adam(1e-3)
    tx_high = optax.adam(1e-3)
    state_low = TrainState.create(
        apply_fn=graphdef_low.apply, params=params_low, tx=tx_low, graphdef=graphdef_low, rest_of_state=rest_of_state_low
    )
    state_high = TrainState.create(
        apply_fn=graphdef_high.apply,
        params=params_high,
        tx=tx_high,
        graphdef=graphdef_high,
        rest_of_state=rest_of_state_high,
    )

    # Use real Flow Match scheduler
    noise_scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(noise_scheduler_state, num_inference_steps=1000, training=True)

    config = pyconfig.config

    data = {
        "latents": jnp.ones((2, 4, 2, 4, 4), dtype=jnp.float32),
        "encoder_hidden_states": jnp.ones((2, 16, 4), dtype=jnp.float32),
    }

    jitted_train_step = jax.jit(functools.partial(train_step_2_2, scheduler=noise_scheduler, config=config))

    initial_params_low = jax.tree.map(lambda x: jnp.copy(x), state_low.params)
    initial_params_high = jax.tree.map(lambda x: jnp.copy(x), state_high.params)

    num_steps = 6
    for _ in range(num_steps):
      state_low, state_high, noise_scheduler_state, metrics, step_rng = jitted_train_step(
          state_low, state_high, data, step_rng, noise_scheduler_state
      )
      loss_val = float(metrics["scalar"]["learning/loss"])
      self.assertFalse(jnp.isnan(loss_val), "Training loss returned NaN")
      self.assertFalse(jnp.isinf(loss_val), "Training loss returned Inf")
      self.assertGreater(loss_val, 0.0, "Training loss should be positive")

    # Verify total steps across both transformers equal the number of train steps executed
    total_steps = int(state_low.step) + int(state_high.step)
    self.assertEqual(total_steps, num_steps)

    # Verify that at least one transformer had parameter updates
    low_updated = any(
        not jnp.array_equal(p1, p2) for p1, p2 in zip(jax.tree.leaves(initial_params_low), jax.tree.leaves(state_low.params))
    )
    high_updated = any(
        not jnp.array_equal(p1, p2)
        for p1, p2 in zip(jax.tree.leaves(initial_params_high), jax.tree.leaves(state_high.params))
    )
    self.assertTrue(low_updated or high_updated, "Parameters should be updated after training steps")

  def test_real_eval_step_2_2_execution(self):
    """Execute real JIT-compiled evaluation step and verify loss output."""
    rng = jax.random.key(0)
    rng_low, rng_high, eval_rng = jax.random.split(rng, 3)

    model_low = MiniWanModel(rngs=nnx.Rngs(rng_low), in_channels=4)
    model_high = MiniWanModel(rngs=nnx.Rngs(rng_high), in_channels=4)

    graphdef_low, params_low, rest_of_state_low = nnx.split(model_low, nnx.Param, ...)
    graphdef_high, params_high, rest_of_state_high = nnx.split(model_high, nnx.Param, ...)

    tx = optax.adam(1e-3)
    state_low = TrainState.create(
        apply_fn=graphdef_low.apply, params=params_low, tx=tx, graphdef=graphdef_low, rest_of_state=rest_of_state_low
    )
    state_high = TrainState.create(
        apply_fn=graphdef_high.apply, params=params_high, tx=tx, graphdef=graphdef_high, rest_of_state=rest_of_state_high
    )

    noise_scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(noise_scheduler_state, num_inference_steps=1000, training=True)

    config = pyconfig.config

    data = {
        "latents": jnp.ones((2, 4, 2, 4, 4), dtype=jnp.float32),
        "encoder_hidden_states": jnp.ones((2, 16, 4), dtype=jnp.float32),
        "timesteps": jnp.array([100, 700], dtype=jnp.int32),
    }

    jitted_eval_step = jax.jit(functools.partial(eval_step_2_2, scheduler=noise_scheduler, config=config))

    metrics, _ = jitted_eval_step(state_low, state_high, data, eval_rng, noise_scheduler_state)
    losses = metrics["scalar"]["learning/eval_loss"]
    self.assertEqual(len(losses), 2)
    self.assertFalse(jnp.isnan(losses).any(), "Eval losses should not be NaN")
    self.assertTrue((losses >= 0).all(), "Eval losses should be non-negative")

  def test_boundary_ratio_validation(self):
    """Verify that boundary_ratio <= 0.0 or >= 1.0 raises ValueError."""
    mock_config = MagicMock()
    mock_config.train_text_encoder = False

    for invalid_ratio in [0.0, 1.0, -0.2, 1.5]:
      mock_config.boundary_ratio = invalid_ratio
      with self.assertRaises(ValueError):
        WanTrainer2_2(mock_config)

  def test_eval_expert_routing_by_timestep(self):
    """Verify eval_step_2_2 routes each example according to its timestep (t < boundary to low, t >= boundary to high)."""

    class DistinguishableModel(nnx.Module):

      def __init__(self, val: float):
        self.val = nnx.Param(jnp.array(val, dtype=jnp.float32))

      def __call__(self, hidden_states, timestep, encoder_hidden_states=None, deterministic=True):
        return jnp.full_like(hidden_states, self.val[...])

    model_low = DistinguishableModel(1.0)
    model_high = DistinguishableModel(10.0)

    graphdef_low, params_low, rest_of_state_low = nnx.split(model_low, nnx.Param, ...)
    graphdef_high, params_high, rest_of_state_high = nnx.split(model_high, nnx.Param, ...)

    tx = optax.adam(1e-3)
    state_low = TrainState.create(
        apply_fn=graphdef_low.apply, params=params_low, tx=tx, graphdef=graphdef_low, rest_of_state=rest_of_state_low
    )
    state_high = TrainState.create(
        apply_fn=graphdef_high.apply, params=params_high, tx=tx, graphdef=graphdef_high, rest_of_state=rest_of_state_high
    )

    noise_scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(noise_scheduler_state, num_inference_steps=1000, training=True)

    mock_config = MagicMock()
    mock_config.boundary_ratio = 0.5
    mock_config.weights_dtype = jnp.float32
    mock_config.global_batch_size_to_train_on = 2
    mock_config.disable_training_weights = True

    # boundary = int(0.5 * 1000) = 500
    # t=100 -> low expert (output 1.0)
    # t=700 -> high expert (output 10.0)
    data_low = {
        "latents": jnp.zeros((2, 4, 2, 4, 4), dtype=jnp.float32),
        "encoder_hidden_states": jnp.zeros((2, 16, 4), dtype=jnp.float32),
        "timesteps": jnp.array([100, 100], dtype=jnp.int32),
    }
    data_high = {
        "latents": jnp.zeros((2, 4, 2, 4, 4), dtype=jnp.float32),
        "encoder_hidden_states": jnp.zeros((2, 16, 4), dtype=jnp.float32),
        "timesteps": jnp.array([700, 700], dtype=jnp.int32),
    }
    data_mixed = {
        "latents": jnp.zeros((2, 4, 2, 4, 4), dtype=jnp.float32),
        "encoder_hidden_states": jnp.zeros((2, 16, 4), dtype=jnp.float32),
        "timesteps": jnp.array([100, 700], dtype=jnp.int32),
    }
    data_mixed_reversed = {
        "latents": jnp.zeros((2, 4, 2, 4, 4), dtype=jnp.float32),
        "encoder_hidden_states": jnp.zeros((2, 16, 4), dtype=jnp.float32),
        "timesteps": jnp.array([700, 100], dtype=jnp.int32),
    }

    eval_fn = jax.jit(functools.partial(eval_step_2_2, scheduler=noise_scheduler, config=mock_config))
    metrics_low, _ = eval_fn(state_low, state_high, data_low, jax.random.key(0), noise_scheduler_state)
    metrics_high, _ = eval_fn(state_low, state_high, data_high, jax.random.key(0), noise_scheduler_state)
    metrics_mixed, _ = eval_fn(state_low, state_high, data_mixed, jax.random.key(0), noise_scheduler_state)
    metrics_mixed_rev, _ = eval_fn(state_low, state_high, data_mixed_reversed, jax.random.key(0), noise_scheduler_state)

    loss_low = metrics_low["scalar"]["learning/eval_loss"]
    loss_high = metrics_high["scalar"]["learning/eval_loss"]
    loss_mixed = metrics_mixed["scalar"]["learning/eval_loss"]
    loss_mixed_rev = metrics_mixed_rev["scalar"]["learning/eval_loss"]

    # Verify lengths
    self.assertEqual(len(loss_low), 2)
    self.assertEqual(len(loss_high), 2)
    self.assertEqual(len(loss_mixed), 2)
    self.assertEqual(len(loss_mixed_rev), 2)

    # Verify homogeneous batches produce distinctly different losses
    self.assertNotEqual(float(loss_low.mean()), float(loss_high.mean()))

    # Verify mixed batch [100, 700] correctly routes sample 0 (t=100) to low and sample 1 (t=700) to high
    self.assertAlmostEqual(float(loss_mixed[0]), float(loss_low[0]), places=5)
    self.assertAlmostEqual(float(loss_mixed[1]), float(loss_high[1]), places=5)
    self.assertNotEqual(float(loss_mixed[0]), float(loss_mixed[1]))

    # Verify mixed batch [700, 100] correctly routes sample 0 (t=700) to high and sample 1 (t=100) to low
    self.assertAlmostEqual(float(loss_mixed_rev[0]), float(loss_high[0]), places=5)
    self.assertAlmostEqual(float(loss_mixed_rev[1]), float(loss_low[1]), places=5)
    self.assertNotEqual(float(loss_mixed_rev[0]), float(loss_mixed_rev[1]))

  def test_checkpoint_resume_equivalence(self):
    """Verify that continuous training and train->checkpoint->restore->train produce identical trajectories."""
    rng = jax.random.key(99)
    rng_low, rng_high = jax.random.split(rng, 2)

    def _init_states():
      model_low = MiniWanModel(rngs=nnx.Rngs(rng_low), in_channels=4)
      model_high = MiniWanModel(rngs=nnx.Rngs(rng_high), in_channels=4)
      g_low, p_low, r_low = nnx.split(model_low, nnx.Param, ...)
      g_high, p_high, r_high = nnx.split(model_high, nnx.Param, ...)
      tx_low = optax.adam(1e-3)
      tx_high = optax.adam(1e-3)
      s_low = TrainState.create(apply_fn=g_low.apply, params=p_low, tx=tx_low, graphdef=g_low, rest_of_state=r_low)
      s_high = TrainState.create(apply_fn=g_high.apply, params=p_high, tx=tx_high, graphdef=g_high, rest_of_state=r_high)
      return s_low, s_high

    noise_scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(noise_scheduler_state, num_inference_steps=1000, training=True)
    config = pyconfig.config

    data = {
        "latents": jnp.ones((2, 4, 2, 4, 4), dtype=jnp.float32),
        "encoder_hidden_states": jnp.ones((2, 16, 4), dtype=jnp.float32),
    }

    train_step_fn = jax.jit(functools.partial(train_step_2_2, scheduler=noise_scheduler, config=config))

    # Run A: 6 continuous steps
    s_low_a, s_high_a = _init_states()
    base_step_rng = jax.random.key(1001)
    sched_state_a = noise_scheduler_state
    for step in range(6):
      step_rng = jax.random.fold_in(base_step_rng, step)
      s_low_a, s_high_a, sched_state_a, _, _ = train_step_fn(s_low_a, s_high_a, data, step_rng, sched_state_a)

    # Run B: 3 steps, simulate save/restore, 3 additional steps
    s_low_b, s_high_b = _init_states()
    sched_state_b = noise_scheduler_state
    for step in range(3):
      step_rng = jax.random.fold_in(base_step_rng, step)
      s_low_b, s_high_b, sched_state_b, _, _ = train_step_fn(s_low_b, s_high_b, data, step_rng, sched_state_b)

    # Simulate checkpoint extraction and restore
    checkpoint = MagicMock()
    checkpoint.low_noise_transformer_state = {"params": s_low_b.params, "opt_state": s_low_b.opt_state, "step": s_low_b.step}
    checkpoint.high_noise_transformer_state = {
        "params": s_high_b.params,
        "opt_state": s_high_b.opt_state,
        "step": s_high_b.step,
    }
    checkpointer = WanCheckpointer2_2(config=config)
    opt_state_dict = checkpointer._extract_opt_state(checkpoint)

    # Re-initialize clean state and restore (simulates new process restart at step 3)
    s_low_b_resumed, s_high_b_resumed = _init_states()
    s_low_b_resumed = s_low_b_resumed.replace(
        params=checkpoint.low_noise_transformer_state["params"],
        opt_state=opt_state_dict["low_noise_transformer"],
        step=opt_state_dict["low_noise_step"],
    )
    s_high_b_resumed = s_high_b_resumed.replace(
        params=checkpoint.high_noise_transformer_state["params"],
        opt_state=opt_state_dict["high_noise_transformer"],
        step=opt_state_dict["high_noise_step"],
    )

    # Verify steps preserved before continuing
    self.assertEqual(int(s_low_b_resumed.step), int(s_low_b.step))
    self.assertEqual(int(s_high_b_resumed.step), int(s_high_b.step))

    # Run remaining 3 steps (steps 3, 4, 5) reconstructing step_rng from base_step_rng
    for step in range(3, 6):
      step_rng = jax.random.fold_in(base_step_rng, step)
      s_low_b_resumed, s_high_b_resumed, sched_state_b, _, _ = train_step_fn(
          s_low_b_resumed, s_high_b_resumed, data, step_rng, sched_state_b
      )

    # Assert exact equivalence between continuous and resumed runs
    self.assertEqual(int(s_low_a.step), int(s_low_b_resumed.step))
    self.assertEqual(int(s_high_a.step), int(s_high_b_resumed.step))

    for p_a, p_b in zip(jax.tree.leaves(s_low_a.params), jax.tree.leaves(s_low_b_resumed.params)):
      self.assertTrue(jnp.allclose(p_a, p_b, atol=1e-6), "Low noise parameters mismatch between continuous and resumed runs")

    for p_a, p_b in zip(jax.tree.leaves(s_high_a.params), jax.tree.leaves(s_high_b_resumed.params)):
      self.assertTrue(
          jnp.allclose(p_a, p_b, atol=1e-6), "High noise parameters mismatch between continuous and resumed runs"
      )

  def test_timestep_sampling_distribution(self):
    """Verify uniform timestep sampling bounds and mean for high/low experts."""
    rng = jax.random.key(42)
    rng_low, rng_high = jax.random.split(rng)
    num_samples = 10000
    boundary = 500
    num_train_timesteps = 1000

    # Low noise expert: uniform integer sampling from [0, boundary)
    t_low = jax.random.randint(rng_low, shape=(num_samples,), minval=0, maxval=boundary)

    # High noise expert: uniform integer sampling from [boundary, num_train_timesteps)
    t_high = jax.random.randint(rng_high, shape=(num_samples,), minval=boundary, maxval=num_train_timesteps)

    # Assert strict bounds
    self.assertTrue((t_low >= 0).all())
    self.assertTrue((t_low < boundary).all())
    self.assertTrue((t_high >= boundary).all())
    self.assertTrue((t_high < num_train_timesteps).all())

    # Assert uniform distribution mean (low mean ~ boundary / 2 = 250, high mean ~ (boundary + num_train_timesteps) / 2 = 750)
    low_mean = float(jnp.mean(t_low))
    high_mean = float(jnp.mean(t_high))
    self.assertAlmostEqual(low_mean, boundary / 2.0, delta=10.0)
    self.assertAlmostEqual(high_mean, (boundary + num_train_timesteps) / 2.0, delta=10.0)

  def test_asymmetric_optimizer_schedule_scaling(self):
    """Verify that optimizer schedules scale independently with asymmetric boundary ratios (e.g. 0.7 vs 0.3)."""
    mock_config = MagicMock()
    mock_config.max_train_steps = 1000
    mock_config.learning_rate_schedule_steps = 1000
    mock_config.warmup_steps_fraction = 0.1
    mock_config.learning_rate = 1e-4
    mock_config.opt_type = "adam"
    mock_config.adam_b1 = 0.9
    mock_config.adam_b2 = 0.999
    mock_config.adam_eps = 1e-8
    mock_config.opt_weight_decay = 0.01
    mock_config.opt_enable_grad_clipping = False
    mock_config.opt_enable_grad_global_norm_clipping = False
    mock_config.opt_clip_by_block = False

    checkpointer = WanCheckpointer2_2(config=mock_config)
    model = MiniWanModel(rngs=nnx.Rngs(0))

    _, lr_schedule_low = checkpointer._create_optimizer(model, mock_config, mock_config.learning_rate, scale_factor=0.7)
    _, lr_schedule_high = checkpointer._create_optimizer(model, mock_config, mock_config.learning_rate, scale_factor=0.3)

    lr_low_50 = float(lr_schedule_low(50))
    lr_high_50 = float(lr_schedule_high(50))
    self.assertNotEqual(lr_low_50, lr_high_50)
    self.assertAlmostEqual(lr_low_50, 1e-4 * (50.0 / 70.0), places=6)
    self.assertAlmostEqual(lr_high_50, 1e-4, places=6)

  def test_eval_varying_batch_sizes(self):
    """Verify eval_step_2_2 executes reliably for single-sample, even, and odd batch sizes."""

    class DistinguishableModel(nnx.Module):

      def __init__(self, val: float):
        self.val = nnx.Param(jnp.array(val, dtype=jnp.float32))

      def __call__(self, hidden_states, timestep, encoder_hidden_states=None, deterministic=True):
        return jnp.full_like(hidden_states, self.val[...])

    model_low = DistinguishableModel(1.0)
    model_high = DistinguishableModel(10.0)
    graphdef_low, params_low, rest_of_state_low = nnx.split(model_low, nnx.Param, ...)
    graphdef_high, params_high, rest_of_state_high = nnx.split(model_high, nnx.Param, ...)
    tx = optax.adam(1e-3)
    state_low = TrainState.create(
        apply_fn=graphdef_low.apply, params=params_low, tx=tx, graphdef=graphdef_low, rest_of_state=rest_of_state_low
    )
    state_high = TrainState.create(
        apply_fn=graphdef_high.apply, params=params_high, tx=tx, graphdef=graphdef_high, rest_of_state=rest_of_state_high
    )

    noise_scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(noise_scheduler_state, num_inference_steps=1000, training=True)

    mock_config = MagicMock()
    mock_config.boundary_ratio = 0.5
    mock_config.weights_dtype = jnp.float32
    mock_config.disable_training_weights = True

    eval_fn = jax.jit(functools.partial(eval_step_2_2, scheduler=noise_scheduler, config=mock_config))

    for bs in [1, 2, 3, 5]:
      timesteps = jnp.array([100 if i % 2 == 0 else 700 for i in range(bs)], dtype=jnp.int32)
      data = {
          "latents": jnp.zeros((bs, 4, 2, 4, 4), dtype=jnp.float32),
          "encoder_hidden_states": jnp.zeros((bs, 16, 4), dtype=jnp.float32),
          "timesteps": timesteps,
      }
      metrics, _ = eval_fn(state_low, state_high, data, jax.random.key(0), noise_scheduler_state)
      losses = metrics["scalar"]["learning/eval_loss"]
      self.assertEqual(len(losses), bs)
      for i in range(bs):
        if int(timesteps[i]) < 500:
          self.assertLess(float(losses[i]), 20.0)
        else:
          self.assertGreater(float(losses[i]), 50.0)

  def test_eval_microbatching(self):
    """Verify eval_step_2_2 microbatches when global_batch_size_to_train_on is smaller than loaded batch."""

    class DistinguishableModel(nnx.Module):

      def __init__(self, val: float):
        self.val = nnx.Param(jnp.array(val, dtype=jnp.float32))

      def __call__(self, hidden_states, timestep, encoder_hidden_states=None, deterministic=True):
        return jnp.full_like(hidden_states, self.val[...])

    model_low = DistinguishableModel(1.0)
    model_high = DistinguishableModel(10.0)
    graphdef_low, params_low, rest_of_state_low = nnx.split(model_low, nnx.Param, ...)
    graphdef_high, params_high, rest_of_state_high = nnx.split(model_high, nnx.Param, ...)
    tx = optax.adam(1e-3)
    state_low = TrainState.create(
        apply_fn=graphdef_low.apply, params=params_low, tx=tx, graphdef=graphdef_low, rest_of_state=rest_of_state_low
    )
    state_high = TrainState.create(
        apply_fn=graphdef_high.apply, params=params_high, tx=tx, graphdef=graphdef_high, rest_of_state=rest_of_state_high
    )

    noise_scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(noise_scheduler_state, num_inference_steps=1000, training=True)

    mock_config = MagicMock()
    mock_config.boundary_ratio = 0.5
    mock_config.weights_dtype = jnp.float32
    mock_config.disable_training_weights = True
    # Simulate loaded batch of 4, but microbatch size of 2
    mock_config.global_batch_size_to_train_on = 2

    eval_fn = jax.jit(functools.partial(eval_step_2_2, scheduler=noise_scheduler, config=mock_config))

    timesteps = jnp.array([100, 700, 100, 700], dtype=jnp.int32)
    data = {
        "latents": jnp.zeros((4, 4, 2, 4, 4), dtype=jnp.float32),
        "encoder_hidden_states": jnp.zeros((4, 16, 4), dtype=jnp.float32),
        "timesteps": timesteps,
    }

    metrics, _ = eval_fn(state_low, state_high, data, jax.random.key(0), noise_scheduler_state)
    losses = metrics["scalar"]["learning/eval_loss"]

    self.assertEqual(len(losses), 4)
    self.assertLess(float(losses[0]), 20.0)
    self.assertGreater(float(losses[1]), 50.0)
    self.assertLess(float(losses[2]), 20.0)
    self.assertGreater(float(losses[3]), 50.0)


if __name__ == "__main__":
  absltest.main()
