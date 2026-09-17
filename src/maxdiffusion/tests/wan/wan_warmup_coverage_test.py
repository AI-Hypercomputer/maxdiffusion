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

"""Warmup must compile the loop's forward pass for both WAN 2.2 transformers.

With flow_shift=12 the 2-step warmup schedule is t=[999, 923], both above the
875 (T2V) and 900 (I2V) boundaries, so the denoise loop alone never reaches the
low-noise transformer. run_inference_2_2 and run_inference_2_2_i2v must compile
it explicitly during warmup, without executing either transformer.
"""

import contextlib
import os
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxdiffusion import aot_cache
from maxdiffusion.pipelines.wan import wan_denoise_utils
from maxdiffusion.pipelines.wan import wan_pipeline_2_2 as p22
from maxdiffusion.pipelines.wan import wan_pipeline_i2v_2p2 as pi2v


class _Step:

  def __init__(self, latents, state):
    self._out = (latents, state)

  def to_tuple(self):
    return self._out


class _WarmupCoverageMixin:
  """Shared by T2V and I2V so the two pipelines are held to the same warmup contract."""

  module = None

  def _invoke(self, scheduler, scheduler_state, guidance_scale_low, guidance_scale_high, num_inference_steps):
    raise NotImplementedError

  def _run(
      self,
      timesteps,
      in_warmup,
      guidance_scale_low=3.0,
      guidance_scale_high=4.0,
      max_inflight=None,
      blocked=None,
  ):
    calls = []
    full_cfg_calls = []

    def fake_forward(graphdef, *args, **kwargs):
      calls.append(graphdef)
      return jnp.zeros((1, 4, 2, 4, 4))

    def fake_full_cfg(graphdef, *args, **kwargs):
      full_cfg_calls.append(graphdef)
      return jnp.zeros((1, 4, 2, 4, 4))

    def fake_block_until_ready(x):
      if blocked is not None:
        blocked.append(x)
      return x

    merged = mock.MagicMock()
    merged.rope.return_value = jnp.zeros((1,))
    scheduler = mock.MagicMock()
    scheduler.step.side_effect = lambda st, noise, t, latents: _Step(latents, st)
    scheduler_state = mock.MagicMock()
    scheduler_state.timesteps = timesteps
    real_execution = mock.MagicMock()

    m = self.module
    env = {} if max_inflight is None else {"MAXD_QUEUE_MAX_INFLIGHT": str(max_inflight)}
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.dict(os.environ, env))
      stack.enter_context(mock.patch.object(m, "transformer_forward_pass", side_effect=fake_forward))
      stack.enter_context(mock.patch.object(m, "transformer_forward_pass_full_cfg", side_effect=fake_full_cfg))
      stack.enter_context(mock.patch.object(m.nnx, "merge", return_value=merged))
      stack.enter_context(mock.patch.object(m.aot_cache, "in_warmup", return_value=in_warmup))
      stack.enter_context(mock.patch.object(m.aot_cache, "real_execution", real_execution))
      stack.enter_context(mock.patch.object(m.jax, "block_until_ready", side_effect=fake_block_until_ready))
      self._invoke(scheduler, scheduler_state, guidance_scale_low, guidance_scale_high, len(timesteps))
    self.assertEqual(full_cfg_calls, [], "the loop must use transformer_forward_pass, not full_cfg")
    real_execution.assert_not_called()  # warmup compiles only; nothing executes for real
    return calls

  def test_warmup_schedule_all_high_still_compiles_low(self):
    blocked = []
    calls = self._run([999, 923], in_warmup=True, blocked=blocked)
    self.assertEqual(calls, ["high", "low", "high", "high"])
    self.assertEqual(blocked, [], "warmup must not wait on device work")

  def test_warmup_schedule_crossing_boundary_compiles_both(self):
    self.assertEqual(self._run([999, 800], in_warmup=True), ["high", "low", "high", "low"])

  def test_warmup_no_cfg_compiles_both(self):
    calls = self._run([999, 923], in_warmup=True, guidance_scale_low=1.0, guidance_scale_high=1.0)
    self.assertEqual(calls, ["high", "low", "high", "high"])

  def test_real_run_is_unchanged(self):
    calls = self._run([999, 923], in_warmup=False)
    self.assertEqual(calls, ["high", "high"])

  def test_loop_bounds_inflight_steps(self):
    blocked = []
    self._run([999, 923, 800, 700], in_warmup=False, max_inflight=1, blocked=blocked)
    self.assertEqual(len(blocked), 3, "with depth 1, each step after the first must wait on its predecessor")
    blocked.clear()
    self._run([999, 923, 800, 700], in_warmup=False, max_inflight=0, blocked=blocked)
    self.assertEqual(blocked, [], "depth 0 disables the bound")


class WarmupCoversBothTransformersTest(_WarmupCoverageMixin, unittest.TestCase):
  """T2V: run_inference_2_2."""

  module = p22

  def _invoke(self, scheduler, scheduler_state, guidance_scale_low, guidance_scale_high, num_inference_steps):
    p22.run_inference_2_2(
        low_noise_graphdef="low",
        low_noise_state=None,
        low_noise_rest=None,
        high_noise_graphdef="high",
        high_noise_state=None,
        high_noise_rest=None,
        latents=jnp.zeros((1, 4, 2, 4, 4)),
        prompt_embeds=jnp.zeros((1, 8, 16)),
        negative_prompt_embeds=jnp.zeros((1, 8, 16)),
        guidance_scale_low=guidance_scale_low,
        guidance_scale_high=guidance_scale_high,
        boundary=875,
        num_inference_steps=num_inference_steps,
        scheduler=scheduler,
        scheduler_state=scheduler_state,
        config=None,
    )


class I2VWarmupCoversBothTransformersTest(_WarmupCoverageMixin, unittest.TestCase):
  """I2V: run_inference_2_2_i2v (latents and condition are BFHWC)."""

  module = pi2v

  def _invoke(self, scheduler, scheduler_state, guidance_scale_low, guidance_scale_high, num_inference_steps):
    pi2v.run_inference_2_2_i2v(
        low_noise_graphdef="low",
        low_noise_state=None,
        low_noise_rest=None,
        high_noise_graphdef="high",
        high_noise_state=None,
        high_noise_rest=None,
        latents=jnp.zeros((1, 2, 4, 4, 4)),
        condition=jnp.zeros((1, 2, 4, 4, 5)),
        prompt_embeds=jnp.zeros((1, 8, 16)),
        negative_prompt_embeds=jnp.zeros((1, 8, 16)),
        image_embeds=jnp.zeros((1, 4, 16)),
        guidance_scale_low=guidance_scale_low,
        guidance_scale_high=guidance_scale_high,
        boundary=900,
        num_inference_steps=num_inference_steps,
        scheduler=scheduler,
        scheduler_state=scheduler_state,
        config=None,
    )


class DenoiseUtilsTest(unittest.TestCase):

  def test_inflight_window_blocks_on_oldest_beyond_depth(self):
    with mock.patch.object(wan_denoise_utils.jax, "block_until_ready") as block:
      window = wan_denoise_utils.InflightWindow(depth=2)
      for i in range(5):
        window.push(i)
    self.assertEqual([c.args[0] for c in block.call_args_list], [0, 1, 2])

  def test_inflight_window_depth_from_env_and_zero_disables(self):
    with mock.patch.dict(os.environ, {"MAXD_QUEUE_MAX_INFLIGHT": "3"}):
      self.assertEqual(wan_denoise_utils.InflightWindow().depth, 3)
    with mock.patch.dict(os.environ):
      os.environ.pop("MAXD_QUEUE_MAX_INFLIGHT", None)
      self.assertEqual(wan_denoise_utils.InflightWindow().depth, 4)
    with mock.patch.object(wan_denoise_utils.jax, "block_until_ready") as block:
      window = wan_denoise_utils.InflightWindow(depth=0)
      for i in range(5):
        window.push(i)
    block.assert_not_called()
    self.assertFalse(window._inflight, "a disabled window must not hold step outputs")

  def test_compile_experts_is_a_noop_outside_warmup(self):
    branch = mock.MagicMock()
    with mock.patch.object(wan_denoise_utils.aot_cache, "in_warmup", return_value=False):
      wan_denoise_utils.compile_experts((branch, branch), "operands")
    branch.assert_not_called()

  def test_compile_experts_compiles_one_shared_executable_without_executing(self):
    """With the real AOT cache: both experts compile once, nothing runs, and the loop reuses the executable."""
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.addCleanup(aot_cache.install, "", {}, None)
    aot_cache.install(tmp.name, meta={"test": "compile_experts"}, mesh=Mesh(np.array(jax.devices()[:1]), ("d",)))

    @aot_cache.cached_jit
    def forward(x, guidance_scale):
      return x * guidance_scale

    outs = []

    def expert(guidance_scale):
      def run(operands):
        outs.append(forward(operands, guidance_scale))
        return outs[-1]

      return run

    x = jnp.arange(1, 5, dtype=jnp.bfloat16)
    high, low = expert(4.0), expert(3.0)
    with aot_cache.warmup_mode():
      wan_denoise_utils.compile_experts((high, low), x)

    self.assertEqual(len(forward._compiled), 1, "guidance 4.0 and 3.0 must share one executable")
    self.assertEqual(len(outs), 2)
    for out in outs:  # compile only: warmup hands back zeros instead of executing
      np.testing.assert_array_equal(np.asarray(out, np.float32), np.zeros(4))
    # The denoise loop then runs each expert on that executable with its own guidance value.
    np.testing.assert_array_equal(np.asarray(high(x), np.float32), np.asarray(x * 4.0, np.float32))
    np.testing.assert_array_equal(np.asarray(low(x), np.float32), np.asarray(x * 3.0, np.float32))
    self.assertEqual(len(forward._compiled), 1, "the loop must not compile a second executable")


if __name__ == "__main__":
  unittest.main()
