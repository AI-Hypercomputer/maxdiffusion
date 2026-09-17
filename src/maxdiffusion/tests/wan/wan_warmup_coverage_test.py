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
875 boundary, so the denoise loop alone never reaches the low-noise
transformer. run_inference_2_2 must compile it explicitly during warmup.
"""

import contextlib
import unittest
from unittest import mock

import jax.numpy as jnp

from maxdiffusion.pipelines.wan import wan_pipeline_2_2 as p22


class _Step:

  def __init__(self, latents, state):
    self._out = (latents, state)

  def to_tuple(self):
    return self._out


class WarmupCoversBothTransformersTest(unittest.TestCase):

  def _run(self, timesteps, in_warmup, guidance_scale_low=3.0, guidance_scale_high=4.0):
    calls = []
    full_cfg_calls = []

    def fake_forward(graphdef, *args, **kwargs):
      calls.append(graphdef)
      return jnp.zeros((1, 4, 2, 4, 4))

    def fake_full_cfg(graphdef, *args, **kwargs):
      full_cfg_calls.append(graphdef)
      return jnp.zeros((1, 4, 2, 4, 4))

    merged = mock.MagicMock()
    merged.rope.return_value = jnp.zeros((1,))
    scheduler = mock.MagicMock()
    scheduler.step.side_effect = lambda st, noise, t, latents: _Step(latents, st)
    scheduler_state = mock.MagicMock()
    scheduler_state.timesteps = timesteps

    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.object(p22, "transformer_forward_pass", side_effect=fake_forward))
      stack.enter_context(mock.patch.object(p22, "transformer_forward_pass_full_cfg", side_effect=fake_full_cfg))
      stack.enter_context(mock.patch.object(p22.nnx, "merge", return_value=merged))
      stack.enter_context(mock.patch.object(p22.aot_cache, "in_warmup", return_value=in_warmup))
      stack.enter_context(mock.patch.object(p22.aot_cache, "real_execution", contextlib.nullcontext))
      stack.enter_context(mock.patch.object(p22.jax, "block_until_ready", side_effect=lambda x: x))
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
          num_inference_steps=len(timesteps),
          scheduler=scheduler,
          scheduler_state=scheduler_state,
          config=None,
      )
    self.assertEqual(full_cfg_calls, [], "warmup priming must use transformer_forward_pass, not full_cfg")
    return calls

  def test_warmup_schedule_all_high_still_compiles_low(self):
    calls = self._run([999, 923], in_warmup=True)
    self.assertIn("low", calls)
    self.assertEqual(calls[:2], ["high", "low"])
    self.assertEqual(calls.count("high"), 3)

  def test_warmup_schedule_crossing_boundary_primes_and_runs_both(self):
    calls = self._run([999, 800], in_warmup=True)
    self.assertEqual(calls, ["high", "low", "high", "low"])

  def test_warmup_no_cfg_schedule_all_high_still_compiles_low(self):
    calls = self._run([999, 923], in_warmup=True, guidance_scale_low=1.0, guidance_scale_high=1.0)
    self.assertIn("low", calls)
    self.assertEqual(calls.count("high"), 2)

  def test_real_run_is_unchanged(self):
    calls = self._run([999, 923], in_warmup=False)
    self.assertEqual(calls, ["high", "high"])


if __name__ == "__main__":
  unittest.main()
