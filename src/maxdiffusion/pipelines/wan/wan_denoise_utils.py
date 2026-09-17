# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warmup compilation and dispatch-queue bounding shared by the Wan 2.2 denoise loops.

T2V (``wan_pipeline_2_2``) and I2V (``wan_pipeline_i2v_2p2``) drive the same
two-expert Python loop and need the same warmup and queueing treatment. It
lives here so the two pipelines cannot drift apart.
"""

import collections
import os
from typing import Any, Callable, Optional, Sequence

import jax

from maxdiffusion import aot_cache


def compile_experts(branches: Sequence[Callable[[Any], Any]], operands: Any) -> None:
  """During warmup, compiles every expert's forward pass without executing it.

  Warmup runs only a couple of denoising steps, and the scheduler can put all
  of them on the high-noise expert (with flow_shift=12 a 2-step schedule is
  t=[999, 923], both above the boundary). Without this, the low-noise expert
  would compile during the first real generation. When the experts share a
  signature (the usual case), the second call is an in-memory cache hit.

  Weights are deliberately not primed with a real forward pass. On the
  current stack priming added 5-7 s to every warmup, while the first
  generation ran within 0.5 s of steady state without it (v6e-8 T2V denoise:
  125.7 s vs 125.6 s primed; tpu7x-8 I2V: 93.3 s either way from a warm AOT
  cache, 93.8 s vs 93.4 s after a cold compile).

  No-op outside ``aot_cache.warmup_mode()``.

  Args:
    branches: One callable per expert. Each takes ``operands`` and dispatches
      the same cached forward pass (same signature) as the denoise loop.
    operands: The branch operands, in the same layout as for ``jax.lax.cond``.
  """
  if not aot_cache.in_warmup():
    return
  for branch in branches:
    branch(operands)


class InflightWindow:
  """Bounds how many dispatched denoise steps are queued on the device.

  The Python loop dispatches asynchronously and otherwise queues every
  remaining step. An unbounded queue once ran 40-step T2V ~20% slower (3.30 vs
  2.69 s/step). On the current stack the bound measures neutral (v6e-8 T2V,
  tpu7x-8 I2V), and it stays as a cheap guard. Blocking on the output of step
  (N - depth) keeps at most ``depth`` steps queued while the pipeline stays
  full; a periodic FULL drain would empty the pipe and cost a bubble per drain.

  ``MAXD_QUEUE_MAX_INFLIGHT`` sets the default depth (4); 0 disables the bound.
  """

  def __init__(self, depth: Optional[int] = None):
    self.depth = int(os.environ.get("MAXD_QUEUE_MAX_INFLIGHT", "4")) if depth is None else depth
    self._inflight = collections.deque()

  def push(self, step_output: Any) -> None:
    """Records a dispatched step's output, blocking on the oldest once more than ``depth`` are queued."""
    if self.depth <= 0:
      return
    self._inflight.append(step_output)
    if len(self._inflight) > self.depth:
      jax.block_until_ready(self._inflight.popleft())
