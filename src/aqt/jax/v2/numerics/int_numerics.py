# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerics for int8, int4, binary and other integer types."""

from typing import Any, Optional
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
from jax import lax
import jax.numpy as jnp


@utils.flax_slots_kw_only_dataclass
class IntNumerics(numerics.AqtNumerics):
  """Numerics for int8, int4, binary, etc."""

  bits: int
  preserve_zero: bool
  # false = map max val on the end of the last bucket
  # true = map max val on the middle of the last
  preserve_max_val: bool
  clip: bool
  clip_gradient: bool
  round: bool
  noise_fn: Optional[stochastic_rounding.NoiseFn]
  dtype: Optional[Any] = None

  # pylint: disable=line-too-long
  # Verifying the correctness of these functions amounts to verifying this table:
  # if preserve_zero == F, zero might be rounded either to [-1, 0] bucket or to [0, 1] bucket
  # preserve_zero, preserve_max_val, 8b, 2b, 1b
  # F, F, 128.0, 2.0, 1.0  # bucket count is even; map onto the far edge of the last bucket
  # F, T, 127.5, 1.5, 0.5  # bucket count is even; map onto the center of the last bucket
  # T, F, 127.5, 1.5, 0.5  # bucket count is odd;  map onto the far edge of the last bucket
  # T, T, 127.0, 1.0, 0.0  # bucket count is odd;  map onto the center of the last bucket
  # pylint: enable=line-too-long

  def get_edge_of_last_int_bucket(self):
    ret = 2.0 ** (self.bits - 1)
    if self.preserve_zero:
      # Lose one bucket.
      ret -= 0.5
    return ret

  def get_center_of_last_int_bucket(self):
    return self.get_edge_of_last_int_bucket() - 0.5

  def abs_val_mapped_to(self):
    if self.preserve_max_val:
      return self.get_center_of_last_int_bucket()
    else:
      return self.get_edge_of_last_int_bucket()

  def _get_fwd_clip_bound(self):
    # If we are not rounding, we just clip to bucket edges.
    fwd_clip_bound = self.get_edge_of_last_int_bucket()
    # If, after clip, we are rounding, we need to make sure that
    # we won't round values at the clip_bound away to the
    # non-existing bucket.
    if self.round:
      # Reducing fwd_clip_bound by any value in (0.0, 1.0) is correct.
      fwd_clip_bound -= 0.5
    return fwd_clip_bound

  def get_dtype(self):
    return self.dtype

  def vjp_fwd(self, x, context):
    """Forward pass."""
    res = (x,)
    input_dtype = x.dtype
    assert self.bits <= 22, 'Too many bits, float32 has less precision.'

    # Maybe noise
    if self.noise_fn:
      assert context.key is not None, (
          'noise_fn is set, requestic stochastic rounding, but RNG was not '
          'passed in Context.key'
      )
      x = (x + self.noise_fn(x.shape, context.key)).astype(input_dtype)

    if self.clip:
      fwd_clip_bound = self._get_fwd_clip_bound()
      x = jnp.clip(x, -fwd_clip_bound, fwd_clip_bound)

    # Maybe round
    if self.round:
      # TODO(lew): Have bucket centers at 2*k + 1, not at halves.
      round_to_halves = not self.preserve_zero
      if round_to_halves:
        x = jnp.floor(x) + 0.5
      else:
        x = lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)

    # Maybe cast: return dtype is either int or the input dtype
    dtype = self.get_dtype()
    x = x.astype(dtype if dtype is not None else input_dtype)
    return x, res

  def vjp_bwd(self, res, grad):
    # Gradient of the clip function.
    # For boundary values we will have full gradient.
    # When using abs(max(x)) scaling, x is always in the interior, and the
    # gradient clip is always 1. So, we can always set clip_gradient to false.
    # However, other types of scaling may result in x being outside (i.e., there
    # is clipping). In that case it may be desirable to make the gradient zero.
    ret = grad
    if self.clip_gradient:
      (x,) = res
      clip_bound = self._get_fwd_clip_bound()
      ret *= (-clip_bound <= x) * (x <= clip_bound)
    return (ret, None)
