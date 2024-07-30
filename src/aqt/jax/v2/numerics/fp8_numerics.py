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
"""Numerics for fp8."""

from typing import Literal, TypeAlias
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
import jax
import jax.numpy as jnp


FP8Dtype: TypeAlias = Literal['e4m3', 'e5m2']
fp8_map = {'e4m3': jnp.float8_e4m3fn, 'e5m2': jnp.float8_e5m2}


def fp_mantissa_round(x, mantissa_bits, key: jax.Array):
  """FP stochastic rounding for a given mantissa."""
  input_dtype = x.dtype
  total_bits = jnp.finfo(input_dtype).bits
  # (sign, exp, mantissa) = (1,8,x)
  msg = f'Unsupported dtype for sthochastic rounding: {input_dtype}.'
  assert input_dtype == jnp.bfloat16 or input_dtype == jnp.float32, msg
  # bf16 bit pattern: seeeeeeeemmmmmmm
  #                   7654321076543210
  mantissa = jnp.finfo(input_dtype).nmant
  bits_dtype = jnp.uint16 if total_bits == 16 else jnp.uint32
  assert mantissa_bits <= mantissa, 'bf16 mantissa bits'
  noise_bit_count = mantissa - mantissa_bits

  noise_mask = bits_dtype((1 << noise_bit_count) - 1)

  # TODO(lew): We can use smaller dtype
  x = jax.lax.bitcast_convert_type(x, bits_dtype)  # adds noise to mantissa
  noise = jax.random.bits(key, x.shape, bits_dtype)
  noise = jax.lax.convert_element_type(noise, bits_dtype) & noise_mask
  # This noise add might overflow up to sign bit if x(bf16) has max exponent.
  # In bf16 this happens if x=nan or x=inf.
  x = x + noise

  # TODO(lew): Is bitwise_and needed? Maybe round_to_nearest_even is ok.
  x = jax.lax.bitwise_and(x, ~noise_mask)  # zero-out lowest bits in mantissa
  x = jax.lax.bitcast_convert_type(x, input_dtype)

  return x


@utils.flax_slots_kw_only_dataclass
class Fp8Numerics(numerics.AqtNumerics):
  """Numerics for fp8."""

  # Storage type.
  dtype: Literal[jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.bfloat16]

  # Requested rounding precision.
  exponent_bits: int = 4
  mantissa_bits: int = 3
  stochastic_rounding: bool = utils.static_field(default=False)

  def _get_edge_of_last_fp8_bucket(self):
    return jnp.finfo(self.dtype).max.astype(jnp.bfloat16)

  def get_dtype(self):
    return self.dtype

  def abs_val_mapped_to(self):
    return self._get_edge_of_last_fp8_bucket()

  def vjp_fwd(self, x, context):
    match self.dtype:
      case jnp.float8_e4m3fn:
        assert (self.exponent_bits, self.mantissa_bits) == (4, 3)
      case jnp.float8_e5m2:
        assert (self.exponent_bits, self.mantissa_bits) == (5, 2)
      case jnp.bfloat16:
        assert self.exponent_bits <= 8
        assert self.mantissa_bits <= 7
      case _:
        assert False, f'Unsupported dtype: {self.dtype}'

    res = (x,)
    if not (
        (self.exponent_bits == 4 and self.mantissa_bits == 3)
        or (self.exponent_bits == 5 and self.mantissa_bits == 2)
    ):
      raise ValueError(
          '(exponent_bits, mantissa_bits) can only be (4,3) or (5,2) but was '
          f'({self.exponent_bits}, {self.mantissa_bits})'
      )

    if self.stochastic_rounding:
      msg = 'stochastic_rounding requires PRNGKey in Context'
      assert context.key is not None, msg
      x = fp_mantissa_round(x, self.mantissa_bits, context.key)

    # TODO(lew):
    #   - Is cliping good enough for exponent rounding?
    #   - Is it needed? Can we do clipping more efficiently?
    fwd_clip_bound = self._get_edge_of_last_fp8_bucket()
    x = jnp.clip(x, -1 * fwd_clip_bound, fwd_clip_bound)
    # TODO(lew): We can round more efficiently if stochastic_rounding == True.
    x = round_to_nearest_even(x, self.dtype)

    return x, res

  def vjp_bwd(self, res, grad):
    # This is gradient of clip.
    # For boundary values we will have full gradient.
    # We might use something like this for calibrations other than abs(max(x))
    # (x,) = res
    # ret = (x <= edge_of_last_bucket) * (x >= -edge_of_last_bucket) * grad
    del res
    ret = grad
    return (ret, None)


def round_to_nearest_even(x: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
  x = x.astype(dtype)
  # TODO(lew): Is this rounding utilizing subnormals range fully?
  # bitcast_convert to uint8 to avoid allow_excess_precision set in XLA
  x = jax.lax.bitcast_convert_type(x, jnp.uint8)
  x = jax.lax.bitcast_convert_type(x, dtype)
  return x
