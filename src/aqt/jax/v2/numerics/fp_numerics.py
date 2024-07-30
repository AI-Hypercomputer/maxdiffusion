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

from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
import jax
import jax.numpy as jnp


@utils.flax_slots_kw_only_dataclass
class FpNumericsConfig:
  nexp: int
  minexp: int
  nmant: int
  has_subnormals: bool
  has_two_nan: bool
  has_naninf: bool

e3m0 = FpNumericsConfig(
    nexp=3,
    minexp=0,
    nmant=0,
    has_subnormals=False,
    has_two_nan=False,
    has_naninf=False,
)

e2m1 = FpNumericsConfig(
    nexp=2,
    minexp=0,
    nmant=1,
    has_subnormals=False,
    has_two_nan=False,
    has_naninf=False,
)

e1m0 = FpNumericsConfig(
    nexp=1,
    minexp=0,
    nmant=0,
    has_subnormals=False,
    has_two_nan=False,
    has_naninf=False,
)

float8_e4m3fn = FpNumericsConfig(
    nexp=4,
    minexp=-6,
    nmant=3,
    has_subnormals=True,
    has_two_nan=True,
    has_naninf=False,
)

float8_e5m2 = FpNumericsConfig(
    nexp=5,
    minexp=-14,
    nmant=2,
    has_subnormals=True,
    has_two_nan=False,
    has_naninf=True,
)

float16 = FpNumericsConfig(
    nexp=5,
    minexp=-14,
    nmant=10,
    has_subnormals=True,
    has_two_nan=False,
    has_naninf=True,
)


# TODO(lew): Add minexp.
def fp_round(
    x,
    *,
    cfg: FpNumericsConfig,
    key: jax.Array,
    stochastic_rounding: bool,
    test_noise_axis=None,
):
  """FP stochastic rounding for a given mantissa and exponent. Returns bf16."""
  nexp = cfg.nexp
  nmant = cfg.nmant
  assert cfg.minexp == 0, 'minexp not implemented'
  assert not cfg.has_subnormals, 'subnormals not implemented'
  assert not cfg.has_two_nan, 'two_nan not implemented'
  assert not cfg.has_naninf, 'naninf not implemented'

  orig_x = x
  input_dtype = x.dtype
  total_bits = jnp.finfo(input_dtype).bits
  # (sign, exp, mantissa) = (1,8,x)
  msg = f'Unsupported dtype for sthochastic rounding: {input_dtype}.'
  assert input_dtype == jnp.bfloat16 or input_dtype == jnp.float32, msg
  # bf16 bit pattern: seeeeeeeemmmmmmm
  #                   7654321076543210
  container_man = jnp.finfo(input_dtype).nmant
  bits_dtype = jnp.uint16 if total_bits == 16 else jnp.uint32
  assert nmant <= container_man, 'bf16 man bits'
  man_trunc_bits = container_man - nmant
  man_trunc_mask = bits_dtype((1 << man_trunc_bits) - 1)

  # TODO(lew): We can use smaller dtype
  x = jax.lax.bitcast_convert_type(x, bits_dtype)  # adds noise to man
  if stochastic_rounding:
    if test_noise_axis is not None:
      noise_axis_size = x.shape[test_noise_axis]
      sh = list((1,) * len(x.shape))
      sh[test_noise_axis] = noise_axis_size
      rnd_bits = jnp.arange(noise_axis_size, dtype=bits_dtype).reshape(sh)
    else:
      rnd_bits = jax.random.bits(key, x.shape, bits_dtype)
    noise = jax.lax.convert_element_type(rnd_bits, bits_dtype) & man_trunc_mask
  else:
    noise = 1 << (man_trunc_bits - 1)
    noise = jax.lax.convert_element_type(noise, bits_dtype)
  # This noise add might overflow up to sign bit if x(bf16) has max exp.
  # In bf16 this happens if x=nan or x=inf.
  x = x + noise
  x = jax.lax.bitwise_and(x, ~man_trunc_mask)  # zero-out lowest bits in man

  if nexp is not None:
    assert input_dtype == jnp.bfloat16
    # min_exp = jnp.uint16(0b0_10000000_0000000)
    # max_exp = jnp.uint16(0b0_10000111_0000000)
    # bf16_exp_bias = jnp.uint16(0b0_01111111)
    min_exp = jnp.uint16(0b0_01111111)
    max_exp = min_exp + 2**nexp - 1

    min_repr = min_exp << container_man
    min_repr_float = jax.lax.bitcast_convert_type(min_repr, input_dtype)
    container_mant_mask = (1 << container_man) - 1
    max_mant = jnp.uint16(container_mant_mask & ~man_trunc_mask)
    max_repr = (max_exp << container_man) + max_mant

    sign_mask = jnp.uint16(0b1_00000000_0000000)
    sign = jax.lax.bitwise_and(x, sign_mask)
    abs_x = jax.lax.bitwise_and(x, ~sign_mask)
    # TODO(lew): we can do faster clipping with bit twiddling
    clip_x = jnp.clip(abs_x, min_repr, max_repr)
    x = jax.lax.bitwise_or(sign, clip_x)
    x = jax.lax.bitcast_convert_type(x, input_dtype)
    if stochastic_rounding:
      assert bits_dtype == jnp.uint16
      if test_noise_axis is not None:
        noise_axis_size = x.shape[test_noise_axis]
        sh = list((1,) * len(x.shape))
        sh[test_noise_axis] = noise_axis_size
        rnd_bits = jnp.arange(noise_axis_size, dtype=bits_dtype).reshape(sh)
        rnd_bits = rnd_bits << 8
      else:
        rnd_bits = jax.random.bits(key, x.shape, bits_dtype)
      rnd_bits = (jnp.uint32(rnd_bits) << 7) | jnp.float32(2).view(jnp.uint32)
      rnd_floats = jax.lax.bitcast_convert_type(rnd_bits, jnp.float32) - (
          3.0 - 2 ** (-16)
      )
      rx = (orig_x > rnd_floats) * 2 - 1.0
      x = jnp.where(jnp.abs(orig_x) < min_repr_float, rx, x)
  else:
    x = jax.lax.bitcast_convert_type(x, input_dtype)

  return x


def fp_largest_representable(cfg: FpNumericsConfig):
  nexp = cfg.nexp
  nmant = cfg.nmant
  assert cfg.minexp == 0, 'minexp not implemented'
  assert not cfg.has_subnormals, 'subnormals not implemented'
  assert not cfg.has_naninf, 'naninf not implemented'
  max_exp = 2**nexp - 1 + cfg.minexp
  max_mant = 2**nmant - 1 - (1 if cfg.has_two_nan else 0)
  return 2**max_exp * (1 + (max_mant / 2**nmant))


@utils.flax_slots_kw_only_dataclass
class FpNumerics(numerics.AqtNumerics):
  """Numerics for fp8."""

  # Requested rounding precision.
  cfg: FpNumericsConfig = utils.static_field()
  stochastic_rounding: bool = utils.static_field(default=False)
  clip_gradient: bool = utils.static_field(default=False)

  def abs_val_mapped_to(self):
    return fp_largest_representable(cfg=self.cfg)

  def get_dtype(self):
    return jnp.bfloat16

  def vjp_fwd(self, x, context):
    x = fp_round(
        x.astype(jnp.bfloat16),
        cfg=self.cfg,
        key=context.key,
        stochastic_rounding=self.stochastic_rounding,
    )
    res = (x,)
    return x, res

  def vjp_bwd(self, res, grad):
    ret = grad
    if self.clip_gradient:
      (x,) = res
      clip_bound = self.abs_val_mapped_to()
      ret *= (-clip_bound <= x) * (x <= clip_bound)
    return (ret, None)
