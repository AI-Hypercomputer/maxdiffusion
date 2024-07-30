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

"""Tests for dot_general."""
import copy
import functools
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import calibration
from aqt.jax.v2 import config
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2 import utils
import aqt.jax.v2.aqt_dot_general as aqt
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
from aqt.jax.v2.numerics import numerics
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats


def test_jaxpr_dtype(f, dg_raws: list[aqt.DotGeneralRaw], float_dtype):
  """Tests whether dot_generals in f conform to dtypes inside of dg_raws."""

  def jaxpr_to_trityp(jaxpr):
    for eq in jaxpr.eqns:
      # This is where dot_generals hide when one uses custom vjp
      if "fun_jaxpr" in eq.params.keys():
        for rec in jaxpr_to_trityp(eq.params["fun_jaxpr"]):
          yield rec

      if eq.primitive.name == "dot_general":
        [lhs, rhs] = eq.invars
        [out] = eq.outvars
        trityp = (lhs.aval, rhs.aval, out.aval)
        yield trityp

  f_jaxpr = jax.make_jaxpr(f)()
  trityps = [trityp for trityp in jaxpr_to_trityp(f_jaxpr)]
  assert len(trityps) == len(dg_raws)
  for (lhs_sa, rhs_sa, out_sa), dg_raw in zip(trityps, dg_raws):
    # If cfg has None, the type is inherited from the arguments' type.
    def assert_dtype_eq(dtype1, dtype2):
      assert dtype1 == dtype2, f"dtype1 != dtype2: {dtype1=} != {dtype2=}"

    # Using str instead of isinstance to work with adhoc import.
    assert (
        str(type(dg_raw.dg_quantizer))
        == "<class 'aqt.jax.v2.aqt_dot_general.DefaultDotGeneralQuantizer'>"
    ), f"Invalid dg_quantizer type. {str(type(dg_raw.dg_quantizer))=}"
    # assert isinstance(
    #     dg_raw.dg_quantizer, aqt.DefaultDotGeneralQuantizer
    # ), f"Invalid dg_quantizer type. {type(dg_raw.dg_quantizer)=}"

    lhs_dtype = dg_raw.dg_quantizer.lhs.numerics.get_dtype()
    rhs_dtype = dg_raw.dg_quantizer.rhs.numerics.get_dtype()
    assert_dtype_eq(lhs_sa.dtype, lhs_dtype or float_dtype)
    assert_dtype_eq(rhs_sa.dtype, rhs_dtype or float_dtype)
    assert_dtype_eq(out_sa.dtype, dg_raw.dg_accumulator_dtype or float_dtype)


def rand_unif(shape, maxval, seed, dtype=jnp.float32):
  key = jax.random.PRNGKey(seed)
  return jax.random.uniform(
      key=key, shape=shape, minval=-maxval, maxval=maxval, dtype=dtype
  )


# The main test strategy is :
# - Test that FQ is sensible (improve fq_noise_test)
#   - Quantization noise (rounding error) is of proper value
#   - Cliping is correct
#   - Stochastic noise is added correctly.
# - Test that we really use int8/int4/bool in XLA/HLO
# - Test of fq vs dot_general equivalence in po2
# - Test that it all applies to gradients
#
#
# TODO(lew): Tests are a bit incomplete. What we need:
# On top of that we shell test:
#  - Gradinent is 1 in the range, 0 outside the abs-max range.


def test_eq(name, a, b):
  # TODO(lew): use library function.
  mean_err = jnp.mean(jnp.abs(a - b))
  if mean_err != 0.0:
    print("mean_err =", mean_err)
    print(a.shape)
    print(a[:3, :3])
    print(b.shape)
    print(b[:3, :3])
    print(f"FAIL: {name}")
    assert False


# Test exact equalities.
def _check_result_eq(dgs, *, lhs, rhs, gra):
  good_lr, good_gl, good_gr = None, None, None
  for name, dg, options in dgs:
    # Default settings:
    if "test_gradient" not in options:
      options["test_gradient"] = True
    if "mult" not in options:
      options["mult"] = (1.0, 1.0, 1.0)
    if "check_fwd_lhs_tricky_clip_and_round" not in options:
      options["check_fwd_lhs_tricky_clip_and_round"] = False

    lrf = dg(lhs, rhs)
    lr, backprop = jax.vjp(dg, lhs, rhs)

    gl, gr = backprop(gra) if options["test_gradient"] else (None, None)

    if options["check_fwd_lhs_tricky_clip_and_round"]:
      # Test that all the expected were zero anyway
      where_zero_gradient_expected = lhs < 0
      assert (gl[where_zero_gradient_expected] == 0.0).all()

    good_lr = lr if good_lr is None else good_lr
    good_gl = gl if good_gl is None else good_gl
    good_gr = gr if good_gr is None else good_gr

    test_eq(f"{name}: lr vs lrf", lr, lrf)

    lr_mult, gl_mult, gr_mult = options["mult"]
    test_eq(f"{name}: lr", good_lr, lr / lr_mult)  # forward pass
    if options["test_gradient"]:
      test_eq(f"{name}: gl", good_gl, gl / gl_mult)  # backward pass  # pytype: disable=unsupported-operands
      test_eq(f"{name}: gr", good_gr, gr / gr_mult)  # backward pass  # pytype: disable=unsupported-operands


def fqt_param_dict(s, use_fwd_quant, **kwargs):
  return dict(
      dg=config.fully_quantized(
          use_fwd_quant=use_fwd_quant,
          use_stochastic_rounding=False,
          **kwargs,
      ),
      seed=s,
  )


class _TrickyNumerics(numerics.AqtNumerics, flax.struct.PyTreeNode):
  # Needed because int8 casting would do additional clip and round.
  dtype: jnp.dtype | None = None

  def get_dtype(self):
    return self.dtype

  def abs_val_mapped_to(self) -> jnp.ndarray:
    return jnp.array(1.0)

  def fwd(self, x, context):
    del context
    return jax.lax.round(x, jax.lax.RoundingMethod.TO_NEAREST_EVEN)

  def vjp_fwd(self, x, context):
    res = (x,)
    return self.fwd(x, context), res

  def vjp_bwd(self, res, grad):
    (x,) = res
    ret = grad * (x >= 0)
    return (ret, None)


def _modify_dg(
    readonly_dg: aqt.DotGeneral,
    *,
    lhs_dequant_mode: aqt.DequantMode = aqt.DequantMode.OUTPUT,
    rhs_dequant_mode: aqt.DequantMode = aqt.DequantMode.OUTPUT,
    lhs_calibration_mode: aqt.CalibrationMode = aqt.CalibrationMode.CONTRACTING_AXIS,
    rhs_calibration_mode: aqt.CalibrationMode = aqt.CalibrationMode.CONTRACTING_AXIS,
    use_fwd_quant: bool | None = None,
    fwd_lhs_tricky_clip_and_round: bool = False,
    local_aqt: aqt.LocalAqt | None = None,
    clip_gradient: bool = False,
) -> aqt.DotGeneral:
  dg = copy.deepcopy(readonly_dg)
  if fwd_lhs_tricky_clip_and_round:
    # Tricky means that we have zero gradient on x < 0
    dg.fwd.dg_quantizer.lhs.numerics = _TrickyNumerics()
    dg.fwd.dg_accumulator_dtype = None

  # Setting po2_scale is ensuring that fake_quant and full dot_general
  # have the same numerics when scales are power of two (po2).
  # We are passing dims to config so that we can reuse it in fake_quant.
  # Power-of-2 scales allow FQ and AQT to be exactly the same.
  def _apply_po2_scale(c):
    c.dg_quantizer.lhs.po2_scale = True
    c.dg_quantizer.rhs.po2_scale = True

  def _apply_dequant_mode(c, lhs_dequant_mode, rhs_dequant_mode):
    c.lhs.dequant_mode = lhs_dequant_mode
    c.rhs.dequant_mode = rhs_dequant_mode

  def _apply_calibration_mode(c, lhs_calibration_mode, rhs_calibration_mode):
    c.lhs.calibration_mode = lhs_calibration_mode
    c.rhs.calibration_mode = rhs_calibration_mode

  def _disable_quant_types(c, on_lhs=True, on_rhs=True):
    if on_lhs:
      c.dg_quantizer.lhs.numerics = (
          c.dg_quantizer.lhs.numerics.replace(dtype=None)
      )
    if on_rhs:
      c.dg_quantizer.rhs.numerics = (
          c.dg_quantizer.rhs.numerics.replace(dtype=None)
      )
    if on_lhs or on_rhs:
      c.dg_accumulator_dtype = None

  disable_lhs_quant = lhs_dequant_mode == aqt.DequantMode.THIS_INPUT
  disable_rhs_quant = rhs_dequant_mode == aqt.DequantMode.THIS_INPUT
  for c in [dg.fwd, dg.dlhs, dg.drhs]:
    _apply_po2_scale(c)
    _apply_dequant_mode(c, lhs_dequant_mode, rhs_dequant_mode)
    _apply_calibration_mode(
        c, lhs_calibration_mode, rhs_calibration_mode
    )
    _disable_quant_types(c, disable_lhs_quant, disable_rhs_quant)

  if use_fwd_quant is not None:
    # If we disable all rounding, we are just testing whether the scales are
    # correct. We don't even need to disable clipping and we are testing
    # that the scales are not too large.
    def disable_quant(c):
      _disable_quant_types(c)
      if isinstance(c.dg_quantizer.lhs.numerics, int_numerics.IntNumerics):
        c.dg_quantizer.lhs.numerics = (
            c.dg_quantizer.lhs.numerics.replace(round=False)
        )
      if isinstance(c.dg_quantizer.rhs.numerics, int_numerics.IntNumerics):
        c.dg_quantizer.rhs.numerics = (
            c.dg_quantizer.rhs.numerics.replace(round=False)
        )

    disable_quant(dg.fwd)
    disable_quant(dg.dlhs)
    disable_quant(dg.drhs)
    lhs_quant = not isinstance(
        dg.fwd.dg_quantizer.lhs.numerics, no_numerics.NoNumerics
    )
    rhs_quant = not isinstance(
        dg.fwd.dg_quantizer.rhs.numerics, no_numerics.NoNumerics
    )
    if lhs_quant:
      dg.drhs.rhs.use_fwd_quant = use_fwd_quant
    if rhs_quant:
      dg.dlhs.rhs.use_fwd_quant = use_fwd_quant
  if local_aqt is not None:
    # Currently we are not supporting local_aqt in fwd pass
    # dg.fwd.local_aqt = local_aqt
    dg.dlhs.local_aqt = local_aqt
    dg.drhs.local_aqt = local_aqt

    # When using abs-max scaling, this should be a no-op.
  if isinstance(dg.fwd.dg_quantizer.lhs.numerics, int_numerics.IntNumerics):
    dg.fwd.dg_quantizer.lhs.numerics = (
        dg.fwd.dg_quantizer.lhs.numerics.replace(clip_gradient=clip_gradient)
    )
  if isinstance(dg.fwd.dg_quantizer.rhs.numerics, int_numerics.IntNumerics):
    dg.fwd.dg_quantizer.rhs.numerics = (
        dg.fwd.dg_quantizer.rhs.numerics.replace(clip_gradient=clip_gradient)
    )

  return dg


def _aqt_dg_full_lr_diff(
    lhs_dequant_mode: aqt.DequantMode,
    rhs_dequant_mode: aqt.DequantMode,
    lhs_calibration_mode: aqt.CalibrationMode = aqt.CalibrationMode.CONTRACTING_AXIS,
    rhs_calibration_mode: aqt.CalibrationMode = aqt.CalibrationMode.CONTRACTING_AXIS,
    use_fwd_quant: bool | None = None,
    fwd_lhs_tricky_clip_and_round: bool = False,
    local_aqt: aqt.LocalAqt | None = None,
    *,
    readonly_dg: aqt.DotGeneral,
    dims: jax.lax.DotDimensionNumbers,
    clip_gradient: bool = False,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  dg = _modify_dg(
      readonly_dg,
      lhs_dequant_mode=lhs_dequant_mode,
      rhs_dequant_mode=rhs_dequant_mode,
      lhs_calibration_mode=lhs_calibration_mode,
      rhs_calibration_mode=rhs_calibration_mode,
      use_fwd_quant=use_fwd_quant,
      fwd_lhs_tricky_clip_and_round=fwd_lhs_tricky_clip_and_round,
      local_aqt=local_aqt,
      clip_gradient=clip_gradient,
  )
  dg = config.set_context(dg, key=jax.random.PRNGKey(4), train_step=None)
  return lambda lhs, rhs: dg(lhs, rhs, dims)


def _aqt_dg_full(
    dequant_mode: aqt.DequantMode,
    calibration_mode: aqt.CalibrationMode = aqt.CalibrationMode.CONTRACTING_AXIS,
    use_fwd_quant: bool | None = None,
    fwd_lhs_tricky_clip_and_round: bool = False,
    local_aqt: aqt.LocalAqt | None = None,
    *,
    readonly_dg: aqt.DotGeneral,
    dims: jax.lax.DotDimensionNumbers,
    clip_gradient: bool = False,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  return _aqt_dg_full_lr_diff(
      dequant_mode,
      dequant_mode,
      calibration_mode,
      calibration_mode,
      use_fwd_quant,
      fwd_lhs_tricky_clip_and_round,
      local_aqt,
      readonly_dg=readonly_dg,
      dims=dims,
      clip_gradient=clip_gradient
  )


def _aqt_dg_raw_lr_diff(
    lhs_dequant_mode: aqt.DequantMode,
    rhs_dequant_mode: aqt.DequantMode,
    lhs_calibration_mode: aqt.CalibrationMode = aqt.CalibrationMode.CONTRACTING_AXIS,
    rhs_calibration_mode: aqt.CalibrationMode = aqt.CalibrationMode.CONTRACTING_AXIS,
    *,
    readonly_dg: aqt.DotGeneral,
    dims: jax.lax.DotDimensionNumbers,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  dg = _modify_dg(
      readonly_dg,
      lhs_dequant_mode=lhs_dequant_mode,
      rhs_dequant_mode=rhs_dequant_mode,
      lhs_calibration_mode=lhs_calibration_mode,
      rhs_calibration_mode=rhs_calibration_mode,
  )
  dg = config.set_context(dg, key=jax.random.PRNGKey(4), train_step=None)
  dg.fwd.dg_quantizer.init_calibration()
  return lambda lhs, rhs: dg.fwd(lhs, rhs, None, None, dims)[0]


def _aqt_dg_raw(
    dequant_mode: aqt.DequantMode,
    calibration_mode: aqt.CalibrationMode = aqt.CalibrationMode.CONTRACTING_AXIS,
    *,
    readonly_dg: aqt.DotGeneral,
    dims: jax.lax.DotDimensionNumbers,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  return _aqt_dg_raw_lr_diff(
      dequant_mode,
      dequant_mode,
      calibration_mode,
      calibration_mode,
      readonly_dg=readonly_dg,
      dims=dims,
  )


class AqtDotGeneralResearchTest(parameterized.TestCase):

  def test_empty(self):
    t = np.random.normal(size=6).reshape((2, 3))
    np.testing.assert_array_equal(t, t)

  @parameterized.product(
      preserve_zero=[True, False],
      prec=[1, 2, 4, 8],
      v=[0.1, 1000.0],
      seed=list(range(10)),
  )
  def test_fq_noise(self, preserve_zero, prec, v, seed):
    key = jax.random.PRNGKey(seed)
    quantizer = config.quantizer_make(prec)
    if isinstance(quantizer.numerics, int_numerics.IntNumerics):
      quantizer.numerics.preserve_zero = preserve_zero
      if not preserve_zero:
        quantizer.numerics.dtype = None
    quantizer.calib_shared_axes = (0,)
    sample_size = 10000
    shape = (sample_size,)
    a = jax.random.uniform(key, shape, minval=-v, maxval=v)
    a_fq = aqt_quantizer.make_fake_quant(quantizer)(a)
    bucket_noise = a_fq - a  #  ~ U(-bucket_size/2, bucket_size/2)
    bucket_count = (2**prec - 1) if preserve_zero else (2**prec)
    bucket_size = (v * 2) / bucket_count
    noise = bucket_noise / bucket_size + 0.5  # ~U(0, 1)
    pvalue = scipy.stats.kstest(noise, "uniform").pvalue
    assert pvalue > 0.01

  def test_stochastic_rounding_noise(self):
    repeats = 1000
    num_values = 100000
    shape = (repeats, num_values)

    def assert_clt(noise: jnp.ndarray):
      # Test if the sample mean of the noise conforms to Central Limit Theorem

      # std of a uniform distribution [-0.5, 0.5]
      noise_std = 1 / jnp.sqrt(12.0)
      noise_mean = jnp.mean(noise, axis=1)
      assert jnp.all(noise_mean * jnp.sqrt(num_values) < 4 * noise_std)

    # jax.uniform implementation
    noise_fn = stochastic_rounding.JaxUniform()
    jax_uniform_noise = noise_fn(shape, jax.random.PRNGKey(10))
    assert_clt(jax_uniform_noise)
    # customized more efficient implementation
    noise_fn = stochastic_rounding.RandomCenteredUniform()
    custom_1_noise = noise_fn(shape, jax.random.PRNGKey(11))
    assert_clt(custom_1_noise)

  @parameterized.parameters([
      dict(bits=1),
  ])
  def test_fake_quant(
      self,
      bits=4,
      maxval=10.0,
      shape=(20, 1),
  ):
    quantizer = config.quantizer_make(bits)
    quantizer.po2_scale = True
    quantizer.calib_shared_axes = (0,)
    x = jnp.linspace(-maxval, maxval, num=shape[0]).reshape(shape)
    grad = jnp.ones(shape) * 12345.0
    x_fq, backprop = jax.vjp(aqt_quantizer.make_fake_quant(quantizer), x)
    gx_fq = backprop(grad)
    # print(f"x     =\n{x}")
    # print(f"x_fq  =\n{x_fq}")
    # print(f"grad  =\n{grad}")
    # print(f"gx_fq =\n{gx_fq}")
    del x
    del x_fq
    del grad
    del gx_fq

    # TODO(lew): test

  @parameterized.parameters([
      dict(
          # TODO(aqt): Change dlhs_bits to 4bit once
          # https://github.com/google/jax/issues/19682 is fixed.
          dg=config.config_v3(
              fwd_bits=3,
              dlhs_bits=6,
              drhs_bits=5,
              drhs_accumulator_dtype=jnp.int32,  # overwriting the default None
          )
      ),
      dict(dg=config.dot_general_make(None, None)),
      dict(dg=config.dot_general_make(1, 1)),
      dict(dg=config.dot_general_make(1, 2)),
      dict(dg=config.dot_general_make(2, 1)),
      dict(dg=config.dot_general_make(2, 2)),
      dict(dg=config.dot_general_make(8, 8)),
      dict(dg=config.dot_general_make(8, 8), clip_gradient=True),
      dict(
          dg=config.dot_general_make(
              8, 8, dlhs_local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2)
          )
      ),
      dict(
          dg=config.dot_general_make(
              8, 8, drhs_local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2)
          )
      ),
      # That test could fail numerically because bf16
      # can't keep in the product of int8*int8 accurately.
      # It just so happens that this test does not fail but others do.
      # We do this test anyway, to catch jax-compilation-time errors.
      dict(dg=config.dot_general_make(2, 2), dtype=jnp.bfloat16),
      dict(dg=config.dot_general_make(8, 8), dtype=jnp.bfloat16),
      dict(dg=config.dot_general_make(None, 8)),
      dict(dg=config.dot_general_make(8, None)),
      dict(
          dg=fqt_param_dict(s=10, use_fwd_quant=True)["dg"],
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
      dict(
          dg=fqt_param_dict(
              s=10,
              use_fwd_quant=True,
              dlhs_local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2),
          )["dg"],
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
      dict(
          dg=config.dot_general_make(2, 2),
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
      *[fqt_param_dict(s, use_fwd_quant=False) for s in range(10)],
      *[fqt_param_dict(s, use_fwd_quant=True) for s in range(10)],
  ])
  def test_dot_general_calibration_with_contracting_axis(
      self,
      dg: aqt.DotGeneral,
      lhs_maxval=10.0,
      rhs_maxval=20.0,
      gra_maxval=30.0,
      dims=(((1,), (0,)), ((), ())),  # classical matmul
      # lhs_shape=(1, 1),
      # rhs_shape=(1, 2),
      # gra_shape=(1, 2),  # has to be the shape of the output
      # lhs_shape=(1, 2),
      # rhs_shape=(2, 3),
      # gra_shape=(1, 3),  # has to be the shape of the output
      lhs_shape=(10, 20),
      rhs_shape=(20, 30),
      gra_shape=(10, 30),  # has to be the shape of the output
      seed=0,
      dtype=jnp.float32,
      clip_gradient=False,
  ):
    readonly_dg = dg
    del dg

    lhs = rand_unif(lhs_shape, lhs_maxval, seed, dtype)
    rhs = rand_unif(rhs_shape, rhs_maxval, seed + 1, dtype)
    gra = rand_unif(gra_shape, gra_maxval, seed + 2, dtype)

    # Prepare utility functions for test.
    aqt_dg_full = functools.partial(
        _aqt_dg_full,
        readonly_dg=readonly_dg,
        dims=dims,
        clip_gradient=clip_gradient,
    )
    aqt_dg_raw = functools.partial(
        _aqt_dg_raw, readonly_dg=readonly_dg, dims=dims
    )
    modify_dg = functools.partial(_modify_dg, readonly_dg=readonly_dg)
    check = functools.partial(_check_result_eq, lhs=lhs, rhs=rhs, gra=gra)

    # Tests for dot_general.
    test_jaxpr_dtype(
        lambda: aqt_dg_full(aqt.DequantMode.OUTPUT)(lhs, rhs),
        [modify_dg().fwd],
        lhs.dtype,
    )
    test_jaxpr_dtype(
        lambda: jax.vjp(aqt_dg_full(aqt.DequantMode.OUTPUT), lhs, rhs),
        [modify_dg().fwd],
        lhs.dtype,
    )
    _, backprop = jax.vjp(aqt_dg_full(aqt.DequantMode.OUTPUT), lhs, rhs)
    test_jaxpr_dtype(
        lambda: backprop(gra),
        [modify_dg().dlhs, modify_dg().drhs],
        gra.dtype,
    )

    check([
        ("default    ", aqt_dg_full(aqt.DequantMode.OUTPUT), dict()),
        ("FQ         ", aqt_dg_full(aqt.DequantMode.THIS_INPUT), dict()),
        (
            "raw fwd    ",
            aqt_dg_raw(aqt.DequantMode.OUTPUT),
            dict(test_gradient=False),
        ),
        (
            "raw fwd FQ ",
            aqt_dg_raw(aqt.DequantMode.THIS_INPUT),
            dict(test_gradient=False),
        ),
    ])

    check([
        (
            "fwd_quant=T",
            aqt_dg_full(aqt.DequantMode.OUTPUT, use_fwd_quant=False),
            dict(),
        ),
        (
            "fwd_quant=F",
            aqt_dg_full(aqt.DequantMode.OUTPUT, use_fwd_quant=True),
            dict(),
        ),
    ])

    check([
        (
            "default    ",
            aqt_dg_full(
                aqt.DequantMode.OUTPUT,
                local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2),
            ),
            dict(),
        ),
        (
            "default    ",
            aqt_dg_full(
                aqt.DequantMode.THIS_INPUT,
                local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2),
            ),
            dict(),
        ),
    ])

    if isinstance(
        readonly_dg.fwd.dg_quantizer.lhs.numerics,
        int_numerics.IntNumerics,
    ):
      check([
          (
              "check_fwd_lhs_tricky_clip_and_round",
              aqt_dg_full(
                  aqt.DequantMode.OUTPUT, fwd_lhs_tricky_clip_and_round=True
              ),
              dict(check_fwd_lhs_tricky_clip_and_round=True),
          ),
      ])

    def unquant_aqt_dg(lhs, rhs):
      dg = config.default_unquantized_config()
      return dg(lhs, rhs, dims)

    def lax_dg(lhs, rhs):
      return jax.lax.dot_general(lhs, rhs, dims)

    check([
        ("unquantized default:", unquant_aqt_dg, dict()),
        ("lax.dot_general:", lax_dg, dict()),
    ])

  @parameterized.parameters([
      dict(
          dg=lambda: config.config_v3(
              fwd_bits=3,
              dlhs_bits=4,
              drhs_bits=5,
              drhs_accumulator_dtype=jnp.int32,  # overwriting the default None
          )
      ),
      dict(dg=config.dot_general_make(None, None)),
      dict(dg=config.dot_general_make(1, 1)),
      dict(dg=config.dot_general_make(1, 2)),
      dict(dg=config.dot_general_make(2, 1)),
      dict(dg=config.dot_general_make(2, 2)),
      dict(dg=config.dot_general_make(8, 8)),
      dict(dg=config.dot_general_make(8, 8), clip_gradient=True),
      dict(
          dg=config.dot_general_make(
              8, 8, dlhs_local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2)
          )
      ),
      dict(
          dg=config.dot_general_make(
              8, 8, drhs_local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2)
          )
      ),
      # That test could fail numerically because bf16
      # can't keep in the product of int8*int8 accurately.
      # It just so happens that this test does not fail but others do.
      # We do this test anyway, to catch jax-compilation-time errors.
      dict(dg=config.dot_general_make(2, 2), dtype=jnp.bfloat16),
      dict(dg=config.dot_general_make(8, 8), dtype=jnp.bfloat16),
      dict(dg=config.dot_general_make(None, 8)),
      dict(dg=config.dot_general_make(8, None)),
      dict(
          dg=config.dot_general_make(2, 2),
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
      *[fqt_param_dict(s, use_fwd_quant=False) for s in range(10)],
  ])
  def test_dot_general_calibration_with_remaining_axis(
      self,
      dg: config.DotGeneral | Callable[[], config.DotGeneral],
      lhs_maxval=10.0,
      rhs_maxval=20.0,
      gra_maxval=30.0,
      dims=(((1,), (0,)), ((), ())),  # classical matmul
      lhs_shape=(10, 20),
      rhs_shape=(20, 30),
      gra_shape=(10, 30),  # has to be the shape of the output
      seed=0,
      dtype=jnp.float32,
      clip_gradient=False,
  ):
    # Deferred evaluation of config function calls. 4-bit config initialization
    # triggers jax.local_devices(), which shouldn't be called before
    # absl.app.run() in some environments.
    if not isinstance(dg, config.DotGeneral):
      dg = dg()
    # Set use_fwd_quant to False.
    dg.drhs.rhs.use_fwd_quant = False
    dg.dlhs.rhs.use_fwd_quant = False
    readonly_dg = dg
    del dg

    lhs = rand_unif(lhs_shape, lhs_maxval, seed, dtype)
    rhs = rand_unif(rhs_shape, rhs_maxval, seed + 1, dtype)
    gra = rand_unif(gra_shape, gra_maxval, seed + 2, dtype)

    # Prepare utility functions for test.
    aqt_dg_full = functools.partial(
        _aqt_dg_full,
        readonly_dg=readonly_dg,
        dims=dims,
        clip_gradient=clip_gradient,
    )
    aqt_dg_full_lr_diff = functools.partial(
        _aqt_dg_full_lr_diff,
        readonly_dg=readonly_dg,
        dims=dims,
        clip_gradient=clip_gradient,
    )
    aqt_dg_raw = functools.partial(
        _aqt_dg_raw, readonly_dg=readonly_dg, dims=dims
    )
    aqt_dg_raw_lr_diff = functools.partial(
        _aqt_dg_raw_lr_diff, readonly_dg=readonly_dg, dims=dims
    )
    check = functools.partial(_check_result_eq, lhs=lhs, rhs=rhs, gra=gra)

    # test dot_general
    check([
        (
            "RA L       ",
            aqt_dg_full_lr_diff(
                aqt.DequantMode.OTHER_INPUT,
                aqt.DequantMode.THIS_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
                aqt.CalibrationMode.REMAINING_AXIS,
            ),
            dict(),
        ),
        (
            "RA R       ",
            aqt_dg_full_lr_diff(
                aqt.DequantMode.THIS_INPUT,
                aqt.DequantMode.OTHER_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
                aqt.CalibrationMode.REMAINING_AXIS,
            ),
            dict(),
        ),
        (
            "RA fake    ",
            aqt_dg_full(
                aqt.DequantMode.THIS_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
            ),
            dict(),
        ),
        (
            "RA L fwd   ",
            aqt_dg_raw_lr_diff(
                aqt.DequantMode.OTHER_INPUT,
                aqt.DequantMode.THIS_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
                aqt.CalibrationMode.REMAINING_AXIS,
            ),
            dict(test_gradient=False),
        ),
        (
            "RA R fwd  ",
            aqt_dg_raw_lr_diff(
                aqt.DequantMode.THIS_INPUT,
                aqt.DequantMode.OTHER_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
                aqt.CalibrationMode.REMAINING_AXIS,
            ),
            dict(test_gradient=False),
        ),
        (
            "RA fake fwd",
            aqt_dg_raw(
                aqt.DequantMode.THIS_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
            ),
            dict(test_gradient=False),
        ),
    ])

    check([
        (
            "RA fake    ",
            aqt_dg_full(
                aqt.DequantMode.THIS_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
                local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2),
            ),
            dict(),
        ),
        (
            "RA L      ",
            aqt_dg_full_lr_diff(
                aqt.DequantMode.OTHER_INPUT,
                aqt.DequantMode.THIS_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
                aqt.CalibrationMode.REMAINING_AXIS,
                local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2),
            ),
            dict(),
        ),
        (
            "RA R       ",
            aqt_dg_full_lr_diff(
                aqt.DequantMode.THIS_INPUT,
                aqt.DequantMode.OTHER_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
                aqt.CalibrationMode.REMAINING_AXIS,
                local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2),
            ),
            dict(),
        ),
    ])

  def test_dot_general_calibrate_dequant_mode_mismatch(self):
    dg = config.dot_general_make(8, 8, use_fwd_quant=False)
    dims = (((1,), (0,)), ((), ()))
    lhs = rand_unif((10, 20), 10.0, 0, jnp.float32)
    rhs = rand_unif((20, 30), 20.0, 1, jnp.float32)

    # 1. Raise error when OTHER_INPUT + CONTRACTING_AXIS
    with self.assertRaisesRegex(
        AssertionError,
        ".*Unsupported calibration mode.*dequant mode combination.*",
    ):
      _aqt_dg_full_lr_diff(
          aqt.DequantMode.THIS_INPUT,
          aqt.DequantMode.OTHER_INPUT,
          aqt.CalibrationMode.CONTRACTING_AXIS,
          aqt.CalibrationMode.CONTRACTING_AXIS,
          readonly_dg=copy.deepcopy(dg),
          dims=dims,
      )(lhs, rhs)

    # 2. Raise error when OUTPUT + REMAINING_AXIS
    with self.assertRaisesRegex(
        AssertionError,
        ".*Unsupported calibration mode.*dequant mode combination.*",
    ):
      _aqt_dg_full_lr_diff(
          aqt.DequantMode.THIS_INPUT,
          aqt.DequantMode.OUTPUT,
          aqt.CalibrationMode.CONTRACTING_AXIS,
          aqt.CalibrationMode.REMAINING_AXIS,
          readonly_dg=copy.deepcopy(dg),
          dims=dims,
      )(lhs, rhs)

  def test_dot_general_prevent_fwd_quant_with_remaining_axis(self):
    """If calibration axis is remaining_axis, use_fwd_quant should be False."""
    dg = config.dot_general_make(8, 8, use_fwd_quant=True)
    dims = (((1,), (0,)), ((), ()))
    lhs = rand_unif((10, 20), 10.0, 0, jnp.float32)
    rhs = rand_unif((20, 30), 20.0, 1, jnp.float32)

    with self.assertRaisesRegex(
        AssertionError,
        ".*use_fwd_quant should be set to False.*",
    ):
      _aqt_dg_full_lr_diff(
          aqt.DequantMode.THIS_INPUT,
          aqt.DequantMode.OTHER_INPUT,
          aqt.CalibrationMode.CONTRACTING_AXIS,
          aqt.CalibrationMode.REMAINING_AXIS,
          readonly_dg=copy.deepcopy(dg),
          dims=dims,
      )(lhs, rhs)

  @parameterized.parameters([
      dict(dg=config.dot_general_make(8, 8)),
      dict(
          dg=config.dot_general_make(
              8, 8, dlhs_local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2)
          )
      ),
      dict(
          dg=config.dot_general_make(
              8, 8, drhs_local_aqt=aqt.LocalAqt(contraction_axis_shard_count=2)
          )
      ),
  ])
  def test_dot_general_equality_between_different_calibration_axes(
      self,
      dg: config.DotGeneral,
  ):
    """Check equality between different calibration axes."""
    dims = (((1,), (0,)), ((), ()))

    # Set use_fwd_quant to False.
    dg.drhs.rhs.use_fwd_quant = False
    dg.dlhs.rhs.use_fwd_quant = False
    readonly_dg = dg
    del dg

    # Set the two arguments as powers of 2, to prevent it from having the
    # quantization loss.
    lhs = 2 ** jnp.floor(rand_unif((10, 20), 3.0, 1, jnp.float32) + 3.0)
    rhs = 2 ** jnp.floor(rand_unif((20, 30), 3.0, 2, jnp.float32) + 3.0)
    gra = 2 ** jnp.floor(rand_unif((10, 30), 3.0, 3, jnp.float32) + 3.0)

    # Prepare utility functions for test.
    aqt_dg_full = functools.partial(
        _aqt_dg_full,
        readonly_dg=readonly_dg,
        dims=dims
    )
    aqt_dg_full_lr_diff = functools.partial(
        _aqt_dg_full_lr_diff,
        readonly_dg=readonly_dg,
        dims=dims
    )
    check = functools.partial(_check_result_eq, lhs=lhs, rhs=rhs, gra=gra)

    # test dot_general
    check([
        (
            "CA         ",
            aqt_dg_full(aqt.DequantMode.OUTPUT, use_fwd_quant=False),
            dict(),
        ),
        (
            "CA fake    ",
            aqt_dg_full(aqt.DequantMode.THIS_INPUT, use_fwd_quant=False),
            dict(),
        ),
        (
            "RA L       ",
            aqt_dg_full_lr_diff(
                aqt.DequantMode.OTHER_INPUT,
                aqt.DequantMode.THIS_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
                aqt.CalibrationMode.REMAINING_AXIS,
            ),
            dict(),
        ),
        (
            "RA R       ",
            aqt_dg_full_lr_diff(
                aqt.DequantMode.THIS_INPUT,
                aqt.DequantMode.OTHER_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
                aqt.CalibrationMode.REMAINING_AXIS,
            ),
            dict(),
        ),
        (
            "RA fake      ",
            aqt_dg_full(
                aqt.DequantMode.THIS_INPUT,
                aqt.CalibrationMode.REMAINING_AXIS,
            ),
            dict(),
        ),
    ])

  def test_dynamic_context(self):
    @jax.jit
    def f(lhs, rhs):
      dg = config.dot_general_make()
      dg = config.set_context(dg, key=jax.random.PRNGKey(4), train_step=None)
      return dg(lhs, rhs, (((0,), (0,)), ((), ())))

    lhs, rhs = jnp.array([3.0, 4.0]), jnp.array([4.0, 5.0])
    jax.value_and_grad(f)(lhs, rhs)

  def test_hardware_int8(self, seed=0):
    dg_raw = config.dot_general_raw_make(8, 8)

    def dg(lhs, rhs):
      ret, _ = dg_raw(
          lhs,
          rhs,
          None,
          None,
          (((1,), (0,)), ((), ())),
      )
      return ret

    lhs = rand_unif((10, 20), 1.0, seed)
    rhs = rand_unif((20, 30), 1.0, seed + 1)
    test_jaxpr_dtype(lambda: dg(lhs, rhs), [dg_raw], lhs.dtype)
    assert dg_raw.dg_quantizer.lhs.numerics.get_dtype() == jnp.int8
    assert dg_raw.dg_quantizer.rhs.numerics.get_dtype() == jnp.int8

  @parameterized.parameters([
      dict(
          shard_count=2,
          lhs=[1270.0, 10.0, 1270000.0, 10000.0],
          expected_product=1281280.0,
      ),
      dict(
          shard_count=1,
          lhs=[1270.0, 10.0, 1270000.0, 10000.0],
          expected_product=1280000.0,
      ),
  ])
  def test_local_aqt(self, shard_count, lhs, expected_product):
    # create a config that quantizes both forward and backward passes to int8
    # set the number of shards (local aqt) to 2
    dg = config.fully_quantized(
        fwd_bits=8,
        bwd_bits=8,
        use_stochastic_rounding=False,
        drhs_local_aqt=aqt.LocalAqt(contraction_axis_shard_count=shard_count),
    )
    dg.fwd.dg_quantizer.lhs.numerics = (
        dg.fwd.dg_quantizer.lhs.numerics.replace(preserve_max_val=True)
    )
    dg.fwd.dg_quantizer.rhs.numerics = (
        dg.fwd.dg_quantizer.rhs.numerics.replace(preserve_max_val=True)
    )
    dg.drhs.dg_quantizer.lhs.numerics = (
        dg.drhs.dg_quantizer.lhs.numerics.replace(preserve_max_val=True)
    )
    dg.drhs.dg_quantizer.rhs.numerics = (
        dg.drhs.dg_quantizer.rhs.numerics.replace(preserve_max_val=True)
    )
    dg_f = lambda lhs, rhs: dg(
        lhs,
        rhs,
        dimension_numbers=(((), ()), ((), ())),
    )
    lhs = jnp.array(lhs)
    rhs = jnp.array([1.0])
    output, bprop = jax.vjp(dg_f, lhs, rhs)
    _, drhs = bprop(jnp.ones_like(output))
    assert drhs == expected_product

  def test_per_tensor(self):
    # TODO(lew): bits=8 started failing in VLP colab due x/x != 1.0 sometimes
    bits = 4
    my_numerics = int_numerics.IntNumerics(
        bits=bits,
        preserve_zero=True,
        preserve_max_val=False,
        clip=True,
        clip_gradient=False,
        round=True,
        noise_fn=None,
        dtype=jnp.int8,
    )
    quantizer = aqt_quantizer.Quantizer(
        numerics=my_numerics,
        calib_shared_axes="per_tensor",
        scale_stop_grad=True,
        calibration=calibration.AbsMaxCalibration,
        po2_scale=False,
        context=utils.Context(key=None, train_step=None),
    )
    # TODO(lew): Perhaps post_init call could work?
    quantizer.init_calibration()

    x = jnp.array([
        [1, 2, 3, 4],
        [5, 6, 7, 2 ** (bits - 1) - 0.5],
        [3, 5, 1, -1],
    ])
    qx, _ = quantizer.quant(x, calibration_axes=None)
    self.assertEqual(qx.scale, [jnp.array([[1.0]])])


if __name__ == "__main__":
  absltest.main()
