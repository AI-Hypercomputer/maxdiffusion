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
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
from aqt.jax.v2 import utils
from aqt.jax.v2.examples import flax_e2e_model_v0
from aqt.jax.v2.numerics import fp8_numerics
import jax
import jax.numpy as jnp
import numpy as np


def averaged_stochastic_rounding(
    numerics,
    key,
    x_count,
    x_min,
    x_max,
    sr_count,
):
  x = jnp.linspace(x_min, x_max, x_count, dtype=jnp.float32)
  bx = x.reshape((x_count, 1)).astype(jnp.bfloat16)
  bx = jnp.ones((x_count, sr_count), dtype=jnp.bfloat16) * bx  # broadcast
  context = utils.Context(key=jax.random.PRNGKey(key), train_step=None)
  qx, _ = numerics.vjp_fwd(bx, context)
  assert qx.dtype == numerics.get_dtype()
  qx = qx.astype(jnp.float32)
  qx = jnp.mean(qx, axis=1)
  return x, qx


class MyTest(parameterized.TestCase):

  def test_2bit_mantissa(self):
    x = jnp.array(
        [
            [0, 1, -1, 0.1234567, 100, 0.5, 0.00000000001, 0.01],
            [
                -2.9098532,
                -0.8508397,
                -0.14903745,
                -3.572757,
                0.63277215,
                0.7478278,
                1.3707844,
                1.3319879,
            ],
        ],
        dtype=jnp.float32,
    )

    result = fp8_numerics.round_to_nearest_even(x, jnp.float8_e5m2)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0, 1, -1, 0.125, 96, 0.5, 0, 10 / 1024],
                [-3, -0.875, -0.15625, -3.5, 0.625, 0.75, 1.25, 1.25],
            ],
            dtype=jnp.float32,
        ),
    )

  def test_3bit_mantissa(self):
    x = jnp.array(
        [
            [0, 1, -1, 0.1234567, 100, 0.5, 0.00000000001, 0.01],
            [
                -2.9098532,
                -0.8508397,
                -0.14903745,
                -3.572757,
                0.63277215,
                0.7478278,
                1.3707844,
                1.3319879,
            ],
        ],
        dtype=jnp.float32,
    )

    result = fp8_numerics.round_to_nearest_even(x, jnp.float8_e4m3fn)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0, 1, -1, 0.125, 96, 0.5, 0, 10 / 1024],
                [-3, -0.875, -0.15625, -3.5, 0.625, 0.75, 1.375, 1.375],
            ],
            dtype=jnp.float32,
        ),
    )

  def test_retains_dtype(self):
    result_f16 = fp8_numerics.round_to_nearest_even(
        jnp.array([1], dtype=jnp.bfloat16), jnp.float8_e5m2
    )
    result_f32 = fp8_numerics.round_to_nearest_even(
        jnp.array([1], dtype=jnp.float32), jnp.float8_e4m3fn
    )

    self.assertEqual(result_f32.dtype, jnp.float8_e4m3fn)
    self.assertEqual(result_f16.dtype, jnp.float8_e5m2)

  @parameterized.parameters([
      dict(fwd_bits="e4m3"),
      dict(fwd_bits="e5m2"),
  ])
  def test_mnist_training(self, fwd_bits: str):
    target_loss = {
        "e4m3": {
            "cpu": [
                # Different CPU models are not bit exact and sometimes produce
                # different losses under the same training setting.
                3.167505979537963867187500000000,
                3.167067050933837890625000000000,  # skylake, cascadelake
                3.167066574096679687500000000000,  # colab
            ],
            "TPU v2": [3.212926864624023437500000000000],
            "TPU v3": [3.212912797927856445312500000000],
            "TPU v4": [3.213006973266601562500000000000],
            "TPU v5 lite": [3.212950229644775390625000000000],
        },
        "e5m2": {
            "cpu": [
                # Different CPU models are not bit exact and sometimes produce
                # different losses under the same training setting.
                3.112026929855346679687500000000,
                3.112026691436767578125000000000,
                3.112027168273925781250000000000,  # colab s
            ],
            "TPU v2": [3.184741020202636718750000000000],
            "TPU v3": [3.184759140014648437500000000000],
            "TPU v4": [3.185783863067626953125000000000],
            "TPU v5 lite": [3.185492992401123046875000000000],
        },
    }

    # RNGs
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    rng, image_rng = jax.random.split(rng)
    rng, label_rng = jax.random.split(rng)
    rng, input_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 8
    ds = {
        "image": jax.random.uniform(key=image_rng, shape=(ds_size, 28, 28, 1)),
        "label": jax.random.randint(
            key=label_rng, shape=(ds_size,), minval=0, maxval=10
        ),
    }

    aqt_cfg = config.config_fwd_fp8(fwd_bits)
    state = flax_e2e_model_v0.create_train_state(init_rng, aqt_cfg)

    _, train_loss, _ = flax_e2e_model_v0.train_epoch(
        state, ds, batch_size=ds_size // 2, rng=input_rng
    )

    device = jax.devices()[0].device_kind
    if train_loss not in target_loss[fwd_bits][device]:
      msg = f"train_loss={train_loss:.30f}, {device=}, {fwd_bits=}"
      self.fail(msg)

  @parameterized.parameters([
      dict(),
      dict(key=1),
      dict(key=2),
      dict(key=3),
      dict(key=4),
      dict(dtype=jnp.float8_e4m3fn, exponent_bits=4, mantissa_bits=3),
      dict(x_min=64 / 2**20, x_max=2 * 64 / 2**20),
      dict(x_min=32 / 2**20, x_max=2 * 32 / 2**20),
      # dict(x_min=16 / 2**20, x_max=2 * 16 / 2**20),  # This one is failing.
  ])
  def test_fp8_stochastic_rounding(
      self,
      key=0,
      x_count=1024,
      x_min=1.0,
      x_max=4.0,
      sr_count=10000,
      dtype=jnp.float8_e5m2,
      exponent_bits=5,
      mantissa_bits=2,
  ):
    numerics = fp8_numerics.Fp8Numerics(
        dtype=dtype,
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        stochastic_rounding=True,
    )
    x, qx = averaged_stochastic_rounding(
        numerics,
        key=key,
        x_count=x_count,
        x_min=x_min,
        x_max=x_max,
        sr_count=sr_count,
    )
    err = x - qx
    mean_err = jnp.mean(err)
    std_err = jnp.std(err) / jnp.sqrt(sr_count)
    mean_in_std_err_units = jnp.abs(mean_err) / std_err
    assert (
        mean_in_std_err_units < 5
    ), f"mean_in_std_err_units: {mean_in_std_err_units}"


def illustrate_bf16():
  def bit1(fro, to):
    # (fro-to) 1s followed by (to) 0s
    return (1 << fro) - (1 << to)

  for i in range(-1, 16):
    bits = jnp.uint16(0)
    if i > 0:
      bits += jnp.uint16(1 << i)
    bits += bit1(15, 8)
    f = float(jax.lax.bitcast_convert_type(bits, jnp.bfloat16))
    print(f"{bits:016b} {i=: 3}, {f:.9f}")


def illustrate_bf16_2():
  def pr(n):
    n = jnp.int16(n)
    bx = jax.lax.bitcast_convert_type(n, jnp.bfloat16)
    # fp8 = fp8_numerics.fp_mantissa_round(bx, 2, key=jax.random.PRNGKey(0))
    # fp8 = bx.astype(jnp.float8_e5m2)
    # fp8_n = jax.lax.bitcast_convert_type(fp8, jnp.uint8)
    ns = f"{n:016b}"
    ns = ns[:1] + "s " + ns[1:9] + "e " + ns[9:] + "m"
    # print("x1", fp8, fp8_n)
    # ns_8 = f"{fp8_n:08b}"
    # ns_8 = ns_8[:1] + "s " + ns_8[1:6] + "e " + ns_8[6:] + "m"
    # print(f"{float(bx):03.8f}[{ns}] -> {float(fp8):03.8}[{ns_8}]")
    print(f"{float(bx):03.8f}[{ns}]")

  pr(0b0_01111111_0000000)
  pr(0b0_01111111_0001000)
  pr(0b0_01111111_0010000)  #
  pr(0b0_01111111_0011000)
  print()

  pr(0b0_01111111_0100000)
  pr(0b0_01111111_0101000)
  pr(0b0_01111111_0110000)  #
  pr(0b0_01111111_0111000)
  print()

  pr(0b0_01111111_1000000)
  pr(0b0_01111111_1001000)
  pr(0b0_01111111_1010000)  #
  pr(0b0_01111111_1011000)
  print()

  pr(0b0_01111111_1100000)
  pr(0b0_01111111_1101000)
  pr(0b0_01111111_1110000)  #
  pr(0b0_01111111_1111000)


# To be used in colab.
def plot_sr_error(x_min=1.0, x_max=2.0, sr_count=64 * 1024, x_count=1024):
  n = fp8_numerics.Fp8Numerics(
      dtype=jnp.float8_e5m2,
      exponent_bits=5,
      mantissa_bits=2,
      stochastic_rounding=True,
  )
  x_min, x_max = 1.0, 2.0
  x, qx = averaged_stochastic_rounding(
      n, key=0, x_min=x_min, x_max=x_max, sr_count=sr_count, x_count=x_count
  )

  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  plt.figure(figsize=(30, 15))
  err = qx - x.astype(jnp.bfloat16).astype(jnp.float32)
  plt.plot(x, err)
  print(jnp.std(err), jnp.std(err) - 0.0004826417)

  plt.show()


if __name__ == "__main__":
  absltest.main()
