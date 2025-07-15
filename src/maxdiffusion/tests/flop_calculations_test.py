import os
import unittest
import jax
from jax.sharding import Mesh
import flax.linen as nn
from absl.testing import absltest
from maxdiffusion.max_utils import calculate_model_tflops
from maxdiffusion.models.attention_flax import FlaxAttention
from .. import pyconfig, max_utils

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class FlopCalculation(unittest.TestCase):

  def setUp(self):
    FlopCalculation.dummy_data = {}
    pyconfig.initialize([None, os.path.join(THIS_DIR, "..", "configs", "base21.yml")], unittest=True)
    self.config = pyconfig.config
    devices_array = max_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)

  def test_dense_layer_model_flops(self):
    class SimpleLinearModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(20)(x)
        x = nn.Dense(15)(x)
        x = nn.Dense(1)(x)
        return x

    model = SimpleLinearModel()
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 10))

    training_tflops = calculate_model_tflops(model, rng, train=True, x=x)
    macs = (10 * 20) + (20 * 15) + (15 * 1)
    forward_tflops = (2 * macs) / 10**12
    calculated_training_tflops = 3 * forward_tflops

    assert abs(1 - (training_tflops / calculated_training_tflops)) * 100 < 5

  def test_conv_layer_model_flops(self):
    class SimpleConv(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.Dense(10)(x)
        return x

    model = SimpleConv()
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 28, 28, 1))
    with self.mesh:
      training_tflops = calculate_model_tflops(model, rng, train=True, x=x)
    macs = (3 * 3 * 28 * 28 * 16) + (3 * 3 * 28 * 28 * 32 * 16) + (28 * 28 * 32 * 10)
    forward_tflops = (2 * macs) / 10**12
    calculated_training_tflops = 3 * forward_tflops

    assert abs(1 - (training_tflops / calculated_training_tflops)) * 100 < 5

  def test_attn_layer_model_flops(self):
    class SimpleAttn(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = FlaxAttention(
            query_dim=320,
            heads=5,
            dim_head=64,
        )(x)

    model = SimpleAttn()
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 9216, 320))
    with self.mesh:
      training_tflops = calculate_model_tflops(model, rng, train=True, x=x)
    # For linears before attn
    qkv_macs = 3 * (320 * 320 * 9216)
    qkv_tflops = 2 * qkv_macs / 10**12

    # Estimation of qk einsum, scaling, softmax and attn_val*v flops.
    attn_tflops = 4 * (320 * 9216**2) / 10**12

    # out proj
    out_proj_mac = 4 * (320 * 9216)
    out_proj_tflops = 2 * out_proj_mac / 10**12

    forward_tflops = qkv_tflops + attn_tflops + out_proj_tflops
    calculated_training_tflops = 3 * forward_tflops

    assert abs(1 - (training_tflops / calculated_training_tflops)) * 100 < 5


if __name__ == "__main__":
  absltest.main()
