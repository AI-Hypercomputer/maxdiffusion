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
"""Mnist example for intercept method API."""

from absl import app
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import utils as aqt_utils
from aqt.jax.v2.examples import flax_e2e_model_v0
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2.flax.intercept import aqt_intercept_methods
from flax import linen as nn
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import optax


class CNN(nn.Module):
  """A simple CNN model."""
  bn_use_stats: bool

  @nn.compact
  def __call__(self, x):
    use_running_avg = not self.bn_use_stats
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=use_running_avg, dtype=x.dtype)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=use_running_avg, dtype=x.dtype)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)

    # TODO(kimjaehong): Support einsum with the intercept method.
    return x


def create_train_state(rng):
  """Creates initial `flax_e2e_model.TrainState`."""
  cnn_train = CNN(bn_use_stats=True)
  model = cnn_train.init({'params': rng}, jnp.ones([1, 28, 28, 1]))
  learning_rate = 0.1
  momentum = 0.9
  tx = optax.sgd(learning_rate, momentum)
  cnn_eval = CNN(bn_use_stats=False)
  return flax_e2e_model_v0.TrainState(
      cnn_train=cnn_train,
      cnn_eval=cnn_eval,
      model=model,
      tx=tx,
      opt_state=tx.init(model['params']),
  )


def train_and_evaluate(
    num_epochs: int, workdir: str, aqt_cfg: aqt_config.DotGeneral,
    tiling_cfg: tiled_dot_general.Cfg,
) -> flax_e2e_model_v0.TrainState:
  """Execute model training and evaluation loop."""
  train_ds, test_ds = flax_e2e_model_v0.get_datasets()
  rng = jax.random.key(0)

  summary_writer = tensorboard.SummaryWriter(workdir)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng)

  batch_size = 128
  with aqt_intercept_methods.intercept_methods(aqt_cfg, tiling_cfg=tiling_cfg):
    for epoch in range(1, num_epochs + 1):
      rng, input_rng = jax.random.split(rng)
      state, train_loss, train_accuracy = flax_e2e_model_v0.train_epoch(
          state, train_ds, batch_size, input_rng
      )
      _, test_loss, test_accuracy, _ = flax_e2e_model_v0.apply_model(
          state, test_ds['image'], test_ds['label'], train=False
      )

      print(
          'epoch:% 3d, train_loss: %.30f, train_accuracy: %.30f, test_loss:'
          ' %.30f, test_accuracy: %.30f'
          % (
              epoch,
              train_loss,
              train_accuracy * 100,
              test_loss,
              test_accuracy * 100,
          ),
          flush=True,
      )

      summary_writer.scalar('train_loss', train_loss, epoch)
      summary_writer.scalar('train_accuracy', train_accuracy, epoch)
      summary_writer.scalar('test_loss', test_loss, epoch)
      summary_writer.scalar('test_accuracy', test_accuracy, epoch)

    summary_writer.flush()
  return state


def serving_conversion(train_state, aqt_cfg, tiling_cfg):
  """Model conversion (quantized weights freezing)."""
  input_shape = (1, 28, 28, 1)
  cnn_freeze = CNN(
      bn_use_stats=False,
  )
  with aqt_intercept_methods.intercept_methods(
      aqt_cfg,
      rhs_quant_mode=aqt_utils.QuantMode.CONVERT,
      lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
      rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
      tiling_cfg=tiling_cfg):
    _, model_serving = cnn_freeze.apply(
        train_state.model,
        jnp.ones(input_shape),
        rngs={'params': jax.random.PRNGKey(0)},
        mutable=True,
    )
  cnn_serve = CNN(
      bn_use_stats=False,
  )

  serve_fn = aqt_intercept_methods.intercept_wrapper(
      cnn_serve.apply,
      aqt_cfg,
      rhs_quant_mode=aqt_utils.QuantMode.SERVE,
      lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
      rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
      tiling_cfg=tiling_cfg)
  return serve_fn, model_serving


@jax.jit
def serve(state,
          aqt_cfg: aqt_config.DotGeneral,
          tiling_cfg: tiled_dot_general.Cfg):
  """Take train state, freeze integer weights, and serve."""
  # get sample serving data
  _, test_ds = flax_e2e_model_v0.get_datasets()
  sample_image, sample_label = test_ds['image'][:64], test_ds['label'][:64]
  # serving
  serve_fn, model_serving = serving_conversion(state, aqt_cfg, tiling_cfg)
  logits = serve_fn(
      model_serving, sample_image, rngs={'params': jax.random.PRNGKey(0)}
  )
  # compute serving loss
  one_hot = jax.nn.one_hot(sample_label, 10)
  loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
  return loss


def main(argv):
  del argv
  aqt_cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
  tiling_cfg = tiled_dot_general.Cfg(
      lhs=tiled_dot_general.TensorTiling(
          contraction_axes=[
              tiled_dot_general.AxisTiling(
                  axis=1, tile_count=2, tile_size=None
              ),
          ],
          remaining_axes=[],
      ),
      rhs=tiled_dot_general.TensorTiling(
          contraction_axes=[
              tiled_dot_general.AxisTiling(
                  axis=0, tile_count=2, tile_size=None
              ),
          ],
          remaining_axes=[],
      ),
  )
  state = train_and_evaluate(
      num_epochs=2, workdir='/tmp/aqt_mnist_example',
      aqt_cfg=aqt_cfg, tiling_cfg=tiling_cfg
  )
  loss = serve(state, aqt_cfg, tiling_cfg)
  print('serve loss on sample ds: {}'.format(loss))


if __name__ == '__main__':
  app.run(main)
