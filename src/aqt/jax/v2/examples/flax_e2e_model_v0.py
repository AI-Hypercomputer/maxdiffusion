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
"""Mnist example."""

import copy
import functools
import sys
from typing import Any, Callable
from absl import app
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import utils
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2.flax import aqt_flax_calibration
from flax import linen as nn
from flax import struct
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds


class CNN(nn.Module):
  """A simple CNN model."""
  bn_use_stats: bool
  aqt_cfg: aqt_config.DotGeneral
  weights_quant_mode: utils.QuantMode = utils.QuantMode.TRAIN
  activation_quant_mode: utils.QuantMode = utils.QuantMode.TRAIN
  use_legacy_freezer: bool = False

  @nn.compact
  def __call__(self, x):
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
    aqt_dg = functools.partial(
        aqt_flax.AqtDotGeneral,
        self.aqt_cfg,
        lhs_quant_mode=self.activation_quant_mode,
        rhs_quant_mode=self.weights_quant_mode,
        tiling_cfg=tiling_cfg,
        use_legacy_freezer=self.use_legacy_freezer,
        lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
    )
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
    x = nn.Dense(features=256, dot_general_cls=aqt_dg)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10, dot_general_cls=aqt_dg)(x)

    # Simple demonstration of how to quantize einsum.
    # Since rhs is activation, we swap the configuration for lhs and rhs, since
    # activation's channelwise config should be set to 'per_tensor'.
    aqt_cfg_rhs_activation = copy.deepcopy(self.aqt_cfg)
    aqt_cfg_rhs_activation.fwd.dg_quantizer.swap_lhs_and_rhs()

    identity = jnp.identity(10, dtype=x.dtype)
    einsum = aqt_flax.AqtEinsum(
        aqt_cfg_rhs_activation,
        lhs_quant_mode=self.weights_quant_mode,
        rhs_quant_mode=self.activation_quant_mode,
        # These assertions are useful when AqtEinsum definition is far away
        # from usage spot (through injection).
        # This is especially useful when specifying tiling.
        assert_eqn='bc,ab->ac',
        assert_lhs_shape=(10, 10),
        assert_rhs_shape=(None, 10),
        tile_sizes={'b': 5},
        use_legacy_freezer=self.use_legacy_freezer,
        lhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION,
    )
    # Note for AQT developers:
    #   This equation is harder because jnp.einsum and einsum swap lhs and rhs.
    x = einsum('bc,ab->ac', identity, x)
    return x


@functools.partial(jax.jit, static_argnums=(3,))
def apply_model(model_params, images, labels, apply_fn):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(model):
    logits, updated_var = apply_fn(
        model,
        images,
        rngs={'params': jax.random.PRNGKey(0)},
        mutable=True,
    )
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, (logits, updated_var)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
  aux, grads = grad_fn(model_params)
  loss, (logits, updated_var) = aux
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy, updated_var


@jax.jit
def update_model(state, grads, updated_var):
  params = state.model['params']
  param_grad = grads['params']
  updates, new_opt_state = state.tx.update(param_grad, state.opt_state, params)
  new_params = optax.apply_updates(params, updates)
  updated_var.update(params=new_params)
  return state.replace(
      model=updated_var,
      opt_state=new_opt_state,
  )


def _prepare_data_perm(ds, batch_size, rng, num_steps=sys.maxsize):
  ds_size = len(ds['image'])
  num_steps = min(num_steps, ds_size // batch_size)

  perms = jax.random.permutation(rng, len(ds['image']))
  perms = perms[: num_steps * batch_size]  # skip incomplete batch
  perms = perms.reshape((num_steps, batch_size))

  return perms


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  perms = _prepare_data_perm(train_ds, batch_size, rng)

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy, updated_var = apply_model(
        state.model, batch_images, batch_labels, state.cnn_train.apply
    )
    state = update_model(state, grads, updated_var)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  print('get_datasets started')
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
  print('get_datasets DONE')
  return train_ds, test_ds


class TrainState(struct.PyTreeNode):
  """Train state."""

  cnn_train: Any = struct.field(pytree_node=False)
  cnn_eval: Any = struct.field(pytree_node=False)
  model: Any = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)


def create_train_state(rng, aqt_cfg):
  """Creates initial `TrainState`."""
  cnn_train = CNN(bn_use_stats=True, aqt_cfg=aqt_cfg)
  model = cnn_train.init({'params': rng}, jnp.ones([1, 28, 28, 1]))
  learning_rate = 0.1
  momentum = 0.9
  tx = optax.sgd(learning_rate, momentum)
  cnn_eval = CNN(bn_use_stats=False, aqt_cfg=aqt_cfg)
  return TrainState(
      cnn_train=cnn_train,
      cnn_eval=cnn_eval,
      model=model,
      tx=tx,
      opt_state=tx.init(model['params']),
  )


def train_and_evaluate(
    num_epochs: int,
    workdir: str,
    aqt_cfg: aqt_config.DotGeneral | None = None,
    state: TrainState | None = None,
) -> TrainState:
  """Execute model training and evaluation loop."""
  train_ds, test_ds = get_datasets()
  rng = jax.random.key(0)

  summary_writer = tensorboard.SummaryWriter(workdir)

  rng, init_rng = jax.random.split(rng)
  if state is None:
    assert aqt_cfg is not None
    state = create_train_state(init_rng, aqt_cfg)

  batch_size = 128
  for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(
        state, train_ds, batch_size, input_rng
    )
    _, test_loss, test_accuracy, _ = apply_model(
        state.model, test_ds['image'], test_ds['label'], state.cnn_eval.apply
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


def serving_conversion(
    train_state: TrainState,
    weight_only: bool = True,
    legacy_for_freeze: bool = False,
    legacy_for_serve: bool = False
) -> tuple[Callable[..., Any], dict[str, Any]]:
  """Model conversion (quantized weights freezing).

  Convert the model parameter for serving. During conversion, quantized weights
  are created as variables, along with their scales.

  Args:
    train_state: TrainState containing model definitions and params.
    weight_only: If set, does not quantize activations.
    legacy_for_freeze: If set, use legacy freezer during freeze.
    legacy_for_serve: If set, use legacy freezer during serve.

  Returns:
    A tuple of serving function, and converted model parameters.
  """
  aqt_cfg = train_state.cnn_eval.aqt_cfg
  input_shape = (1, 28, 28, 1)
  activation_quant_mode = (
      utils.QuantMode.TRAIN if weight_only else utils.QuantMode.CONVERT
  )
  cnn_freeze = CNN(
      bn_use_stats=False,
      aqt_cfg=aqt_cfg,
      weights_quant_mode=utils.QuantMode.CONVERT,
      activation_quant_mode=activation_quant_mode,
      use_legacy_freezer=legacy_for_freeze
  )
  _, model_serving = cnn_freeze.apply(
      train_state.model,
      jnp.ones(input_shape),
      rngs={'params': jax.random.PRNGKey(0)},
      mutable=True,
  )

  activation_quant_mode = (
      utils.QuantMode.TRAIN if weight_only else utils.QuantMode.SERVE
  )
  cnn_serve = CNN(
      bn_use_stats=False,
      aqt_cfg=aqt_cfg,
      weights_quant_mode=utils.QuantMode.SERVE,
      activation_quant_mode=activation_quant_mode,
      use_legacy_freezer=legacy_for_serve
  )

  return cnn_serve.apply, model_serving


def _merge_pytrees(from_model, to_model):
  """Copies the parameters from from_model to to_model."""
  from_model_flattened, _ = jax.tree_util.tree_flatten_with_path(from_model)
  to_model_flattened, to_model_treedef = jax.tree_util.tree_flatten_with_path(
      to_model
  )
  from_model_kp_to_val = {kp: val for kp, val in from_model_flattened}
  merged_flattened = []
  for kp, val in to_model_flattened:
    if kp in from_model_kp_to_val:
      merged_flattened.append((kp, from_model_kp_to_val[kp]))
    else:
      merged_flattened.append((kp, val))

  merged_model = jax.tree_util.tree_unflatten(
      to_model_treedef, [v for _, v in merged_flattened]
  )

  return merged_model


def update_cfg_with_calibration(aqt_cfg):
  """Updates aqt_cfg for static range calibration."""
  sr_calibration_cls = functools.partial(
      aqt_flax_calibration.MeanOfAbsMaxCalibration,
      quant_collection='qc',
  )

  aqt_config.set_fwd_calibration(aqt_cfg, sr_calibration_cls)

  # For static range calibration, the calibration axis for activation should
  # be set to per_tensor, since its dimensions could be different during
  # training and during inference.
  aqt_cfg.fwd.dg_quantizer.lhs.calib_shared_axes = 'per_tensor'


def calibration_conversion(
    train_state: TrainState,
) -> tuple[Callable[..., Any], dict[str, Any]]:
  """Model conversion (initializing calibration parameters).

  Newly initialize variables to store the quantization statistics collected
  during calibration process.

  Args:
    train_state: TrainState containing model definitions and params.
  Returns:
    A tuple of calibration function, and an updated model parameters with new
    variables to store the gathered quantization statistics.
  """
  cnn_calibrate = CNN(
      bn_use_stats=False,
      aqt_cfg=train_state.cnn_eval.aqt_cfg,
      # Both side should be calibrated.
      weights_quant_mode=utils.QuantMode.CALIBRATE,
      activation_quant_mode=utils.QuantMode.CALIBRATE,
  )

  # Initialize the model, and then load the checkpoint into the initialized
  # parameter dict.
  model_calibrated_init = cnn_calibrate.init(
      jax.random.PRNGKey(0), jnp.ones([1, 28, 28, 1])
  )
  model_calibrated = _merge_pytrees(train_state.model, model_calibrated_init)

  return cnn_calibrate.apply, model_calibrated


def calibrate_epoch(
    calibrate_func,
    model_calibrated,
    train_ds,
    batch_size,
    rng,
    calibration_steps,
):
  """Calibrates for a single epoch."""
  perms = _prepare_data_perm(train_ds, batch_size, rng, calibration_steps)

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]

    # Calibration simply updates model during inference; it does NOT apply any
    # gradients.
    _, _, _, model_calibrated = apply_model(
        model_calibrated, batch_images, batch_labels, calibrate_func
    )

  return model_calibrated


def calibrate(state: TrainState, calibration_steps: int) -> TrainState:
  """Calibrate."""
  train_ds, _ = get_datasets()
  rng = jax.random.key(0)
  batch_size = 128
  calibration_func, model_calibrated = calibration_conversion(state)

  model_calibrated = calibrate_epoch(
      calibration_func,
      model_calibrated,
      train_ds,
      batch_size,
      rng,
      calibration_steps,
  )

  return state.replace(model=model_calibrated)


@functools.partial(jax.jit, static_argnums=(1,))
def serve(state: TrainState, weight_only: bool = True):
  """Take train state, freeze integer weights, and serve."""
  # get sample serving data
  _, test_ds = get_datasets()
  sample_image, sample_label = test_ds['image'][:64], test_ds['label'][:64]
  # serving
  serve_fn, model_serving = serving_conversion(state, weight_only=weight_only)
  logits = serve_fn(
      model_serving, sample_image, rngs={'params': jax.random.PRNGKey(0)}
  )
  # compute serving loss
  one_hot = jax.nn.one_hot(sample_label, 10)
  loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
  return loss


def serve_fn_hlo(state):
  """Example on how to inspect HLO to verify if everything was quantized."""
  # get sample serving data
  _, test_ds = get_datasets()
  sample_image = test_ds['image'][:64]
  # serving
  serve_fn, model_serving = serving_conversion(state)
  # The following XLA graph is only needed for debugging purpose
  hlo = jax.jit(serve_fn).lower(
      model_serving,
      sample_image,
      rngs={'params': jax.random.PRNGKey(0)},
  ).compiler_ir('hlo').as_hlo_module()
  return hlo


def main(argv):
  del argv

  # 1. TRAIN.
  aqt_cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
  state = train_and_evaluate(
      num_epochs=1, workdir='/tmp/aqt_mnist_example', aqt_cfg=aqt_cfg
  )

  # 2. Calibration.
  update_cfg_with_calibration(state.cnn_train.aqt_cfg)
  update_cfg_with_calibration(state.cnn_eval.aqt_cfg)
  state = calibrate(state, calibration_steps=10)

  # 3. TRAIN with the calibrated stats.
  state = train_and_evaluate(
      num_epochs=1, workdir='/tmp/aqt_mnist_example', state=state
  )

  # 4. CONVERT & SERVE.
  loss = serve(state, weight_only=False)
  print('serve loss on sample ds: {}'.format(loss))


if __name__ == '__main__':
  app.run(main)
