# Copyright 2024 Google LLC
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

"""Utilities for model training and serving."""

from collections.abc import Callable
import functools
import sys
from typing import Any

from flax import linen as nn
from flax import struct
from flax import typing as flax_typing
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import numpy as np
import optax


class TrainState(struct.PyTreeNode):
  """Train state."""

  train_model: nn.Module = struct.field(pytree_node=False)
  eval_model: nn.Module = struct.field(pytree_node=False)
  model_vars: Any = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)


def create_train_state(
    rng: jax.Array,
    train_model: nn.Module,
    eval_model: nn.Module,
) -> TrainState:
  """Creates initial `TrainState`."""
  model_vars = train_model.init({'params': rng}, jnp.ones([1, 28, 28, 1]))
  learning_rate = 0.1
  momentum = 0.9
  tx = optax.sgd(learning_rate, momentum)
  return TrainState(
      train_model=train_model,
      eval_model=eval_model,
      model_vars=model_vars,
      tx=tx,
      opt_state=tx.init(model_vars['params']),
  )


def prepare_data_perm(
    ds: dict[str, jax.Array],
    batch_size: int,
    rng: jax.Array,
    num_steps: int = sys.maxsize,
) -> jax.Array:
  """Creates random permutation of data."""
  ds_size = len(ds['image'])
  num_steps = min(num_steps, ds_size // batch_size)
  perms = jax.random.permutation(rng, len(ds['image']))
  perms = perms[: num_steps * batch_size]  # skip incomplete batch
  return perms.reshape((num_steps, batch_size))


@functools.partial(jax.jit, static_argnums=(3,))
def _apply_model(
    model_vars: flax_typing.FrozenVariableDict,
    images: Any,
    labels: Any,
    apply_fn: Callable[..., Any],
) -> tuple[jax.Array, jax.Array, jax.Array, flax_typing.FrozenVariableDict]:
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(model_vars):
    logits, updated_vars = apply_fn(
        model_vars,
        images,
        rngs={'params': jax.random.PRNGKey(0)},
        mutable=True,
    )
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, (logits, updated_vars)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
  aux, grads = grad_fn(model_vars)
  loss, (logits, updated_vars) = aux
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

  return grads, loss, accuracy, updated_vars


@jax.jit
def _update_model(state, grads, model_vars) -> TrainState:
  """Updates parameters in model_vars with given gradients."""
  params = state.model_vars['params']
  param_grad = grads['params']
  updates, new_opt_state = state.tx.update(param_grad, state.opt_state, params)
  new_params = optax.apply_updates(params, updates)
  model_vars.update(params=new_params)
  return state.replace(
      model_vars=model_vars,
      opt_state=new_opt_state,
  )


def _train_epoch(
    state: TrainState,
    train_ds: dict[str, jax.Array],
    batch_size: int,
    rng: jax.Array,
) -> tuple[TrainState, float, float]:
  """Trains for a single epoch."""
  perms = prepare_data_perm(train_ds, batch_size, rng)

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy, updated_var = _apply_model(
        state.model_vars, batch_images, batch_labels, state.train_model.apply
    )
    state = _update_model(state, grads, updated_var)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


def train_and_evaluate(
    num_epochs: int,
    workdir: str,
    train_ds: dict[str, jax.Array],
    test_ds: dict[str, jax.Array],
    state: TrainState,
) -> TrainState:
  """Execute model training and evaluation loop."""
  rng = jax.random.key(0)
  summary_writer = tensorboard.SummaryWriter(workdir)

  batch_size = 128
  for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = _train_epoch(
        state, train_ds, batch_size, input_rng
    )
    _, test_loss, test_accuracy, _ = _apply_model(
        state.model_vars,
        test_ds['image'],
        test_ds['label'],
        state.eval_model.apply,
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


def serve(
    serve_model: nn.Module,
    model_vars: flax_typing.FrozenVariableDict,
    test_ds: Any,
) -> jax.Array:
  """Serves a given model with given vars and computes test loss."""
  sample_image, sample_label = test_ds['image'][:64], test_ds['label'][:64]
  logits = jax.jit(serve_model.apply)(
      model_vars, sample_image, rngs={'params': jax.random.PRNGKey(0)}
  )
  # compute test loss
  one_hot = jax.nn.one_hot(sample_label, 10)
  loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
  return loss
