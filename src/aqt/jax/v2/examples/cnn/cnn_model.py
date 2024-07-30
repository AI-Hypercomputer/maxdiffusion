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

"""A simple CNN model and MNIST dataset function."""

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

ModuleDef = Any


class CNN(nn.Module):
  """A simple CNN model."""

  bn_use_stats: bool
  dot_general_cls: ModuleDef = lambda: jax.lax.dot_general
  einsum_cls: ModuleDef = lambda: jnp.einsum

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
    x = nn.Dense(features=256, dot_general_cls=self.dot_general_cls)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10, dot_general_cls=self.dot_general_cls)(x)

    identity = jnp.identity(10, dtype=x.dtype)
    x = self.einsum_cls()('ab,bc->ac', x, identity)
    return x


def get_datasets() -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
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
