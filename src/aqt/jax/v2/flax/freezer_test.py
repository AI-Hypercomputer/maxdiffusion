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
"""Tests for freezer."""

from typing import Any, Mapping, Sequence, Callable

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2.flax import freezer
import flax
from flax import linen as nn
from flax.core import meta as nn_meta
import jax
from jax import numpy as jnp


@flax.struct.dataclass
class _CustomStructure:
  member: jnp.ndarray
  member_list: Sequence[jnp.ndarray]
  member_dict: Mapping[str, jnp.ndarray]


class TestModel(nn.Module):
  freezer_mode: freezer.FreezerMode
  axis_metadata_wrapper: Callable[..., nn_meta.AxisMetadata] | None = None

  def setup(self):
    self.f = freezer.Freezer(
        'freezer',
        mode=self.freezer_mode,
        axis_metadata_wrapper=self.axis_metadata_wrapper,
    )

  def __call__(self, x):
    """Emulates basic routine on how to use the freezer."""
    self.f.get()
    self.f.set(x)
    return x

  def freezer_get(self):
    return self.f.get()

  def freezer_set(self, x):
    return self.f.set(x)


class FreezerTest(parameterized.TestCase):
  def _create_custom_structure(self, prngkey):
    subkeys = jax.random.split(prngkey, 7)
    value1 = jax.random.normal(subkeys[0], (3, 3))
    value2 = [jax.random.normal(subkeys[i], (2, 2)) for i in [1, 2]]
    value3 = {'sub1': jax.random.normal(subkeys[4], (1, 4)),
              'sub2': jax.random.normal(subkeys[5], (3, 6)),
              'sub3': jax.random.normal(subkeys[6], (7, 7))}
    return _CustomStructure(value1, value2, value3)

  def _assert_same_tree_shape_dtype(self, tree1, tree2):
    """Checks if the two given pytrees have the same structure with the same leaves' shapes and dtypes."""
    leaves1, treedef1 = jax.tree.flatten(tree1)
    leaves2, treedef2 = jax.tree.flatten(tree2)

    # 1. Check if they have the same tree structure.
    self.assertEqual(treedef1, treedef2)

    # 2. Check if the leaves are with the same shapes and dtypes.
    self.assertEqual(len(leaves1), len(leaves2))
    for leaf1, leaf2 in zip(leaves1, leaves2):
      self.assertEqual(leaf1.shape, leaf2.shape)
      self.assertEqual(leaf1.dtype, leaf2.dtype)

  def test_freezer_get_set(self):
    class CustomWrapper(flax.struct.PyTreeNode, nn_meta.AxisMetadata):
      value: Any
      metadata: str = flax.struct.field(default=None, pytree_node=False)

      def unbox(self):
        return self.value

      def replace_boxed(self, val):
        return self.replace(value=val)

      def add_axis(self, index, params):
        return self

      def remove_axis(self, index, params):
        return self

    def axis_metadata_wrapper(x: _CustomStructure):
      ret = x.replace(
          member=CustomWrapper(x.member, 'member metadata'),
          member_list=[
              CustomWrapper(v, 'member list metadata') for v in x.member_list
          ],
      )
      return ret

    def unbox(x):
      if isinstance(x, CustomWrapper):
        return x.value
      return x

    subkeys = jax.random.split(jax.random.PRNGKey(0), 6)
    cs_for_init = self._create_custom_structure(subkeys[0])
    cs = self._create_custom_structure(subkeys[1])
    cs2 = self._create_custom_structure(subkeys[2])

    # 1. NONE mode test.
    tm_none = TestModel(freezer_mode=freezer.FreezerMode.NONE)
    param_init_none = tm_none.init(subkeys[3], cs_for_init)
    get_none = tm_none.apply(param_init_none, method=TestModel.freezer_get)
    _, param_set_none = tm_none.apply(
        param_init_none, cs, method=TestModel.freezer_set, mutable=True
    )
    get_after_set_none = tm_none.apply(
        param_set_none, method=TestModel.freezer_get
    )

    self.assertEqual(dict(), param_init_none)
    self.assertIsNone(get_none)
    self.assertIsNone(get_after_set_none)
    self.assertEqual(dict(), param_set_none)

    # 2. WRITE mode test.
    tm_write = TestModel(
        freezer_mode=freezer.FreezerMode.WRITE,
        axis_metadata_wrapper=axis_metadata_wrapper,
    )
    param_init_write = tm_write.init(subkeys[4], cs_for_init)

    # Check if the init parameters are properly wrapped.
    cs_frozen = param_init_write['freezer']['f']['frozen']
    self.assertIsInstance(cs_frozen.member, CustomWrapper)
    self.assertEqual(cs_frozen.member.metadata, 'member metadata')
    for value in cs_frozen.member_list:
      self.assertIsInstance(value, CustomWrapper)
      self.assertEqual(value.metadata, 'member list metadata')

    # Unbox the initialization parameters.
    # The metadata in the box is used to shard the variables here.
    param_init_write = jax.tree.map(
        unbox, param_init_write, is_leaf=lambda x: isinstance(x, CustomWrapper)
    )

    get_write = tm_write.apply(param_init_write, method=TestModel.freezer_get)
    _, param_set_write = tm_write.apply(
        param_init_write, cs, method=TestModel.freezer_set, mutable=True
    )
    get_after_set_write = tm_write.apply(
        param_set_write, method=TestModel.freezer_get
    )

    # Check if the init parameters are properly initialized with the init value.
    self._assert_same_tree_shape_dtype(
        cs_for_init, param_init_write['freezer']['f']['frozen']
    )

    # Since WRITE mode, get method should always return None.
    self.assertIsNone(get_write)
    self.assertIsNone(get_after_set_write)

    # Check if the parameter is properly updated after the set.
    self.assertEqual(cs, param_set_write['freezer']['f']['frozen'])

    # 3. READ mode test.
    tm_read = TestModel(
        freezer_mode=freezer.FreezerMode.READ,
        axis_metadata_wrapper=axis_metadata_wrapper,
    )
    param_init_read = tm_read.init(subkeys[5], cs_for_init)

    # Check if the init parameters are properly wrapped.
    cs_frozen = param_init_read['freezer']['f']['frozen']
    self.assertIsInstance(cs_frozen.member, CustomWrapper)
    self.assertEqual(cs_frozen.member.metadata, 'member metadata')
    for value in cs_frozen.member_list:
      self.assertIsInstance(value, CustomWrapper)
      self.assertEqual(value.metadata, 'member list metadata')

    # Unbox the initialization parameters.
    param_init_read = jax.tree.map(
        unbox, param_init_read, is_leaf=lambda x: isinstance(x, CustomWrapper)
    )

    # The tree structure initialized with READ should be the same with the one
    # after WRITE.
    self._assert_same_tree_shape_dtype(param_set_write, param_init_read)

    # get method should return the value specified by set.
    get_read = tm_read.apply(param_set_write, method=TestModel.freezer_get)

    self.assertEqual(cs, get_read)

    # set method should be ineffective.
    _, param_set_read = tm_read.apply(
        param_set_write, cs2, method=TestModel.freezer_set, mutable=True
    )
    get_after_set_read = tm_read.apply(
        param_set_read, method=TestModel.freezer_get
    )

    self.assertEqual(cs, get_after_set_read)


if __name__ == '__main__':
  absltest.main()
