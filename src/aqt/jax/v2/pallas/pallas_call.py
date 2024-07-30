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
"""Module for calling pallas functions from JAX."""

import collections
from typing import Any, Callable

from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.pallas import pallas_tensor
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import lazy_object_proxy


BlockSpec = pl.BlockSpec
no_block_spec = pl.no_block_spec
tree_util = jax.tree_util

QTensor = aqt_tensor.QTensor
TransposedTensor = pallas_tensor.TransposedTensor

ArgAndBlockSpec = collections.namedtuple('ArgAndBlockSpec', 'arg block_spec')


def _make_qtensor_blockspec(arg, block_spec) -> QTensor | Any:
  if isinstance(arg, QTensor) and isinstance(block_spec, BlockSpec):
    return pallas_tensor.make_qtensor_blockspec(arg, block_spec)
  else:
    return block_spec


def _transpose_tensor_for_memory_saving(
    arg: Any, block_spec: BlockSpec
) -> ArgAndBlockSpec:
  """Transposes tensor for memory optimization."""
  arg, block_spec = pallas_tensor.transpose_tensor_for_memory_saving(
      arg, block_spec
  )
  return ArgAndBlockSpec(arg, block_spec)


def _is_qtensor(x):
  return isinstance(x, QTensor)


def _is_transposed_tensor(x):
  return isinstance(x, TransposedTensor)


def _is_arg_and_block_spec(x):
  return isinstance(x, ArgAndBlockSpec)


def pallas_call(
    f: Callable[..., None],
    *pl_call_args,
    grid_spec=None,
    in_specs=no_block_spec,
    **pl_call_kwrags,
):
  """pl.pallas_call wrapper that can pass QTensor as input."""

  # If grid spec is given, use in_specs from grid spec.
  if grid_spec is not None:
    in_specs = tree_util.tree_unflatten(
        grid_spec.in_specs_tree, grid_spec.in_specs
    )

  @jax.jit
  def wrapped(*args):

    prefetch_args = ()
    if isinstance(grid_spec, pltpu.PrefetchScalarGridSpec):
      prefetch_args, args = (
          args[: grid_spec.num_scalar_prefetch],
          args[grid_spec.num_scalar_prefetch :],
      )

    # Flatten args and its inspecs.
    flat_args, args_treedef = tree_util.tree_flatten(
        args, is_leaf=_is_qtensor
    )
    flat_inspecs, inspecs_treedef = tree_util.tree_flatten(in_specs)

    # Build block spec for each QTensor in arguments.
    flat_inspecs = tree_util.tree_map(
        _make_qtensor_blockspec,
        flat_args,
        flat_inspecs,
        is_leaf=_is_qtensor,
    )

    # Transpose tensor for memory optimization.
    flat_args_and_inspecs = tree_util.tree_map(
        _transpose_tensor_for_memory_saving,
        flat_args,
        flat_inspecs,
    )

    flat_args = tree_util.tree_map(
        lambda x: x.arg, flat_args_and_inspecs, is_leaf=_is_arg_and_block_spec
    )
    flat_inspecs = tree_util.tree_map(
        lambda x: x.block_spec,
        flat_args_and_inspecs,
        is_leaf=_is_arg_and_block_spec,
    )

    # Unflatten args and its inspecs.
    args = tree_util.tree_unflatten(
        args_treedef, flat_args
    )
    kernel_inspecs = tree_util.tree_unflatten(
        inspecs_treedef, flat_inspecs
    )

    def kernel(*args):
      # [Inside kernel] untranspose tensor to restore to the original shape.
      args = jax.tree_util.tree_map(
          # Lazily untranspose tensor when the transposed tensor is actually
          # used. This is to avoid unnecesary load & permutation when the tensor
          # is not used.
          lambda arg: lazy_object_proxy.Proxy(lambda: arg.untransposed)
          if isinstance(arg, TransposedTensor)
          else arg,
          args,
          is_leaf=_is_transposed_tensor,
      )
      return f(*args)

    if grid_spec is not None:
      kernel_inspecs, inspecs_treedef = tree_util.tree_flatten(kernel_inspecs)
      grid_spec.in_specs = tuple(kernel_inspecs)
      grid_spec.in_specs_tree = inspecs_treedef

    func = pl.pallas_call(
        kernel,
        *pl_call_args,
        grid_spec=grid_spec,
        in_specs=kernel_inspecs if grid_spec is None else None,
        **pl_call_kwrags,
    )
    return func(*prefetch_args, *args)

  return wrapped
