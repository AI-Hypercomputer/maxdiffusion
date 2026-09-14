"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Head-local placement for pure-Ulysses SVG inference.
"""

from functools import partial

import jax
from jax.sharding import PartitionSpec as P


@jax.custom_jvp
def inference_only(x):
  return x


@inference_only.defjvp
def _reject_derivative(primals, tangents):
  raise NotImplementedError("SVG attention supports inference only.")


def exchange_local(q, k, v, route, *, mesh, qspec, kvspec, ulysses_axis, place, restore, core):
  """Route sharding follows the exact head blocks selected by all_to_all."""
  if qspec != kvspec or qspec[1] is not None:
    raise ValueError("Require identical sequence-sharded QKV with unsharded heads")
  route_spec = P(qspec[0], ulysses_axis)

  @partial(jax.shard_map, mesh=mesh, in_specs=(qspec, kvspec, kvspec, route_spec), out_specs=qspec, check_vma=False)
  def local(q, k, v, route):
    a2a = partial(jax.lax.all_to_all, axis_name=ulysses_axis, tiled=True)
    q, k, v = (a2a(x, split_axis=1, concat_axis=2) for x in (q, k, v))
    with jax.named_scope("head_local_placement"):
      q, k, v = place(q, k, v, route)
    out = core(q, k, v)
    with jax.named_scope("head_local_restore"):
      out = restore(out, route)
    out = a2a(out, split_axis=2, concat_axis=1)
    return out

  return local(q, k, v, route)
