# Copyright 2026 Google LLC / The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feed-forward network modules for NNX / Flax models."""

from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion.max_utils import safe_getattr
from maxdiffusion.models.modeling_flax_utils import get_activation

Array = jax.Array


class NNXSimpleFeedForward(nnx.Module):
  """Simple feed-forward network with linear projections and activation."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      dim_out: Optional[int] = None,
      mult: int = 4,
      activation_fn: str = "gelu",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
      sharding_specs: Optional[Any] = None,
  ):
    inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim

    net_0_kernel = safe_getattr(sharding_specs, "net_0_kernel", ("embed", "mlp"))
    net_0_bias = safe_getattr(sharding_specs, "net_0_bias", ("mlp",))
    net_2_kernel = safe_getattr(sharding_specs, "net_2_kernel", ("mlp", "embed"))
    net_2_bias = safe_getattr(sharding_specs, "net_2_bias", ("embed",))

    self.net_0 = nnx.Linear(
        dim,
        inner_dim,
        rngs=rngs,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), net_0_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, net_0_bias),
    )
    self.act = get_activation(activation_fn)
    self.net_2 = nnx.Linear(
        inner_dim,
        dim_out,
        rngs=rngs,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), net_2_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, net_2_bias),
    )

  def __call__(self, hidden_states: Array) -> Array:
    hidden_states = self.net_0(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.net_2(hidden_states)
    return hidden_states
