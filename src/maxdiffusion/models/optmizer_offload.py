
import functools
from typing import Any, Mapping

from absl import flags

from absl import logging
import jax
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
import numpy as np
import optax

def with_memory_kind(t, memory_kind):
  return jax.tree_util.tree_map(
      lambda x: x.with_memory_kind(kind=memory_kind), t
  )

def optimizable(ps: Mapping[str, Mapping[str, Any]]):
      if freeze_embeddings:
        return {
            'params': {
                k: v for k, v in ps['params'].items() if k != 'embeddings'
            }
        }
      else:
        return ps

def merge_frozen_params(
    new_params: Mapping[str, Mapping[str, Any]],
    old_params: Mapping[str, Mapping[str, Any]],
):
    if freeze_embeddings:
    return dict(
        params=dict(
            embeddings=old_params['params']['embeddings'],
            **new_params['params'],
        )
    )
    return new_params
    
def optimizer_update(params, opt_state, grads):
    updates, opt_state = optimizer.update(grads, opt_state)
    updated_params = optax.apply_updates(optimizable(params), updates)
    params = merge_frozen_params(updated_params, params)
    return params, opt_state

def cast_to_bf16(params):
    return cast_dtype_from_to(params, np.float32, jnp.bfloat16)

def update(params, opt_state, data):
    if offload_state and not offload_compute:
    device_params = jax.device_put(
        params, with_memory_kind(param_sharding, 'device')
    )
    cast_params = cast_to_bf16(device_params)
    elif offload_state:
    cast_params = compute_on('device_host')(jax.jit(cast_to_bf16))(params)
    cast_params = jax.device_put(
        cast_params, with_memory_kind(param_sharding, 'device')
    )
    else:
    cast_params = params
    loss, grads = jax.value_and_grad(transformer.loss_fn(model))(
        cast_params, data
    )
    grads = optimizable(grads)
    if offload_state and offload_compute:
    grads = jax.device_put(
        grads, with_memory_kind(optimizable_param_sharding, 'pinned_host')
    )
    elif offload_state:
    params = jax.device_put(
        params, with_memory_kind(param_sharding, 'device')
    )
    opt_state = jax.device_put(
        opt_state, with_memory_kind(opt_state_sharding, 'device')
    )
    if offload_compute:
    params, opt_state = compute_on('device_host')(
        jax.jit(optimizer_update)
    )(params, opt_state, grads)
    else:
    params, opt_state = optimizer_update(params, opt_state, grads)
    if offload_state and not offload_compute:
    params = jax.device_put(
        params, with_memory_kind(param_sharding, 'pinned_host')
    )
    opt_state = jax.device_put(
        opt_state, with_memory_kind(opt_state_sharding, 'pinned_host')
    )
    return params, opt_state, loss

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data')
    )
    jitted_update = jax.jit(
        update,
        in_shardings=(param_sharding, opt_state_sharding, data_sharding),
        out_shardings=(param_sharding, opt_state_sharding, None),
        # donate_argnums=(0, 1),
    )
