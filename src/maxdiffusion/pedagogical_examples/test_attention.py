import os
import time

import jax
from jax.sharding import Mesh
import jax.numpy as jnp
from maxdiffusion.models.attention_flax import FlaxAttention
from maxdiffusion import max_utils, pyconfig

def make_data():
  key = jax.random.PRNGKey(0)
  query_states = jax.random.normal(key, (batch, length, heads * head_depth))

  key = jax.random.PRNGKey(1)
  key_states = jax.random.normal(key, (batch, length, heads * head_depth))

  key = jax.random.PRNGKey(2)
  value_states = jax.random.normal(key, (batch, length, heads * head_depth))

  return query_states, key_states, value_states

def run_time_comparison():
  query_states, key_states, value_states = make_data()

  this_dir = os.path.dirname(os.path.abspath(__file__))
  pyconfig.initialize([None,os.path.join(this_dir,'..','configs','base_2_base.yml')])
  config = pyconfig.config

  attention = FlaxAttention(
    heads * head_depth,
    heads,
    head_depth,
    split_head_dim = True,
    attention="dot_product",
    mesh=None,
    dtype=jnp.bfloat16
  )
  key1, key2 = jax.random.split(jax.random.PRNGKey(0))
  x = jax.random.normal(key1, (batch, length, heads * head_depth))
  params = attention.init(key2, x)['params']

  p_apply = jax.jit(attention.apply).lower({"params" : params}, x).compile()

  start_time = time.perf_counter()
  for _ in range(n_trials):
    dot_attention_out = p_apply({"params" : params}, x).block_until_ready()

  end_time = time.perf_counter()
  total_time = end_time - start_time
  average_time = total_time / n_trials
  print("\n\n Dot product attention Average time:", average_time, "seconds per run")

  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  attention = FlaxAttention(
    heads * head_depth,
    heads,
    head_depth,
    split_head_dim = False,
    attention="flash",
    mesh=mesh,
    dtype=jnp.bfloat16
  )

  params = attention.init(key2, x)['params']

  p_apply = jax.jit(attention.apply).lower({"params" : params}, x).compile()

  start_time = time.perf_counter()
  for _ in range(n_trials):
    flash_attention_out = p_apply({"params" : params}, x).block_until_ready()

  end_time = time.perf_counter()
  total_time = end_time - start_time
  average_time = total_time / n_trials
  print("\n\n Flash attention Average time:", average_time, "seconds per run")

  diff_norm = jnp.linalg.norm(dot_attention_out - flash_attention_out)
  print(f"{diff_norm=}")

  regular_norm = jnp.linalg.norm(dot_attention_out)
  print(f"{regular_norm=}")

  flash_norm = jnp.linalg.norm(flash_attention_out)
  print(f"{flash_norm=}")


global batch
batch = 8
global heads
heads = 10
global length
length = 4096
global head_depth
head_depth = 64
global n_trials
n_trials = 10
global split_head_dim
split_head_dim = True
global scale
scale = 1.0 / jnp.sqrt(head_depth)

run_time_comparison()
