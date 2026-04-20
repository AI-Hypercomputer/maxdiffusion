"""
Script to tune flash block sizes for LTX2 model in MaxDiffusion.
"""
import os
import time
import jax
import jax.numpy as jnp
import flax
from flax import nnx
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning
import flax.linen as nn
from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh, device_put_replicated
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.common_types import BlockSizes

jax.config.update("jax_use_shardy_partitioner", True)
try:
  flax.config.update("flax_always_shard_variable", False)
except LookupError:
  pass


def create_model(config, mesh, block_sizes):
  key = jax.random.key(42)  # Fixed seed for identical weights
  rngs = nnx.Rngs(key)

  def model_factory(rngs):
    return LTX2VideoTransformer3DModel(
        rngs=rngs,
        in_channels=128,
        out_channels=128,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=32,
        attention_head_dim=128,
        cross_attention_dim=4096,
        caption_channels=3840,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_cross_attention_dim=2048,
        num_layers=48,  # Full model
        mesh=mesh,
        attention_kernel="flash",
        flash_block_sizes=block_sizes,
        flash_min_seq_length=4096,
        dtype=jnp.bfloat16,
        weights_dtype=jnp.bfloat16,
    )

  # Use eval_shape to avoid allocating full parameters on default device
  transformer = nnx.eval_shape(model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(transformer, nnx.Param, ...)

  logical_state_spec = nnx.get_partition_spec(state)
  logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
  logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))

  flat_state = dict(nnx.to_flat_state(state))
  for path, shape_dtype in flat_state.items():
    sharding = logical_state_sharding[path].value
    val = jnp.zeros(shape_dtype.shape, dtype=shape_dtype.dtype)
    flat_state[path].value = device_put_replicated(val, sharding)

  state = nnx.from_flat_state(flat_state)

  def init_dummy_shape(node):
    if isinstance(node, jax.ShapeDtypeStruct):
      if jax.dtypes.issubdtype(node.dtype, jax.dtypes.prng_key):
        dummy_key = jax.random.key(0)
        if node.shape == ():
          return dummy_key
        return jax.random.split(dummy_key, node.shape[0])
      return jnp.zeros(node.shape, dtype=node.dtype)
    return node

  rest_of_state = jax.tree_util.tree_map(init_dummy_shape, rest_of_state)

  model = nnx.merge(graphdef, state, rest_of_state)
  return model


def run_tuning(block_q=None, block_kv_compute=None, block_kv=None):
  # Initialize config
  script_dir = os.path.dirname(os.path.abspath(__file__))
  config_path = os.path.join(script_dir, "configs", "ltx2_video.yml")

  print(f"Loading config from: {config_path}")
  pyconfig.initialize(
      [
          None,
          config_path,
          "per_device_batch_size=0.125",
          "ici_data_parallelism=2",
          "ici_context_parallelism=4",
          "ici_tensor_parallelism=1",
          "ici_fsdp_parallelism=1",
          "attention=flash",
      ],
      unittest=True,
  )
  config = pyconfig.config

  # Create mesh
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  print(f"Mesh created: {mesh}")

  # Define search space for elaborate grid search (multiples of 256)
  block_q_options = [512, 1024, 1536, 2048]
  block_kv_compute_options = [512, 1024, 1536, 2048]
  block_kv_options = [1024, 2048, 3072, 4096]

  if block_q is not None:
    block_q_options = [block_q]
  if block_kv_compute is not None:
    block_kv_compute_options = [block_kv_compute]
  if block_kv is not None:
    block_kv_options = [block_kv]

  best_time = float("inf")
  best_comb = None

  # Dummy inputs
  # User runs with per_device_batch_size = 0.125, which gives global_batch_size = 1.
  # But CFG (Classifier-Free Guidance) doubles the batch size to 2.
  # So we use global_batch_size = 2 here to match the actual tensor shape.
  per_device_batch_size = 0.125
  global_batch_size = 2
  seq_len = 6144  # Updated to match user's actual sequence length
  audio_seq_len = 126

  hidden_states = jnp.zeros((global_batch_size, seq_len, 128), dtype=jnp.bfloat16)
  audio_hidden_states = jnp.zeros((global_batch_size, audio_seq_len, 128), dtype=jnp.bfloat16)
  timestep = jnp.ones((global_batch_size,), dtype=jnp.bfloat16)
  encoder_hidden_states = jnp.zeros((global_batch_size, 128, 3840), dtype=jnp.bfloat16)
  audio_encoder_hidden_states = jnp.zeros((global_batch_size, 128, 3840), dtype=jnp.bfloat16)

  for bq in block_q_options:
    for bkv_c in block_kv_compute_options:
      for bkv in block_kv_options:
        # Enforce that block_kv must be a multiple of block_kv_compute
        if bkv % bkv_c != 0:
          continue

        print(f"\nTrying combination: block_q={bq}, block_kv_compute={bkv_c}, block_kv={bkv}")

        block_sizes = BlockSizes(
            block_q=bq,
            block_kv_compute=bkv_c,
            block_kv=bkv,
            block_q_dkv=bq,
            block_kv_dkv=bkv,
            block_kv_dkv_compute=bkv_c,
            block_q_dq=None,
            block_kv_dq=None,
            use_fused_bwd_kernel=True,
        )

        try:
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            model = create_model(config, mesh, block_sizes)

            graphdef, state = nnx.split(model)

            @nnx.jit
            def step_fn(
                graphdef, state, hidden_states, audio_hidden_states, encoder_hidden_states, audio_encoder_hidden_states
            ):
              model_local = nnx.merge(graphdef, state)
              return model_local(
                  hidden_states=hidden_states,
                  audio_hidden_states=audio_hidden_states,
                  encoder_hidden_states=encoder_hidden_states,
                  audio_encoder_hidden_states=audio_encoder_hidden_states,
                  timestep=timestep,
                  num_frames=6,
                  height=32,
                  width=32,
                  audio_num_frames=audio_seq_len,
                  return_dict=True,
              )

            # Warmup / Compilation
            print("  Compiling...")
            res = step_fn(
                graphdef, state, hidden_states, audio_hidden_states, encoder_hidden_states, audio_encoder_hidden_states
            )
            jax.block_until_ready(res)

            # Run 12 steps
            times = []
            for i in range(12):
              start = time.time()
              res = step_fn(
                  graphdef, state, hidden_states, audio_hidden_states, encoder_hidden_states, audio_encoder_hidden_states
              )
              jax.block_until_ready(res)
              end = time.time()
              times.append(end - start)
              print(f"  Step {i}: {end - start:.4f}s")

            avg_time = sum(times[2:]) / 10
            print(f"  Average time (last 10 steps): {avg_time:.4f}s")

            # Append to a results file to track across processes
            results_file = "flash_attention_tuning_results.csv"
            file_exists = os.path.exists(results_file)
            with open(results_file, "a") as f:
              if not file_exists:
                f.write("block_q,block_kv_compute,block_kv,average_time\n")
              f.write(f"{bq},{bkv_c},{bkv},{avg_time:.4f}\n")

            if avg_time < best_time:
              best_time = avg_time
              best_comb = (bq, bkv_c, bkv)

        except Exception as e:
          print(f"  invalid combination. Error: {e}")
          import traceback

          traceback.print_exc()
        finally:
          # Clear memory to avoid OOM between iterations
          if "model" in locals():
            del model
          import gc

          gc.collect()
          jax.clear_caches()

  print(f"\n{'='*40}")
  if best_comb:
    print(f"Best combination: block_q={best_comb[0]}, block_kv_compute={best_comb[1]}, block_kv={best_comb[2]}")
    print(f"Best average time: {best_time:.4f}s")
  else:
    print("No valid combination found.")
  print(f"{'='*40}")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--block_q", type=int, default=None)
  parser.add_argument("--block_kv_compute", type=int, default=None)
  parser.add_argument("--block_kv", type=int, default=None)
  args = parser.parse_args()

  run_tuning(block_q=args.block_q, block_kv_compute=args.block_kv_compute, block_kv=args.block_kv)
