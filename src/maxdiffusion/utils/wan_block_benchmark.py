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

WAN implementation of the model-agnostic `BlockBenchmark` (see tile_size_grid_search.py).

Benchmarks ONE real WanTransformerBlock: it builds a real WanModel with num_layers=1 and
runs its actual forward (patch-embed -> rope -> 1 DiT block -> proj_out), so compilation and
the attention/sharding path match generate_wan.py exactly.  Only the (block_q, block_kv)
change between runs; params are random (we time latency, not correctness).

Usage:
  # standalone
  python -m maxdiffusion.utils.wan_block_benchmark --attention ulysses_ring_custom \
      --ulysses-shards 1 --smart-search --out-dir /tmp/tile_search
  # in generate_wan.py: WanBlockBenchmark.from_config(config, mesh); grid_search(bench, ...)
"""

import os

# Perf flags mirror the wan22 run script; setdefault so we NEVER clobber flags a caller
# (e.g. generate_wan.py) already set before its own TPU init.  Only takes effect for the
# standalone CLI, where this module is imported before jax.
os.environ.setdefault(
    "LIBTPU_INIT_ARGS",
    " ".join([
        "--xla_tpu_dvfs_p_state=7",
        "--xla_tpu_spmd_rng_bit_generator_unsafe=true",
        "--xla_tpu_enable_dot_strength_reduction=true",
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
        "--xla_enable_async_collective_permute=true",
        "--xla_tpu_enable_async_collective_fusion=true",
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
        "--xla_enable_async_all_gather=true",
        "--xla_tpu_scoped_vmem_limit_kib=65536",
        "--xla_tpu_enable_async_all_to_all=true",
        "--xla_tpu_enable_all_experimental_scheduler_features=true",
        "--xla_tpu_enable_latency_hiding_scheduler=true",
        "--xla_tpu_enable_megacore_fusion=true",
    ]),
)
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "bfloat16")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.models.wan.transformers.transformer_wan import WanModel
from maxdiffusion.utils.tile_size_grid_search import (
    BenchResult,
    BlockBenchmark,
    grid_search,
    time_callable,
)

# WAN2.1/2.2 latent geometry (matches bench_block_worker / generate_wan defaults).
_IN_CHANNELS = 16
_VAE_T, _VAE_S = 4, 8  # VAE temporal / spatial compression
_PATCH_T, _PATCH_H, _PATCH_W = 1, 2, 2
_RING_VARIANTS = {
    "ulysses_ring_custom",
    "ulysses_ring_custom_bidir",
    "tokamax_ring",
    "ring",
}


def latent_seq_len(num_frames: int, height: int, width: int) -> int:
  """Full (unsharded) patch-token sequence length the transformer sees."""
  lf = (num_frames - 1) // _VAE_T + 1
  lh, lw = height // _VAE_S, width // _VAE_S
  return (lf // _PATCH_T) * (lh // _PATCH_H) * (lw // _PATCH_W)


def tiled_seq_len(full_seq: int, attention: str, context_shards: int, ulysses_shards: int) -> int:
  """Per-shard sequence the kernel actually TILES (what the bq/bkv candidate math runs on).

  Ring shards the sequence, so each ring step tiles full_seq / R (R = CP / U).  Pure ulysses
  gathers the full sequence per device (heads sharded), and non-CP flash sees the full seq.
  """
  if attention in _RING_VARIANTS and ulysses_shards >= 1 and context_shards >= 1:
    ring_shards = max(1, context_shards // max(1, ulysses_shards))
    return full_seq // ring_shards
  return full_seq


@nnx.jit
def _forward(model, latents, timestep, prompt_embeds):
  return model(
      hidden_states=latents,
      timestep=timestep,
      encoder_hidden_states=prompt_embeds,
      encoder_hidden_states_image=None,
      deterministic=True,
      rngs=None,
  )


class WanBlockBenchmark(BlockBenchmark):
  """BlockBenchmark for a single WAN DiT block.  One-time setup (config, mesh, HF transformer
  config, dummy inputs) in __init__; each `run` rebuilds the 1-layer model with the given block
  sizes (they are baked in at construction) and times a forward with compile excluded.
  """

  def __init__(
      self,
      config,
      mesh,
      *,
      num_frames,
      height,
      width,
      batch=None,
      vmem_limit_bytes=64 * 1024 * 1024,
  ):
    self._config = config
    self._mesh = mesh
    self._rules = config.logical_axis_rules
    self._attention = config.attention
    self._context_shards = int(mesh.shape.get("context", 1))
    self._ulysses_shards = int(getattr(config, "ulysses_shards", 1) or 1)
    self._vmem = int(vmem_limit_bytes)
    self.label = f"wan/{self._attention}/u{self._ulysses_shards}"

    lf = (num_frames - 1) // _VAE_T + 1
    self._lf, self._lh, self._lw = lf, height // _VAE_S, width // _VAE_S
    self._full_seq = latent_seq_len(num_frames, height, width)
    # The 'batch' logical axis maps to ('data', 'fsdp'), so the dummy batch must be divisible
    # by data*fsdp -- default to exactly that (e.g. 2 on dp2) so batch=1 doesn't fail the ring
    # shard_map on dp>1 meshes.
    data_shards = int(mesh.shape.get("data", 1)) * int(mesh.shape.get("fsdp", 1))
    self._batch = batch if batch is not None else max(1, data_shards)
    # HF transformer config fetched once, reused for every rebuild.
    self._hf_cfg = WanModel.load_config(config.pretrained_model_name_or_path, subfolder="transformer")
    self._inputs = self._make_inputs()

  @classmethod
  def from_config(cls, config, mesh, **overrides):
    return cls(
        config,
        mesh,
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        **overrides,
    )

  # --- BlockBenchmark interface ---------------------------------------------------
  def tiled_seq_lens(self):
    s = tiled_seq_len(self._full_seq, self._attention, self._context_shards, self._ulysses_shards)
    return (s, s)

  def vmem_bytes(self):
    return self._vmem

  def run(self, bq, bkv, *, bkv_compute=None, iters=10, warmup=2):
    cmp = bkv_compute or bkv
    try:
      with self._mesh:
        model = self._build_model(bq, bkv, cmp)
      latents, timestep, prompt_embeds = self._inputs
      with self._mesh, nn_partitioning.axis_rules(self._rules):
        mean, std, times, compile_ms = time_callable(
            lambda: _forward(model, latents, timestep, prompt_embeds),
            iters=iters,
            warmup=warmup,
            sync=jax.block_until_ready,
        )
      return BenchResult(
          bq,
          bkv,
          cmp,
          "ok",
          mean_ms=mean,
          std_ms=std,
          times_ms=times,
          compile_ms=compile_ms,
      )
    except Exception as e:  # pylint: disable=broad-except
      msg = str(e)
      oom = any(t in msg for t in ("RESOURCE_EXHAUSTED", "out of memory", "Mosaic", "VMEM"))
      return BenchResult(bq, bkv, cmp, "oom" if oom else "error", detail=msg[:200])

  # --- WAN model build (adapted from scripts/bench_block_worker.py) ----------------
  def _flash_block_sizes(self, bq, bkv, cmp):
    from maxdiffusion.max_utils import CustomFlashBlockSizes

    return CustomFlashBlockSizes(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=cmp,
        block_kv_compute_in=cmp,
        heads_per_tile=1,
        vmem_limit_bytes=self._vmem,
    )

  def _build_model(self, bq, bkv, cmp):
    c = self._config
    wan_config = dict(self._hf_cfg)
    wan_config.update(
        mesh=self._mesh,
        dtype=c.activations_dtype,
        weights_dtype=c.weights_dtype,
        attention=c.attention,
        precision=max_utils.get_precision(c),
        flash_block_sizes=self._flash_block_sizes(bq, bkv, cmp),
        remat_policy=c.remat_policy,
        names_which_can_be_saved=c.names_which_can_be_saved,
        names_which_can_be_offloaded=c.names_which_can_be_offloaded,
        flash_min_seq_length=c.flash_min_seq_length,
        dropout=c.dropout,
        mask_padding_tokens=c.mask_padding_tokens,
        scan_layers=False,
        num_layers=1,
        enable_jax_named_scopes=c.enable_jax_named_scopes,
        attention_config={
            "use_base2_exp": c.use_base2_exp,
            "use_experimental_scheduler": c.use_experimental_scheduler,
            "ulysses_shards": c.ulysses_shards,
        },
    )
    model = WanModel(**wan_config, rngs=nnx.Rngs(params=0))
    gd, state, rest = nnx.split(model, nnx.Param, ...)
    shardings = nn.logical_to_mesh_sharding(nnx.get_partition_spec(state), self._mesh, self._rules)
    return nnx.merge(gd, jax.device_put(state, shardings), rest)

  def _make_inputs(self):
    dtype = self._config.activations_dtype
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    latents = jax.random.normal(k1, (self._batch, _IN_CHANNELS, self._lf, self._lh, self._lw), dtype)
    prompt_embeds = jax.random.normal(k2, (self._batch, 512, 4096), dtype)
    timestep = jnp.zeros((self._batch,), jnp.int32)
    repl = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
    return (
        jax.device_put(latents, repl),
        jax.device_put(timestep, repl),
        jax.device_put(prompt_embeds, repl),
    )


# ================================================================================
# CLI: standalone tile-size search for WAN
# ================================================================================
def _build_cli_config(argv_yaml, attention, ulysses_shards, num_frames, height, width):
  pyconfig.initialize([
      "wan_block_benchmark",
      argv_yaml,
      f"attention={attention}",
      f"ulysses_shards={ulysses_shards}",
      "skip_jax_distributed_system=True",
      "weights_dtype=bfloat16",
      "activations_dtype=bfloat16",
      "flash_min_seq_length=0",
      "per_device_batch_size=1",
      f"num_frames={num_frames}",
      f"height={height}",
      f"width={width}",
      "ici_data_parallelism=1",
      "ici_fsdp_parallelism=1",
      f"ici_context_parallelism={jax.device_count()}",
      "ici_tensor_parallelism=1",
      "allow_split_physical_axes=True",
      "use_base2_exp=true",
      "use_experimental_scheduler=true",
      "enable_jax_named_scopes=false",
  ])
  return pyconfig.config


def main():
  import argparse

  p = argparse.ArgumentParser(description="WAN tile-size (block_q/block_kv) grid search")
  p.add_argument("--config-yml", default="src/maxdiffusion/configs/base_wan_27b.yml")
  p.add_argument("--attention", default="ulysses_ring_custom")
  p.add_argument("--ulysses-shards", type=int, default=1)
  p.add_argument("--num-frames", type=int, default=81)
  p.add_argument("--height", type=int, default=720)
  p.add_argument("--width", type=int, default=1280)
  p.add_argument("--mode", choices=["smart", "full"], default="smart")
  p.add_argument("--smart-search", action="store_true", help="alias for --mode smart")
  p.add_argument("--k", type=int, default=3)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--warmup", type=int, default=2)
  p.add_argument("--step", type=int, default=256, help="full-mode step")
  p.add_argument("--max-configs", type=int, default=None)
  p.add_argument("--vmem-mb", type=int, default=64)
  p.add_argument("--out-dir", default=None)
  args = p.parse_args()

  config = _build_cli_config(
      args.config_yml,
      args.attention,
      args.ulysses_shards,
      args.num_frames,
      args.height,
      args.width,
  )
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
  bench = WanBlockBenchmark.from_config(config, mesh, vmem_limit_bytes=args.vmem_mb * 1024 * 1024)
  mode = "smart" if args.smart_search else args.mode
  grid_search(
      bench,
      mode=mode,
      out_dir=args.out_dir,
      iters=args.iters,
      warmup=args.warmup,
      k=args.k,
      step=args.step,
      max_configs=args.max_configs,
      log=max_logging.log,
  )


if __name__ == "__main__":
  main()
