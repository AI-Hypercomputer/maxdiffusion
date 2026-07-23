"""
LTX2 implementation of the model-agnostic `BlockBenchmark` (see tile_size_grid_search.py).
"""
import os
import functools

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
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.utils.tile_size_grid_search import (
    BenchResult,
    BlockBenchmark,
    grid_search,
    time_callable,
)

_IN_CHANNELS = 128  # LTX2 uses 128 channels in latent space
_VAE_T, _VAE_S = (
    8,
    32,
)  # LTX2 spatial/temporal compression: patch_size_t=1, patch_size=1 (it actually operates on 32x32 compressed) Wait, LTX2 spatial downsampling is 32x32 in pixel space, so in latent space it's 1x1 patch? No, let's just use the latent shape.


def latent_seq_len(num_frames: int, height: int, width: int) -> int:
  lf = (num_frames - 1) // 8 + 1
  lh, lw = height // 32, width // 32
  return lf * lh * lw


def tiled_seq_len(full_seq: int, attention: str, context_shards: int, ulysses_shards: int) -> int:
  _RING_VARIANTS = {"ulysses_ring_custom", "ulysses_ring_custom_bidir", "tokamax_ring", "ring"}
  if attention in _RING_VARIANTS and ulysses_shards >= 1 and context_shards >= 1:
    ring_shards = max(1, context_shards // max(1, ulysses_shards))
    return full_seq // ring_shards
  return full_seq


@functools.partial(nnx.jit, static_argnames=("num_frames", "height", "width"))
def _forward(
    model,
    latents,
    timestep,
    prompt_embeds,
    prompt_attention_mask,
    audio_latents,
    audio_prompt_embeds,
    audio_prompt_attention_mask,
    *,
    num_frames,
    height,
    width,
):
  return model(
      hidden_states=latents,
      audio_hidden_states=audio_latents,
      encoder_hidden_states=prompt_embeds,
      audio_encoder_hidden_states=audio_prompt_embeds,
      timestep=timestep,
      audio_timestep=None,
      sigma=None,
      audio_sigma=None,
      encoder_attention_mask=prompt_attention_mask,
      audio_encoder_attention_mask=audio_prompt_attention_mask,
      num_frames=num_frames,
      height=height,
      width=width,
      fps=24.0,
      audio_num_frames=128,
      video_coords=None,
      audio_coords=None,
      attention_kwargs=None,
      use_cross_timestep=False,
  )


class LTX2BlockBenchmark(BlockBenchmark):

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
    self._attention = getattr(config, "attention", "flash")
    self._context_shards = int(mesh.shape.get("context", 1))
    self._ulysses_shards = int(getattr(config, "ulysses_shards", 1) or 1)
    self._vmem = int(vmem_limit_bytes)
    self.label = f"ltx2/{self._attention}/u{self._ulysses_shards}"

    self._lf = (num_frames - 1) // 8 + 1
    self._lh, self._lw = height // 32, width // 32
    self._full_seq = latent_seq_len(num_frames, height, width)
    self._num_frames_orig = num_frames
    self._height_orig = height
    self._width_orig = width

    data_shards = int(mesh.shape.get("data", 1)) * int(mesh.shape.get("fsdp", 1))
    self._batch = batch if batch is not None else max(1, data_shards)
    self._hf_cfg = LTX2VideoTransformer3DModel.load_config(config.pretrained_model_name_or_path, subfolder="transformer")
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
      (
          latents,
          timestep,
          prompt_embeds,
          prompt_attention_mask,
          audio_latents,
          audio_prompt_embeds,
          audio_prompt_attention_mask,
      ) = self._inputs
      with self._mesh, nn_partitioning.axis_rules(self._rules):
        mean, std, times, compile_ms = time_callable(
            lambda: _forward(
                model,
                latents,
                timestep,
                prompt_embeds,
                prompt_attention_mask,
                audio_latents,
                audio_prompt_embeds,
                audio_prompt_attention_mask,
                num_frames=self._lf,
                height=self._lh,
                width=self._lw,
            ),
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
    except Exception as e:
      import traceback

      traceback.print_exc()
      msg = str(e)
      oom = any(t in msg for t in ("RESOURCE_EXHAUSTED", "out of memory", "Mosaic", "VMEM"))
      return BenchResult(bq, bkv, cmp, "oom" if oom else "error", detail=msg[:200])

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
    ltx2_config = dict(self._hf_cfg)
    if ltx2_config.get("activation_fn") == "gelu-approximate":
      ltx2_config["activation_fn"] = "gelu"
    ltx2_config.update(
        mesh=self._mesh,
        dtype=c.activations_dtype,
        weights_dtype=c.weights_dtype,
        attention_kernel=c.attention,
        a2v_attention_kernel=getattr(c, "a2v_attention_kernel", "flash"),
        v2a_attention_kernel=getattr(c, "v2a_attention_kernel", "dot_product"),
        precision=max_utils.get_precision(c),
        flash_block_sizes=self._flash_block_sizes(bq, bkv, cmp),
        flash_min_seq_length=getattr(c, "flash_min_seq_length", 4096),
        ulysses_shards=getattr(c, "ulysses_shards", -1),
        ulysses_attention_chunks=getattr(c, "ulysses_attention_chunks", 1),
        remat_policy=getattr(c, "remat_policy", "NONE"),
        scan_layers=False,
        num_layers=1,
    )
    model = LTX2VideoTransformer3DModel(**ltx2_config, rngs=nnx.Rngs(params=0))
    gd, state, rest = nnx.split(model, nnx.Param, ...)
    shardings = nn.logical_to_mesh_sharding(nnx.get_partition_spec(state), self._mesh, self._rules)
    return nnx.merge(gd, jax.device_put(state, shardings), rest)

  def _make_inputs(self):
    dtype = self._config.activations_dtype
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    seq_len = self._lf * self._lh * self._lw
    latents = jax.random.normal(k1, (self._batch, seq_len, _IN_CHANNELS), dtype)

    # Audio latents
    audio_seq_len = 128
    audio_latents = jax.random.normal(k3, (self._batch, audio_seq_len, _IN_CHANNELS), dtype)

    # Prompts
    prompt_embeds = jax.random.normal(k2, (self._batch, 1024, 3840), dtype)
    prompt_attention_mask = jnp.ones((self._batch, 1024), dtype=jnp.int32)

    audio_prompt_embeds = jax.random.normal(k4, (self._batch, 1024, 3840), dtype)
    audio_prompt_attention_mask = jnp.ones((self._batch, 1024), dtype=jnp.int32)

    timestep = jnp.zeros((self._batch,), jnp.float32)
    repl = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
    return (
        jax.device_put(latents, repl),
        jax.device_put(timestep, repl),
        jax.device_put(prompt_embeds, repl),
        jax.device_put(prompt_attention_mask, repl),
        jax.device_put(audio_latents, repl),
        jax.device_put(audio_prompt_embeds, repl),
        jax.device_put(audio_prompt_attention_mask, repl),
    )


def _build_cli_config(argv_yaml, attention, ulysses_shards, num_frames, height, width):
  pyconfig.initialize([
      "ltx2_block_benchmark",
      argv_yaml,
      f"attention={attention}",
      f"ulysses_shards={ulysses_shards}",
      "skip_jax_distributed_system=True",
      "weights_dtype=bfloat16",
      "activations_dtype=bfloat16",
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

  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="Path to yaml config (e.g. configs/ltx2_video.yml)")
  parser.add_argument("--attention", default="ulysses_ring_custom")
  parser.add_argument("--ulysses-shards", type=int, default=1)
  parser.add_argument("--num-frames", type=int, default=161)
  parser.add_argument("--height", type=int, default=512)
  parser.add_argument("--width", type=int, default=768)
  parser.add_argument("--smart-search", action="store_true")
  parser.add_argument("--full-search", action="store_true")
  parser.add_argument("--out-dir", default="")
  args = parser.parse_args()

  config = _build_cli_config(args.config, args.attention, args.ulysses_shards, args.num_frames, args.height, args.width)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
  bench = LTX2BlockBenchmark.from_config(config, mesh)

  mode = "full" if args.full_search else "smart"
  grid_search(bench, mode=mode, out_dir=args.out_dir or None, log=max_logging.log)


if __name__ == "__main__":
  main()
