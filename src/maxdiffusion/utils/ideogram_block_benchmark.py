"""One-block Ideogram 4 benchmark for the tile-size grid search.

The WAN analogue (`wan_block_benchmark.py`) learned the hard way that a
hand-reconstructed DiT block crashes XLA/Mosaic at compile time, so this builds
a REAL `Ideogram4Transformer` with `num_layers=1` and synthetic weights. That
keeps the kernel, the mask plumbing and the RoPE path identical to inference
while skipping the ~27.5 GB checkpoint load, so a candidate costs seconds.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..models.ideogram.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from ..models.ideogram.transformer_ideogram import Ideogram4Config, Ideogram4Transformer
from .tile_size_grid_search import BenchResult, BlockBenchmark, time_callable

# Image tokens are (H/patch)*(W/patch) with patch = patch_size * ae_scale_factor
# = 2 * 8 = 16, matching IdeogramPipeline._build_inputs_cpu.
_IMAGE_PATCH = 16

# TPU7x VMEM per core. q/k/v are bf16 but the splash score tile is f32, which is
# why the candidate math is run with dtype_bytes=4.
_VMEM_BYTES = 128 * 1024 * 1024


def divisor_blocks(seq_len: int, align: int = 128) -> list:
  """Block sizes the splash kernel accepts: the kernel asserts
  `kv_block_size should divide kv_seq_len`, and a block must be lane-aligned.
  For seq_len=4608 this is [128, 256, 384, 512, 768, 1152, 1536, 2304, 4608].
  """
  return [b for b in range(align, seq_len + 1, align) if seq_len % b == 0]


def _shard_like_inference(model, logical_axis_rules, mesh):
  """Same eager device_put path generate_ideogram uses, so block latency is
  measured under the sharding production actually runs."""
  from flax import nnx

  def add_rule(vs):
    vs.set_metadata(sharding_rules=logical_axis_rules)
    return vs

  graphdef, state, rest = nnx.split(model, nnx.Param, ...)
  state = jax.tree.map(add_rule, state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  pspecs = nnx.get_partition_spec(state)
  state = jax.tree.map(lambda x, p: jax.device_put(x, jax.sharding.NamedSharding(mesh, p)), state, pspecs)
  return nnx.merge(graphdef, state, rest)


class IdeogramBlockBenchmark(BlockBenchmark):
  """Times one real Ideogram 4 layer for a given (block_q, block_kv)."""

  label = "ideogram4_block"

  def __init__(
      self,
      mesh,
      *,
      height: int = 1024,
      width: int = 1024,
      text_tokens: int = 512,
      valid_text_tokens: Optional[int] = None,
      batch: int = 1,
      dtype=jnp.bfloat16,
      config: Optional[Ideogram4Config] = None,
      logical_axis_rules=None,
      with_llm_features: bool = False,
  ):
    self.mesh = mesh
    self.batch = batch
    self.dtype = dtype
    # Shard exactly like inference does, otherwise the block runs replicated on
    # one device and the linear layers dominate, compressing the tile-to-tile
    # differences this search exists to resolve.
    self.logical_axis_rules = logical_axis_rules
    # llm_cond_proj is 53248 -> 4608 over the text region (~251 GFLOP), paid ONCE
    # per forward rather than per layer. Including it in a one-layer measurement
    # adds a large constant that buries the tile effect. The unconditional branch
    # genuinely passes llm_features=None, so this is a real configuration.
    self.with_llm_features = with_llm_features
    self.text_tokens = text_tokens
    # Left-padded text, exactly like the pipeline: the real tokens occupy the
    # LAST `valid_text_tokens` columns of the text region.
    self.valid_text_tokens = valid_text_tokens if valid_text_tokens is not None else text_tokens
    self.image_tokens = (height // _IMAGE_PATCH) * (width // _IMAGE_PATCH)
    self.seq_len = self.text_tokens + self.image_tokens
    self.base_config = config or Ideogram4Config()
    self._inputs = None

  @classmethod
  def from_config(cls, config, mesh, text_tokens: int = 512):
    """Build from a maxdiffusion config. `text_tokens` must match the bucket the
    real prompt lands in (`ceil(len/TEXT_TOKEN_BUCKET) * TEXT_TOKEN_BUCKET`);
    it changes the sequence length and therefore the tile candidates."""
    return cls(
        mesh,
        height=config.height,
        width=config.width,
        text_tokens=text_tokens,
        batch=config.per_device_batch_size,
        dtype=jnp.bfloat16 if config.activations_dtype == "bfloat16" else jnp.float32,
    )

  def tiled_seq_lens(self):
    # No context parallelism on this path: every device tiles the full sequence.
    return self.seq_len, self.seq_len

  def vmem_bytes(self):
    return _VMEM_BYTES

  def dtype_bytes(self):
    return 2

  def _build_inputs(self):
    if self._inputs is not None:
      return self._inputs
    b, t, i, n = self.batch, self.text_tokens, self.image_tokens, self.seq_len
    key = jax.random.PRNGKey(0)

    # The transformer takes the FULL sequence: _denoise_step concatenates a
    # zeroed text region onto the image latents before calling it (see
    # ideogram_pipeline._denoise_step), so x is (B, seq, in_channels), not
    # (B, image_tokens, in_channels).
    z_image = jax.random.normal(key, (b, i, self.base_config.in_channels), dtype=jnp.float32)
    z = jnp.concatenate([jnp.zeros((b, t, self.base_config.in_channels), dtype=z_image.dtype), z_image], axis=1)
    llm = jax.random.normal(key, (b, t, self.base_config.llm_features_dim), dtype=jnp.float32)

    # segment_ids: 0 = padding, 1 = valid. Text is left-padded, image is all valid.
    segment_ids = np.zeros((b, n), dtype=np.int32)
    segment_ids[:, t - self.valid_text_tokens :] = 1

    indicator = np.zeros((b, n), dtype=np.int32)
    indicator[:, :t] = LLM_TOKEN_INDICATOR
    indicator[:, t:] = OUTPUT_IMAGE_INDICATOR

    position_ids = np.zeros((b, n, 3), dtype=np.int32)
    position_ids[:, :, 0] = np.arange(n)[None, :]

    self._inputs = (
        llm if self.with_llm_features else None,
        z,
        jnp.full((b,), 0.5, dtype=jnp.float32),
        jnp.asarray(position_ids),
        jnp.asarray(segment_ids),
        jnp.asarray(indicator),
    )
    return self._inputs

  def run(self, bq, bkv, *, bkv_compute=None, iters: int = 10, warmup: int = 2) -> BenchResult:
    from flax import nnx

    try:
      config = Ideogram4Config(
          emb_dim=self.base_config.emb_dim,
          num_heads=self.base_config.num_heads,
          in_channels=self.base_config.in_channels,
          llm_features_dim=self.base_config.llm_features_dim,
          adanln_dim=self.base_config.adanln_dim,
          intermediate_size=self.base_config.intermediate_size,
          num_layers=1,
          attention="flash",
          weights_dtype=self.dtype,
          activations_dtype=self.dtype,
          flash_block_q=bq,
          flash_block_kv=bkv,
      )
      model = Ideogram4Transformer(nnx.Rngs(0), config, mesh=self.mesh)
      if self.logical_axis_rules:
        model = _shard_like_inference(model, self.logical_axis_rules, self.mesh)
      graphdef, state, rest = nnx.split(model, nnx.Param, ...)
      inputs = self._build_inputs()

      @jax.jit
      def fwd(state, *args):
        return nnx.merge(graphdef, state, rest)(*args)

      # time_callable runs call #1 untimed and reports it as compile_ms, so the
      # mean never contains compilation or warmup.
      mean_ms, std_ms, times, compile_ms = time_callable(
          lambda: fwd(state, *inputs), iters=iters, warmup=warmup, sync=jax.block_until_ready
      )
      del model, state
      return BenchResult(
          bq=bq,
          bkv=bkv,
          bkv_compute=bkv_compute or bkv,
          status="ok",
          mean_ms=mean_ms,
          std_ms=std_ms,
          times_ms=list(times),
          compile_ms=compile_ms,
      )
    except Exception as e:  # noqa: BLE001 - VMEM overflow surfaces as several distinct types
      msg = str(e)
      lowered = msg.lower()
      status = "oom" if ("vmem" in lowered or "out of memory" in lowered or "resource exhausted" in lowered) else "error"
      return BenchResult(bq=bq, bkv=bkv, bkv_compute=bkv_compute or bkv, status=status, detail=msg[:300])
