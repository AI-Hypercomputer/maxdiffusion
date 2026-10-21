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
"""

"""End-to-end WAN 2.2 I2V padding + self-attention masking harness.

Cases (selected via ELM_RUNS, saved to ELM_OUT_DIR):
  - base / base_rerun / pipeline_base: unpadded 5s (81 frames = 75,600 tokens).
  - t1_{shard,end}_{masked,nomask}[_alt]: 5s video on a 7.5s grid (111,600 tokens).
  - t2_{shard,end}_{masked,nomask}[_alt]: 2x image tokens (+1 latent frame, 79,200 tokens).
  - p<T>_{shard,end}_{masked,nomask}: the 5s video zero-padded to exactly T tokens.
    Mask flags beyond masked/nomask: lean (kernel-only KV bound), leanq (lean + the
    kernel computes only real query tiles), fast (leanq + block reorder: the shard
    layout is built with reshape/pad instead of an index gather).
  - <case>_alt: same shapes/mask as <case>, with random values in the padding.
  - tf_<case>: single-step teacher-forced comparison against recorded base steps.
  - <case>_t<BQ>x<BKV>: <case> with self-attention block_q=BQ, block_kv=block_kv_compute=BKV
    (<case>_bq<N> keeps the config block_kv).
  - <case>_xf[_t<BQ>x<BKV>]: cross-attention through the tokamax Pallas flash kernel
    (flash_min_seq_length=0 on the cross-attention ops, block_q=ELM_XF_BQ, default 2048)
    instead of the XLA dot-product fallback for the 512-token text KV.
  - <case>_tsearch: two-stage end-to-end tile search for <case> (utils/seqlen_tile_search.py);
    writes <case>_t<BQ>x<BKV> (screen), <case>_t<BQ>x<BKV>_confirm and <case>_tbest.
  - <case>_tbase: <case> timed with the confirmed tiles of base_tsearch (for shard cases,
    whose kernel sees exactly base's per-chunk tokens).

ELM_MODE=speed times ELM_SPEED_STEPS denoise steps after ELM_SPEED_WARMUP warmup
steps per case (no decode, no outputs saved). An existing results.json in ELM_OUT_DIR is
resumed: finished cases are skipped, and a case the process died in is recorded as failed.
"""

import dataclasses
import hashlib
import json
import os
import time
import traceback
from typing import Sequence

import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion import max_logging, pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
from maxdiffusion.models.attention_flax import NNXAttentionOp
from maxdiffusion.models.token_padding import LAYOUT_END, LAYOUT_SHARD, TokenPadding
from maxdiffusion.pipelines.wan.wan_pipeline import transformer_forward_pass
from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p2 import WanPipelineI2V_2_2
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils import seqlen_tile_search as tsearch
from maxdiffusion.utils.loading_utils import load_image

jax.config.update("jax_use_shardy_partitioner", True)

REAL_FRAMES = int(os.environ.get("ELM_REAL_FRAMES", "81"))  # default 81 = 5 s @ 16 fps
PADDED_FRAMES = 121  # 7.5 s @ 16 fps
# Clips longer than this get synthetic speed inputs (the VAE encode is slow and memory-hungry).
VAE_MAX_FRAMES = int(os.environ.get("ELM_VAE_MAX_FRAMES", "121"))
TF_STEPS = (0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 39)  # base steps recorded for tf_<case>
ALT_PADDING_SALT = 7777  # rng fold-in for the <case>_alt padding content
DEFAULT_RUNS = (
    "base",
    "base_rerun",
    "t2_shard_masked",
    "t1_shard_masked",
    "t2_end_masked",
    "t1_end_masked",
    "t2_end_nomask",
    "t1_end_nomask",
    "pipeline_base",
)


def _md5(a) -> str:
  return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _file_md5(path: str) -> str:
  with open(path, "rb") as f:
    return hashlib.md5(f.read()).hexdigest()


def _log(msg: str) -> None:
  max_logging.log(f"[elm-test] {msg}")


def _self_attention_ops(transformer) -> list:
  return [node for path, node in nnx.iter_graph(transformer) if isinstance(node, NNXAttentionOp) and "attn1" in path]


def _cross_attention_ops(transformer) -> list:
  return [node for path, node in nnx.iter_graph(transformer) if isinstance(node, NNXAttentionOp) and "attn2" in path]


XF_BLOCK_Q = int(os.environ.get("ELM_XF_BQ", "2048"))  # cross-attention block_q for <case>_xf
MASK_FLAGS = {  # maskflag -> (mask, lean, q_split, block_reorder)
    "nomask": (False, False, False, False),
    "masked": (True, False, False, False),
    "lean": (True, True, False, False),
    "leanq": (True, True, True, False),
    "fast": (True, True, True, True),
}


def _split_case(name: str):
  """'<case>[_xf][_t<bq>x<bkv>]' -> (case, Tile or None, cross_flash)."""
  case, tile = tsearch.parse_tile_suffix(name)
  cross_flash = case.endswith("_xf")
  return (case[: -len("_xf")] if cross_flash else case), tile, cross_flash


def load_pipeline(config) -> WanPipelineI2V_2_2:
  """Same loading path as generate_wan.run for WAN 2.2 I2V."""
  loader = WanCheckpointerI2V_2_2(config=config)
  step = loader.checkpoint_manager.latest_step()
  if step is not None:
    pipeline, _, _ = loader.load_checkpoint(step)
    return pipeline
  return loader.load_pretrained_pipeline_or_diffusers(
      config,
      WanPipelineI2V_2_2,
      (
          ("low_noise_transformer_state", "low_noise_transformer"),
          ("high_noise_transformer_state", "high_noise_transformer"),
      ),
      "low_noise_transformer",
  )


class Harness:
  """Holds the pipeline + shared inputs and runs each test case."""

  # Tile-search step counts (ELM_SCREEN_STEPS / ELM_CONFIRM_STEPS override in __init__).
  screen_steps = tsearch.SCREEN_STEPS
  confirm_steps = tsearch.CONFIRM_STEPS

  def __init__(self, config, pipeline: WanPipelineI2V_2_2, out_dir: str):
    self.config = config
    self.pipe = pipeline
    self.out_dir = out_dir
    self.steps = config.num_inference_steps
    self.ring_chunks = pipeline.mesh.shape["context"] // config.ulysses_shards
    self.mode = os.environ.get("ELM_MODE", "full")
    self.speed_warmup = int(os.environ.get("ELM_SPEED_WARMUP", "2"))
    self.speed_steps = int(os.environ.get("ELM_SPEED_STEPS", "5"))
    self.screen_steps = int(os.environ.get("ELM_SCREEN_STEPS", tsearch.SCREEN_STEPS))
    self.confirm_steps = int(os.environ.get("ELM_CONFIRM_STEPS", tsearch.CONFIRM_STEPS))
    self.results = {}
    self.base = None  # (real latents f32, frames uint8) of the reference run
    self.refs = {}  # 'base' / 'base_xf' -> (denormalized latents, frames) of the first unpadded run
    self.base_records = {}  # step -> (real latents in, real noise pred, use_high) of the base run

    # Shared inputs, built as WanPipelineI2V_2_2.__call__ does.
    self.image = load_image(config.image_url)
    batch_size = config.global_batch_size_to_train_on
    self.prompt = [config.prompt] * batch_size
    self.negative_prompt = [config.negative_prompt] * batch_size
    pe, npe, image_embeds, self.batch = pipeline._prepare_model_inputs_i2v(  # pylint: disable=protected-access
        self.prompt, self.image, self.negative_prompt, 1, config.max_sequence_length, None, None, None, None
    )
    assert image_embeds is None, "WAN 2.2 I2V should not produce CLIP image embeddings"
    self.prompt_embeds, self.negative_prompt_embeds = pe, npe
    tensor = pipeline.video_processor.preprocess(self.image, height=config.height, width=config.width)
    self.image_tensor = jnp.array(tensor.cpu().numpy())
    if self.image_tensor.ndim == 3:
      self.image_tensor = self.image_tensor[None, ...]
    latents_rng, _ = jax.random.split(jax.random.key(config.seed))
    self.latents_rng = latents_rng
    dtype = pe.dtype

    def prep(num_frames):
      lat, cond, _ = pipeline.prepare_latents(
          image=self.image_tensor,
          batch_size=self.batch,
          height=config.height,
          width=config.width,
          num_frames=num_frames,
          dtype=dtype,
          rng=latents_rng,
      )
      return lat, cond

    if REAL_FRAMES <= VAE_MAX_FRAMES:
      self.lat81, self.cond81 = prep(REAL_FRAMES)  # real-clip latents/condition (any length)
    else:
      # Synthetic speed inputs: noise latents at the real shape; the condition is extended
      # with its last latent frame.
      lat_s, cond_s = prep(VAE_MAX_FRAMES)
      num_lat = (REAL_FRAMES - 1) // 4 + 1
      self.lat81 = jax.random.normal(latents_rng, (lat_s.shape[0], num_lat) + lat_s.shape[2:], dtype=lat_s.dtype)
      tail = jnp.repeat(cond_s[:, -1:], num_lat - cond_s.shape[1], axis=1)
      self.cond81 = jnp.concatenate([cond_s, tail], axis=1)
      _log(f"synthetic speed inputs: {REAL_FRAMES} frames -> {num_lat} latent frames (VAE ran {VAE_MAX_FRAMES})")
    self.real_lat_frames = self.lat81.shape[1]  # 21 at the default 81 frames
    tokens_per_frame = (config.height // 16) * (config.width // 16)  # VAE /8, patch /2
    self.tokens_per_frame = tokens_per_frame
    self.num_real_tokens = self.real_lat_frames * tokens_per_frame
    _log(f"real clip: {REAL_FRAMES} frames, {config.width}x{config.height}, {self.num_real_tokens} tokens")
    with pipeline.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      self.rope21 = pipeline.low_noise_transformer.rope(jnp.zeros(self.lat81.shape))

    # t1/t2 cases: the 121-frame condition, checked against the real clip's first latent frames.
    self.cond121 = self.rope31 = None
    self.side_checks = {}
    if os.environ.get("ELM_SIDE_CHECKS", "1" if REAL_FRAMES == 81 else "0") == "1":
      _, self.cond121 = prep(PADDED_FRAMES)
      c81 = np.asarray(self.cond81, np.float32)
      c121 = np.asarray(self.cond121[:, : self.real_lat_frames], np.float32)
      self.side_checks = {
          "cond121_first21_equals_cond81": bool(np.array_equal(c81, c121)),
          "cond121_first21_max_abs_diff": float(np.max(np.abs(c81 - c121))),
          "cond_shapes": {"81": list(self.cond81.shape), "121": list(self.cond121.shape)},
      }
      with pipeline.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        rope = pipeline.low_noise_transformer.rope
        self.rope31 = rope(jnp.zeros(self.lat81.shape[:1] + (self.cond121.shape[1],) + self.lat81.shape[2:]))
      self.side_checks["rope31_prefix_equals_rope21"] = bool(
          np.array_equal(np.asarray(self.rope31[:, :, : self.num_real_tokens]), np.asarray(self.rope21))
      )
      _log(f"side checks: {self.side_checks}")

    # Per-run constants of run_inference_2_2_i2v's default path.
    self.low = nnx.split(pipeline.low_noise_transformer, nnx.Param, ...)
    self.high = nnx.split(pipeline.high_noise_transformer, nnx.Param, ...)
    self.default_block_sizes = _self_attention_ops(pipeline.high_noise_transformer)[0].flash_block_sizes
    self.tile = None  # self-attention tile override (None = config tiles)
    cross_op = _cross_attention_ops(pipeline.high_noise_transformer)[0]
    self.default_cross = (cross_op.flash_min_seq_length, cross_op.flash_block_sizes)
    self.cross_flash = False  # True: cross-attention through the tokamax flash kernel
    self.boundary = pipeline.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
    self.data_sharding = NamedSharding(pipeline.mesh, P())
    if config.global_batch_size_to_train_on // config.per_device_batch_size == 0:
      self.data_sharding = NamedSharding(pipeline.mesh, P(*config.data_sharding))
    with pipeline.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      self.prompt_embeds_combined = jnp.concatenate(
          [jax.device_put(pe, self.data_sharding), jax.device_put(npe, self.data_sharding)], axis=0
      )

  def set_tile(self, tile):
    """Sets the self-attention tile on both experts (None = config tiles); recompiles, no reload.

    tile.block_kv=None keeps the config block_kv; otherwise block_kv_compute = block_kv and
    block_kv_compute_in = min(config value, block_kv).
    """
    if tile == self.tile:
      return
    sizes = self.default_block_sizes
    if tile is not None:
      sizes = dataclasses.replace(sizes, block_q=tile.block_q)
      if tile.block_kv is not None:
        sizes = dataclasses.replace(
            sizes,
            block_kv=tile.block_kv,
            block_kv_compute=tile.block_kv,
            block_kv_compute_in=min(sizes.block_kv_compute_in or tile.block_kv, tile.block_kv),
        )
    for transformer in (self.pipe.low_noise_transformer, self.pipe.high_noise_transformer):
      for op in _self_attention_ops(transformer):
        op.flash_block_sizes = sizes
    self.low = nnx.split(self.pipe.low_noise_transformer, nnx.Param, ...)
    self.high = nnx.split(self.pipe.high_noise_transformer, nnx.Param, ...)
    self.tile = tile

  def set_cross_flash(self, on: bool):
    """Routes cross-attention through the tokamax flash kernel (True) or the config default."""
    if on == self.cross_flash:
      return
    min_len, sizes = self.default_cross
    if on:
      min_len, sizes = 0, dataclasses.replace(sizes, block_q=XF_BLOCK_Q)
    for transformer in (self.pipe.low_noise_transformer, self.pipe.high_noise_transformer):
      for op in _cross_attention_ops(transformer):
        op.flash_min_seq_length = min_len
        op.flash_block_sizes = sizes
    self.low = nnx.split(self.pipe.low_noise_transformer, nnx.Param, ...)
    self.high = nnx.split(self.pipe.high_noise_transformer, nnx.Param, ...)
    self.cross_flash = on

  def _tile_fields(self):
    sizes = self.default_block_sizes
    tile = self.tile
    return {
        "block_q": tile.block_q if tile else sizes.block_q,
        "block_kv": tile.block_kv if tile and tile.block_kv else sizes.block_kv,
        "cross_flash": self.cross_flash,
        **({"cross_block_q": XF_BLOCK_Q} if self.cross_flash else {}),
    }

  def _inflight(self, key):
    """Marks `key` as running (None clears): a restart records it as a crash."""
    path = os.path.join(self.out_dir, "inflight.json")
    if key is None:
      if os.path.exists(path):
        os.remove(path)
      return
    with open(path, "w", encoding="utf-8") as fh:
      json.dump({"key": key}, fh)

  def resume(self):
    """Reloads results.json. The case the previous process died in is retried once (spot
    preemption also kills it) and recorded as failed if it was in flight at a second death."""
    path = os.path.join(self.out_dir, "results.json")
    if os.path.exists(path):
      with open(path, encoding="utf-8") as fh:
        self.results = json.load(fh).get("runs", {})
    marker = os.path.join(self.out_dir, "inflight.json")
    if os.path.exists(marker):
      with open(marker, encoding="utf-8") as fh:
        key = json.load(fh)["key"]
      if key not in self.results:
        deaths_path = os.path.join(self.out_dir, "crashes.json")
        deaths = {}
        if os.path.exists(deaths_path):
          with open(deaths_path, encoding="utf-8") as fh:
            deaths = json.load(fh)
        deaths[key] = deaths.get(key, 0) + 1
        with open(deaths_path, "w", encoding="utf-8") as fh:
          json.dump(deaths, fh)
        if deaths[key] >= 2:
          self.results[key] = {"error": f"process died {deaths[key]}x while running this case (fatal compile error?)"}
          _log(f"resume: {key} was running at {deaths[key]} process deaths; recorded as failed")
        else:
          _log(f"resume: {key} was running when the previous process died (crash or preemption); retrying")
      self._inflight(None)
    if self.results:
      _log(f"resume: {len(self.results)} finished records loaded")

  def _measure(self, key, case, tile, timed_steps, abort_above=None, warmup=None):
    """One speed record under `key` (reused if already recorded); errors are recorded, not raised."""
    if key in self.results:
      return self.results[key]
    self._inflight(key)
    try:
      self.set_tile(tile)
      rec = self.time_steps(case, warmup=warmup, steps=timed_steps, abort_above=abort_above)
      rec.update(self._tile_fields())
    except Exception as e:  # pylint: disable=broad-exception-caught
      # A failing tile must not end the search.
      _log(f"{key} FAILED: {e}\n{traceback.format_exc()}")
      rec = {"error": repr(e)[:2000], **({"block_q": tile.block_q, "block_kv": tile.block_kv} if tile else {})}
    self.results[key] = rec
    self._inflight(None)
    _log(f"{key}: {json.dumps(rec)}")
    self.save()
    return rec

  def tile_search(self, case):
    """Two-stage end-to-end (block_q, block_kv) search for `case`; records <case>_tbest."""
    chunk = self.case_inputs(case)[3]
    chunk_len = (chunk.num_total if chunk is not None else self.num_real_tokens) // self.ring_chunks
    cands = tsearch.tile_candidates(chunk_len)
    _log(f"{case}: tile search over {len(cands)} candidates for chunk {chunk_len}: {[t.name for t in cands]}")

    def measure(tile, steps, abort_above):
      confirm = steps == self.confirm_steps
      key = f"{case}_{tile.name}" + ("_confirm" if confirm else "")
      return self._measure(key, case, tile, steps, abort_above, warmup=1 if confirm else None)

    out = tsearch.run_search(cands, measure, screen_steps=self.screen_steps, confirm_steps=self.confirm_steps)
    self._record_best(case, out["winner"], out["confirm"], len(cands))

  def tiles_from_base(self, case):
    """Times `case` with base_tsearch's confirmed tiles; records <case>_tbest."""
    base = self.results.get("base_tbest", {})
    tiles = [tsearch.Tile(*t) for t in base.get("finalists", [])] or [None]
    confirm = {}
    for tile in tiles:
      key = f"{case}_{tile.name if tile else 'tconfig'}_confirm"
      confirm[tile] = self._measure(key, case, tile, self.confirm_steps)
    done = [t for t, r in confirm.items() if "error" not in r]
    winner = min(done, key=lambda t: confirm[t]["step_median_s"]) if done else None
    self._record_best(case, winner, confirm, len(tiles))

  def _record_best(self, case, winner, confirm, num_candidates):
    finalists = [list(t) for t in confirm if t is not None and "error" not in confirm[t]]
    if winner is None and None not in confirm:
      rec = {"error": "no tile candidate ran", "finalists": finalists}
    else:
      rec = dict(confirm[winner])
      rec.update({"tile": list(winner) if winner else None, "finalists": finalists, "num_candidates": num_candidates})
    self.results[f"{case}_tbest"] = rec
    _log(f"{case}_tbest: {json.dumps(rec)}")
    self.save()

  def case_inputs(self, name: str):
    """Returns (latents BFHWC, condition BFHWC, rotary_emb, token_padding)."""
    if name in ("base", "base_rerun", "pipeline_base"):
      return self.lat81, self.cond81, self.rope21, None
    parts = name.split("_")
    alt = parts[-1] == "alt"
    if len(parts) != 3 + alt:
      raise ValueError(name)
    test, layout, maskflag = parts[:3]
    layout = {"end": LAYOUT_END, "shard": LAYOUT_SHARD}[layout]
    flags = MASK_FLAGS[maskflag]
    if test[0] == "p" and test[1:].isdigit() and not alt:
      padding = TokenPadding(self.num_real_tokens, int(test[1:]), layout, self.ring_chunks, *flags).validate()
      return self.lat81, self.cond81, self.rope21, padding
    if test == "t1":
      if self.cond121 is None:
        raise ValueError(f"{name} needs ELM_SIDE_CHECKS=1 (the 121-frame condition)")
      extra = self.cond121.shape[1] - self.real_lat_frames  # 10 latent frames
      noise_extra = jax.random.normal(
          jax.random.fold_in(self.latents_rng, PADDED_FRAMES),
          (self.batch, extra) + self.lat81.shape[2:],
          dtype=self.lat81.dtype,
      )
      latents = jnp.concatenate([self.lat81, noise_extra], axis=1)
      condition = jnp.concatenate([self.cond81, self.cond121[:, self.real_lat_frames :]], axis=1)
      rotary = self.rope31
    elif test == "t2":
      latents = jnp.concatenate([self.lat81, self.lat81[:, :1]], axis=1)
      condition = jnp.concatenate([self.cond81, self.cond81[:, :1]], axis=1)
      rotary = jnp.concatenate([self.rope21, self.rope21[:, :, : self.tokens_per_frame]], axis=2)
    else:
      raise ValueError(name)
    if alt:
      f = self.real_lat_frames
      k_lat, k_cond = jax.random.split(jax.random.fold_in(self.latents_rng, ALT_PADDING_SALT))
      pad_lat = 2.0 * jax.random.normal(k_lat, latents[:, f:].shape, dtype=latents.dtype)
      pad_cond = jax.random.normal(k_cond, condition[:, f:].shape, dtype=jnp.float32).astype(condition.dtype)
      latents = jnp.concatenate([latents[:, :f], pad_lat], axis=1)
      condition = jnp.concatenate([condition[:, :f], pad_cond], axis=1)
    total = latents.shape[1] * self.tokens_per_frame
    padding = TokenPadding(self.num_real_tokens, total, layout, self.ring_chunks, *flags).validate()
    return latents, condition, rotary, padding

  def _prepare(self, latents, condition):
    """Scheduler state and sharded device inputs."""
    pipe = self.pipe
    scheduler_state = pipe.scheduler.set_timesteps(pipe.scheduler_state, num_inference_steps=self.steps, shape=latents.shape)
    latents = jax.device_put(latents, self.data_sharding)
    condition = jax.device_put(condition, self.data_sharding)
    with pipe.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      cond = jnp.transpose(jnp.concatenate([condition] * 2), (0, 4, 1, 2, 3))
    return scheduler_state, latents, cond

  def _noise_pred(self, latents, cond, rotary_emb, token_padding, step, t, use_high):
    """One CFG model call of the denoise loop; returns BFHWC noise prediction."""
    (graphdef, state, rest), guidance = (
        (self.high, self.config.guidance_scale_high) if use_high else (self.low, self.config.guidance_scale_low)
    )
    extra = {} if token_padding is None else {"token_padding": token_padding}
    latents_input = jnp.transpose(jnp.concatenate([latents, latents], axis=0), (0, 4, 1, 2, 3))
    latent_model_input = jnp.concatenate([latents_input, cond], axis=1)
    timestep = jnp.broadcast_to(t, latents_input.shape[0])
    noise_pred = transformer_forward_pass(
        graphdef,
        state,
        rest,
        latent_model_input,
        timestep,
        self.prompt_embeds_combined,
        do_classifier_free_guidance=True,
        guidance_scale=guidance,
        encoder_hidden_states_image=None,
        kv_cache=None,
        rotary_emb=rotary_emb,
        encoder_attention_mask=None,
        svg_step_index=jnp.asarray(step, dtype=jnp.int32),
        **extra,
    )
    return jnp.transpose(noise_pred, (0, 2, 3, 4, 1))

  def denoise(self, latents, condition, rotary_emb, token_padding, record_steps=()):
    """Denoise loop with optional per-step recording for teacher-forced eval."""
    pipe, f = self.pipe, self.real_lat_frames
    scheduler_state, latents, cond = self._prepare(latents, condition)
    records = {}
    with pipe.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      timesteps = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)
      timesteps_np = np.asarray(scheduler_state.timesteps)
      for step in range(self.steps):
        t = timesteps[step]
        use_high = bool(timesteps_np[step] >= np.asarray(self.boundary))
        noise_pred = self._noise_pred(latents, cond, rotary_emb, token_padding, step, t, use_high)
        if step in record_steps:
          records[step] = (np.asarray(latents[:, :f], np.float32), np.asarray(noise_pred[:, :f], np.float32), use_high)
        latents, scheduler_state = pipe.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
      latents.block_until_ready()
    return latents, records

  def time_steps(self, name: str, warmup=None, steps=None, abort_above=None, profile_dir=None):
    """Per-step wall time of the denoise loop; the first warmup step includes compile.

    With abort_above, stops after the first timed step if it took longer (record "aborted").
    With profile_dir, captures a jax.profiler trace of the timed steps only (not warmup).
    """
    pipe = self.pipe
    warmup = self.speed_warmup if warmup is None else warmup
    steps = self.speed_steps if steps is None else steps
    latents, condition, rotary, padding = self.case_inputs(name)
    scheduler_state, latents, cond = self._prepare(latents, condition)
    times = []
    aborted = False
    with pipe.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      timesteps = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)
      timesteps_np = np.asarray(scheduler_state.timesteps)
      for step in range(warmup + steps):
        if profile_dir and step == warmup:
          jax.profiler.start_trace(profile_dir)
        t0 = time.perf_counter()
        use_high = bool(timesteps_np[step] >= np.asarray(self.boundary))
        with jax.profiler.StepTraceAnnotation("denoise", step_num=step):
          noise_pred = self._noise_pred(latents, cond, rotary, padding, step, timesteps[step], use_high)
          latents, scheduler_state = pipe.scheduler.step(scheduler_state, noise_pred, timesteps[step], latents).to_tuple()
          latents.block_until_ready()
        times.append(time.perf_counter() - t0)
        if abort_above is not None and step == warmup and times[-1] > abort_above:
          aborted = True
          break
      if profile_dir:
        jax.profiler.stop_trace()
    timed = np.asarray(times[warmup:])
    rec = {
        "mode": "speed",
        "num_total": padding.num_total if padding is not None else self.num_real_tokens,
        "token_padding": padding._asdict() if padding is not None else None,
        "first_step_s": round(times[0], 3),
        "step_s": [round(float(x), 4) for x in timed],
        "step_median_s": round(float(np.median(timed)), 4),
        "step_mean_s": round(float(timed.mean()), 4),
        "step_std_s": round(float(timed.std()), 4),
        "projected_denoise_s": round(float(np.median(timed)) * self.steps, 2),
    }
    peaks = [(d.memory_stats() or {}).get("peak_bytes_in_use") for d in jax.local_devices()]
    if any(p is not None for p in peaks):  # process-wide peak so far, max over devices
      rec["hbm_peak_gib"] = round(max(p for p in peaks if p is not None) / 2**30, 2)
    if aborted:
      rec["aborted"] = True
      rec["abort_above_s"] = round(float(abort_above), 4)
    return rec

  def compare_step0(self, case: str) -> dict:
    """One step-0 model call for the unpadded base and `case` on the same inputs; real-frame diff."""
    preds = {}
    for label, name in (("base", "base"), ("case", case)):
      latents, condition, rotary, padding = self.case_inputs(name)
      scheduler_state, lat, cond = self._prepare(latents, condition)
      with self.pipe.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        timesteps = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)
        use_high = bool(np.asarray(scheduler_state.timesteps)[0] >= np.asarray(self.boundary))
        pred = self._noise_pred(lat, cond, rotary, padding, 0, timesteps[0], use_high)
      preds[label] = np.asarray(pred[:, : self.real_lat_frames], np.float32)
    a, b = preds["base"], preds["case"]
    diff = b - a
    return {
        "mode": "eq",
        "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(a)),
        "max_abs": float(np.abs(diff).max()),
        "cosine": float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))),
        "finite": bool(np.isfinite(b).all()),
    }

  def teacher_forced(self, case: str):
    """Per recorded base step: base latents in, compare the real-frame noise prediction."""
    if not self.base_records:
      raise RuntimeError("teacher forcing needs the base run's recorded steps; run base first")
    pipe, f = self.pipe, self.real_lat_frames
    lat0, condition, rotary, padding = self.case_inputs(case)
    scheduler_state, _, cond = self._prepare(lat0, condition)
    per_step = {}
    with pipe.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      timesteps = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)
      for step in sorted(self.base_records):
        x_real, base_pred, use_high = self.base_records[step]
        lat = jnp.asarray(x_real)
        if lat0.shape[1] > f:  # padding keeps the case's initial content
          lat = jnp.concatenate([lat, jnp.asarray(lat0[:, f:], lat.dtype)], axis=1)
        lat = jax.device_put(lat, self.data_sharding)
        pred = np.asarray(self._noise_pred(lat, cond, rotary, padding, step, timesteps[step], use_high)[:, :f], np.float32)
        diff = (pred - base_pred).astype(np.float64)
        ref = base_pred.astype(np.float64)
        per_step[str(step)] = {
            "expert": "high" if use_high else "low",
            "bit_identical": bool(np.array_equal(pred, base_pred)),
            "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(ref)),
            "max_abs": float(np.abs(diff).max()),
            "cosine": float(
                np.sum(pred.astype(np.float64) * ref) / (np.linalg.norm(pred.astype(np.float64)) * np.linalg.norm(ref))
            ),
        }
    return per_step

  def decode(self, real_latents_bfhwc):
    """Pipeline post-processing: BFHWC -> BCFHW, denormalize, VAE decode."""
    with self.pipe.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      den = self.pipe._denormalize_latents(jnp.transpose(real_latents_bfhwc, (0, 4, 1, 2, 3)))  # pylint: disable=protected-access
      den.block_until_ready()
    frames = self.pipe._decode_latents_to_video(den)  # pylint: disable=protected-access
    return den, np.asarray(frames)

  def run_case(self, name: str):
    """Runs one case (speed, eq_, tf_, _xprof, tile search or full generation) and records it."""
    _log(f"=== {name} ===")
    t0 = time.perf_counter()
    if name.endswith("_tsearch"):
      self.set_cross_flash(False)
      self.tile_search(name[: -len("_tsearch")])
      return
    if name.endswith("_tbase"):
      self.set_cross_flash(False)
      self.tiles_from_base(name[: -len("_tbase")])
      return
    if name.startswith("eq_"):  # eq_<case>[_xf][_t<BQ>x<BKV>]: step-0 real-token output vs unpadded base
      case, tile, cross_flash = _split_case(name[3:])
      self.set_tile(tile)
      self.set_cross_flash(cross_flash)
      rec = self.compare_step0(case)
      rec.update(self._tile_fields())
      self.results[name] = rec
      _log(f"{name}: {json.dumps(rec)}")
      return
    if name.endswith("_xprof"):  # <case>[_xf]_t<BQ>x<BKV>_xprof: trace the timed steps
      case, tile, cross_flash = _split_case(name[: -len("_xprof")])
      self.set_tile(tile)
      self.set_cross_flash(cross_flash)
      profile_dir = os.path.join(self.out_dir, "xprof", name)
      os.makedirs(profile_dir, exist_ok=True)
      rec = self.time_steps(case, profile_dir=profile_dir)
      rec.update(self._tile_fields())
      rec["profile_dir"] = profile_dir
      self.results[name] = rec
      _log(f"{name}: {json.dumps(rec)}")
      return
    case, tile, cross_flash = _split_case(name)
    self.set_tile(tile)
    self.set_cross_flash(cross_flash)
    if self.mode == "speed":
      rec = self.time_steps(case)
      rec.update(self._tile_fields())
      self.results[name] = rec
      _log(f"{name}: {json.dumps(rec)}")
      return
    if name.startswith("tf_"):
      per_step = self.teacher_forced(name[3:])
      rels = [v["rel_l2"] for v in per_step.values()]
      rec = {
          "wall_s": round(time.perf_counter() - t0, 1),
          "all_bit_identical": all(v["bit_identical"] for v in per_step.values()),
          "max_rel_l2": max(rels),
          "mean_rel_l2": float(np.mean(rels)),
          "min_cosine": min(v["cosine"] for v in per_step.values()),
          "per_step": per_step,
      }
      self.results[name] = rec
      _log(f"{name}: {json.dumps(rec)}")
      return
    if name == "pipeline_base":
      den, _ = self.pipe(
          prompt=self.prompt,
          image=self.image,
          negative_prompt=self.negative_prompt,
          height=self.config.height,
          width=self.config.width,
          num_frames=REAL_FRAMES,
          num_inference_steps=self.steps,
          guidance_scale_low=self.config.guidance_scale_low,
          guidance_scale_high=self.config.guidance_scale_high,
          output_type="latent",
      )
      denoise_s = time.perf_counter() - t0
      real = None
      frames = np.asarray(self.pipe._decode_latents_to_video(den))  # pylint: disable=protected-access
      padding = None
    else:
      latents, condition, rotary, padding = self.case_inputs(case)
      out, records = self.denoise(latents, condition, rotary, padding, record_steps=TF_STEPS if name == "base" else ())
      if name == "base":
        self.base_records = records
      denoise_s = time.perf_counter() - t0
      real = out[:, : self.real_lat_frames]
      den, frames = self.decode(real)
    den_np = np.asarray(den, np.float32)
    rec = {
        "denoise_wall_s": round(denoise_s, 1),
        "token_padding": padding._asdict() if padding is not None else None,
        "md5_denorm_latents": _md5(den_np),
        "md5_frames_uint8": _md5(frames),
        "frames_shape": list(frames.shape),
    }
    if real is not None:
      real_np = np.asarray(real, np.float32)
      rec["md5_latents"] = _md5(real_np)
      np.save(os.path.join(self.out_dir, f"{name}_latents.npy"), real_np)
    np.save(os.path.join(self.out_dir, f"{name}_frames.npy"), frames)  # raw uint8, for SSIM/LPIPS
    mp4 = os.path.join(self.out_dir, f"{name}.mp4")
    try:
      export_to_video(frames[0], mp4, fps=self.config.fps)
      rec["md5_mp4"] = _file_md5(mp4)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # The mp4 is a convenience; the frames MD5 is authoritative.
      rec["md5_mp4"] = f"export failed: {e}"
    # Compare with the first unpadded run of each cross-attention kind recorded so far.
    for label, ref in self.refs.items():
      rec[f"vs_{label}"] = self.compare(den_np, frames, ref)
    if case == "base":
      self.refs.setdefault("base_xf" if cross_flash else "base", (den_np, frames))
    if self.base is None and name == "base":
      self.base = (den_np, frames)
    if name.endswith("_alt"):
      ref = self.results.get(name[: -len("_alt")], {})
      rec["vs_ref"] = {
          "ref": name[: -len("_alt")],
          "ref_ran": "md5_frames_uint8" in ref,
          "latents_md5_equal": rec.get("md5_latents") == ref.get("md5_latents"),
          "frames_md5_equal": rec["md5_frames_uint8"] == ref.get("md5_frames_uint8"),
          "mp4_md5_equal": rec["md5_mp4"] == ref.get("md5_mp4"),
      }
    self.results[name] = rec
    _log(f"{name}: {json.dumps(rec)}")

  def compare(self, den_np, frames, ref):
    base_den, base_frames = ref
    d = np.abs(den_np - base_den)
    f = np.abs(frames.astype(np.int16) - base_frames.astype(np.int16))
    mse = float(np.mean(f.astype(np.float64) ** 2))
    per_frame_mse = np.mean(f.astype(np.float64) ** 2, axis=(0, 2, 3, 4))
    worst = float(np.max(per_frame_mse))
    return {
        "latents_bit_identical": bool(np.array_equal(den_np, base_den)),
        "latents_max_abs_diff": float(d.max()),
        "latents_mean_abs_diff": float(d.mean()),
        "latents_frac_elems_differ": float(np.mean(d > 0)),
        "frames_bit_identical": bool(np.array_equal(frames, base_frames)),
        "frames_frac_values_differ": float(np.mean(f > 0)),
        "frames_max_abs_diff": int(f.max()),
        "frames_psnr_db": (float("inf") if mse == 0 else round(10 * np.log10(255.0**2 / mse), 2)),
        "frames_worst_frame_psnr_db": (float("inf") if worst == 0 else round(10 * np.log10(255.0**2 / worst), 2)),
    }

  def save(self):
    """Writes results.json atomically."""
    summary = {
        "config": {
            "model": self.config.pretrained_model_name_or_path,
            "attention": self.config.attention,
            "ulysses_shards": self.config.ulysses_shards,
            "mesh": dict(self.pipe.mesh.shape),
            "flash_block_sizes": dict(self.config.flash_block_sizes),
            "height": self.config.height,
            "width": self.config.width,
            "real_frames": REAL_FRAMES,
            "steps": self.steps,
            "mode": self.mode,
            "speed_warmup": self.speed_warmup,
            "speed_steps": self.speed_steps,
            "seed": self.config.seed,
            "jax": jax.__version__,
        },
        "side_checks": self.side_checks,
        "runs": self.results,
    }
    path = os.path.join(self.out_dir, "results.json")
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
      json.dump(summary, fh, indent=2, default=str)
    os.replace(path + ".tmp", path)


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  config = pyconfig.config
  if config.model_name != "wan2.2" or config.model_type != "I2V":
    raise ValueError("elm-test expects the WAN 2.2 I2V config.")
  out_dir = os.environ.get("ELM_OUT_DIR") or f"/tmp/elm_test_{time.strftime('%m%d-%H%M%S')}"
  os.makedirs(out_dir, exist_ok=True)
  runs = [r for r in os.environ.get("ELM_RUNS", ",".join(DEFAULT_RUNS)).split(",") if r]
  _log(f"out_dir={out_dir} runs={runs}")

  t0 = time.perf_counter()
  pipeline = load_pipeline(config)
  _log(f"load_time {time.perf_counter() - t0:.1f}s, mesh {dict(pipeline.mesh.shape)}")
  harness = Harness(config, pipeline, out_dir)
  harness.resume()
  harness.save()
  if runs and runs[0] != "base" and "base" in runs:
    runs.remove("base")
    runs.insert(0, "base")
  for name in runs:
    done_key = name[: -len("_tsearch")] + "_tbest" if name.endswith("_tsearch") else name
    done_key = name[: -len("_tbase")] + "_tbest" if name.endswith("_tbase") else done_key
    if done_key in harness.results:
      _log(f"{name}: already done, skipping")
      continue
    harness._inflight(name)  # pylint: disable=protected-access
    try:
      harness.run_case(name)
    except Exception as e:  # pylint: disable=broad-exception-caught
      _log(f"{name} FAILED: {e}\n{traceback.format_exc()}")
      harness.results[name] = {"error": repr(e)}
    harness._inflight(None)  # pylint: disable=protected-access
    harness.save()

  lines = [f"{'run':22} {'latents md5':34} {'frames md5':34} {'vs base':10} {'psnr_db':8} vs_ref / teacher-forced"]
  for name, rec in harness.results.items():
    if "error" in rec:
      lines.append(f"{name:22} ERROR {rec['error'][:120]}")
      continue
    if rec.get("mode") == "speed":
      lines.append(
          f"{name:30} T={rec['num_total']:>6} bq={rec['block_q']} step median={rec['step_median_s']:.3f}s "
          f"(std {rec['step_std_s']:.3f}) projected {rec['projected_denoise_s']:.1f}s first step={rec['first_step_s']:.1f}s"
      )
      continue
    if name.startswith("tf_"):
      lines.append(
          f"{name:22} teacher-forced: bit-identical={rec['all_bit_identical']} "
          f"max rel_l2={rec['max_rel_l2']:.3e} mean rel_l2={rec['mean_rel_l2']:.3e} min cos={rec['min_cosine']:.6f}"
      )
      continue
    if rec.get("mode") == "eq":
      lines.append(f"{name:30} vs base: rel_l2={rec['rel_l2']:.3e} max_abs={rec['max_abs']:.3e} cos={rec['cosine']:.6f}")
      continue
    vs = rec.get("vs_base", {})
    ident = "reference" if name == "base" else str(vs.get("frames_bit_identical"))
    ref = rec.get("vs_ref")
    ref_s = f"same MD5 as {ref['ref']}: {ref['frames_md5_equal']}" if ref else ""
    lines.append(
        f"{name:22} {rec['md5_denorm_latents']:34} {rec['md5_frames_uint8']:34} {ident:10} "
        f"{str(vs.get('frames_psnr_db', '')):8} {ref_s}"
    )
  _log("SUMMARY\n" + "\n".join(lines))
  _log("ELM TEST COMPLETE")


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
