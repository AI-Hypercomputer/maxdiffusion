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
  - <case>_alt: same shapes/mask as <case>, with random values in the padding.
  - tf_<case>: single-step teacher-forced comparison against recorded base steps.
"""

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
from maxdiffusion.models.token_padding import LAYOUT_END, LAYOUT_SHARD, TokenPadding
from maxdiffusion.pipelines.wan.wan_pipeline import transformer_forward_pass
from maxdiffusion.pipelines.wan.wan_pipeline_i2v_2p2 import WanPipelineI2V_2_2
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image

jax.config.update("jax_use_shardy_partitioner", True)

REAL_FRAMES = 81  # 5 s @ 16 fps
PADDED_FRAMES = 121  # 7.5 s @ 16 fps
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

  def __init__(self, config, pipeline: WanPipelineI2V_2_2, out_dir: str):
    self.config = config
    self.pipe = pipeline
    self.out_dir = out_dir
    self.steps = config.num_inference_steps
    self.ring_chunks = pipeline.mesh.shape["context"] // config.ulysses_shards
    self.results = {}
    self.base = None  # (real latents f32, frames uint8) of the reference run
    self.base_records = {}  # step -> (real latents in, real noise pred, use_high) of the base run

    # ---- Shared inputs: exactly what WanPipelineI2V_2_2.__call__ builds ----
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

    self.lat81, self.cond81 = prep(REAL_FRAMES)
    _, self.cond121 = prep(PADDED_FRAMES)
    self.real_lat_frames = self.lat81.shape[1]  # 21
    tokens_per_frame = (config.height // 16) * (config.width // 16)  # VAE /8, patch /2
    self.tokens_per_frame = tokens_per_frame
    self.num_real_tokens = self.real_lat_frames * tokens_per_frame

    # Side check: does a 7.5 s request condition its first 21 latent frames
    # exactly like a 5 s request? (Causal VAE encoder, different length.)
    c81 = np.asarray(self.cond81, np.float32)
    c121 = np.asarray(self.cond121[:, : self.real_lat_frames], np.float32)
    self.side_checks = {
        "cond121_first21_equals_cond81": bool(np.array_equal(c81, c121)),
        "cond121_first21_max_abs_diff": float(np.max(np.abs(c81 - c121))),
        "cond_shapes": {"81": list(self.cond81.shape), "121": list(self.cond121.shape)},
    }
    with pipeline.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      rope = pipeline.low_noise_transformer.rope
      self.rope21 = rope(jnp.zeros(self.lat81.shape))
      self.rope31 = rope(jnp.zeros(self.lat81.shape[:1] + (self.cond121.shape[1],) + self.lat81.shape[2:]))
    self.side_checks["rope31_prefix_equals_rope21"] = bool(
        np.array_equal(np.asarray(self.rope31[:, :, : self.num_real_tokens]), np.asarray(self.rope21))
    )
    _log(f"side checks: {self.side_checks}")

    # ---- Per-run constants of run_inference_2_2_i2v's default path ----
    self.low = nnx.split(pipeline.low_noise_transformer, nnx.Param, ...)
    self.high = nnx.split(pipeline.high_noise_transformer, nnx.Param, ...)
    self.boundary = pipeline.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
    self.data_sharding = NamedSharding(pipeline.mesh, P())
    if config.global_batch_size_to_train_on // config.per_device_batch_size == 0:
      self.data_sharding = NamedSharding(pipeline.mesh, P(*config.data_sharding))
    with pipeline.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      self.prompt_embeds_combined = jnp.concatenate(
          [jax.device_put(pe, self.data_sharding), jax.device_put(npe, self.data_sharding)], axis=0
      )

  # ------------------------------------------------------------------ cases
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
    mask = maskflag == "masked"
    if test == "t1":
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
    padding = TokenPadding(self.num_real_tokens, total, layout, self.ring_chunks, mask).validate()
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
            "cosine": float(np.sum(pred.astype(np.float64) * ref) / (np.linalg.norm(pred.astype(np.float64)) * np.linalg.norm(ref))),
        }
    return per_step

  def decode(self, real_latents_bfhwc):
    """Pipeline post-processing: BFHWC -> BCFHW, denormalize, VAE decode."""
    with self.pipe.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      den = self.pipe._denormalize_latents(jnp.transpose(real_latents_bfhwc, (0, 4, 1, 2, 3)))  # pylint: disable=protected-access
      den.block_until_ready()
    frames = self.pipe._decode_latents_to_video(den)  # pylint: disable=protected-access
    return den, np.asarray(frames)

  # -------------------------------------------------------------- run/score
  def run_case(self, name: str):
    _log(f"=== {name} ===")
    t0 = time.perf_counter()
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
      latents, condition, rotary, padding = self.case_inputs(name)
      out, records = self.denoise(
          latents, condition, rotary, padding, record_steps=TF_STEPS if name == "base" else ()
      )
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
    except Exception as e:  # noqa: BLE001 - mp4 is a convenience; frames MD5 is authoritative
      rec["md5_mp4"] = f"export failed: {e}"
    if self.base is None and name == "base":
      self.base = (den_np, frames)
    elif self.base is not None:
      rec["vs_base"] = self.compare(den_np, frames)
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

  def compare(self, den_np, frames):
    base_den, base_frames = self.base
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
  harness.save()
  if runs and runs[0] != "base" and "base" in runs:
    runs.remove("base")
    runs.insert(0, "base")
  for name in runs:
    try:
      harness.run_case(name)
    except Exception as e:  # noqa: BLE001 - keep going; record the failure
      _log(f"{name} FAILED: {e}\n{traceback.format_exc()}")
      harness.results[name] = {"error": repr(e)}
    harness.save()

  lines = [f"{'run':22} {'latents md5':34} {'frames md5':34} {'vs base':10} {'psnr_db':8} vs_ref / teacher-forced"]
  for name, rec in harness.results.items():
    if "error" in rec:
      lines.append(f"{name:22} ERROR {rec['error'][:120]}")
      continue
    if name.startswith("tf_"):
      lines.append(
          f"{name:22} teacher-forced: bit-identical={rec['all_bit_identical']} "
          f"max rel_l2={rec['max_rel_l2']:.3e} mean rel_l2={rec['mean_rel_l2']:.3e} min cos={rec['min_cosine']:.6f}"
      )
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
