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

End-to-end verification of the tile-size auto-tuner's top-K candidates.

The one-DiT-block proxy (WanBlockBenchmark) ranks tile sizes cheaply but does not reliably
predict real end-to-end denoise speed, increasingly so as the search space widens. This pass
times the proxy's top-K candidates with a few REAL denoise steps of the loaded pipeline and
commits the fastest to the config and to every transformer of the pipeline.

generate_wan.run() calls `maybe_verify` after the pipeline (and LoRA) is loaded and BEFORE
aot_cache.install(), so AOT metadata and executables only ever see the verified winner.
Per candidate:
  1. jax.clear_caches() (between candidates); write the candidate into
     config.flash_block_sizes, starting from the pre-verification block sizes.
  2. `rebuild_with_block_sizes` on every transformer: block sizes are static attributes of the
     nnx GraphDef that the jitted denoise steps take as a static argument, so the pipeline gets a
     new module graph that shares the loaded nnx.Variables (no second copy of the weights) with
     the static `flash_block_sizes` fields retagged.
  3. One untimed pass, then one timed pass, both generating with the same N-step schedule
     (N = tile_search_e2e_steps) and output_type="latent" (no VAE decode). s/step is the
     pipeline's own denoise timer (it ends with block_until_ready) divided by N. The untimed
     pass runs the full N-step schedule, not a single step, because a short WAN 2.2 schedule
     reaches the low-noise expert only in its last step(s) (3 steps: t=[999, 960, 857] vs. the
     T2V boundary 875) and the experts' graphdefs are distinct jit cache keys, so a 1-step warmup
     would leave its compilation inside the timed pass. If anything is still traced or compiled
     during the timed pass, the candidate is timed once more.
A candidate that fails (e.g. compile-time VMEM OOM) is recorded as failed and never chosen; if
every candidate fails, RuntimeError is raised. Must not import generate_wan (it imports this).
"""

import contextlib
import csv
import dataclasses
import gc
import os
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import numpy as np
from flax import nnx
from jax.experimental import multihost_utils

from maxdiffusion import max_logging, max_utils

DEFAULT_TOP_K = 3
DEFAULT_STEPS = 3
CSV_NAME = "e2e_verify.csv"
# Pipeline attributes that can hold a transformer: WAN 2.1 `transformer`, WAN 2.2 high/low noise.
TRANSFORMER_ATTRS = ("transformer", "high_noise_transformer", "low_noise_transformer")
# Candidate fields reported per measurement (the candidate dict also carries proxy_ms).
SIZE_KEYS = ("block_q", "block_kv", "block_kv_compute", "block_q_sub", "block_q_outer")
# Fired by JAX whenever it traces a function / compiles an executable (jit-cache misses).
_COMPILE_EVENTS = frozenset({
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/backend_compile_duration",
})
_PROFILER_KEYS = ("enable_profiler", "enable_ml_diagnostics")
CSV_FIELDS = (
    "proxy_rank",
    *SIZE_KEYS,
    "proxy_ms",
    "status",
    "s_per_step",
    "denoise_s",
    "steps",
    "e2e_rank",
    "winner",
    "retimed",
    "compile_events",
    "detail",
)

RunSteps = Callable[[int], float]  # run_steps(n) -> seconds the pipeline spent denoising n steps


@dataclasses.dataclass
class CandidateResult:
  """One candidate's e2e measurement. `index` is its 1-based proxy rank."""

  index: int
  candidate: dict
  steps: int
  sizes: Optional[dict] = None  # SIZE_KEYS as written to config.flash_block_sizes (what ran)
  denoise_s: Optional[float] = None
  retimed: bool = False
  compile_events: int = 0  # traces/compiles seen during the (final) timed pass
  error: str = ""

  @property
  def ok(self) -> bool:
    return not self.error and self.denoise_s is not None

  @property
  def s_per_step(self) -> Optional[float]:
    return self.denoise_s / self.steps if self.ok else None

  def csv_row(self) -> dict:
    sizes = self.sizes or self.candidate
    return {
        "proxy_rank": self.index,
        **{k: sizes.get(k) for k in SIZE_KEYS},
        "proxy_ms": self.candidate.get("proxy_ms"),
        "status": "ok" if self.ok else "failed",
        "s_per_step": self.s_per_step,
        "denoise_s": self.denoise_s if self.ok else None,
        "steps": self.steps,
        "retimed": self.retimed,
        "compile_events": self.compile_events,
        "detail": self.error,
    }


def top_k(config) -> int:
  return int(getattr(config, "tile_search_e2e_top_k", DEFAULT_TOP_K))


def num_steps(config) -> int:
  return int(getattr(config, "tile_search_e2e_steps", DEFAULT_STEPS))


def maybe_verify(config, pipeline, candidates, *, call_pipeline, log=max_logging.log) -> Optional[CandidateResult]:
  """generate_wan.run() hook, called between pipeline load and aot_cache.install().

  `candidates` is maybe_tune_block_sizes()'s return value: proxy-ranked dicts, best first (None
  when the tile search is off). No-op unless there are >= 2 candidates and
  tile_search_e2e_top_k >= 2; otherwise verifies the top-K, commits the winner to `config` and to
  every transformer of `pipeline`, and returns the winner's result. `call_pipeline` is
  generate_wan.call_pipeline (passed in: this module must not import generate_wan).
  """
  if not candidates:
    return None
  k = top_k(config)
  if k < 2 or len(candidates) < 2:
    log(f"[tile-e2e] skipped: {len(candidates)} candidate(s), tile_search_e2e_top_k={k} (needs >= 2 of each)")
    return None
  # Like the generation warmup: never profile the verification passes.
  keys = config.get_keys()
  saved = {name: keys[name] for name in _PROFILER_KEYS if name in keys}
  keys.update(dict.fromkeys(saved, False))
  try:
    return verify_candidates(
        config,
        pipeline,
        candidates,
        run_steps=make_pipeline_runner(config, pipeline, call_pipeline),
        k=k,
        steps=num_steps(config),
        out_dir=getattr(config, "tile_search_out", "") or None,
        log=log,
    )
  finally:
    keys.update(saved)


def verify_candidates(
    config,
    pipeline,
    candidates: Sequence[Mapping[str, Any]],
    *,
    run_steps: RunSteps,
    k: int = DEFAULT_TOP_K,
    steps: int = DEFAULT_STEPS,
    out_dir: Optional[str] = None,
    log=max_logging.log,
) -> CandidateResult:
  """Times the top-`k` `candidates` end to end and commits the fastest (see module docstring).

  `run_steps(n)` runs the pipeline as currently configured for an n-step generation and returns
  the seconds spent denoising. On return, config.flash_block_sizes and every transformer of
  `pipeline` carry the winner. Raises RuntimeError (with config and pipeline restored) if every
  candidate fails.
  """
  if steps < 1:
    raise ValueError(f"tile_search_e2e_steps must be >= 1, got {steps}")
  cands = [dict(c) for c in candidates[:k]]
  originals = {attr: getattr(pipeline, attr) for attr in pipeline_transformer_attrs(pipeline)}
  baseline = dict(config.flash_block_sizes)
  apply_fn = _resolve_apply_fn()
  log(
      f"[tile-e2e] verifying the tile search's top-{len(cands)} end to end: "
      f"{steps} timed denoise step(s) each, transformers: {', '.join(originals)}"
  )
  results = []
  for i, cand in enumerate(cands, 1):
    if i > 1:
      jax.clear_caches()  # drop the previous candidate's executables
      gc.collect()
    res = CandidateResult(index=i, candidate=cand, steps=steps)
    try:
      res.sizes = _set_block_sizes(config, baseline, cand, apply_fn)
      _install_block_sizes(pipeline, originals, max_utils.get_flash_block_sizes(config))
      run_steps(steps)  # untimed: compiles/warms every executable the timed pass runs
      res.denoise_s, res.compile_events = _timed(run_steps, steps)
      if res.compile_events:
        res.retimed = True
        res.denoise_s, res.compile_events = _timed(run_steps, steps)
    except Exception as e:  # pylint: disable=broad-exception-caught
      res.denoise_s, res.error = None, _short_error(e)
    res = _aggregate_across_processes(res)
    results.append(res)
    log(_candidate_line(res, len(cands)))
    if res.retimed:
      log(f"[tile-e2e]   cand {i}/{len(cands)} re-timed: the first timed pass traced/compiled")

  ok = [r for r in results if r.ok]
  winner = min(ok, key=lambda r: (r.s_per_step, r.index)) if ok else None
  _write_csv(results, winner, out_dir, log)
  if winner is None:
    config.get_keys()["flash_block_sizes"] = baseline
    for attr, model in originals.items():
      setattr(pipeline, attr, model)
    reasons = "; ".join(f"cand {r.index}: {r.error}" for r in results)
    raise RuntimeError(f"[tile-e2e] every tile-size candidate failed end to end: {reasons}")

  sizes = _set_block_sizes(config, baseline, winner.candidate, apply_fn)
  _install_block_sizes(pipeline, originals, max_utils.get_flash_block_sizes(config))
  if winner is not results[-1]:
    jax.clear_caches()  # the last candidate's executables are dead weight for the real run
    gc.collect()
  log(_winner_line(winner, results))
  log(format_using_line(sizes, winner.s_per_step))  # must stay the pass's last line
  return winner


def make_pipeline_runner(config, pipeline, call_pipeline) -> RunSteps:
  """run_steps(n) for `verify_candidates`: one real n-step generation with the run's prompt,
  shapes and flags, stopping at latents. Returns the pipeline's `denoise_total` trace entry,
  which it measures around the denoise loop ending in block_until_ready."""
  prompts = max_utils.load_prompts(getattr(config, "prompt_file", ""), default_prompt=config.prompt)
  batch_size = config.global_batch_size_to_train_on
  prompt = [prompts[0]] * batch_size
  negative_prompt = [config.negative_prompt] * batch_size
  latent_pipeline = _LatentOutput(pipeline)

  def run_steps(n: int) -> float:
    out = call_pipeline(config, latent_pipeline, prompt, negative_prompt, num_inference_steps=n)
    trace = out[1] if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict) else {}
    if "denoise_total" not in trace:
      raise RuntimeError("pipeline returned no 'denoise_total' timing trace")
    return float(trace["denoise_total"])

  return run_steps


class _LatentOutput:
  """Calls the wrapped pipeline with output_type="latent" (skips the VAE decode, which does not
  depend on the attention block sizes); forwards attribute access unchanged."""

  def __init__(self, pipeline):
    self._pipeline = pipeline

  def __getattr__(self, name):
    return getattr(self._pipeline, name)

  def __call__(self, *args, **kwargs):
    kwargs.setdefault("output_type", "latent")
    return self._pipeline(*args, **kwargs)


def pipeline_transformer_attrs(pipeline) -> list[str]:
  attrs = [a for a in TRANSFORMER_ATTRS if isinstance(getattr(pipeline, a, None), nnx.Module)]
  if not attrs:
    raise ValueError(f"[tile-e2e] {type(pipeline).__name__} has none of the transformer attributes {TRANSFORMER_ATTRS}")
  return attrs


def rebuild_with_block_sizes(model: nnx.Module, block_sizes: Any) -> nnx.Module:
  """Returns a copy of `model` whose attention ops use `block_sizes`; `model` is untouched.

  `flash_block_sizes` is baked in at construction as a static module attribute, i.e. it lives in
  the nnx GraphDef, not in the state. nnx.split + nnx.merge builds a new module graph around the
  SAME nnx.Variables (no copy of the weights); every module attribute equal to the model's
  configured `flash_block_sizes` (WanModel: each attention op) is then retagged, and so is the
  `register_to_config` record, so the copy has the AOT-cache signature of a model constructed
  with `block_sizes`.
  """
  model_config = getattr(model, "config", None)
  if not isinstance(model_config, Mapping) or "flash_block_sizes" not in model_config:
    raise ValueError(f"[tile-e2e] {type(model).__name__} does not record flash_block_sizes in its config")
  configured = model_config["flash_block_sizes"]
  clone = nnx.merge(*nnx.split(model))
  retagged = 0
  for _, node in nnx.iter_graph(clone):
    if isinstance(node, nnx.Module) and "flash_block_sizes" in vars(node) and node.flash_block_sizes == configured:
      node.flash_block_sizes = block_sizes
      retagged += 1
  if not retagged:
    raise ValueError(f"[tile-e2e] no module of {type(model).__name__} uses flash_block_sizes={configured!r}")
  internal = getattr(clone, "_internal_dict", None)  # ConfigMixin's record behind `model.config`
  if isinstance(internal, Mapping):
    clone._internal_dict = type(internal)({**internal, "flash_block_sizes": block_sizes})  # pylint: disable=protected-access
  return clone


def _install_block_sizes(pipeline, originals: Mapping[str, nnx.Module], block_sizes) -> None:
  """Points each transformer attribute of `pipeline` at a rebuild of its ORIGINAL module."""
  rebuilt = {}  # by id: a module shared by two attributes stays shared
  for attr, model in originals.items():
    if id(model) not in rebuilt:
      rebuilt[id(model)] = rebuild_with_block_sizes(model, block_sizes)
    setattr(pipeline, attr, rebuilt[id(model)])


def _resolve_apply_fn():
  try:
    from maxdiffusion.utils.tile_size_grid_search import apply_candidate_to_config  # pylint: disable=import-outside-toplevel
  except ImportError:
    return _apply_candidate_to_config
  return apply_candidate_to_config


def _apply_candidate_to_config(config, cand: Mapping[str, Any]) -> None:
  """Fallback for tile_size_grid_search.apply_candidate_to_config: writes `cand` into
  config.flash_block_sizes exactly the way maybe_tune_block_sizes writes its winner."""
  updates = {
      "block_q": cand.get("block_q"),
      "block_kv": cand.get("block_kv"),
      "block_kv_compute": cand.get("block_kv_compute"),
      "block_kv_compute_in": cand.get("block_kv_compute"),
      "block_q_sub": cand.get("block_q_sub"),
      "block_q_outer": cand.get("block_q_outer"),
  }
  fbs = dict(config.flash_block_sizes)
  fbs.update({k: v for k, v in updates.items() if v is not None})
  config.get_keys()["flash_block_sizes"] = fbs  # config is immutable via setattr; mutate raw dict


def _set_block_sizes(config, baseline: Mapping[str, Any], cand: Mapping[str, Any], apply_fn) -> dict:
  """Applies `cand` on top of the pre-verification block sizes (never on top of a previous
  candidate) and returns the resulting SIZE_KEYS, i.e. what actually runs."""
  config.get_keys()["flash_block_sizes"] = dict(baseline)
  apply_fn(config, cand)
  return {key: config.flash_block_sizes.get(key) for key in SIZE_KEYS}


def _timed(run_steps: RunSteps, steps: int) -> tuple[float, int]:
  with _count_compile_events() as events:
    seconds = run_steps(steps)
  return float(seconds), events[0]


@contextlib.contextmanager
def _count_compile_events():
  """Counts JAX trace/compile events while active (yields a 1-element list)."""
  events = [0]
  active = True

  def listener(event, duration_secs, **kwargs):
    del duration_secs, kwargs
    if active and event in _COMPILE_EVENTS:
      events[0] += 1

  jax.monitoring.register_event_duration_secs_listener(listener)
  try:
    yield events
  finally:
    active = False
    unregister = getattr(jax.monitoring, "unregister_event_duration_listener", None)
    if unregister is not None:
      unregister(listener)


def _aggregate_across_processes(res: CandidateResult) -> CandidateResult:
  """Multi-host: a candidate that failed on any process failed; s/step is the slowest process's.
  Mirrors tile_size_grid_search._aggregate_process_result so every process picks the same winner."""
  if jax.process_count() == 1:
    return res
  local = np.asarray(
      [0.0 if res.ok else 1.0, res.denoise_s if res.ok else np.inf, res.compile_events, float(res.retimed)],
      dtype=np.float32,
  )
  gathered = np.asarray(multihost_utils.process_allgather(local, tiled=False)).reshape((-1, 4))
  failed = int(np.count_nonzero(gathered[:, 0]))
  if failed:
    res.error = res.error or f"failed on {failed}/{jax.process_count()} process(es)"
    res.denoise_s = None
  else:
    res.denoise_s = float(np.max(gathered[:, 1]))
  res.compile_events = int(np.max(gathered[:, 2]))
  res.retimed = bool(np.max(gathered[:, 3]))
  return res


def _short_error(e: BaseException, limit: int = 300) -> str:
  lines = [line.strip() for line in str(e).splitlines() if line.strip()]
  msg = f"{type(e).__name__}: {lines[0]}" if lines else type(e).__name__
  return msg if len(msg) <= limit else msg[: limit - 3] + "..."


def _fmt_sizes(sizes: Mapping[str, Any]) -> str:
  def fmt(key, unset):
    value = sizes.get(key)
    return unset if value is None else str(int(value))

  return (
      f"block_q={fmt('block_q', 'none')} block_kv={fmt('block_kv', 'none')} "
      f"block_kv_compute={fmt('block_kv_compute', 'none')} block_q_sub={fmt('block_q_sub', 'auto')} "
      f"block_q_outer={fmt('block_q_outer', 'none')}"
  )


def format_using_line(sizes: Mapping[str, Any], s_per_step: float) -> str:
  """The '[tile-search] using' line; log parsers read the LAST one as the block sizes in use."""
  return f"[tile-search] using {_fmt_sizes(sizes)} (e2e-verified {s_per_step:.3f} s/step)"


def _candidate_line(res: CandidateResult, k: int) -> str:
  proxy_ms = res.candidate.get("proxy_ms")
  proxy = "n/a" if proxy_ms is None else f"{proxy_ms:.2f}"
  outcome = f"{res.s_per_step:.3f} s/step" if res.ok else f"FAILED: {res.error}"
  return f"[tile-e2e] cand {res.index}/{k}: {_fmt_sizes(res.sizes or res.candidate)} proxy={proxy} ms -> {outcome}"


def _winner_line(winner: CandidateResult, results: Sequence[CandidateResult]) -> str:
  line = f"[tile-e2e] winner: cand {winner.index}/{len(results)} -> {winner.s_per_step:.3f} s/step"
  proxy_top1 = results[0]
  if winner is proxy_top1:
    return line + " (agrees with the proxy's top-1)"
  if not proxy_top1.ok:
    return line + " (the proxy's top-1 FAILED end to end)"
  slower = 100.0 * (proxy_top1.s_per_step / winner.s_per_step - 1.0)
  return line + f" (the proxy's top-1 ran {proxy_top1.s_per_step:.3f} s/step, {slower:.1f}% slower)"


def _write_csv(results: Sequence[CandidateResult], winner: Optional[CandidateResult], out_dir: Optional[str], log) -> None:
  """Writes <out_dir>/e2e_verify.csv (process 0), next to the tile search's own CSV."""
  if not out_dir or jax.process_index() != 0:
    return
  ranked = sorted((r for r in results if r.ok), key=lambda r: (r.s_per_step, r.index))
  e2e_rank = {r.index: rank for rank, r in enumerate(ranked, 1)}
  path = os.path.join(out_dir, CSV_NAME)
  try:
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
      writer.writeheader()
      for r in results:
        writer.writerow({**r.csv_row(), "e2e_rank": e2e_rank.get(r.index), "winner": r is winner})
  except OSError as e:
    log(f"[tile-e2e] could not write {path}: {e}")
    return
  log(f"[tile-e2e] wrote {path}")
