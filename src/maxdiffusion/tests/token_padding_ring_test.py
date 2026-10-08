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

"""TPU unit test for masked token padding in ulysses_ring_custom[_fixed_m].

Verifies on a v7x-8 mesh (data=2, context=4, ulysses_shards=2) that:
  - layout="shard" with mask=True is bit-identical to the unpadded kernel call;
  - layout="end" with mask=True matches the dense fp32 reference over real tokens;
  - mask=False alters the output when padded tokens contain random values.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxdiffusion.common_types import BATCH, D_KV, SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, SELF_ATTN_Q_LENGTH
from maxdiffusion.models import attention_flax
from maxdiffusion.models.token_padding import LAYOUT_END, LAYOUT_SHARD, TokenPadding

HEADS, HEAD_DIM, BATCH_SIZE = 4, 128, 2
ULYSSES = 2
BLOCKS = {
    "block_q": 6400,
    "block_kv": 2048,
    "block_kv_compute": 2048,
    "block_kv_compute_in": 1024,
    "heads_per_tile": 1,
    "vmem_limit_bytes": 67108864,
}
RULES = (
    (BATCH, ("data", "fsdp")),
    (SELF_ATTN_HEAD, None),
    (SELF_ATTN_Q_LENGTH, "context"),
    (SELF_ATTN_KV_LENGTH, "context"),
)


def _attention(mesh, q, k, v, *, token_padding, use_fixed_m):
  def f(q, k, v):
    return attention_flax._ulysses_ring_custom_attention(  # pylint: disable=protected-access
        q,
        k,
        v,
        heads=HEADS,
        mesh=mesh,
        axis_names_q=(BATCH, SELF_ATTN_HEAD, SELF_ATTN_Q_LENGTH, D_KV),
        axis_names_kv=(BATCH, SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, D_KV),
        flash_block_sizes=BLOCKS,
        dtype=jnp.bfloat16,
        ulysses_shards=ULYSSES,
        use_base2_exp=True,
        use_experimental_scheduler=True,
        use_fixed_m=use_fixed_m,
        token_padding=token_padding,
    )

  with mesh, nn_partitioning.axis_rules(RULES):
    sharding = NamedSharding(mesh, P(("data", "fsdp"), "context", None))
    q, k, v = (jax.device_put(x, sharding) for x in (q, k, v))
    return np.asarray(jax.jit(f)(q, k, v), np.float32)


def _dense_reference(q, k, v, logit_scale):
  """softmax(logit_scale * q k^T) v per head in f32."""
  b, s, _ = q.shape
  qh, kh, vh = (np.asarray(x, np.float32).reshape(b, s, HEADS, HEAD_DIM) for x in (q, k, v))
  logits = logit_scale * np.einsum("bqhd,bkhd->bhqk", qh, kh)
  logits -= logits.max(-1, keepdims=True)
  p = np.exp(logits)
  p /= p.sum(-1, keepdims=True)
  return np.einsum("bhqk,bkhd->bqhd", p, vh).reshape(b, s, HEADS * HEAD_DIM)


def main() -> int:
  devices = np.array(jax.devices())
  assert devices.size == 8, f"needs 8 devices, got {devices.size}"
  mesh = Mesh(devices.reshape(2, 1, 4, 1), ("data", "fsdp", "context", "tensor"))
  ring_chunks = mesh.shape["context"] // ULYSSES
  rng = np.random.default_rng(0)
  n_real = 9600
  scale = HEAD_DIM**-0.25  # keeps unscaled logits ~ N(0, 1)

  def rand(n, std):
    return jnp.asarray(rng.normal(0.0, std, (BATCH_SIZE, n, HEADS * HEAD_DIM)), jnp.bfloat16)

  q_real, k_real, v_real = rand(n_real, scale), rand(n_real, scale), rand(n_real, 1.0)
  refs = {s: _dense_reference(q_real, k_real, v_real, s) for s in (1.0, HEAD_DIM**-0.5)}
  failures = []
  for use_fixed_m in (True, False):
    kernel = "ulysses_ring_custom_fixed_m" if use_fixed_m else "ulysses_ring_custom"
    unpadded = _attention(mesh, q_real, k_real, v_real, token_padding=None, use_fixed_m=use_fixed_m)
    errs = {s: float(np.max(np.abs(unpadded - r))) for s, r in refs.items()}
    logit_scale = min(errs, key=errs.get)
    ref, err_ref = refs[logit_scale], errs[logit_scale]
    print(f"[{kernel}] unpadded vs dense f32 reference (logit scale {logit_scale:.4f}): max|diff|={err_ref:.3e}", flush=True)
    if err_ref > 5e-2:
      failures.append(f"{kernel} unpadded vs reference {err_ref:.3e}")
    for n_pad in (4800, 800):  # Test-1-like (+50%) and Test-2-like (+1/12) padding
      garbage = [rand(n_pad, 3.0) for _ in range(3)]
      q_nat, k_nat, v_nat = (jnp.concatenate([r, g], axis=1) for r, g in zip((q_real, k_real, v_real), garbage))
      for layout, mask in ((LAYOUT_SHARD, True), (LAYOUT_END, True), (LAYOUT_END, False)):
        tp = TokenPadding(n_real, n_real + n_pad, layout, ring_chunks, mask).validate()
        perm, inv = tp.natural_to_physical(), tp.physical_to_natural()
        q, k, v = (x if perm is None else x[:, perm] for x in (q_nat, k_nat, v_nat))
        out = _attention(mesh, q, k, v, token_padding=tp, use_fixed_m=use_fixed_m)
        out_nat = out if inv is None else out[:, inv]
        real = out_nat[:, :n_real]
        d_unpadded = float(np.max(np.abs(real - unpadded)))
        d_ref = float(np.max(np.abs(real - ref)))
        identical = bool(np.array_equal(real, unpadded))
        pad_rows_zero = bool(np.all(out_nat[:, n_real:] == 0)) if mask else None
        tag = f"[{kernel}] pad={n_pad} layout={layout} mask={mask}"
        print(
            f"{tag}: bit-identical-to-unpadded={identical} max|diff| unpadded={d_unpadded:.3e} "
            f"ref={d_ref:.3e} pad_rows_zero={pad_rows_zero}",
            flush=True,
        )
        if mask:
          if layout == LAYOUT_SHARD and not identical:
            failures.append(f"{tag} not bit-identical to unpadded ({d_unpadded:.3e})")
          if d_ref > 5e-2 or not pad_rows_zero:
            failures.append(f"{tag} wrong (ref {d_ref:.3e}, pad rows zero {pad_rows_zero})")
        elif d_unpadded < 1e-2:
          failures.append(f"{tag} control unexpectedly matches unpadded ({d_unpadded:.3e})")
  print("TOKEN PADDING RING TEST", "FAILED:\n  " + "\n  ".join(failures) if failures else "PASSED", flush=True)
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
