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

Production entry points for SVG attention.

This module is the top layer of the SVG kernel stack and exists to keep that
stack acyclic:

  custom_splash_attention                       (dense primitives)
    -> custom_svg_static_range_attention        (tiling, partial kernels,
                                                 exact reference builder)
      -> custom_svg_balanced_rounding_attention (boundary selection, production
                                                 builder)
        -> custom_svg_attention_dispatch        (this module)

Each arrow points from a dependency to its dependent, and no arrow runs the
other way. Callers that want "the SVG kernel we ship" use this module; callers
that want a specific implementation import the corresponding layer directly.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.kernels import custom_svg_balanced_rounding_attention as balanced
from maxdiffusion.kernels import custom_svg_static_range_attention as static_range

SVGBlockSizes = static_range.SVGBlockSizes


def make_svg_static_range_mha(
    *,
    block_sizes,
    orig_q_seq_len,
    orig_kv_seq_len,
    band_width,
    frame_size,
    include_first_frame=True,
    bkv_compute_in=None,
    use_base2_exp=True,
    use_experimental_scheduler=False,
    vmem_limit_bytes=None,
):
  """Build the production balanced-rounding SVG implementation.

  The upstream path intentionally has no environment-variable policy switch.
  Production uses the validated `global_balanced`, scale=1.0 policy. The exact
  reference remains available explicitly via
  `custom_svg_static_range_attention.make_svg_exact_static_range_mha`.
  """
  return balanced.make_svg_balanced_rounding_mha(
      policy="global_balanced",
      budget_scale=1.0,
      block_sizes=block_sizes,
      orig_q_seq_len=orig_q_seq_len,
      orig_kv_seq_len=orig_kv_seq_len,
      band_width=band_width,
      frame_size=frame_size,
      include_first_frame=include_first_frame,
      bkv_compute_in=bkv_compute_in,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      vmem_limit_bytes=vmem_limit_bytes,
  )


def custom_svg_static_range_attention(
    query,
    key,
    value,
    band_width,
    anchor_width=0,
    global_stride=0,
    global_offset=0,
    mesh: Any = None,
    axis_names_q=None,
    axis_names_kv=None,
    dtype=jnp.bfloat16,
    block_sizes=None,
    use_base2_exp=True,
    use_experimental_scheduler=False,
):
  del global_stride, global_offset, mesh, axis_names_q, axis_names_kv, dtype
  if block_sizes is None:
    block_sizes = SVGBlockSizes()
  qlen = query.shape[2]
  klen = key.shape[2]
  bw = int(np.asarray(band_width).max()) if isinstance(band_width, (jax.Array, np.ndarray)) else int(band_width)
  kernel = make_svg_static_range_mha(
      block_sizes=block_sizes,
      orig_q_seq_len=qlen,
      orig_kv_seq_len=klen,
      band_width=bw,
      frame_size=int(anchor_width) if anchor_width > 0 else 0,
      include_first_frame=anchor_width > 0,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
  )
  out = jax.vmap(kernel, in_axes=(0, 0, 0))(query, key, value)
  return jnp.swapaxes(out, 2, 3)
