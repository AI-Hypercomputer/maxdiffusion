# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections.abc import Callable
import dataclasses
import functools
from typing import Any, TypeVar

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from . import base
from . import splash_attention_kernel as splash
from . import splash_attention_mask as mask_lib
from . import splash_attention_test_utils as test_utils


jax.config.parse_flags_with_absl()


hp.settings.register_profile(
    name="deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=15,
    print_blob=True,
    verbosity=hp.Verbosity.verbose,
)
hp.settings.load_profile(name="deterministic")

partial = functools.partial
Draw = TypeVar("Draw", bound=Callable[[hps.SearchStrategy[Any]], Any])


@dataclasses.dataclass
class ModelConfig:
  q_seq_len: int
  kv_seq_len: int
  num_q_heads: int
  num_kv_heads: int
  head_dim_qk: int
  head_dim_v: int
  dtype: np.dtype


@hps.composite
def segment_ids_strategy(draw, seq_len: int) -> base.SegmentIds:
  boundaries = hps.sets(hps.integers(1, seq_len - 1), min_size=1, max_size=4)
  bounds = sorted(draw(boundaries))
  ids_array = np.empty((seq_len,), dtype=np.int32)
  for i, (start, end) in enumerate(zip((0, *bounds), (*bounds, seq_len))):
    # Not sure why, but short segments can trip things up
    if end - start < 2:
      end = start + 2
    ids_array[start:end] = i
  return base.SegmentIds(ids_array, ids_array)


def seed_strategy() -> hps.SearchStrategy[int]:
  return hps.integers(min_value=0, max_value=4)


class Mask:

  def get_mask(self) -> mask_lib.Mask:
    raise NotImplementedError()


def full_mask_strategy(q_seq_len: int, kv_seq_len: int) -> hps.SearchStrategy[Mask]:
  return hps.just(FullMask(q_seq_len, kv_seq_len))


@dataclasses.dataclass
class SplitMask(Mask):
  q_seq_len: int
  kv_seq_len: int

  def get_mask(self) -> mask_lib.Mask:
    mask = np.ones((self.q_seq_len, self.kv_seq_len)).astype(np.bool_)
    mask[:, mask.shape[1] // 2 :] = False
    return mask_lib.NumpyMask(mask)


def split_mask_strategy(q_seq_len: int, kv_seq_len: int) -> hps.SearchStrategy[Mask]:
  return hps.just(SplitMask(q_seq_len, kv_seq_len))


@dataclasses.dataclass
class FullMask(Mask):
  q_seq_len: int
  kv_seq_len: int

  def get_mask(self) -> mask_lib.Mask:
    return mask_lib.FullMask((self.q_seq_len, self.kv_seq_len))


def causal_mask_strategy(q_seq_len: int, kv_seq_len: int) -> hps.SearchStrategy[Mask]:
  return hps.just(CausalMask(q_seq_len, kv_seq_len))


@dataclasses.dataclass
class CausalMask(Mask):
  q_seq_len: int
  kv_seq_len: int

  def get_mask(self) -> mask_lib.Mask:
    return mask_lib.CausalMask((self.q_seq_len, self.kv_seq_len))


@dataclasses.dataclass
class LocalAttentionMask(Mask):
  seq_len: int
  left: int | None
  right: int | None
  offset: int

  def get_mask(self) -> mask_lib.Mask:
    mask = mask_lib.LocalMask(
        (self.seq_len, self.seq_len),
        (self.left, self.right),
        offset=self.offset,
    )
    # Make sure that no row is full of zeros as this is leads to undefined
    # softmax.
    diagonal = mask_lib.NumpyMask(np.identity(self.seq_len, dtype=np.bool_))
    return mask | diagonal


@hps.composite
def local_attention_mask_strategy(draw: Draw, seq_len: int) -> Mask:
  left_window = draw(hps.one_of(hps.none(), hps.integers(min_value=0, max_value=seq_len)))
  right_window = draw(hps.one_of(hps.none(), hps.integers(min_value=0, max_value=seq_len)))
  offset = draw(hps.integers(min_value=-seq_len, max_value=seq_len - 1))
  return LocalAttentionMask(seq_len, left_window, right_window, offset=offset)


@dataclasses.dataclass
class RandomMask(Mask):
  q_seq_len: int
  kv_seq_len: int
  sparsity: float
  seed: int

  def get_mask(self) -> mask_lib.Mask:
    mask = mask_lib.make_random_mask((self.q_seq_len, self.kv_seq_len), self.sparsity, self.seed)
    # Make sure that no row is full of zeros as this is leads to undefined
    # softmax.
    mask[:, 0] = True

    return mask_lib.NumpyMask(mask)


@hps.composite
def random_mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int) -> Mask:
  rand = draw(hps.randoms())
  seed = rand.randint(0, 2**32 - 1)
  sparsity = rand.uniform(0.01, 0.5)
  return RandomMask(q_seq_len, kv_seq_len, sparsity, seed)


@dataclasses.dataclass
class ComposeMask(Mask):
  left: Mask
  right: Mask
  op: Callable[[mask_lib.Mask, mask_lib.Mask], mask_lib.Mask]

  def get_mask(self) -> mask_lib.Mask:
    return self.op(self.left.get_mask(), self.right.get_mask())


@hps.composite
def compose_mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int) -> Mask:
  mask1 = draw(mask_strategy(q_seq_len, kv_seq_len))
  mask2 = draw(mask_strategy(q_seq_len, kv_seq_len))
  op = draw(hps.one_of(hps.just(mask_lib.LogicalOr), hps.just(mask_lib.LogicalAnd)))
  return ComposeMask(mask1, mask2, op)


@hps.composite
def mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int) -> Mask:
  oneof = [
      causal_mask_strategy(q_seq_len, kv_seq_len),
      full_mask_strategy(q_seq_len, kv_seq_len),
      split_mask_strategy(q_seq_len, kv_seq_len),
      random_mask_strategy(q_seq_len, kv_seq_len),
      # TODO Composing masks creates masks that produce minor numerical
      # differences. We should investigate this in the future.
      # compose_mask_strategy(q_seq_len, kv_seq_len),
  ]

  if q_seq_len == kv_seq_len:
    oneof.append(local_attention_mask_strategy(q_seq_len))

  return draw(hps.one_of(oneof))


@hps.composite
def model_config_strategy(draw: Draw) -> ModelConfig:
  q_seq_len = draw(hps.sampled_from([1024, 2048, 4096]))
  kv_seq_len = draw(hps.sampled_from([1024, 2048, 4096]))
  head_dim_qk, head_dim_v = draw(hps.sampled_from([(64, 128), (64, 64), (128, 128), (256, 256), (192, 128)]))
  if q_seq_len >= 4096 and kv_seq_len >= 4096:
    dtype = np.dtype("float32")
  else:
    dtype = draw(hps.sampled_from([np.dtype("float32"), np.dtype(jnp.bfloat16)]))

  num_q_heads, num_kv_heads = draw(hps.sampled_from([(1, 1), (2, 2), (4, 1), (8, 4), (6, 2)]))
  return ModelConfig(
      q_seq_len,
      kv_seq_len,
      num_q_heads,
      num_kv_heads,
      head_dim_qk,
      head_dim_v,
      dtype,
  )


def check_mask_no_empty_rows(mask: mask_lib.Mask, segment_ids: splash.SegmentIds | None):
  effective_mask = np.array(mask[:, :])

  if segment_ids is not None:
    segment_mask = segment_ids.q[:, None] == segment_ids.kv[None, :]
    effective_mask = effective_mask & segment_mask

  hp.assume(np.all(np.any(effective_mask, axis=1)))


@hps.composite
def block_sizes_strategy(
    draw: Draw,
    q_seq_len: int,
    kv_seq_len: int,
    include_bwd_blocks: bool = False,
) -> splash.SplashConfig:
  all_block_shapes = [128, 256, 512]
  q_layout = draw(hps.sampled_from(splash.QKVLayout))
  k_layout = draw(hps.sampled_from(splash.QKVLayout))
  v_layout = draw(hps.sampled_from(splash.QKVLayout))
  layouts = {"q_layout": q_layout, "k_layout": k_layout, "v_layout": v_layout}
  q_valid_block_shapes = [bs for bs in all_block_shapes if bs <= q_seq_len]
  kv_valid_block_shapes = [bs for bs in all_block_shapes if bs <= kv_seq_len]
  bq, bkv = (
      draw(hps.sampled_from(q_valid_block_shapes)),
      draw(hps.sampled_from(kv_valid_block_shapes)),
  )
  bkv_compute = draw(hps.sampled_from([None, *[b for b in kv_valid_block_shapes if b <= bkv]]))
  if not include_bwd_blocks:
    return splash.SplashConfig(block_q=bq, block_kv=bkv, block_kv_compute=bkv_compute, **layouts)
  all_block_shapes = [128, 256]
  q_valid_block_shapes = [bs for bs in all_block_shapes if bs <= q_seq_len]
  kv_valid_block_shapes = [bs for bs in all_block_shapes if bs <= kv_seq_len]
  bq_dkv, bkv_dkv = (
      draw(hps.sampled_from(q_valid_block_shapes)),
      draw(hps.sampled_from(kv_valid_block_shapes)),
  )
  block_kv_dkv_compute = draw(hps.sampled_from([None, *[b for b in kv_valid_block_shapes if b <= bkv_dkv]]))
  return splash.SplashConfig(
      block_q=bq,
      block_kv=bkv,
      block_kv_compute=bkv_compute,
      block_q_dkv=bq_dkv,
      block_kv_dkv=bkv_dkv,
      block_kv_dkv_compute=block_kv_dkv_compute,
      **layouts,
  )


def _generate_inputs(
    data,
    config: ModelConfig,
    is_mqa: bool,
    is_segmented: bool,
    use_sinks: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, splash.SegmentIds | None, jax.Array,]:
  seed = data.draw(seed_strategy())
  key = random.key(seed)
  k1, k2, k3, k_sinks, k_do = random.split(key, 5)

  q_shape = (config.num_q_heads, config.q_seq_len, config.head_dim_qk)
  if is_mqa:
    k_shape = (config.kv_seq_len, config.head_dim_qk)
    v_shape = (config.kv_seq_len, config.head_dim_v)
  else:
    k_shape = (config.num_kv_heads, config.kv_seq_len, config.head_dim_qk)
    v_shape = (config.num_kv_heads, config.kv_seq_len, config.head_dim_v)

  q = random.uniform(k1, q_shape, dtype=config.dtype)
  k = random.uniform(k2, k_shape, dtype=config.dtype)
  v = random.uniform(k3, v_shape, dtype=config.dtype)

  sinks = None
  if use_sinks:
    sinks = random.uniform(k_sinks, (config.num_q_heads,), dtype=config.dtype)

  segment_ids = None
  if is_segmented:
    hp.assume(config.q_seq_len == config.kv_seq_len)
    segment_ids = data.draw(segment_ids_strategy(config.q_seq_len))

  o_shape = (config.num_q_heads, config.q_seq_len, config.head_dim_v)
  do = random.uniform(k_do, o_shape, dtype=config.dtype)
  return (q, k, v, sinks, segment_ids, do)


def attn_logits_soft_cap_strategy() -> hps.SearchStrategy[float | None]:
  return hps.one_of(hps.just(None), hps.floats(min_value=1.0, max_value=50.0))


@test_utils.thread_unsafe_test_class()  # hypothesis is not thread safe
class SplashAttentionTest(test_utils.SplashAttentionTestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.product(
      is_mqa=(False, True),
      is_segmented=(False, True),
      is_dynamic_mask=(False, True),
  )
  @hp.given(hps.data())
  def test_splash_attention(self, is_mqa, is_segmented, is_dynamic_mask, data):
    model_config = data.draw(model_config_strategy())
    q_seq_len, kv_seq_len = model_config.q_seq_len, model_config.kv_seq_len
    q, k, v, _, segment_ids, _ = _generate_inputs(data, model_config, is_mqa, is_segmented)
    attn_logits_soft_cap = data.draw(attn_logits_soft_cap_strategy())
    mask_obj = data.draw(mask_strategy(q_seq_len, kv_seq_len))
    mask = mask_obj.get_mask()
    # Skip edge case: single attention head + random mask triggers JAX/Mosaic compilation bug
    hp.assume(not (model_config.num_q_heads == 1 and isinstance(mask_obj, RandomMask)))
    check_mask_no_empty_rows(mask, segment_ids)
    if is_dynamic_mask:
      mask = jnp.array(mask[:, :])
    config = data.draw(block_sizes_strategy(q_seq_len, kv_seq_len))
    config = dataclasses.replace(
        config,
        attn_logits_soft_cap=attn_logits_soft_cap,
        interpret=self.INTERPRET,
    )

    attn_ref = partial(base.attention_reference, is_mqa=is_mqa)
    if is_mqa:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mqa_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mqa
    else:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mha_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mha

    attn = make_mask_fn(mask, config=config)

    o = attn(q, k, v, segment_ids)
    o_ref = attn_ref(
        q.astype(np.float32),
        k.astype(np.float32),
        v.astype(np.float32),
        jnp.array(mask[:, :]),
        segment_ids,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    self._assert_allclose(o, o_ref, atol=6e-3, rtol=3e-3)

  @parameterized.product(
      is_mqa=(False, True),
      is_segmented=(False, True),
      is_dynamic_mask=(False, True),
      use_base2_exp=(False, True),
      use_max_logit_estimate=(None, "const", "value_1d", "value_2d"),
      fuse_reciprocal=(True, False),
      use_sinks=(False, True),
  )
  @hp.given(hps.data())
  def test_splash_attention_fwd(
      self, is_mqa, is_segmented, is_dynamic_mask, use_base2_exp, use_max_logit_estimate, fuse_reciprocal, use_sinks, data
  ):
    model_config = data.draw(model_config_strategy())
    q_seq_len, kv_seq_len = model_config.q_seq_len, model_config.kv_seq_len
    q, k, v, sinks, segment_ids, _ = _generate_inputs(data, model_config, is_mqa, is_segmented, use_sinks)
    attn_logits_soft_cap = data.draw(attn_logits_soft_cap_strategy())
    mask = data.draw(mask_strategy(q_seq_len, kv_seq_len)).get_mask()
    check_mask_no_empty_rows(mask, segment_ids)
    if is_dynamic_mask:
      mask = jnp.array(mask[:, :])
    config = data.draw(block_sizes_strategy(q_seq_len, kv_seq_len))
    if is_mqa:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mqa_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mqa
    else:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mha_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mha

    config = dataclasses.replace(
        config,
        fuse_reciprocal=fuse_reciprocal,
        attn_logits_soft_cap=attn_logits_soft_cap,
        use_base2_exp=use_base2_exp,
        interpret=self.INTERPRET,
    )

    max_logit_value, max_val = None, 30.0
    if use_max_logit_estimate == "const":
      config = dataclasses.replace(config, max_logit_const=max_val)
    elif use_max_logit_estimate == "value_1d":
      max_logit_value = max_val * jnp.ones((1,), dtype=jnp.bfloat16)
    elif use_max_logit_estimate == "value_2d":
      max_logit_value = max_val * jnp.ones((model_config.num_q_heads,), dtype=jnp.bfloat16)
    attn = make_mask_fn(mask, config=config, save_residuals=True)
    attn_ref = partial(
        base.attention_reference,
        is_mqa=is_mqa,
        save_residuals=True,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    o, stats = attn(q, k, v, segment_ids, sinks, max_logit_value=max_logit_value)

    o_ref, stats_ref = attn_ref(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        jnp.array(mask[:, :]),
        segment_ids,
        sinks,
    )

    lse_tol = {"atol": 1e-3, "rtol": 3e-3}
    max_logits_tol = {"atol": 1e-3, "rtol": 4e-3}
    if use_sinks:
      o_tol = {"atol": 8e-2, "rtol": 1e-1}
      lse_tol["rtol"] = 6e-2
    elif use_base2_exp or use_max_logit_estimate is not None or not fuse_reciprocal:
      o_tol = {"atol": 8e-3, "rtol": 3e-3}
    else:
      o_tol = {"atol": 4e-3, "rtol": 3e-3}

    self._assert_allclose(o, o_ref, **o_tol)
    self._assert_allclose(stats["logsumexp"], stats_ref["logsumexp"], **lse_tol)
    if use_max_logit_estimate is None:
      self._assert_allclose(stats["max_logits"], stats_ref["max_logits"], **max_logits_tol)

  @parameterized.product(
      is_mqa=(False, True),
      is_segmented=(False, True),
      is_dynamic_mask=(False, True),
      # use_max_logit_estimate=(None, "const", "value_1d", "value_2d"),
      use_max_logit_estimate=(None,),
      use_sinks=(False, True),
      dq_reduction_steps=(None, 3),
  )
  @hp.given(hps.data())
  def test_splash_attention_bwd(
      self,
      is_mqa,
      is_segmented,
      is_dynamic_mask,
      use_max_logit_estimate,
      dq_reduction_steps,
      use_sinks,
      data,
  ):
    downcast_smem_data = data.draw(hp.strategies.booleans())
    fuse_reciprocal = data.draw(hp.strategies.booleans())
    use_base2_exp = data.draw(hp.strategies.booleans())

    model_config = data.draw(model_config_strategy())
    q_seq_len, kv_seq_len = model_config.q_seq_len, model_config.kv_seq_len
    q, k, v, sinks, segment_ids, do = _generate_inputs(data, model_config, is_mqa, is_segmented, use_sinks=use_sinks)
    attn_logits_soft_cap = data.draw(attn_logits_soft_cap_strategy())
    mask = data.draw(mask_strategy(q_seq_len, kv_seq_len)).get_mask()
    check_mask_no_empty_rows(mask, segment_ids)
    if is_dynamic_mask:
      mask = jnp.array(mask[:, :])
    config = data.draw(block_sizes_strategy(q_seq_len, kv_seq_len, include_bwd_blocks=True))

    config = dataclasses.replace(
        config,
        fuse_reciprocal=fuse_reciprocal,
        attn_logits_soft_cap=attn_logits_soft_cap,
        interpret=self.INTERPRET,
        use_base2_exp=use_base2_exp,
        dq_reduction_steps=dq_reduction_steps,
    )
    if is_mqa:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mqa_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mqa
    else:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mha_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mha

    max_logit_value, max_val = None, 30.0
    if use_max_logit_estimate == "const":
      config = dataclasses.replace(config, max_logit_const=max_val)
    elif use_max_logit_estimate == "value_1d":
      max_logit_value = max_val * jnp.ones((1,), dtype=jnp.bfloat16)
    elif use_max_logit_estimate == "value_2d":
      max_logit_value = max_val * jnp.ones((model_config.num_q_heads,), dtype=jnp.bfloat16)

    attn = make_mask_fn(mask, config=config, downcast_smem_data=downcast_smem_data)

    o, attn_vjp = jax.vjp(partial(attn, max_logit_value=max_logit_value), q, k, v, segment_ids, sinks)
    q32, k32, v32 = jax.tree.map(lambda x: x.astype(jnp.float32), (q, k, v))
    o_ref, stats_ref = base.attention_reference(
        q32,
        k32,
        v32,
        jnp.array(mask[:, :]),
        segment_ids,
        sinks,
        is_mqa=is_mqa,
        save_residuals=True,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    if use_sinks:
      o_tol = {"atol": 1e-2, "rtol": 1e-1}
    elif use_base2_exp or use_max_logit_estimate is not None or not fuse_reciprocal:
      o_tol = {"atol": 8e-3, "rtol": 1e-2}
    else:
      o_tol = {"atol": 4e-3, "rtol": 3e-3}
    self._assert_allclose(o, o_ref, **o_tol)

    dq, dk, dv, _, dsinks = attn_vjp(do)
    dq_ref, dk_ref, dv_ref, dsinks_ref = base.attention_reference_vjp(
        do.astype(jnp.float32),
        q32,
        k32,
        v32,
        jnp.array(mask[:, :]),
        segment_ids,
        sinks,
        o.astype(jnp.float32),
        stats_ref["logsumexp"],
        is_mqa=is_mqa,
        backward_impl="flash",
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    dq_atol = 8e-2 if use_base2_exp else 2e-2
    dk_atol = 7e-2 if use_base2_exp else 2e-2
    dv_atol = 2e-2 if use_base2_exp else 2e-2
    self._assert_allclose(dq, dq_ref, atol=dq_atol, rtol=3e-2)
    self._assert_allclose(dk, dk_ref, atol=dk_atol, rtol=3e-2)
    self._assert_allclose(dv, dv_ref, atol=dv_atol, rtol=3e-2)
    if use_sinks:
      self._assert_allclose(dsinks, dsinks_ref, atol=4e-3, rtol=6e-3)


if __name__ == "__main__":
  absltest.main()
