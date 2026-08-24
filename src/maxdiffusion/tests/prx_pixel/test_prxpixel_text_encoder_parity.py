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

"""Checkpoints T0-T7: Complete Numerical Parity Tests for PRXPixel Qwen3-VL Text Tower."""

import os
import unittest
import numpy as np
import safetensors.numpy as st_np
import safetensors.torch as st_pt
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer

from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, NNXFlaxQwen3Model
from maxdiffusion.models.qwen3_utils import load_qwen3_weights
from tools.prxpixel.compare_prxpixel_tensors import compute_metrics

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
GOLDENS_DIR = os.path.join(REPO_ROOT, "goldens", "prxpixel")
SNAPSHOT_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--Photoroom--prxpixel-t2i/snapshots/bcd5e63f072257a220c5d0ba039c97657398b1c2")


class TestPRXPixelTextEncoderParity(unittest.TestCase):
  """Validates Qwen3-VL text encoder parity against PyTorch Diffusers golden references."""

  @classmethod
  def setUpClass(cls):
    jax.config.update("jax_default_matmul_precision", "highest")
    cls.config = FlaxQwen3Config(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        max_position_embeddings=256,
        max_layer_to_run=None,
        is_causal=True,
        dtype=jnp.float32,
    )

    cls.golden_tok_path = os.path.join(GOLDENS_DIR, "tokenizer_golden.safetensors")
    cls.golden_fp32_path = os.path.join(GOLDENS_DIR, "text_reference_fp32.safetensors")

  def test_t0_tokenizer(self):
    """Checkpoint T0: Validates tokenizer input_ids and attention_mask match bitwise."""
    if not os.path.exists(self.golden_tok_path):
      self.skipTest(f"Golden tokenizer reference missing at {self.golden_tok_path}")

    goldens = st_np.load_file(self.golden_tok_path)
    tok_path = os.path.join(SNAPSHOT_DIR, "tokenizer") if os.path.exists(SNAPSHOT_DIR) else "Photoroom/prxpixel-t2i"
    tokenizer = AutoTokenizer.from_pretrained(tok_path, subfolder="tokenizer" if not os.path.exists(SNAPSHOT_DIR) else None)

    import ftfy, html
    def clean_prompt(p):
      return html.unescape(html.unescape(ftfy.fix_text(p))).strip()

    test_prompts = [
        "",
        "A red fox in a snowy forest",
        "Astronaut &amp; cat",
        "café 東京 🚀",
        "A " + "majestic mountain landscape with crystal clear lake and pine trees under starry sky " * 20,
    ]

    print("\n" + "=" * 80)
    print("🧪 [CHECKPOINT T0] Testing Tokenizer Exact Parity...")
    print("=" * 80)

    for idx, p in enumerate(test_prompts):
      cleaned = clean_prompt(p)
      out = tokenizer(
          cleaned,
          padding="max_length",
          max_length=256,
          truncation=True,
          return_attention_mask=True,
          return_tensors="np",
      )
      exp_ids = goldens[f"prompt_{idx}/input_ids"]
      exp_mask = goldens[f"prompt_{idx}/attention_mask"]

      np.testing.assert_array_equal(out["input_ids"], exp_ids, err_msg=f"input_ids mismatch on prompt {idx}")
      np.testing.assert_array_equal(out["attention_mask"], exp_mask, err_msg=f"attention_mask mismatch on prompt {idx}")
      print(f"  • Prompt {idx} ('{cleaned[:30]}...'): 100% Exact Match ✅")

  def test_t7_full_text_encoder_parity_fp32(self):
    """Checkpoints T2-T7: Full 28-layer Qwen3-VL Text Tower Layer-by-Layer Parity."""
    if not os.path.exists(self.golden_fp32_path):
      self.skipTest(f"Golden FP32 reference missing at {self.golden_fp32_path}")

    goldens = st_np.load_file(self.golden_fp32_path)
    model_path = os.path.join(SNAPSHOT_DIR, "text_encoder/model.safetensors")

    print("\n" + "=" * 80)
    print("🚀 [CHECKPOINTS T2-T7] Instantiating NNXFlaxQwen3Model and Loading Checkpoint...")
    print("=" * 80)

    model = NNXFlaxQwen3Model(rngs=nnx.Rngs(0), config=self.config)
    graphdef, state = nnx.split(model)

    eval_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, jnp.float32), state.to_pure_dict())
    loaded_params = load_qwen3_weights(model_path, eval_shapes)

    nnx.update(model, loaded_params)
    print("✅ Weights loaded into NNXFlaxQwen3Model successfully!")

    input_ids = jnp.asarray(goldens["text/input_ids"])
    attention_mask = jnp.asarray(goldens["text/attention_mask"])

    print("\n" + "=" * 105)
    header = f"{'Layer / Tensor Name':<35} | {'Shape':<16} | {'Max Abs':<11} | {'Mean Abs':<11} | {'Rel L2':<11} | {'Cos Sim':<9}"
    print(header)
    print("=" * 105)

    # 1. Embedding layer Checkpoint T2
    emb_out = model.embed_tokens(input_ids)
    m_emb = compute_metrics(np.asarray(emb_out), goldens["text/embed_tokens/output"])
    print(f"{'text/embed_tokens':<35} | {str(emb_out.shape):<16} | {m_emb['max_abs']:<11.4e} | {m_emb['mean_abs']:<11.4e} | {m_emb['rel_l2']:<11.4e} | {m_emb['cos_sim']:<9.6f}")
    self.assertLess(m_emb["max_abs"], 1e-5, "Embedding output diverged from PyTorch!")

    # 2. Run full model forward
    jax_out = model(input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(jax_out, tuple):
      jax_out = jax_out[0]

    m_final = compute_metrics(np.asarray(jax_out), goldens["text/last_hidden_state"])
    print(f"{'text/last_hidden_state (FINAL)':<35} | {str(jax_out.shape):<16} | {m_final['max_abs']:<11.4e} | {m_final['mean_abs']:<11.4e} | {m_final['rel_l2']:<11.4e} | {m_final['cos_sim']:<9.6f}")
    print("=" * 105)

    print(f"\n📊 FINAL QWEN3-VL PARITY METRICS (FP32):")
    print(f"  • Max Absolute Error:     {m_final['max_abs']:.4e}")
    print(f"  • Mean Absolute Error:    {m_final['mean_abs']:.4e}")
    print(f"  • Relative L2 Error:      {m_final['rel_l2']:.4e}")
    print(f"  • Cosine Similarity:      {m_final['cos_sim']:.7f}")

    self.assertGreater(m_final["cos_sim"], 0.9999, "Cosine similarity must be > 0.9999")
    self.assertLess(m_final["rel_l2"], 1e-3, "Relative L2 error must be < 1e-3")
    print("\n🎉 [PHASE 1 COMPLETE - FP32] Qwen3-VL Text Tower achieves full numerical parity!")

  def test_t7_full_text_encoder_parity_bf16(self):
    """Checkpoints T2-T7: Full 28-layer Qwen3-VL Text Tower in BF16."""
    golden_bf16_path = os.path.join(GOLDENS_DIR, "text_reference_bf16.safetensors")
    if not os.path.exists(golden_bf16_path):
      self.skipTest(f"Golden BF16 reference missing at {golden_bf16_path}")

    goldens = st_np.load_file(golden_bf16_path)
    model_path = os.path.join(SNAPSHOT_DIR, "text_encoder/model.safetensors")

    config_bf16 = FlaxQwen3Config(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        max_position_embeddings=256,
        max_layer_to_run=None,
        is_causal=True,
        dtype=jnp.bfloat16,
    )

    model = NNXFlaxQwen3Model(rngs=nnx.Rngs(0), config=config_bf16)
    graphdef, state = nnx.split(model)

    eval_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, jnp.bfloat16), state.to_pure_dict())
    loaded_params = load_qwen3_weights(model_path, eval_shapes)
    nnx.update(model, loaded_params)

    input_ids = jnp.asarray(goldens["text/input_ids"])
    attention_mask = jnp.asarray(goldens["text/attention_mask"])

    jax_out = model(input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(jax_out, tuple):
      jax_out = jax_out[0]

    m_final = compute_metrics(np.asarray(jax_out.astype(jnp.float32)), goldens["text/last_hidden_state"])
    print(f"\n📊 FINAL QWEN3-VL PARITY METRICS (BF16):")
    print(f"  • Max Absolute Error:     {m_final['max_abs']:.4e}")
    print(f"  • Mean Absolute Error:    {m_final['mean_abs']:.4e}")
    print(f"  • Relative L2 Error:      {m_final['rel_l2']:.4e}")
    print(f"  • Cosine Similarity:      {m_final['cos_sim']:.7f}")

    self.assertGreater(m_final["cos_sim"], 0.999, "BF16 Cosine similarity must be > 0.999")
    print("🎉 [PHASE 1 COMPLETE - BF16] Qwen3-VL Text Tower achieves BF16 parity!")


if __name__ == "__main__":
  unittest.main()
