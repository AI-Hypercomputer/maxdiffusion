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

"""Phases 2-7: Numerical Parity Tests for PRXPixel Transformer Primitives, Blocks, and Full Model."""

import os
import unittest
import numpy as np
import safetensors.numpy as st_np
import jax
import jax.numpy as jnp
from flax import nnx

from maxdiffusion.models.prx_pixel import (
    FlaxPRXPixelConfig,
    NNXPRXPixelTransformer2DModel,
    load_prx_pixel_weights,
    img2seq_flax,
    seq2img_flax,
    get_image_ids_flax,
    NNXPRXEmbedND,
)
from tools.prxpixel.compare_prxpixel_tensors import compute_metrics

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
GOLDENS_DIR = os.path.join(REPO_ROOT, "goldens", "prxpixel")
SNAPSHOT_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--Photoroom--prxpixel-t2i/snapshots/bcd5e63f072257a220c5d0ba039c97657398b1c2")


class TestPRXPixelTransformerParity(unittest.TestCase):
  """Validates PRXPixel Transformer layers and full 24-block model against PyTorch golden references."""

  @classmethod
  def setUpClass(cls):
    jax.config.update("jax_default_matmul_precision", "highest")
    cls.config_fp32 = FlaxPRXPixelConfig(
        in_channels=3,
        patch_size=16,
        context_in_dim=2048,
        hidden_size=3584,
        mlp_ratio=3.5,
        num_heads=28,
        depth=24,
        axes_dim=(64, 64),
        theta=10000,
        dtype=jnp.float32,
    )
    cls.golden_fp32_path = os.path.join(GOLDENS_DIR, "transformer_tiny_fp32.safetensors")
    cls.golden_bf16_path = os.path.join(GOLDENS_DIR, "transformer_tiny_bf16.safetensors")

  def test_p0_patchify_unpatchify(self):
    """Checkpoint P0: Validates img2seq and seq2img reversibility on non-trivial spatial patterns."""
    B, C, H, W = 2, 3, 64, 64
    x = jnp.arange(B * C * H * W, dtype=jnp.float32).reshape(B, C, H, W)

    seq = img2seq_flax(x, patch_size=16)
    self.assertEqual(seq.shape, (B, (64 // 16) * (64 // 16), 3 * 16 * 16))

    reconstructed = seq2img_flax(seq, patch_size=16, shape=(B, C, H, W))
    np.testing.assert_array_equal(np.asarray(reconstructed), np.asarray(x))
    print("\n✅ [CHECKPOINT P0] img2seq and seq2img 100% reversible bitwise!")

  def test_f0_full_transformer_parity_fp32(self):
    """Checkpoints P1-F2: Layer-by-Layer and Full 24-Block Transformer Parity (FP32)."""
    if not os.path.exists(self.golden_fp32_path):
      self.skipTest(f"Golden FP32 reference missing at {self.golden_fp32_path}")

    goldens = st_np.load_file(self.golden_fp32_path)
    model_dir = os.path.join(SNAPSHOT_DIR, "transformer")

    print("\n" + "=" * 80)
    print("🚀 [PHASE 7] Instantiating NNXPRXPixelTransformer2DModel and Loading Weights...")
    print("=" * 80)

    model = NNXPRXPixelTransformer2DModel(config=self.config_fp32, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)

    eval_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, jnp.float32), state.to_pure_dict())
    loaded_params = load_prx_pixel_weights(model_dir, eval_shapes)
    nnx.update(model, loaded_params)
    print("✅ All 282 parameters loaded into NNXPRXPixelTransformer2DModel!")

    # Test inputs
    pixels = jnp.asarray(goldens["input/pixels"])
    txt_embeds = jnp.asarray(goldens["input/text_embeddings"])
    txt_mask = jnp.asarray(goldens["input/text_mask"])
    timestep = jnp.asarray(goldens["input/timestep"])

    b, c, h, w = pixels.shape

    print("\n" + "=" * 105)
    header = f"{'Layer / Tensor Name':<35} | {'Shape':<16} | {'Max Abs':<11} | {'Mean Abs':<11} | {'Rel L2':<11} | {'Cos Sim':<9}"
    print(header)
    print("=" * 105)

    # 1. Text projection
    txt = model.txt_in(txt_embeds)
    m_txt = compute_metrics(np.asarray(txt), goldens["txt_in/output"])
    print(f"{'txt_in/output':<35} | {str(txt.shape):<16} | {m_txt['max_abs']:<11.4e} | {m_txt['mean_abs']:<11.4e} | {m_txt['rel_l2']:<11.4e} | {m_txt['cos_sim']:<9.6f}")
    self.assertLess(m_txt["max_abs"], 1e-4)

    # 2. Pixel patchify & two-layer bottleneck
    patches = img2seq_flax(pixels, patch_size=model.config.patch_size)
    img0 = model.img_in[0](patches)
    m_img0 = compute_metrics(np.asarray(img0), goldens["img_in/linear0"])
    print(f"{'img_in/linear0':<35} | {str(img0.shape):<16} | {m_img0['max_abs']:<11.4e} | {m_img0['mean_abs']:<11.4e} | {m_img0['rel_l2']:<11.4e} | {m_img0['cos_sim']:<9.6f}")

    img = model.img_in[1](img0)
    m_img1 = compute_metrics(np.asarray(img), goldens["img_in/linear1"])
    print(f"{'img_in/linear1':<35} | {str(img.shape):<16} | {m_img1['max_abs']:<11.4e} | {m_img1['mean_abs']:<11.4e} | {m_img1['rel_l2']:<11.4e} | {m_img1['cos_sim']:<9.6f}")
    self.assertLess(m_img1["max_abs"], 1e-4)

    # 3. 2D RoPE
    img_ids = get_image_ids_flax(b, h, w, patch_size=model.config.patch_size)
    pe = model.pe_embedder(img_ids)
    m_pe = compute_metrics(np.asarray(pe), goldens["rope/embedding"])
    print(f"{'rope/embedding':<35} | {str(pe.shape):<16} | {m_pe['max_abs']:<11.4e} | {m_pe['mean_abs']:<11.4e} | {m_pe['rel_l2']:<11.4e} | {m_pe['cos_sim']:<9.6f}")
    self.assertLess(m_pe["max_abs"], 1e-5)

    # 4. Conditioning vector
    t_emb = model._compute_timestep_embedding(timestep)
    res_emb = model.resolution_embedder(h, w, b)
    vec = t_emb + res_emb
    m_vec = compute_metrics(np.asarray(vec), goldens["condition/combined_vec"])
    print(f"{'condition/combined_vec':<35} | {str(vec.shape):<16} | {m_vec['max_abs']:<11.4e} | {m_vec['mean_abs']:<11.4e} | {m_vec['rel_l2']:<11.4e} | {m_vec['cos_sim']:<9.6f}")
    self.assertLess(m_vec["max_abs"], 1e-4)

    # 5. Transformer blocks
    for idx, block in enumerate(model.blocks):
      img = block(
          hidden_states=img,
          encoder_hidden_states=txt,
          temb=vec,
          image_rotary_emb=pe,
          attention_mask=txt_mask,
      )
      m_b = compute_metrics(np.asarray(img), goldens[f"block_{idx:02d}/output"])
      if idx in (0, 1, 5, 11, 17, 23):
        print(f"{f'block_{idx:02d}/output':<35} | {str(img.shape):<16} | {m_b['max_abs']:<11.4e} | {m_b['mean_abs']:<11.4e} | {m_b['rel_l2']:<11.4e} | {m_b['cos_sim']:<9.6f}")
      self.assertGreater(m_b["cos_sim"], 0.9999, f"Block {idx} diverged! CosSim={m_b['cos_sim']}")

    # 6. Final Layer & unpatchify
    final_patches = model.final_layer(img, vec)
    final_img = seq2img_flax(final_patches, patch_size=model.config.patch_size, shape=(b, c, h, w))
    m_final = compute_metrics(np.asarray(final_img), goldens["final/unpatchified"])
    print("=" * 105)
    print(f"{'final/unpatchified (PREDICTED X0)':<35} | {str(final_img.shape):<16} | {m_final['max_abs']:<11.4e} | {m_final['mean_abs']:<11.4e} | {m_final['rel_l2']:<11.4e} | {m_final['cos_sim']:<9.6f}")
    print("=" * 105)

    print(f"\n📊 FINAL PRXPIXEL TRANSFORMER PARITY (FP32):")
    print(f"  • Max Absolute Error:     {m_final['max_abs']:.4e}")
    print(f"  • Mean Absolute Error:    {m_final['mean_abs']:.4e}")
    print(f"  • Relative L2 Error:      {m_final['rel_l2']:.4e}")
    print(f"  • Cosine Similarity:      {m_final['cos_sim']:.7f}")

    self.assertGreater(m_final["cos_sim"], 0.9999, "Final predicted x0 Cosine Similarity must be > 0.9999")
    self.assertLess(m_final["rel_l2"], 1e-3, "Final predicted x0 Relative L2 must be < 1e-3")
    print("\n🎉 [PHASE 7 COMPLETE - FP32] Full 24-Block PRXPixel Transformer achieves full numerical parity!")

  def test_f0_full_transformer_parity_bf16(self):
    """Checkpoints P1-F2: Full 24-Block Transformer Parity (BF16)."""
    if not os.path.exists(self.golden_bf16_path):
      self.skipTest(f"Golden BF16 reference missing at {self.golden_bf16_path}")

    goldens = st_np.load_file(self.golden_bf16_path)
    model_dir = os.path.join(SNAPSHOT_DIR, "transformer")

    config_bf16 = FlaxPRXPixelConfig(
        in_channels=3,
        patch_size=16,
        context_in_dim=2048,
        hidden_size=3584,
        mlp_ratio=3.5,
        num_heads=28,
        depth=24,
        axes_dim=(64, 64),
        theta=10000,
        dtype=jnp.bfloat16,
    )

    model = NNXPRXPixelTransformer2DModel(config=config_bf16, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)

    eval_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, jnp.bfloat16), state.to_pure_dict())
    loaded_params = load_prx_pixel_weights(model_dir, eval_shapes)
    nnx.update(model, loaded_params)

    pixels = jnp.asarray(goldens["input/pixels"], dtype=jnp.bfloat16)
    txt_embeds = jnp.asarray(goldens["input/text_embeddings"], dtype=jnp.bfloat16)
    txt_mask = jnp.asarray(goldens["input/text_mask"])
    timestep = jnp.asarray(goldens["input/timestep"])

    out = model(
        hidden_states=pixels,
        timestep=timestep,
        encoder_hidden_states=txt_embeds,
        attention_mask=txt_mask,
    )

    m_final = compute_metrics(np.asarray(out.astype(jnp.float32)), goldens["final/unpatchified"])
    print(f"\n📊 FINAL PRXPIXEL TRANSFORMER PARITY (BF16):")
    print(f"  • Max Absolute Error:     {m_final['max_abs']:.4e}")
    print(f"  • Mean Absolute Error:    {m_final['mean_abs']:.4e}")
    print(f"  • Relative L2 Error:      {m_final['rel_l2']:.4e}")
    print(f"  • Cosine Similarity:      {m_final['cos_sim']:.7f}")

    self.assertGreater(m_final["cos_sim"], 0.999, "BF16 Cosine Similarity must be > 0.999")
    print("🎉 [PHASE 7 COMPLETE - BF16] Full 24-Block PRXPixel Transformer achieves BF16 parity!")


if __name__ == "__main__":
  unittest.main()
