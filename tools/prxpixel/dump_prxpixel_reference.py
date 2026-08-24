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

"""Dump golden reference activations from PyTorch PRXPixel Transformer and Pipeline."""

import argparse
import os
import torch
import numpy as np
import safetensors.numpy as st_np
import safetensors.torch as st_pt
from diffusers import PRXPipeline, PRXTransformer2DModel
from diffusers.models.transformers.transformer_prx import (
    img2seq, seq2img, get_image_ids, get_timestep_embedding, apply_rope, MLPEmbedder
)

MODEL_ID = "Photoroom/prxpixel-t2i"
REVISION = "bcd5e63f072257a220c5d0ba039c97657398b1c2"


def build_pytorch_transformer_model(checkpoint_dir: str, dtype=torch.float32) -> PRXTransformer2DModel:
  """Instantiates the exact PRXTransformer2DModel matching the 282 checkpoint weights."""
  import torch.nn as nn

  model = PRXTransformer2DModel(
      in_channels=3,
      patch_size=16,
      context_in_dim=2048,
      hidden_size=3584,
      mlp_ratio=3.5,
      num_heads=28,
      depth=24,
      axes_dim=[64, 64],
      theta=10000,
      time_factor=1000.0,
      time_max_period=10000,
  )

  # Fix two-layer img_in bottleneck
  model.img_in = nn.Sequential(
      nn.Linear(768, 768, bias=True),
      nn.Linear(768, 3584, bias=True),
  )

  # Fix resolution embedder
  class ResEmbedder(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=3584):
      super().__init__()
      self.mlp = MLPEmbedder(in_dim=in_dim, hidden_dim=hidden_dim)

    def forward(self, h: int, w: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
      h_emb = get_timestep_embedding(torch.tensor([float(h)], device=device), 128, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=1.0)
      w_emb = get_timestep_embedding(torch.tensor([float(w)], device=device), 128, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=1.0)
      res_emb = torch.cat([h_emb, w_emb], dim=-1).repeat(batch_size, 1).to(dtype)
      return self.mlp(res_emb)

  model.resolution_embedder = ResEmbedder(256, 3584)

  # Load weights from safetensors
  import glob
  shards = sorted(glob.glob(os.path.join(checkpoint_dir, "transformer/*.safetensors")))
  state_dict = {}
  for s in shards:
    state_dict.update(st_pt.load_file(s))

  load_res = model.load_state_dict(state_dict, strict=True)
  print(f"✅ Loaded PyTorch PRXTransformer2DModel strictly with 0 missing/unexpected keys: {load_res}")
  model.eval()
  model.to(dtype)
  return model


def dump_tiny_transformer_golden(model: PRXTransformer2DModel, output_dir: str, dtype_str: str = "fp32"):
  """Dumps layer-by-layer golden reference on a small 64x64 spatial grid (16 image tokens)."""
  print(f"\n================================================================================")
  print(f"🚀 Dumping Tiny PRX Transformer Golden Reference (64x64, {dtype_str.upper()})...")
  print(f"================================================================================")

  dtype = torch.float32 if dtype_str == "fp32" else torch.bfloat16
  model = model.to(dtype)

  B, C, H, W = 1, 3, 64, 64
  torch.manual_seed(42)

  # Deterministic test inputs
  pixels = torch.linspace(-1.0, 1.0, B * C * H * W, dtype=dtype).reshape(B, C, H, W)
  txt_embeds = torch.linspace(-0.5, 0.5, B * 256 * 2048, dtype=dtype).reshape(B, 256, 2048)
  txt_mask = torch.ones((B, 256), dtype=torch.bool)
  timestep = torch.tensor([0.5], dtype=torch.float32)

  activations = {}
  activations["input/pixels"] = pixels.detach().cpu().float().numpy()
  activations["input/text_embeddings"] = txt_embeds.detach().cpu().float().numpy()
  activations["input/text_mask"] = txt_mask.detach().cpu().numpy().astype(np.int32)
  activations["input/timestep"] = timestep.detach().cpu().numpy()

  with torch.no_grad():
    # 1. Text projection
    txt = model.txt_in(txt_embeds)
    activations["txt_in/output"] = txt.detach().cpu().float().numpy()

    # 2. Pixel patchify & bottleneck
    patches = img2seq(pixels, model.patch_size)
    activations["patchify/output"] = patches.detach().cpu().float().numpy()

    img0 = model.img_in[0](patches)
    activations["img_in/linear0"] = img0.detach().cpu().float().numpy()
    img = model.img_in[1](img0)
    activations["img_in/linear1"] = img.detach().cpu().float().numpy()

    # 3. Positional embeddings
    img_ids = get_image_ids(B, H, W, patch_size=model.patch_size, device=pixels.device)
    activations["rope/image_ids"] = img_ids.detach().cpu().float().numpy()
    pe = model.pe_embedder(img_ids)
    activations["rope/embedding"] = pe.detach().cpu().float().numpy()

    # 4. Timestep & Resolution conditioning
    t_emb = model.time_in(
        get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=256,
            max_period=model.time_max_period,
            scale=model.time_factor,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        ).to(dtype)
    )
    activations["condition/timestep_mlp"] = t_emb.detach().cpu().float().numpy()

    res_emb = model.resolution_embedder(H, W, B, pixels.device, dtype)
    activations["condition/resolution_mlp"] = res_emb.detach().cpu().float().numpy()

    vec = t_emb + res_emb
    activations["condition/combined_vec"] = vec.detach().cpu().float().numpy()

    # 5. Transformer blocks
    for idx, block in enumerate(model.blocks):
      activations[f"block_{idx:02d}/input"] = img.detach().cpu().float().numpy()
      img = block(
          hidden_states=img,
          encoder_hidden_states=txt,
          temb=vec,
          image_rotary_emb=pe,
          attention_mask=txt_mask,
      )
      activations[f"block_{idx:02d}/output"] = img.detach().cpu().float().numpy()

    # 6. Final layer & unpatchify
    final_patches = model.final_layer(img, vec)
    activations["final/linear"] = final_patches.detach().cpu().float().numpy()

    final_img = seq2img(final_patches, model.patch_size, pixels.shape)
    activations["final/unpatchified"] = final_img.detach().cpu().float().numpy()

  out_path = os.path.join(output_dir, f"transformer_tiny_{dtype_str}.safetensors")
  st_np.save_file(activations, out_path)
  print(f"✅ Saved {len(activations)} tiny transformer activations ({dtype_str}) to: {out_path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--snapshot_dir", type=str, default=os.path.expanduser("~/.cache/huggingface/hub/models--Photoroom--prxpixel-t2i/snapshots/bcd5e63f072257a220c5d0ba039c97657398b1c2"))
  parser.add_argument("--output_dir", type=str, default="goldens/prxpixel")
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  model = build_pytorch_transformer_model(args.snapshot_dir, dtype=torch.float32)
  dump_tiny_transformer_golden(model, args.output_dir, dtype_str="fp32")
  dump_tiny_transformer_golden(model, args.output_dir, dtype_str="bf16")


if __name__ == "__main__":
  main()
