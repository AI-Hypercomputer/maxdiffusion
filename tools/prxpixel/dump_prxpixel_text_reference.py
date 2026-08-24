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

"""Dump golden reference activations from PyTorch Qwen3-VL text encoder for PRXPixel."""

import argparse
import html
import json
import os
import ftfy
import numpy as np
import safetensors.numpy as st_np
import torch
from transformers import AutoTokenizer, AutoConfig

MODEL_ID = "Photoroom/prxpixel-t2i"
REVISION = "bcd5e63f072257a220c5d0ba039c97657398b1c2"


def clean_prompt(prompt: str) -> str:
  """Exact prompt preprocessing used by PRXPixel pipeline."""
  prompt = ftfy.fix_text(prompt)
  prompt = html.unescape(html.unescape(prompt))
  return prompt.strip()


def dump_tokenizer_golden(tokenizer, output_dir: str):
  """Dump tokenizer golden tensors for multiple edge-case prompts."""
  test_prompts = [
      "",
      "A red fox in a snowy forest",
      "Astronaut &amp; cat",
      "café 東京 🚀",
      "A " + "majestic mountain landscape with crystal clear lake and pine trees under starry sky " * 20,
  ]

  data = {}
  for idx, p in enumerate(test_prompts):
    cleaned = clean_prompt(p)
    tok_out = tokenizer(
        cleaned,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    data[f"prompt_{idx}/input_ids"] = tok_out["input_ids"].astype(np.int32)
    data[f"prompt_{idx}/attention_mask"] = tok_out["attention_mask"].astype(np.int32)

  out_path = os.path.join(output_dir, "tokenizer_golden.safetensors")
  st_np.save_file(data, out_path)
  print(f"✅ Saved tokenizer golden reference to: {out_path}")


def dump_text_encoder_golden(model_dir: str, output_dir: str, dtype_str: str = "fp32"):
  """Instrument and dump all intermediate activations from PyTorch Qwen3-VL text encoder."""
  from transformers import Qwen2_5_VLForConditionalGeneration, AutoModel

  print(f"\n================================================================================")
  print(f"🚀 Dumping PyTorch Qwen3-VL Text Tower Reference ({dtype_str.upper()})...")
  print(f"================================================================================")

  tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer", revision=REVISION)
  dump_tokenizer_golden(tokenizer, output_dir)

  # Load text model
  dtype = torch.float32 if dtype_str == "fp32" else torch.bfloat16
  
  # Try loading text encoder directly from subfolder
  text_encoder = None
  try:
    from diffusers import PRXPipeline
    pipe = PRXPipeline.from_pretrained(model_dir, torch_dtype=dtype, revision=REVISION)
    text_encoder = pipe.text_encoder
  except Exception as e:
    print(f"ℹ️ Could not load via PRXPipeline: {e}. Trying direct subfolder...")
    try:
      text_encoder = AutoModel.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=dtype, revision=REVISION)
    except Exception as e2:
      print(f"ℹ️ Direct AutoModel load: {e2}")

  if text_encoder is None:
    raise RuntimeError("Failed to load Qwen3-VL text encoder!")

  text_encoder.eval()
  text_encoder.to("cpu")
  if dtype_str == "fp32":
    text_encoder = text_encoder.float()
  else:
    text_encoder = text_encoder.bfloat16()

  prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
  neg_prompt = ""
  
  cleaned_prompts = [clean_prompt(neg_prompt), clean_prompt(prompt)]
  tok_out = tokenizer(
      cleaned_prompts,
      padding="max_length",
      max_length=256,
      truncation=True,
      return_attention_mask=True,
      return_tensors="pt",
  )

  input_ids = tok_out["input_ids"].to("cpu")
  attention_mask = tok_out["attention_mask"].to("cpu")

  activations = {}
  activations["text/input_ids"] = input_ids.numpy().astype(np.int32)
  activations["text/attention_mask"] = attention_mask.numpy().astype(np.int32)

  # Register forward hooks on layers
  hooks = []
  
  # Base text model
  base_model = getattr(text_encoder, "model", text_encoder)

  def hook_fn(name):
    def fn(module, input, output):
      if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor):
        inp_np = input[0].detach().cpu().float().numpy()
        activations[f"{name}/input"] = inp_np
      if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        out_np = output[0].detach().cpu().float().numpy()
        activations[f"{name}/output"] = out_np
      elif isinstance(output, torch.Tensor):
        out_np = output.detach().cpu().float().numpy()
        activations[f"{name}/output"] = out_np
    return fn

  # Embeddings
  if hasattr(base_model, "embed_tokens"):
    hooks.append(base_model.embed_tokens.register_forward_hook(hook_fn("text/embed_tokens")))

  # Layers
  layers = getattr(base_model, "layers", [])
  for idx, layer in enumerate(layers):
    hooks.append(layer.register_forward_hook(hook_fn(f"layer_{idx:02d}")))
    if hasattr(layer, "input_layernorm"):
      hooks.append(layer.input_layernorm.register_forward_hook(hook_fn(f"layer_{idx:02d}/input_layernorm")))
    if hasattr(layer, "self_attn"):
      hooks.append(layer.self_attn.register_forward_hook(hook_fn(f"layer_{idx:02d}/self_attn")))
      if hasattr(layer.self_attn, "q_proj"):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_fn(f"layer_{idx:02d}/q_proj")))
      if hasattr(layer.self_attn, "k_proj"):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(hook_fn(f"layer_{idx:02d}/k_proj")))
      if hasattr(layer.self_attn, "v_proj"):
        hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_fn(f"layer_{idx:02d}/v_proj")))
      if hasattr(layer.self_attn, "o_proj"):
        hooks.append(layer.self_attn.o_proj.register_forward_hook(hook_fn(f"layer_{idx:02d}/o_proj")))
      if hasattr(layer.self_attn, "q_norm"):
        hooks.append(layer.self_attn.q_norm.register_forward_hook(hook_fn(f"layer_{idx:02d}/q_norm")))
      if hasattr(layer.self_attn, "k_norm"):
        hooks.append(layer.self_attn.k_norm.register_forward_hook(hook_fn(f"layer_{idx:02d}/k_norm")))
    if hasattr(layer, "post_attention_layernorm"):
      hooks.append(layer.post_attention_layernorm.register_forward_hook(hook_fn(f"layer_{idx:02d}/post_attention_layernorm")))
    if hasattr(layer, "mlp"):
      hooks.append(layer.mlp.register_forward_hook(hook_fn(f"layer_{idx:02d}/mlp")))
      if hasattr(layer.mlp, "gate_proj"):
        hooks.append(layer.mlp.gate_proj.register_forward_hook(hook_fn(f"layer_{idx:02d}/mlp_gate")))
      if hasattr(layer.mlp, "up_proj"):
        hooks.append(layer.mlp.up_proj.register_forward_hook(hook_fn(f"layer_{idx:02d}/mlp_up")))
      if hasattr(layer.mlp, "down_proj"):
        hooks.append(layer.mlp.down_proj.register_forward_hook(hook_fn(f"layer_{idx:02d}/mlp_down")))

  if hasattr(base_model, "norm"):
    hooks.append(base_model.norm.register_forward_hook(hook_fn("text/final_norm")))

  with torch.no_grad():
    out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(out, "last_hidden_state"):
      activations["text/last_hidden_state"] = out.last_hidden_state.detach().cpu().float().numpy()
    elif isinstance(out, torch.Tensor):
      activations["text/last_hidden_state"] = out.detach().cpu().float().numpy()

  for h in hooks:
    h.remove()

  out_path = os.path.join(output_dir, f"text_reference_{dtype_str}.safetensors")
  st_np.save_file(activations, out_path)
  print(f"✅ Saved {len(activations)} text tower activations ({dtype_str}) to: {out_path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", type=str, default=MODEL_ID)
  parser.add_argument("--output_dir", type=str, default="goldens/prxpixel")
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  dump_text_encoder_golden(args.model_dir, args.output_dir, dtype_str="fp32")
  dump_text_encoder_golden(args.model_dir, args.output_dir, dtype_str="bf16")


if __name__ == "__main__":
  main()
