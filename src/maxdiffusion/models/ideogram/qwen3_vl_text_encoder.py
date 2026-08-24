"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Qwen3-VL prompt encoder for Ideogram 4.

This is **eager PyTorch on CPU**, not torchax/torch_xla2 -- the file was
previously named `torchax_text_encoder.py`, which it never was. It runs once per
image, outside the denoise loop, but it is still seconds of CPU time on an ~8B
model. Porting it to JAX (or at minimum running it under torchax on TPU) is the
obvious next optimization; see `Qwen3VLTextEncoder.__call__` for the
jax -> numpy -> torch -> numpy -> jax round trip that would also disappear.
"""

import jax


class Qwen3VLTextEncoder:
  """Eager-PyTorch (CPU) Qwen3-VL encoder producing Ideogram's LLM conditioning."""

  def __init__(self, model):
    self.model = model

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: str = "text_encoder"):
    from transformers import AutoModel, AutoConfig
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json
    import os
    import torch
    from .quantized_loading import swap_linears_to_fp8, load_fp8_state_dict

    kwargs = {"trust_remote_code": True}
    if subfolder:
      kwargs["subfolder"] = subfolder

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    if getattr(config, "text_config", None) is not None:
      if getattr(config.text_config, "rope_scaling", None) is None:
        config.text_config.rope_scaling = {}

    # Instantiate from config (random weights, but creates all non-persistent buffers)
    model = AutoModel.from_config(config, trust_remote_code=True)

    # Download and merge sharded state dict
    index_filename = f"{subfolder}/model.safetensors.index.json" if subfolder else "model.safetensors.index.json"
    index_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=index_filename)
    with open(index_path) as f:
      index = json.load(f)

    shard_dir = os.path.dirname(index_filename)
    shard_filenames = sorted(set(index["weight_map"].values()))

    state_dict = {}
    for shard in shard_filenames:
      shard_repo_path = os.path.join(shard_dir, shard) if shard_dir else shard
      shard_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=shard_repo_path)
      state_dict.update(load_file(shard_path))

    # Swap to Fp8Linear for quantized weights, and load state dict.
    compute_dtype = torch.bfloat16
    swap_linears_to_fp8(model, state_dict, compute_dtype=compute_dtype)
    # We don't move the entire model to bfloat16 explicitly before loading,
    # because load_fp8_state_dict(assign=True) takes care of floating tensors.
    load_fp8_state_dict(model, state_dict, device=torch.device("cpu"), dtype=compute_dtype, assign=True, strict=False)
    model.eval()
    return cls(model)

  def __call__(
      self,
      input_ids: jax.Array,
      attention_mask: jax.Array,
      pos_2d: jax.Array,
  ) -> jax.Array:
    import torch
    import numpy as np

    # Run natively in PyTorch
    device = next(self.model.parameters()).device
    pt_input_ids = torch.from_numpy(np.array(input_ids)).to(device)
    pt_attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
    pt_pos_2d = torch.from_numpy(np.array(pos_2d)).to(device)

    with torch.no_grad():
      output = self._forward_inner(
          self.model,
          input_ids=pt_input_ids,
          attention_mask=pt_attention_mask,
          pos_2d=pt_pos_2d,
      )

    return jax.numpy.array(output.to(torch.float32).cpu().numpy())

  @staticmethod
  def _forward_inner(model, input_ids, attention_mask, pos_2d):
    from transformers.masking_utils import create_causal_mask
    import torch

    language_model = model.language_model
    inputs_embeds = language_model.embed_tokens(input_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    import inspect

    sig = inspect.signature(create_causal_mask)
    mask_kwargs = {
        "config": language_model.config,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    if "input_embeds" in sig.parameters:
      mask_kwargs["input_embeds"] = inputs_embeds
    else:
      mask_kwargs["inputs_embeds"] = inputs_embeds
    if "cache_position" in sig.parameters:
      mask_kwargs["cache_position"] = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

    causal_mask = create_causal_mask(**mask_kwargs)
    if language_model.rotary_emb.inv_freq.device.type != "jax":
      language_model.rotary_emb.inv_freq = language_model.rotary_emb.inv_freq.to(inputs_embeds.device)
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    from .constants import QWEN3_VL_ACTIVATION_LAYERS

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
      layer_out = decoder_layer(
          hidden_states,
          attention_mask=causal_mask,
          position_ids=text_position_ids,
          past_key_values=None,
          position_embeddings=position_embeddings,
      )
      hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
      if layer_idx in tap_set:
        captured[layer_idx] = hidden_states

    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

    # Interleave features per token by stacking and permuting
    batch_size, seq_len, _ = selected[0].shape
    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))
    stacked = stacked.reshape(batch_size, seq_len, -1)

    return stacked
