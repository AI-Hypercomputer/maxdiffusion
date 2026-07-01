# Copyright 2026 Google LLC
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

from absl import app
import numpy as np
import os

# Set HF_HOME immediately before any HF/MaxDiffusion imports to ensure
# the cache directory and token are correctly resolved.
if not os.environ.get("HF_HOME"):
    if os.path.exists("/mnt/data/hf_cache"):
        os.environ["HF_HOME"] = "/mnt/data/hf_cache"

import jax
from typing import List, Union
import jax.numpy as jnp
import flax
from flax.linen import partitioning as nn_partitioning

from maxdiffusion import pyconfig
import flax.linen as nn
from maxdiffusion.max_utils import create_device_mesh
from jax.sharding import Mesh

from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.models.vae_flax import FlaxAutoencoderKL
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model, load_and_convert_qwen3_weights

# -----------------------------------------------------------------------------
# FlowMatch Scheduler Helpers
# -----------------------------------------------------------------------------

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)

# -----------------------------------------------------------------------------
# Latent Packing & Unpacking Helpers
# -----------------------------------------------------------------------------

def unpack_latents(latents, batch_size, num_channels_latents, height, width):
    """
    Unpacks packed Flux latents of shape (batch_size, (height//16)*(width//16), channels*4)
    back to the unpacked shape (batch_size, channels, height//8, width//8).
    """
    h_latent = height // 8
    w_latent = width // 8
    
    # 1. Reshape to split spatial grid and packed channel blocks
    latents = np.reshape(latents, (batch_size, h_latent // 2, w_latent // 2, num_channels_latents, 2, 2))
    # 2. Permute dimensions back to unpacked order
    latents = np.transpose(latents, (0, 3, 1, 4, 2, 5))
    # 3. Reshape to merge 2x2 blocks back into spatial height and width
    latents = np.reshape(latents, (batch_size, num_channels_latents, h_latent, w_latent))
    return latents

def prepare_latent_image_ids(batch_size, height, width):
    """Generates 4D position coordinates (T, H, W, L) for latent tensors."""
    grid = jnp.zeros((height, width, 4), dtype=jnp.int32)
    grid = grid.at[..., 1].set(jnp.arange(height)[:, None])
    grid = grid.at[..., 2].set(jnp.arange(width)[None, :])
    latent_ids = grid.reshape(-1, 4)
    latent_ids = jnp.expand_dims(latent_ids, axis=0)
    latent_ids = jnp.repeat(latent_ids, batch_size, axis=0)
    return latent_ids

def prepare_text_ids(batch_size, seq_len):
    """Generates 4D position coordinates (0, 0, 0, l) for text sequence tokens, matching PyTorch."""
    coords = jnp.zeros((seq_len, 4), dtype=jnp.int32)
    coords = coords.at[:, 3].set(jnp.arange(seq_len))
    coords = jnp.expand_dims(coords, axis=0)
    coords = jnp.repeat(coords, batch_size, axis=0)
    return coords

def patchify_latents(latents):
    """Groups 2x2 spatial patches into channels: [B, C, H, W] -> [B, C*4, H/2, W/2]"""
    batch_size, num_channels, height, width = latents.shape
    x = jnp.reshape(latents, (batch_size, num_channels, height // 2, 2, width // 2, 2))
    x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))
    x = jnp.reshape(x, (batch_size, num_channels * 4, height // 2, width // 2))
    return x

def pack_latents(latents):
    """[B, C, H, W] -> [B, H*W, C] after patchifying"""
    patchified = patchify_latents(latents)
    batch_size, num_channels, height, width = patchified.shape
    x = jnp.reshape(patchified, (batch_size, num_channels, height * width))
    x = jnp.transpose(x, (0, 2, 1))
    return x

def unpack_latents_with_ids(x, x_ids, height, width):
    """[B, H*W, C] -> [B, C, H, W] using coordinate IDs."""
    batch_size, seq_len, ch = x.shape
    x_list = []
    for b in range(batch_size):
        data = x[b]
        pos = x_ids[b]
        h_ids = pos[:, 1].astype(jnp.int32)
        w_ids = pos[:, 2].astype(jnp.int32)
        flat_ids = h_ids * width + w_ids
        out = jnp.zeros((height * width, ch), dtype=x.dtype)
        out = out.at[flat_ids].set(data)
        out = jnp.transpose(jnp.reshape(out, (height, width, ch)), (2, 0, 1))
        x_list.append(out)
    return jnp.stack(x_list, axis=0)

def unpatchify_latents(latents):
    """Reverses the 2x2 spatial patch grouping: [B, C, H, W] -> [B, C/4, H*2, W*2]"""
    batch_size, num_channels_latents, height, width = latents.shape
    x = jnp.reshape(latents, (batch_size, num_channels_latents // 4, 2, 2, height, width))
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
    x = jnp.reshape(x, (batch_size, num_channels_latents // 4, height * 2, width * 2))
    return x

def load_or_generate_latents(config):
    """
    Loads saved latents if use_latents is True, otherwise generates random latents.
    """
    if isinstance(config, dict):
        from types import SimpleNamespace
        config = SimpleNamespace(**config)
        
    batch_size = config.batch_size
    height = config.height
    width = config.width
    use_latents = config.use_latents

    # Flux latents typically have 32 channels and are downsampled by 8
    num_channels_latents = 32
    latent_height = height // 8
    latent_width = width // 8
    latent_shape = (batch_size, num_channels_latents, latent_height, latent_width)

    if use_latents:
        print("use_latents is True. Loading latents from disk...")
        bundle_path = "src/maxdiffusion/tests/flux2_klein_complete_diagnostic_bundle.npz"
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Expected to find {bundle_path} but it was not found.")
        
        bundle = np.load(bundle_path)
        # Look for initial latents under two possible key names
        if "initial_pipeline_latents" in bundle:
            packed_latents = bundle["initial_pipeline_latents"]
        elif "step_0_cond_transformer_input_latents" in bundle:
            packed_latents = bundle["step_0_cond_transformer_input_latents"]
        else:
            raise KeyError(f"Neither 'initial_pipeline_latents' nor 'step_0_cond_transformer_input_latents' was found in {bundle_path}")

        print(f"Successfully loaded initial latents with shape: {packed_latents.shape}")
        
        # Unpack the latents to match the expected unpacked shape
        latents = unpack_latents(packed_latents, batch_size, num_channels_latents, height, width)
        print(f"Successfully unpacked latents to shape: {latents.shape}")
        
        # Ensure shape matches what we expect
        if latents.shape != latent_shape:
            print(f"Warning: Unpacked latent shape {latents.shape} does not match expected shape {latent_shape}.")
    else:
        print(f"use_latents is False. Generating random gaussian noise with shape: {latent_shape}...")
        # Fix seed for reproducibility in testing
        np.random.seed(42)  
        latents = np.random.randn(*latent_shape).astype(np.float32)

    return latents

# -----------------------------------------------------------------------------
# Qwen3 Prompt Encoder Helpers
# -----------------------------------------------------------------------------

_tokenizer = None
_text_encoder = None

def get_qwen3_models(repo_id="black-forest-labs/FLUX.2-klein-4B"):
    """
    Lazily loads the tokenizer and text encoder from the cached repo path.
    """
    global _tokenizer, _text_encoder
    if _tokenizer is None:
        import os
        import torch
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
        
        # Resolve absolute local path from HF cache if it exists to bypass buggy from_pretrained resolving
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        cache_dir = os.path.join(hf_home, "hub", f"models--{repo_id.replace('/', '--')}", "snapshots")
        if os.path.exists(cache_dir):
            snapshots = os.listdir(cache_dir)
            if snapshots:
                snapshot_dir = os.path.join(cache_dir, snapshots[0])
                print(f"Detected local cache directory: {snapshot_dir}")
                repo_id = snapshot_dir

        print(f"Loading Qwen3 models from repo path: {repo_id}...")
        try:
            _tokenizer = Qwen2TokenizerFast.from_pretrained(repo_id, local_files_only=True)
        except Exception:
            # Fallback if tokenizer is in a subfolder
            _tokenizer = Qwen2TokenizerFast.from_pretrained(repo_id, subfolder="tokenizer", local_files_only=True)
        
        _text_encoder = Qwen3ForCausalLM.from_pretrained(
            repo_id,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
            local_files_only=True
        )
        # Keep text encoder on CPU to conserve TPU/GPU memory
        _text_encoder.to("cpu")
        print("Successfully loaded Qwen3 models!")
    return _tokenizer, _text_encoder

def encode_prompt(
    prompt: Union[str, List[str]],
    repo_id: str = "black-forest-labs/FLUX.2-klein-4B",
    max_sequence_length: int = 512,
):
    """
    Encodes the prompt(s) using Qwen3 and extracts concatenated hidden states
    from layers 9, 18, and 27.
    Returns a numpy array of shape (batch, 512, 7680).
    """
    import torch
    tokenizer, text_encoder = get_qwen3_models(repo_id)
    
    # Standardize to list of strings
    prompts = [prompt] if isinstance(prompt, str) else prompt
    
    print(f"Encoding {len(prompts)} prompt(s) in batch...")
    templated_texts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        templated_texts.append(text)
        
    inputs = tokenizer(
        templated_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        
    # Extract layers 9, 18, 27
    hidden_states_layers = (9, 18, 27)
    # Stack layers: shape (batch, 3, 512, 2560)
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    
    # Reshape to concatenate layer features: (batch, 512, 3 * 2560) = (batch, 512, 7680)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
    
    embeds_np = prompt_embeds.cpu().numpy()
    print(f"Generated prompt embeddings shape: {embeds_np.shape}")
    return embeds_np

def encode_prompt_jax(
    prompt: Union[str, List[str]],
    qwen3_params,
    jitted_qwen3_fn,
    repo_id: str = "black-forest-labs/FLUX.2-klein-4B",
    max_sequence_length: int = 512,
):
    """
    Encodes the prompt(s) using the JAX Qwen3 model and extracts concatenated
    hidden states from layers 8, 17, and 26 (indices 9, 18, 27).
    Returns a JAX array of shape (batch, 512, 7680).
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import Qwen2TokenizerFast
        print(f"Loading Qwen3 tokenizer from: {repo_id}...")
        try:
            _tokenizer = Qwen2TokenizerFast.from_pretrained(repo_id, local_files_only=True)
        except Exception:
            _tokenizer = Qwen2TokenizerFast.from_pretrained(repo_id, subfolder="tokenizer", local_files_only=True)

    # Standardize to list of strings
    prompts = [prompt] if isinstance(prompt, str) else prompt
    
    templated_texts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        templated_texts.append(text)
        
    inputs = _tokenizer(
        templated_texts,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    
    input_ids = jnp.array(inputs["input_ids"])
    attention_mask = jnp.array(inputs["attention_mask"])
    
    hidden_states, all_hidden_states = jitted_qwen3_fn(qwen3_params, input_ids, attention_mask)
    
    # Extract layers 8, 17, 26 (indices 9, 18, 27 in all_hidden_states)
    h_9 = all_hidden_states[9]
    h_18 = all_hidden_states[18]
    h_27 = all_hidden_states[27]
    
    # Stack along channels: shape (batch, 3, 512, 2560)
    out = jnp.stack([h_9, h_18, h_27], axis=1)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len, num_channels * hidden_dim))
    
    return prompt_embeds

# -----------------------------------------------------------------------------
# Weight Mapping & Conversion
# -----------------------------------------------------------------------------

def load_and_convert_weights(safetensors_path, params, num_double_layers=8, num_single_layers=24):
    """
    Loads PyTorch weights from safetensors (supporting shards) and converts them to JAX parameter dictionary.
    """
    from safetensors.torch import load_file
    import torch
    import glob
    import os
    
    pt_state_dict = {}
    if os.path.isdir(safetensors_path):
        shards = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        print(f"Loading sharded PyTorch weights from directory: {safetensors_path} (Found {len(shards)} shards)...")
        for shard in sorted(shards):
            print(f"Loading shard: {shard}...")
            pt_state_dict.update(load_file(shard, device="cpu"))
    else:
        print(f"Loading PyTorch weights from: {safetensors_path}")
        pt_state_dict = load_file(safetensors_path, device="cpu")
        
    print("Mapping PyTorch weights to JAX parameters...")
    
    # Global layers
    params["txt_in"]["kernel"] = jnp.array(pt_state_dict["context_embedder.weight"].to(torch.float32).cpu().numpy().T)
    params["img_in"]["kernel"] = jnp.array(pt_state_dict["x_embedder.weight"].to(torch.float32).cpu().numpy().T)
    params["double_stream_modulation_img"]["kernel"] = jnp.array(pt_state_dict["double_stream_modulation_img.linear.weight"].to(torch.float32).cpu().numpy().T)
    params["double_stream_modulation_txt"]["kernel"] = jnp.array(pt_state_dict["double_stream_modulation_txt.linear.weight"].to(torch.float32).cpu().numpy().T)
    params["single_stream_modulation"]["kernel"] = jnp.array(pt_state_dict["single_stream_modulation.linear.weight"].to(torch.float32).cpu().numpy().T)
    params["proj_out"]["kernel"] = jnp.array(pt_state_dict["proj_out.weight"].to(torch.float32).cpu().numpy().T)
    
    # norm_out
    params["norm_out"]["Dense_0"]["kernel"] = jnp.array(pt_state_dict["norm_out.linear.weight"].to(torch.float32).cpu().numpy().T)
    
    # time_text_embed (Timestep Embedding)
    params["time_text_embed"]["FlaxTimestepEmbedding_0"]["linear_1"]["kernel"] = jnp.array(pt_state_dict["time_guidance_embed.timestep_embedder.linear_1.weight"].to(torch.float32).cpu().numpy().T)
    params["time_text_embed"]["FlaxTimestepEmbedding_0"]["linear_2"]["kernel"] = jnp.array(pt_state_dict["time_guidance_embed.timestep_embedder.linear_2.weight"].to(torch.float32).cpu().numpy().T)

    # Double Blocks
    print(f"Mapping {num_double_layers} double-stream attention blocks...")
    for block_idx in range(num_double_layers):
        jax_db = params[f"double_blocks_{block_idx}"]
        prefix = f"transformer_blocks.{block_idx}."
        
        # Concatenate QKV projections
        to_q = pt_state_dict[prefix + "attn.to_q.weight"].to(torch.float32).T.cpu().numpy()
        to_k = pt_state_dict[prefix + "attn.to_k.weight"].to(torch.float32).T.cpu().numpy()
        to_v = pt_state_dict[prefix + "attn.to_v.weight"].to(torch.float32).T.cpu().numpy()
        jax_db["attn"]["i_qkv"]["kernel"] = jnp.array(np.concatenate([to_q, to_k, to_v], axis=1))

        add_q = pt_state_dict[prefix + "attn.add_q_proj.weight"].to(torch.float32).T.cpu().numpy()
        add_k = pt_state_dict[prefix + "attn.add_k_proj.weight"].to(torch.float32).T.cpu().numpy()
        add_v = pt_state_dict[prefix + "attn.add_v_proj.weight"].to(torch.float32).T.cpu().numpy()
        jax_db["attn"]["e_qkv"]["kernel"] = jnp.array(np.concatenate([add_q, add_k, add_v], axis=1))

        # Projections out
        jax_db["attn"]["i_proj"]["kernel"] = jnp.array(pt_state_dict[prefix + "attn.to_out.0.weight"].to(torch.float32).T.cpu().numpy())
        jax_db["attn"]["e_proj"]["kernel"] = jnp.array(pt_state_dict[prefix + "attn.to_add_out.weight"].to(torch.float32).T.cpu().numpy())

        # Norm scales
        jax_db["attn"]["query_norm"]["scale"] = jnp.array(pt_state_dict[prefix + "attn.norm_q.weight"].to(torch.float32).cpu().numpy())
        jax_db["attn"]["key_norm"]["scale"] = jnp.array(pt_state_dict[prefix + "attn.norm_k.weight"].to(torch.float32).cpu().numpy())
        jax_db["attn"]["encoder_query_norm"]["scale"] = jnp.array(pt_state_dict[prefix + "attn.norm_added_q.weight"].to(torch.float32).cpu().numpy())
        jax_db["attn"]["encoder_key_norm"]["scale"] = jnp.array(pt_state_dict[prefix + "attn.norm_added_k.weight"].to(torch.float32).cpu().numpy())

        # SwiGLU MLPs
        jax_db["img_mlp"]["linear_in"]["kernel"] = jnp.array(pt_state_dict[prefix + "ff.linear_in.weight"].to(torch.float32).T.cpu().numpy())
        jax_db["img_mlp"]["linear_out"]["kernel"] = jnp.array(pt_state_dict[prefix + "ff.linear_out.weight"].to(torch.float32).T.cpu().numpy())
        jax_db["txt_mlp"]["linear_in"]["kernel"] = jnp.array(pt_state_dict[prefix + "ff_context.linear_in.weight"].to(torch.float32).T.cpu().numpy())
        jax_db["txt_mlp"]["linear_out"]["kernel"] = jnp.array(pt_state_dict[prefix + "ff_context.linear_out.weight"].to(torch.float32).T.cpu().numpy())

    # Single Blocks
    print(f"Mapping {num_single_layers} single-stream attention blocks...")
    for block_idx in range(num_single_layers):
        jax_sb = params[f"single_blocks_{block_idx}"]
        s_prefix = f"single_transformer_blocks.{block_idx}."

        # Joint projections
        jax_sb["linear1"]["kernel"] = jnp.array(pt_state_dict[s_prefix + "attn.to_qkv_mlp_proj.weight"].to(torch.float32).T.cpu().numpy())
        jax_sb["linear2"]["kernel"] = jnp.array(pt_state_dict[s_prefix + "attn.to_out.weight"].to(torch.float32).T.cpu().numpy())

        # Norm scales
        jax_sb["attn"]["query_norm"]["scale"] = jnp.array(pt_state_dict[s_prefix + "attn.norm_q.weight"].to(torch.float32).cpu().numpy())
        jax_sb["attn"]["key_norm"]["scale"] = jnp.array(pt_state_dict[s_prefix + "attn.norm_k.weight"].to(torch.float32).cpu().numpy())

    print("Weight conversion complete!")
    return params

def load_and_convert_vae_weights(safetensors_path, jax_params):
    """Loads PyTorch VAE weights from safetensors, maps them to JAX, and extracts BN stats."""
    from safetensors.torch import load_file
    import torch
    import numpy as np
    import flax
    
    print(f"Loading PyTorch VAE weights from: {safetensors_path}")
    pt_state_dict = load_file(safetensors_path)
    
    # Helper to safely convert PyTorch bfloat16 tensors to numpy float32
    def get_w(key):
        return pt_state_dict[key].to(torch.float32).cpu().numpy()
        
    # Unfreeze JAX params so we can load the weights
    jax_params = flax.core.unfreeze(jax_params)
    
    # Map weights (identical to our unit test!)
    print("Mapping VAE decoder weights to JAX parameters...")
    
    # post_quant_conv
    jax_params["post_quant_conv"]["kernel"] = jnp.array(get_w("post_quant_conv.weight").transpose(2, 3, 1, 0))
    jax_params["post_quant_conv"]["bias"] = jnp.array(get_w("post_quant_conv.bias"))
    
    # decoder.conv_in
    jax_params["decoder"]["conv_in"]["kernel"] = jnp.array(get_w("decoder.conv_in.weight").transpose(2, 3, 1, 0))
    jax_params["decoder"]["conv_in"]["bias"] = jnp.array(get_w("decoder.conv_in.bias"))
    
    # decoder.mid_block
    # resnets
    for idx in [0, 1]:
        res_jax = jax_params["decoder"]["mid_block"][f"resnets_{idx}"]
        res_pt_prefix = f"decoder.mid_block.resnets.{idx}"
        
        res_jax["norm1"]["scale"] = jnp.array(get_w(f"{res_pt_prefix}.norm1.weight"))
        res_jax["norm1"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.norm1.bias"))
        res_jax["conv1"]["kernel"] = jnp.array(get_w(f"{res_pt_prefix}.conv1.weight").transpose(2, 3, 1, 0))
        res_jax["conv1"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.conv1.bias"))
        
        res_jax["norm2"]["scale"] = jnp.array(get_w(f"{res_pt_prefix}.norm2.weight"))
        res_jax["norm2"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.norm2.bias"))
        res_jax["conv2"]["kernel"] = jnp.array(get_w(f"{res_pt_prefix}.conv2.weight").transpose(2, 3, 1, 0))
        res_jax["conv2"]["bias"] = jnp.array(get_w(f"{res_pt_prefix}.conv2.bias"))
        
    # attentions
    attn_pt_prefix = "decoder.mid_block.attentions.0"
    attn_jax = jax_params["decoder"]["mid_block"]["attentions_0"]
    
    attn_jax["group_norm"]["scale"] = jnp.array(get_w(f"{attn_pt_prefix}.group_norm.weight"))
    attn_jax["group_norm"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.group_norm.bias"))
    
    attn_jax["query"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_q.weight").T)
    attn_jax["query"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_q.bias"))
    attn_jax["key"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_k.weight").T)
    attn_jax["key"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_k.bias"))
    attn_jax["value"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_v.weight").T)
    attn_jax["value"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_v.bias"))
    
    attn_jax["proj_attn"]["kernel"] = jnp.array(get_w(f"{attn_pt_prefix}.to_out.0.weight").T)
    attn_jax["proj_attn"]["bias"] = jnp.array(get_w(f"{attn_pt_prefix}.to_out.0.bias"))
    
    # decoder.up_blocks
    for b_idx in range(4):
        up_block_jax = jax_params["decoder"][f"up_blocks_{b_idx}"]
        up_block_pt = f"decoder.up_blocks.{b_idx}"
        
        for r_idx in range(3):
            res_jax = up_block_jax[f"resnets_{r_idx}"]
            res_pt = f"{up_block_pt}.resnets.{r_idx}"
            
            res_jax["norm1"]["scale"] = jnp.array(get_w(f"{res_pt}.norm1.weight"))
            res_jax["norm1"]["bias"] = jnp.array(get_w(f"{res_pt}.norm1.bias"))
            res_jax["conv1"]["kernel"] = jnp.array(get_w(f"{res_pt}.conv1.weight").transpose(2, 3, 1, 0))
            res_jax["conv1"]["bias"] = jnp.array(get_w(f"{res_pt}.conv1.bias"))
            
            res_jax["norm2"]["scale"] = jnp.array(get_w(f"{res_pt}.norm2.weight"))
            res_jax["norm2"]["bias"] = jnp.array(get_w(f"{res_pt}.norm2.bias"))
            res_jax["conv2"]["kernel"] = jnp.array(get_w(f"{res_pt}.conv2.weight").transpose(2, 3, 1, 0))
            res_jax["conv2"]["bias"] = jnp.array(get_w(f"{res_pt}.conv2.bias"))
            
            shortcut_key = f"{res_pt}.conv_shortcut.weight"
            if shortcut_key in pt_state_dict:
                res_jax["conv_shortcut"]["kernel"] = jnp.array(get_w(shortcut_key).transpose(2, 3, 1, 0))
                res_jax["conv_shortcut"]["bias"] = jnp.array(get_w(f"{res_pt}.conv_shortcut.bias"))
                
        if b_idx < 3:
            upsampler_jax = up_block_jax["upsamplers_0"]
            upsampler_pt = f"{up_block_pt}.upsamplers.0"
            
            upsampler_jax["conv"]["kernel"] = jnp.array(get_w(f"{upsampler_pt}.conv.weight").transpose(2, 3, 1, 0))
            upsampler_jax["conv"]["bias"] = jnp.array(get_w(f"{upsampler_pt}.conv.bias"))
            
    # decoder.conv_norm_out & conv_out
    jax_params["decoder"]["conv_norm_out"]["scale"] = jnp.array(get_w("decoder.conv_norm_out.weight"))
    jax_params["decoder"]["conv_norm_out"]["bias"] = jnp.array(get_w("decoder.conv_norm_out.bias"))
    jax_params["decoder"]["conv_out"]["kernel"] = jnp.array(get_w("decoder.conv_out.weight").transpose(2, 3, 1, 0))
    jax_params["decoder"]["conv_out"]["bias"] = jnp.array(get_w("decoder.conv_out.bias"))
    
    # Freeze parameters
    jax_params = flax.core.freeze(jax_params)
    
    # Extract Batch Normalization running stats
    print("Extracting VAE Batch Normalization running stats...")
    bn_mean = jnp.array(get_w("bn.running_mean")).reshape(1, -1, 1, 1)
    bn_var = jnp.array(get_w("bn.running_var")).reshape(1, -1, 1, 1)
    batch_norm_eps = 0.0001
    bn_std = jnp.sqrt(bn_var + batch_norm_eps)
    
    print("VAE weights and BN stats loaded successfully!")
    return jax_params, bn_mean, bn_std

# -----------------------------------------------------------------------------
# Prompt Partitioning Helper
# -----------------------------------------------------------------------------

def partition_prompts(prompt_str: str, batch_size: int) -> List[str]:
    """
    Splits a prompt string by '||' and replicates/truncates them to fill the batch_size.
    """
    raw_prompts = [p.strip() for p in prompt_str.split("||") if p.strip()]
    if not raw_prompts:
        raw_prompts = ["A detailed vector illustration of a robotic hummingbird"]
        
    num_prompts = len(raw_prompts)
    
    if num_prompts == 1:
        active_prompts = raw_prompts * batch_size
    elif num_prompts <= batch_size:
        reps = batch_size // num_prompts
        active_prompts = []
        for p in raw_prompts:
            active_prompts.extend([p] * reps)
        if len(active_prompts) < batch_size:
            active_prompts.extend([raw_prompts[-1]] * (batch_size - len(active_prompts)))
    else:
        print(f"⚠️ Warning: Found {num_prompts} prompts in config, but batch_size is {batch_size}. Truncating to the first {batch_size} prompts.")
        active_prompts = raw_prompts[:batch_size]
        
    return active_prompts

# -----------------------------------------------------------------------------
# Parameter In-place Casting Helper
# -----------------------------------------------------------------------------

def cast_dict_to_bfloat16_inplace(d):
    """Casts a nested dictionary of JAX/numpy arrays to bfloat16 in-place, freeing memory immediately."""
    import gc
    for k, v in list(d.items()):
        if isinstance(v, dict):
            cast_dict_to_bfloat16_inplace(v)
        elif hasattr(v, "astype"):
            # Force conversion to JAX array on the active default device (CPU during init)
            d[k] = jnp.array(v, dtype=jnp.bfloat16)
            if hasattr(d[k], "block_until_ready"):
                d[k].block_until_ready()
            del v
            gc.collect()

# -----------------------------------------------------------------------------
# Main Generation Entry Point
# -----------------------------------------------------------------------------

def main(argv):
    # Use default matmul precision to activate native TPU hardware matrix units (bfloat16)
    # (Only uncomment and set to 'highest' when debugging strict numerical parity with PyTorch CPU)
    import jax
    # jax.config.update("jax_default_matmul_precision", "highest")
    
    # 1. Load configurations
    if getattr(pyconfig, "config", None) is None:
        # If running as standalone script, initialize config from default base_flux2klein.yml
        config_path = "src/maxdiffusion/configs/base_flux2klein.yml"
        
        # Robustly separate custom config path from key=value overrides in argv
        custom_overrides = []
        if len(argv) > 1:
            if argv[1].endswith(".yml") or argv[1].endswith(".yaml"):
                config_path = argv[1]
                if len(argv) > 2:
                    custom_overrides = argv[2:]
            else:
                custom_overrides = argv[1:]
                
        print(f"Initializing pyconfig with base config: {config_path}")
        default_args = [
            None,
            config_path,
            "run_name=flux2klein_generation",
            "output_dir=output/",
            "jax_cache_dir=/tmp/cache_dir",
        ]
        default_args.extend(custom_overrides)
            
        # Dynamically force use_latents=False in interactive mode to avoid shape conflicts
        is_interactive = False
        for arg in default_args:
            if arg and "interactive=True" in arg.replace(" ", ""):
                is_interactive = True
                
        if is_interactive:
            print("ℹ️ Interactive mode detected: overriding use_latents=False to support dynamic generation and batching.")
            default_args.append("use_latents=False")
            
        pyconfig.initialize(default_args)
    config = pyconfig.config
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 2. Setup device mesh
    print("Setting up JAX device mesh...")
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    
    # 3. Load or generate initial latents
    # Unpacked shape: (batch_size, 32, height//8, width//8)
    latents_unpacked = load_or_generate_latents(config)
    batch_size = config.batch_size
    height = config.height
    width = config.width
    
    # Pack latents: [B, 32, 64, 64] -> [B, 1024, 128]
    print(f"Packing latents from unpacked shape {latents_unpacked.shape}...")
    latents_packed = pack_latents(latents_unpacked)
    print(f"Packed latents shape: {latents_packed.shape}")
    
    # 5. Locate cached PyTorch weights (with automatic download if missing)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            
    repo_id = config.pretrained_model_name_or_path
    cache_dir = os.path.join(hf_home, "hub", f"models--{repo_id.replace('/', '--')}", "snapshots")
    
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print(f"\n📢 Model cache not found at {cache_dir}.")
        print(f"🚀 Downloading '{repo_id}' from Hugging Face Hub (this may take a few minutes)...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_files_only=False)
        
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Hugging Face cache directory still not found after download: {cache_dir}")
        
    snapshots = os.listdir(cache_dir)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in Hugging Face cache directory: {cache_dir}")
    snapshot_dir = os.path.join(cache_dir, snapshots[0])
    safetensors_path = os.path.join(snapshot_dir, "transformer")
    vae_safetensors_path = os.path.join(snapshot_dir, "vae", "diffusion_pytorch_model.safetensors")

    # Load Qwen3 configuration early to get the correct hidden size
    from transformers import AutoConfig
    text_encoder_path = os.path.join(snapshot_dir, "text_encoder")
    print(f"Loading Qwen3 config from text_encoder path: {text_encoder_path}...")
    pt_config = AutoConfig.from_pretrained(text_encoder_path, local_files_only=True)
    
    # 6. Instantiate JAX FluxTransformer2DModel
    print("Instantiating JAX FluxTransformer2DModel for Flux.2-klein-9B...")
    transformer = FluxTransformer2DModel(
        in_channels=128,
        num_layers=config.num_double_layers,
        num_single_layers=config.depth,
        attention_head_dim=128,
        num_attention_heads=config.num_attention_heads,
        joint_attention_dim=3 * pt_config.hidden_size,  # concatenated 3 layers
        pooled_projection_dim=768,   # CFG pooled dim
        mlp_ratio=3.0,
        qkv_bias=False,
        joint_attention_bias=False,
        x_embedder_bias=False,
        proj_out_bias=False,
        use_global_modulation=True,  # Global modulation enabled
        use_swiglu=True,             # SwiGLU enabled
        axes_dims_rope=(32, 32, 32, 32), # 4D RoPE
        theta=2000,                  # Theta = 2000
        mesh=mesh,
        dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
        weights_dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
        attention_kernel=config.attention,
        scale_shift_order=getattr(config, "scale_shift_order", "shift_scale"),
    )
    
    # 6b. Instantiate JAX FlaxAutoencoderKL VAE
    print("Instantiating JAX FlaxAutoencoderKL VAE...")
    vae = FlaxAutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=32,
        norm_num_groups=32,
        sample_size=512,
        use_quant_conv=True,
        use_post_quant_conv=True,
        dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
    )
    
    # 7. Evaluate shapes and extract TPU shardings using jax.eval_shape
    print("Evaluating shapes and extracting TPU shardings...")
    
    # Determine sequence lengths based on resolution
    h_packed = height // 16
    w_packed = width // 16
    seq_len_img = h_packed * w_packed
    seq_len_txt = config.max_sequence_length

    # Define dummy inputs once for reuse
    img_dummy = jnp.zeros((batch_size, seq_len_img, 128))
    img_ids_dummy = jnp.zeros((batch_size, seq_len_img, 4))
    txt_dummy = jnp.zeros((batch_size, seq_len_txt, 3 * pt_config.hidden_size))
    txt_ids_dummy = jnp.zeros((batch_size, seq_len_txt, 4))
    vec_dummy = jnp.zeros((batch_size, 768))
    t_vec_dummy = jnp.zeros((batch_size,))
    guidance_vec_dummy = jnp.zeros((batch_size,))
    dummy_img = jnp.zeros((batch_size, 3, 512, 512)) # for VAE
    
    # Initialize JAX Qwen3 Model (config already loaded)
    qwen3_config = FlaxQwen3Config(
        vocab_size=pt_config.vocab_size,
        hidden_size=pt_config.hidden_size,
        intermediate_size=pt_config.intermediate_size,
        num_hidden_layers=pt_config.num_hidden_layers,
        num_attention_heads=pt_config.num_attention_heads,
        num_key_value_heads=pt_config.num_key_value_heads,
        max_position_embeddings=pt_config.max_position_embeddings,
        rms_norm_eps=pt_config.rms_norm_eps,
        rope_theta=pt_config.rope_theta,
        dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
    )
    qwen3_model = FlaxQwen3Model(qwen3_config)

    # Dummy inputs for Qwen3 init
    dummy_ids = jnp.zeros((batch_size, seq_len_txt), dtype=jnp.int32)
    dummy_mask = jnp.zeros((batch_size, seq_len_txt), dtype=jnp.int32)

    key = jax.random.PRNGKey(0)
    key, vae_key, qwen_key = jax.random.split(key, 3)

    def transformer_init_fn():
        return transformer.init(
            key,
            hidden_states=img_dummy,
            img_ids=img_ids_dummy,
            encoder_hidden_states=txt_dummy,
            txt_ids=txt_ids_dummy,
            pooled_projections=vec_dummy,
            timestep=t_vec_dummy,
            guidance=guidance_vec_dummy,
        )
    def vae_init_fn():
        return vae.init(vae_key, dummy_img)
    def qwen3_init_fn():
        return qwen3_model.init(qwen_key, dummy_ids, dummy_mask)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        abstract_transformer_vars = jax.eval_shape(transformer_init_fn)
        abstract_vae_vars = jax.eval_shape(vae_init_fn)
        abstract_qwen3_vars = jax.eval_shape(qwen3_init_fn)
        
        logical_transformer_specs = nn.get_partition_spec(abstract_transformer_vars)
        logical_vae_specs = nn.get_partition_spec(abstract_vae_vars)
        logical_qwen3_specs = nn.get_partition_spec(abstract_qwen3_vars)
        
        transformer_mesh_shardings = nn.logical_to_mesh_sharding(logical_transformer_specs, mesh, config.logical_axis_rules)
        vae_mesh_shardings = nn.logical_to_mesh_sharding(logical_vae_specs, mesh, config.logical_axis_rules)
        qwen3_mesh_shardings = nn.logical_to_mesh_sharding(logical_qwen3_specs, mesh, config.logical_axis_rules)
        
    transformer_shardings = flax.core.freeze(transformer_mesh_shardings['params'])
    vae_shardings = flax.core.freeze(vae_mesh_shardings['params'])
    qwen3_shardings = flax.core.freeze(qwen3_mesh_shardings['params'])

    # 8. Initialize JAX parameters on CPU
    print("Initializing JAX parameters on CPU...")
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            # Initialize Transformer
            variables = transformer.init(
                key,
                hidden_states=img_dummy,
                img_ids=img_ids_dummy,
                encoder_hidden_states=txt_dummy,
                txt_ids=txt_ids_dummy,
                pooled_projections=vec_dummy,
                timestep=t_vec_dummy,
                guidance=guidance_vec_dummy,
            )
            params = variables["params"]

            # Initialize VAE
            vae_variables = vae.init(vae_key, dummy_img)
            vae_params = vae_variables["params"]

            # Initialize Qwen3 parameters
            print("Initializing JAX Qwen3 parameters...")
            qwen3_variables = qwen3_model.init(qwen_key, dummy_ids, dummy_mask)
            qwen3_params = qwen3_variables["params"]

            # Unbox LogicallyPartitioned parameters for all
            import flax.linen.spmd as flax_spmd
            params = jax.tree_util.tree_map(
                lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
                params,
                is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
            )
            params = flax.core.unfreeze(params)

            vae_params = jax.tree_util.tree_map(
                lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
                vae_params,
                is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
            )
            vae_params = flax.core.unfreeze(vae_params)

            qwen3_params = jax.tree_util.tree_map(
                lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
                qwen3_params,
                is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
            )
            qwen3_params = flax.core.unfreeze(qwen3_params)

            # Convert and load weights for all
            print("Loading weights for Flux, VAE, and Qwen3...")
            params = load_and_convert_weights(safetensors_path, params, num_double_layers=config.num_double_layers, num_single_layers=config.depth)
            vae_params, vae_bn_mean, vae_bn_std = load_and_convert_vae_weights(vae_safetensors_path, vae_params)
            qwen3_params = load_and_convert_qwen3_weights(text_encoder_path, qwen3_params, qwen3_config)

            # In-place parameter casting to prevent TPU HBM OOM
            if config.weights_dtype == "bfloat16":
                print("Casting JAX parameters and BN stats to bfloat16 in-place...")
                cast_dict_to_bfloat16_inplace(params)
                cast_dict_to_bfloat16_inplace(vae_params)
                cast_dict_to_bfloat16_inplace(qwen3_params)
                vae_bn_mean = vae_bn_mean.astype(jnp.bfloat16)
                vae_bn_std = vae_bn_std.astype(jnp.bfloat16)

            params = flax.core.freeze(params)
            vae_params = flax.core.freeze(vae_params)
            qwen3_params = flax.core.freeze(qwen3_params)



            # Dynamic Offloading Auto-Detection
            device = jax.devices()[0]
            device_kind = device.device_kind.lower()
            default_offload = "v6e" in device_kind or "v5e" in device_kind or "v4" in device_kind or "lite" in device_kind or "v6" in device_kind
            dynamic_offload = getattr(config, "dynamic_offload", default_offload)

            if dynamic_offload:
                print("\n" + "="*80)
                print("🚀 DYNAMIC PARAMETER OFFLOADING ENABLED! Swapping parameters to Host CPU...")
                print("="*80 + "\n")
                cpu_device = jax.devices("cpu")[0]

                # Move param dictionaries to Host CPU memory
                params = jax.device_put(params, cpu_device)
                vae_params = jax.device_put(vae_params, cpu_device)
                qwen3_params = jax.device_put(qwen3_params, cpu_device)

                import gc
                gc.collect()
                # Synchronize device to ensure HBM is freed
                jax.effects_barrier()
            else:
                print("\n" + "="*80)
                print("🚀 Dynamic parameter offloading disabled. Moving all parameters to TPU HBM permanently...")
                print("="*80 + "\n")
                params = jax.device_put(params, transformer_shardings)
                vae_params = jax.device_put(vae_params, vae_shardings)
                qwen3_params = jax.device_put(qwen3_params, qwen3_shardings)
                
                import gc
                gc.collect()
                jax.effects_barrier()

    # 8. Set up Flow Match Scheduler
    from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
    
    print("Setting up FlowMatch Scheduler...")
    num_inference_steps = 4
    mu = compute_empirical_mu(seq_len_img, num_inference_steps)
    print(f"Computed empirical mu (shift) for image_seq_len={seq_len_img}: {mu}")
    
    jax_scheduler = FlaxFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=mu,
        sigma_max=1.0,
        sigma_min=0.001,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
    )
    scheduler_state = jax_scheduler.create_state()
    # Explicitly pass linear sigmas matching the PyTorch pipeline to align timesteps!
    explicit_sigmas = jnp.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler_state = jax_scheduler.set_timesteps_ltx2(
        state=scheduler_state,
        num_inference_steps=num_inference_steps,
        shift=mu,
        sigmas=explicit_sigmas,
    )
    
    # 9. Prepare Coordinate IDs for RoPE
    txt_ids_val = prepare_text_ids(batch_size, seq_len_txt)
    img_ids_val = prepare_latent_image_ids(batch_size, h_packed, w_packed)
    
    # Define JIT-compiled TPU step functions passing parameters explicitly
    @jax.jit
    def jitted_transformer_step(transformer_params, latents, img_ids, prompt_embeds, txt_ids, vec, timestep, guidance):
        return transformer.apply(
            {"params": transformer_params},
            hidden_states=latents,
            img_ids=img_ids,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            pooled_projections=vec,
            timestep=timestep,
            guidance=guidance,
        )

    @jax.jit
    def jitted_vae_decode(v_params, latents_unpatched):
        return vae.apply(
            {"params": v_params},
            latents=latents_unpatched,
            method=vae.decode,
        )

    @jax.jit
    def jitted_qwen3(q_params, ids, mask):
        return qwen3_model.apply(
            {"params": q_params},
            input_ids=ids,
            attention_mask=mask,
        )
        
    # Define a reusable generation function
    def run_generation(current_prompts: List[str], output_name: str, measure_time: bool = False):
        tpu_device = jax.devices("tpu")[0]
        
        t_embed = 0.0
        t_denoise = 0.0
        t_vae = 0.0
        
        # 4. Encode prompt using JAX Qwen3
        print(f"\n[PHASE A] Encoding {len(current_prompts)} prompt(s) using JAX Qwen3 on TPU...")
        if measure_time:
            import time
            jax.effects_barrier()
            t0 = time.time()
            
        if dynamic_offload:
            print("  Moving Qwen3 parameters to TPU HBM...")
            q_params_tpu = jax.device_put(qwen3_params, qwen3_shardings)
        else:
            q_params_tpu = qwen3_params
            
        # Run JAX Qwen3 forward pass
        prompt_embeds_jax = encode_prompt_jax(
            current_prompts,
            qwen3_params=q_params_tpu,
            jitted_qwen3_fn=jitted_qwen3,
            repo_id=snapshot_dir,
            max_sequence_length=config.max_sequence_length,
        )
        
        # Force completion to push activations to device and block
        prompt_embeds_jax.block_until_ready()
        
        if measure_time:
            t_embed = time.time() - t0
            print(f" -> [TIMING] Prompt Encoding (Qwen3): {t_embed:.4f} seconds ⏱️")
            
        if dynamic_offload:
            print("  Releasing Qwen3 parameters from TPU HBM...")
            del q_params_tpu
            import gc
            gc.collect()
            jax.effects_barrier()
            
        # 10. Run E2E Denoising Loop (4 steps)
        print(f"\n[PHASE B] Running 4-step E2E Denoising Loop on a batch of {batch_size} images...")
        if measure_time:
            jax.effects_barrier()
            t0 = time.time()
            
        if dynamic_offload:
            print("  Moving Flux Transformer parameters to TPU HBM...")
            t_params_tpu = jax.device_put(params, transformer_shardings)
        else:
            t_params_tpu = params
            
        latents = jnp.array(latents_packed)
        
        # Reset scheduler state for a fresh run
        nonlocal scheduler_state
        scheduler_state = jax_scheduler.create_state()
        scheduler_state = jax_scheduler.set_timesteps_ltx2(
            state=scheduler_state,
            num_inference_steps=num_inference_steps,
            shift=mu,
            sigmas=explicit_sigmas,
        )
        
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            guidance_vec_val = jnp.array([4.0] * batch_size)
            vec_val = jnp.zeros((batch_size, 768))
            
            for step_idx in range(num_inference_steps):
                step_t = jnp.array([scheduler_state.timesteps[step_idx]])
                print(f" -> Step {step_idx}: Timestep = {step_t[0]:.4f}, Sigma = {scheduler_state.sigmas[step_idx]:.4f}")
                
                model_output = jitted_transformer_step(
                    t_params_tpu, # Pass parameter dict!
                    latents,
                    img_ids_val,
                    prompt_embeds_jax,
                    txt_ids_val,
                    vec_val,
                    step_t,
                    guidance_vec_val,
                )
                
                step_output = jax_scheduler.step(
                    state=scheduler_state,
                    model_output=model_output.sample,
                    timestep=step_t[0],
                    sample=latents,
                )
                latents = step_output.prev_sample
                scheduler_state = step_output.state
                
        # Force completion to block denoising loop timer
        latents.block_until_ready()
        
        if measure_time:
            t_denoise = time.time() - t0
            print(f" -> [TIMING] Denoising Loop (Flux): {t_denoise:.4f} seconds ⏱️")
            
        if dynamic_offload:
            print("  Releasing Flux Transformer parameters from TPU HBM...")
            del t_params_tpu
            import gc
            gc.collect()
            jax.effects_barrier()
            
        # 11. Unpack, Apply VAE Batch Normalization, and Unpatchify Final Latents
        print("\n[POST-PROCESS] Unpacking and postprocessing final denoised latents...")
        latents_unpacked = unpack_latents_with_ids(latents, img_ids_val, h_packed, w_packed)
        latents_bn = latents_unpacked * vae_bn_std + vae_bn_mean
        final_latents_unpatched = unpatchify_latents(latents_bn)
        
        # 12. Decode Latents to RGB Image via JAX VAE Decoder
        print("\n[PHASE C] Decoding final latents to RGB image using JAX VAE decoder on TPU...")
        if measure_time:
            jax.effects_barrier()
            t0 = time.time()
            
        if dynamic_offload:
            print("  Moving VAE parameters to TPU HBM...")
            v_params_tpu = jax.device_put(vae_params, vae_shardings)
        else:
            v_params_tpu = vae_params
            
        with mesh:
            jax_image_out = jitted_vae_decode(v_params_tpu, final_latents_unpatched)
            # Force JAX to complete VAE decoding before stopping the timer
            jax_image_out.sample.block_until_ready()
            
        if measure_time:
            t_vae = time.time() - t0
            print(f" -> [TIMING] VAE Decoding: {t_vae:.4f} seconds ⏱️")
            
        if dynamic_offload:
            print("  Releasing VAE parameters from TPU HBM...")
            del v_params_tpu
            import gc
            gc.collect()
            jax.effects_barrier()
            
        if measure_time:
            total_time = t_embed + t_denoise + t_vae
            print(f"\n" + "="*55)
            print(f" ⏱️ END-TO-END TIMING BREAKDOWN (After JIT Warmup):")
            print(f"="*55)
            print(f"  * Prompt Encoding (Qwen3):   {t_embed:.4f}s")
            print(f"  * Denoising Loop (Flux):     {t_denoise:.4f}s")
            print(f"  * VAE Decoding:             {t_vae:.4f}s")
            print(f"  -----------------------------------------------------")
            print(f"  * TOTAL SUMMED LATENCY:      {total_time:.4f} seconds ⚡")
            print(f"=======================================================\n")
            
        # 13. Postprocess and Save Images in Batch
        print("Postprocessing and saving generated images...")
        image = (jax_image_out.sample / 2.0 + 0.5)
        image = jnp.clip(image, 0.0, 1.0)
        image = jnp.transpose(image, (0, 2, 3, 1)) # NHWC
        
        from PIL import Image
        for b_idx in range(batch_size):
            image_np = np.array(image[b_idx] * 255.0, dtype=np.uint8)
            img = Image.fromarray(image_np)
            
            # Formulate output filename for this batch index
            if batch_size > 1:
                batch_output_name = output_name.replace(".png", f"_b{b_idx}.png")
            else:
                batch_output_name = output_name
                
            output_png_path = os.path.join(config.output_dir, batch_output_name)
            os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
            img.save(output_png_path)
            
            # Print success log for each image with its corresponding prompt
            p_text = current_prompts[b_idx] if b_idx < len(current_prompts) else "N/A"
            print(f" -> Saved image: {os.path.abspath(output_png_path)} | Prompt: '{p_text}'")
            
        # Also save raw latents
        output_npy_path = os.path.join(config.output_dir, output_name.replace(".png", "_latents.npy"))
        np.save(output_npy_path, np.array(final_latents_unpatched))
        
        print(f"\n=======================================================")
        print(f"SUCCESS! Batched generation complete for {batch_size} images! 🎨🎉")
        print(f"Saved raw denoised latents to: {os.path.abspath(output_npy_path)}")
        print(f"=======================================================")

    # 14. Execution Phase: One-shot or Interactive Loop
    if getattr(config, "interactive", False):
        print("\n" + "="*80)
        print("   BATCHED INTERACTIVE GENERATION MODE ENABLED 🎮")
        print("The model has been fully loaded and JAX-compiled on the TPU.")
        print(f"Batch size is locked at: {batch_size} parallel images.")
        print("You can enter a single prompt, or multiple prompts separated by '||'!")
        print("Example: A cute kitten || A roaring lion || A racing sports car")
        print("Type 'exit' or 'quit' to end the session.")
        print("="*80)
        
        image_idx = 1
        while True:
            try:
                user_input = input("\nEnter prompt(s): ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode...")
                break
                
            if user_input.strip().lower() in ('exit', 'quit'):
                print("Exiting interactive mode...")
                break
                
            if not user_input.strip():
                continue
                
            active_prompts = partition_prompts(user_input, batch_size)
                
            output_file = f"generated_{image_idx:03d}.png"
            run_generation(active_prompts, output_file)
            image_idx += 1
    else:
        # Run one-shot generation supporting multiple prompts separated by '||'
        active_prompts = partition_prompts(config.prompt, batch_size)
        
        # Pass 1: Warmup pass to compile XLA graphs
        print("\n" + "="*80)
        print("🚀 Running initial dry run (Warmup Pass) to compile XLA graphs...")
        print("="*80)
        run_generation(active_prompts, "flux2klein_warmup.png")
        
        # Pass 2: Timed pass
        print("\n" + "="*80)
        print("⏱️ Running timed pass at full TPU speed...")
        print("="*80)
        run_generation(active_prompts, "flux2klein_generated_image.png", measure_time=True)

if __name__ == "__main__":
    app.run(main)
