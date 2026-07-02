import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding
import numpy as np
import time
import argparse
import os

# =====================================================================
# Setup: 8-Device Mesh
# =====================================================================
def get_mesh():
  devices = jax.devices()
  return Mesh(np.array(devices), ('x',))

# =====================================================================
# Helper: Sinusoidal Timestep Embedding
# =====================================================================
def timestep_embedding(timesteps, embedding_dim=256):
  half_dim = embedding_dim // 2
  exponent = jnp.exp(-jnp.log(10000) * jnp.arange(half_dim) / half_dim)
  args = timesteps[:, None] * exponent[None, :]
  embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
  return embedding

# =====================================================================
# Helper: Initialize Weights
# =====================================================================
def init_dit_params(key):
  keys = jax.random.split(key, 10)
  params = {
      "t_embed_w1": jax.random.normal(keys[0], (256, 2048)) * 0.02,
      "t_embed_w2": jax.random.normal(keys[1], (2048, 2048)) * 0.02,
      
      "ada_ln_w": jax.random.normal(keys[2], (2048, 6 * 2048)) * 0.02,
      
      "w_q": jax.random.normal(keys[3], (2048, 2048)) * 0.02,
      "w_k": jax.random.normal(keys[4], (2048, 2048)) * 0.02,
      "w_v": jax.random.normal(keys[5], (2048, 2048)) * 0.02,
      "w_out": jax.random.normal(keys[6], (2048, 2048)) * 0.02,
      
      "w_gate": jax.random.normal(keys[7], (2048, 8192)) * 0.02,
      "w_up": jax.random.normal(keys[8], (2048, 8192)) * 0.02,
      "w_down": jax.random.normal(keys[9], (8192, 2048)) * 0.02,
  }
  return params

# =====================================================================
# Shard Parameters
# =====================================================================
def shard_dit_params(params, mesh):
  t_sharding = NamedSharding(mesh, P(None, None))
  
  gate_sharding = NamedSharding(mesh, P('x', None)) 
  up_sharding = NamedSharding(mesh, P('x', None))   
  down_sharding = NamedSharding(mesh, P(None, 'x')) 
  
  sharded_params = {
      "t_embed_w1": jax.device_put(params["t_embed_w1"], t_sharding),
      "t_embed_w2": jax.device_put(params["t_embed_w2"], t_sharding),
      "ada_ln_w": jax.device_put(params["ada_ln_w"], t_sharding),
      
      "w_q": jax.device_put(params["w_q"], t_sharding),
      "w_k": jax.device_put(params["w_k"], t_sharding),
      "w_v": jax.device_put(params["w_v"], t_sharding),
      "w_out": jax.device_put(params["w_out"], t_sharding),
      
      "w_gate": jax.device_put(params["w_gate"], gate_sharding),
      "w_up": jax.device_put(params["w_up"], up_sharding),
      "w_down": jax.device_put(params["w_down"], down_sharding),
  }
  return sharded_params

# =====================================================================
# Attention Block
# =====================================================================
def local_flash_attention(q, k, v, num_heads=16, head_dim=128):
  batch_size, seq_len, _ = q.shape
  
  q_heads = q.reshape(batch_size, seq_len, num_heads, head_dim)
  k_heads = k.reshape(batch_size, seq_len, num_heads, head_dim)
  v_heads = v.reshape(batch_size, seq_len, num_heads, head_dim)
  
  q_t = jnp.transpose(q_heads, (0, 2, 1, 3))
  k_t = jnp.transpose(k_heads, (0, 2, 1, 3))
  v_t = jnp.transpose(v_heads, (0, 2, 1, 3))
  
  with jax.named_scope("local_flash_attention_core"):
    attn_out_t = jax.nn.dot_product_attention(q_t, k_t, v_t)
    
  attn_out = jnp.transpose(attn_out_t, (0, 2, 1, 3))
  return attn_out.reshape(batch_size, seq_len, -1)

# =====================================================================
# DiT Block (Single Layer)
# =====================================================================
def dit_block_ineefficient(x, t_cond, params, layer_idx):
  with jax.named_scope(f"dit_layer_{layer_idx}"):
    # 1. AdaLN Modulation Parameters
    with jax.named_scope("ada_ln_modulation"):
      modulation = jnp.dot(t_cond, params["ada_ln_w"])
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)
      
    # 2. LayerNorm + Attention
    with jax.named_scope("attn_path"):
      x_norm = (x - x.mean(axis=-1, keepdims=True)) / jnp.std(x, axis=-1, keepdims=True)
      x_modulated = x_norm * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
      
      q = jnp.dot(x_modulated, params["w_q"])
      k = jnp.dot(x_modulated, params["w_k"])
      v = jnp.dot(x_modulated, params["w_v"])
      
      attn_out = local_flash_attention(q, k, v)
      
      with jax.named_scope("attn_out_projection"):
        attn_out_t = jnp.transpose(attn_out, (0, 2, 1))
        out_projected_t = jnp.matmul(params["w_out"], attn_out_t)
        attn_out_proj = jnp.transpose(out_projected_t, (0, 2, 1))
      
      with jax.named_scope("residual_shape_alignment"):
        attn_out_padded = jnp.pad(attn_out_proj, ((0, 0), (0, 8), (0, 0))) 
        attn_out_sliced = attn_out_padded[:, :32768, :] 
      
      x = x + gate_msa[:, None, :] * attn_out_sliced
      
    # 3. LayerNorm + MLP (SwiGLU)
    with jax.named_scope("mlp_path"):
      x_norm = (x - x.mean(axis=-1, keepdims=True)) / jnp.std(x, axis=-1, keepdims=True)
      x_modulated = x_norm * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
      
      gate = jnp.dot(x_modulated, params["w_gate"])
      up = jnp.dot(x_modulated, params["w_up"])
      
      with jax.named_scope("swiglu_activation"):
        gate_fp32 = gate.astype(jnp.float32)
        gate_activated_fp32 = gate_fp32 * jax.nn.sigmoid(gate_fp32)
        gate_activated = gate_activated_fp32.astype(jnp.bfloat16)
        
      mlp_hidden = gate_activated * up
      
      mlp_out = jnp.dot(mlp_hidden, params["w_down"])
      x = x + gate_mlp[:, None, :] * mlp_out
      
  return x

# =====================================================================
# Deep DiT Model Step (Simulates a 24-Layer Model)
# =====================================================================
@jax.jit(static_argnums=(1,))
def model_step_ineefficient(x, t_scalar, params):
  t_emb = timestep_embedding(jnp.array([t_scalar]), embedding_dim=256)
  with jax.named_scope("t_embedding_projection"):
    t_hidden = jnp.dot(t_emb, params["t_embed_w1"])
    t_hidden = jax.nn.silu(t_hidden)
    t_cond = jnp.dot(t_hidden, params["t_embed_w2"])
  
  current_hidden = x
  for layer_idx in range(24):
    current_hidden = dit_block_ineefficient(current_hidden, t_cond, params, layer_idx)
    
  return current_hidden

# =====================================================================
# Sequential CFG Step
# =====================================================================
@jax.jit(static_argnums=(2,))
def sequential_cfg_step(x_cond, x_uncond, t_scalar, params):
  with jax.named_scope("sequential_cfg_step"):
    out_cond = model_step_ineefficient(x_cond, t_scalar, params)
    out_uncond = model_step_ineefficient(x_uncond, t_scalar, params)
    return out_cond + 0.5 * (out_cond - out_uncond)

# =====================================================================
# MAIN
# =====================================================================
def main():
  parser = argparse.ArgumentParser(description="TPU Inefficient ML Workload (24-Layer DiT)")
  parser.add_argument("--profile", action="store_true", help="Enable JAX XProf profiling")
  parser.add_argument("--profile_dir", type=str, default="/tmp/dit_workload_profile", help="Directory to save XProf traces")
  args = parser.parse_args()

  key = jax.random.PRNGKey(0)
  mesh = get_mesh()
  print(f"Initialized JAX mesh with {len(jax.devices())} devices.")
  
  # Initialize and shard parameters
  raw_params = init_dit_params(key)
  params = shard_dit_params(raw_params, mesh)
  
  x_sharding = NamedSharding(mesh, P(None, None, None))
  x_cond = jax.device_put(jax.random.normal(key, (1, 32768, 2048)), x_sharding)
  x_uncond = jax.device_put(jax.random.normal(key, (1, 32768, 2048)), x_sharding)
  
  print("=" * 60)
  print("RUNNING 24-LAYER INEFFICIENT DiT WORKLOAD (SeqLen = 32768, EmbedDim = 2048)")
  print("=" * 60)
  
  if args.profile:
    os.makedirs(args.profile_dir, exist_ok=True)
    print(f"Starting XProf trace in: {args.profile_dir}")
    jax.profiler.start_trace(args.profile_dir)

  start_time = time.time()
  
  current_x = x_cond
  for step in range(5):
    with jax.profiler.StepTraceAnnotation("dit-denoise-step", step_num=step):
      step_start = time.time()
      
      t = 1000.0 - (step * 200.0)
      current_x = sequential_cfg_step(current_x, x_uncond, t, params)
      
      if jnp.isnan(current_x).any():
        print("NaNs detected! Stopping.")
        break
        
      print(f"Step {step} | Timestep: {t:.1f} | Step Time: {time.time() - step_start:.4f}s")
    
  current_x.block_until_ready()
  
  if args.profile:
    jax.profiler.stop_trace()
    print("Stopped XProf trace.")

  total_inefficient_time = time.time() - start_time
  print(f"\nTotal Inefficient Workload Time: {total_inefficient_time:.4f}s\n")
  print("=" * 60)

if __name__ == "__main__":
  main()
