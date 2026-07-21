import os
import sys
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np
from transformers import AutoConfig, Qwen2TokenizerFast

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Dynamically resolve maxdiffusion/src directory relative to this test file
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model, load_and_convert_qwen3_weights

def main():
  print("=" * 80)
  print(" QWEN3 TEXT EMBEDDING DIAGNOSTIC CHECK ")
  print("=" * 80)
  print(f"JAX Devices: {jax.devices()}")

  prompt = "anime corgi eating sushi in the mountains"
  seq_len_txt = 512

  repo_id = "black-forest-labs/FLUX.2-klein-4B"
  hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
  cache_dir = os.path.join(hf_home, "hub", f"models--{repo_id.replace('/', '--')}", "snapshots")

  if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
    print(f"Model cache not found at {cache_dir}. Resolving via huggingface_hub...")
    from huggingface_hub import snapshot_download
    snapshot_dir = snapshot_download(repo_id=repo_id)
  else:
    snapshots = [s for s in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, s))]
    snapshot_dir = os.path.join(cache_dir, snapshots[0])

  text_encoder_path = os.path.join(snapshot_dir, "text_encoder")

  print(f"Loading Qwen3 config from: {text_encoder_path}")
  pt_config = AutoConfig.from_pretrained(text_encoder_path, local_files_only=True)
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
      dtype=jnp.float32,
  )

  try:
    tokenizer = Qwen2TokenizerFast.from_pretrained(snapshot_dir, local_files_only=True)
  except Exception:
    tokenizer = Qwen2TokenizerFast.from_pretrained(snapshot_dir, subfolder="tokenizer", local_files_only=True)

  templated_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
  inputs = tokenizer([templated_text], return_tensors="np", padding="max_length", truncation=True, max_length=seq_len_txt)
  
  prompt_ids = jnp.array(inputs["input_ids"])
  prompt_mask = jnp.array(inputs["attention_mask"])

  print("\n1. TOKEN IDS CHECK:")
  print(f"   prompt_ids[0][:15] = {prompt_ids[0][:15].tolist()}")
  print(f"   prompt_ids sum = {prompt_ids.sum()}")

  print("\n2. MODEL INITIALIZATION & WEIGHT CONVERSION:")
  qwen3_model = FlaxQwen3Model(qwen3_config)
  key = jax.random.PRNGKey(0)
  init_vars = qwen3_model.init(key, prompt_ids, prompt_mask)
  qwen3_params = init_vars["params"]
  qwen3_params = load_and_convert_qwen3_weights(text_encoder_path, qwen3_params, qwen3_config)
  qwen3_params = jax.tree_util.tree_map(lambda leaf: leaf.astype(jnp.float32), qwen3_params)

  print("\n3. FORWARD PASS EXECUTION:")
  @jax.jit
  def forward_fn(params, ids, mask):
    return qwen3_model.apply({"params": params}, input_ids=ids, attention_mask=mask)

  hidden_states, all_hidden_states = forward_fn(qwen3_params, prompt_ids, prompt_mask)

  l9 = all_hidden_states[9]
  l18 = all_hidden_states[18]
  l27 = all_hidden_states[27]

  print(f"   Layer 9  -> min: {float(l9.min()):.4f}, max: {float(l9.max()):.4f}, mean: {float(l9.mean()):.6f}, sum: {float(l9.sum()):.4f}")
  print(f"   Layer 18 -> min: {float(l18.min()):.4f}, max: {float(l18.max()):.4f}, mean: {float(l18.mean()):.6f}, sum: {float(l18.sum()):.4f}")
  print(f"   Layer 27 -> min: {float(l27.min()):.4f}, max: {float(l27.max()):.4f}, mean: {float(l27.mean()):.6f}, sum: {float(l27.sum()):.4f}")

  prompt_embeds = jnp.concatenate([l9, l18, l27], axis=-1)
  print(f"\n4. FINAL STACKED PROMPT EMBEDDINGS (3072 dims):")
  print(f"   min: {float(prompt_embeds.min()):.4f}")
  print(f"   max: {float(prompt_embeds.max()):.4f}")
  print(f"   mean: {float(prompt_embeds.mean()):.6f}")
  print(f"   sum: {float(prompt_embeds.sum()):.4f}")
  print("=" * 80)

if __name__ == "__main__":
  main()
