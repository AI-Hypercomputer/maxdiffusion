import os
import sys

# Append MaxDiffusion path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
import jax.numpy as jnp
from flax import nnx
import time

from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import (
    LTX2VideoAutoencoderKL,
)

def main():
    print("Initializing MaxDiffusion LTX-2 VAE for JIT Benchmarking...")
    rngs = nnx.Rngs(0)
    
    model = LTX2VideoAutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=128,
        block_out_channels=(256, 512, 1024, 2048),
        decoder_block_out_channels=(256, 512, 1024),
        layers_per_block=(4, 6, 6, 2, 2),
        decoder_layers_per_block=(5, 5, 5, 5),
        rngs=rngs,
    )
    
    # 1 batch, 9 frames, 128x128 spatial
    # Random input to prevent any constant folding optimizations
    key = jax.random.PRNGKey(42)
    fake_input = jax.random.normal(key, (1, 9, 128, 128, 3))
    
    print("\n--- Benchmarking Encoder JIT Compilation ---")
    
    # First execution triggers PyTree flattening, Tracing, XLA Compilation, and finally execution
    start = time.time()
    latent_dist = model.encode(fake_input, return_dict=False)[0]
    latents = latent_dist.sample(key=key)
    # block_until_ready forces async JAX execution to finish so we can measure device time
    latents.block_until_ready()
    encode_compile_time = time.time() - start
    print(f"[{encode_compile_time.__format__('.4f')}s] 1st Encoder pass (Trace + Compile + Execute)")
    
    # Second execution skips python tracing & compiling - executes the cached HLO directly on device
    start = time.time()
    latent_dist = model.encode(fake_input, return_dict=False)[0]
    latents2 = latent_dist.sample(key=key)
    latents2.block_until_ready()
    encode_exec_time = time.time() - start
    print(f"[{encode_exec_time.__format__('.4f')}s] 2nd Encoder pass (Pure Execute)")
    
    print("\n--- Benchmarking Decoder JIT Compilation ---")
    
    # First decoder pass
    start = time.time()
    recon = model.decode(latents).sample
    recon.block_until_ready()
    decode_compile_time = time.time() - start
    print(f"[{decode_compile_time.__format__('.4f')}s] 1st Decoder pass (Trace + Compile + Execute)")
    
    # Second decoder pass
    start = time.time()
    recon2 = model.decode(latents).sample
    recon2.block_until_ready()
    decode_exec_time = time.time() - start
    print(f"[{decode_exec_time.__format__('.4f')}s] 2nd Decoder pass (Pure Execute)")
    
    print("\n=================")
    print(f"Encoder JIT Speedup: {encode_compile_time / encode_exec_time:.2f}x")
    print(f"Decoder JIT Speedup: {decode_compile_time / decode_exec_time:.2f}x")
    print("=================")

if __name__ == "__main__":
    main()
