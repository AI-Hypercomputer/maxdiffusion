import time
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache

def run_benchmark():
    mesh = Mesh(jax.devices(), ('vae_spatial',))
    
    # 1080p latent shape for WAN: (B, T, H, W, C)
    # T=1 for 1 frame. 1080p spatial latents are roughly 135x240
    batch, t, h, w, c = 1, 1, 135, 240, 16 
    dummy_latent = jnp.ones((batch, t, h, w, c), dtype=jnp.float32)

    with mesh:
        # Init dummy variables
        rng = jax.random.PRNGKey(0)
        
        # Initialize the actual AutoencoderKLWan
        vae = AutoencoderKLWan(
            rngs=nnx.Rngs(0),
            mesh=mesh,
            vae_decode_chunk=1,
            dtype=jnp.float32,
            weights_dtype=jnp.float32,
        )
        
        # We need a cache for decoding
        cache = AutoencoderKLWanCache(vae)

        @nnx.jit
        def decode_step(model, z, cache_):
            return model.decode(z, cache_)

        print("Compiling model...")
        out = decode_step(vae, dummy_latent, cache)
        jax.block_until_ready(out)

        print("Running Benchmark...")
        start = time.perf_counter()
        
        iters = 5
        for _ in range(iters):
            out = decode_step(vae, dummy_latent, cache)
        jax.block_until_ready(out)
        
        end = time.perf_counter()
        
    print(f"Average Decode Time: {(end - start) / iters * 1000:.2f} ms")

if __name__ == "__main__":
    run_benchmark()
