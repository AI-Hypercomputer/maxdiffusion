
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel

def debug_eval_shapes():
    rngs = nnx.Rngs(0)
    transformer = LTX2VideoTransformer3DModel(
        rngs=rngs,
        in_channels=128,
        out_channels=128,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=4, # Small for speed
        attention_head_dim=32,
        cross_attention_dim=64,
        audio_dim=32,
        audio_num_attention_heads=4,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=32,
        num_layers=2, # Small for speed
        scan_layers=True
    )
    
    state = nnx.state(transformer)
    eval_shapes = state.to_pure_dict()
    
    from flax.traverse_util import flatten_dict
    flat_shapes = flatten_dict(eval_shapes)
    
    print("--- EVAL SHAPES DEBUG ---")
    keys = sorted(list(flat_shapes.keys()))
    
    for k in keys:
        k_str = str(k)
        if "norm_out" in k_str:
            print(f"NORM_OUT: {k}")
        if "audio_caption_projection" in k_str:
            print(f"AUDIO_CAP_PROJ: {k}")
        if "scale_shift_table" in k_str:
            print(f"SCALE_SHIFT: {k}")
        if "transformer_blocks" in k_str and "audio_to_video_attn" in k_str and "norm_k" in k_str:
             print(f"BLOCK_KEY: {k}")

if __name__ == "__main__":
    debug_eval_shapes()
