
import jax
import torch
import re
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
from maxdiffusion.models.ltx2.ltx2_utils import rename_for_ltx2_transformer, get_key_and_value
from maxdiffusion.models.modeling_flax_pytorch_utils import rename_key, rename_key_and_reshape_tensor

# Mock random_flax_state_dict (expected Flax keys)
random_flax_state_dict = {
    ('audio_caption_projection', 'linear_1', 'kernel'): "PLACEHOLDER",
    ('audio_caption_projection', 'linear_1', 'bias'): "PLACEHOLDER",
    ('transformer_blocks', 'audio_to_video_attn', 'norm_k', 'scale'): "PLACEHOLDER",
    ('transformer_blocks', 'scale_shift_table',): "PLACEHOLDER",
    ('transformer_blocks', '0', 'scale_shift_table'): "PLACEHOLDER", # If scanned, expected to be mapped here?
}

# Values for "random_flax_state_dict" are not used by rename logic EXCEPT for checks relative to it.
# We need to make sure we populate it enough for rename_key_and_reshape_tensor to work if it checks existence.

# Checkpoint keys to test
checkpoint_keys = [
    "audio_caption_projection.linear_1.weight",
    "audio_caption_projection.linear_1.bias",
    "transformer_blocks.0.audio_to_video_attn.norm_k.weight",
    "transformer_blocks.0.scale_shift_table", # Expected in checkpoint? Index JSON says "transformer_blocks.0.scale_shift_table"?
    # JSON has: "audio_scale_shift_table" (global), and maybe block ones?
    # Let's check a block key from JSON if possible, but we only have global ones in snippet.
    # We saw "transformer_blocks.0.scale_shift_table" in debug prints?
    # Actually debug prints showed: "transformer_blocks.0.scale_shift_table"
]

print("--- START DEBUG ---")

for pt_key in checkpoint_keys:
    print(f"\nProcessing Checkpoint Key: {pt_key}")
    
    # 1. rename_key
    renamed_pt_key = rename_key(pt_key)
    print(f"After rename_key: {renamed_pt_key}")
    
    # 2. rename_for_ltx2_transformer
    renamed_pt_key = rename_for_ltx2_transformer(renamed_pt_key)
    print(f"After rename_for_ltx2: {renamed_pt_key}")
    
    pt_tuple_key = tuple(renamed_pt_key.split("."))
    print(f"Tuple Key: {pt_tuple_key}")
    
    # 3. get_key_and_value
    # We need dummy tensor
    dummy_tensor = torch.zeros((10, 10))
    flax_state_dict = {} # Mock
    
    # Need to simulate scan_layers=True
    scan_layers = True
    num_layers = 48
    
    flax_key, flax_tensor = get_key_and_value(
        pt_tuple_key, dummy_tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers
    )
    
    print(f"Final Flax Key: {flax_key}")
    
    # Check if match
    if flax_key in random_flax_state_dict:
        print(">> MATCH FOUUND in random_flax_state_dict")
    else:
        print(">> MISSING in random_flax_state_dict")
        # Try finding partial match
        possible = [k for k in random_flax_state_dict if k[-1] == flax_key[-1]]
        if possible:
            print(f"   Did you mean: {possible}?")
