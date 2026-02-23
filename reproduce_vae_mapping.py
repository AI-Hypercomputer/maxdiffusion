
import sys
import os
import torch
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from maxdiffusion.models.modeling_flax_pytorch_utils import rename_key, rename_key_and_reshape_tensor
from maxdiffusion.models.ltx2.ltx2_utils import _tuple_str_to_int

def test_vae_key(pt_key):
    print(f"\nProcessing Checkpoint Key: {pt_key}")
    
    # Logic copied/adapted from load_vae_weights in ltx2_utils.py
    renamed_pt_key = rename_key(pt_key)
    # print(f"After rename_key: {renamed_pt_key}")
    
    pt_tuple_key = tuple(renamed_pt_key.split("."))
    
    pt_list = []
    
    for i, part in enumerate(pt_tuple_key):
        if "_" in part and part.split("_")[-1].isdigit():
            name = "_".join(part.split("_")[:-1])
            idx = int(part.split("_")[-1])
            
            if name == "resnets":
                pt_list.append("resnets")
                pt_list.append(str(idx))
            elif name == "upsamplers":
                pt_list.append("upsampler")
            elif name in ["down_blocks", "up_blocks", "downsamplers"]:
                pt_list.append(name)
                pt_list.append(str(idx))
            else:
                pt_list.append(part)
        elif part == "upsampler":
            pt_list.append("upsampler") 
        elif part in ["conv1", "conv2", "conv"]:
            pt_list.append(part)
            # Logic from ltx2_utils.py
            if i + 1 < len(pt_tuple_key) and pt_tuple_key[i+1] == "conv":
                pass
            elif pt_list[-1] == "conv": 
                pass
            elif len(pt_list) >= 2 and pt_list[-2] == "conv":
                 pass
            elif part == "conv":
                pass
            else:
                pt_list.append("conv")
        else:
            pt_list.append(part)
    
    pt_tuple_key = tuple(pt_list)
    print(f"Constructed PT Tuple Key: {pt_tuple_key}")
    
    # Mock random_flax_state_dict for rename_key_and_reshape_tensor check
    # We pretend the target key exists
    # If pt_tuple_key ends in 'weight', we look for 'kernel'
    # If logic generates 'conv1.conv', we check compatibility
    
    mock_flax_key = list(pt_tuple_key)
    if mock_flax_key[-1] == "weight":
        mock_flax_key[-1] = "kernel"
    if mock_flax_key[-1] == "bias":
        pass 
        
    mock_flax_key_tuple = tuple(mock_flax_key)
    random_flax_state_dict = {mock_flax_key_tuple: 1} # Dummy Exists
    
    # dummy tensor
    import torch
    tensor = torch.zeros(1)
    
    flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict)
    flax_key = _tuple_str_to_int(flax_key)
    
    print(f"Final Flax Key: {flax_key}")

if __name__ == "__main__":
    # Test cases from inspection (WITH .conv.)
    test_keys = [
        "decoder.up_blocks.0.resnets.2.conv2.conv.weight",
        "decoder.mid_block.resnets.0.conv1.conv.weight",
    ]
    
    for k in test_keys:
        test_vae_key(k)
