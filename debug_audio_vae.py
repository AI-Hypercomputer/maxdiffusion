
import jax
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from maxdiffusion.models.ltx2.audio_vae import FlaxAutoencoderKLLTX2Audio
from maxdiffusion.models.ltx2.ltx2_utils import load_audio_vae_weights, rename_for_ltx2_audio_vae, load_sharded_checkpoint
from flax import nnx

def debug_keys():
    print("Initializing Model...")
    config = {
      "base_channels": 128,
      "ch_mult": (1, 2, 4),
      "double_z": True,
      "dropout": 0.0,
      "in_channels": 2,
      "latent_channels": 8,
      "mel_bins": 64,
      "mel_hop_length": 160,
      "mid_block_add_attention": False,
      "norm_type": "pixel",
      "num_res_blocks": 2,
      "output_channels": 2,
      "resolution": 256,
      "sample_rate": 16000,
      "rngs": nnx.Rngs(0)
    }
    
    with jax.default_device(jax.devices("cpu")[0]):
        model = FlaxAutoencoderKLLTX2Audio(**config)
        
    state = nnx.state(model)
    eval_shapes = state.to_pure_dict()
    
    # Print some expected Flax keys
    print("\nSample Flax Keys (Expected):")
    
    def flatten(d, parent_key=()):
        items = []
        for k, v in d.items():
            new_key = parent_key + (k,)
            if isinstance(v, dict):
                items.extend(flatten(v, new_key))
            else:
                items.append(new_key)
        return items

    flax_keys = flatten(eval_shapes)
    for k in flax_keys[:20]:
        print(k)
        
    print("\nTotal Flax Keys:", len(flax_keys))

    # Load PyTorch keys
    print("\nLoading PyTorch SafeTensors Keys...")
    pretrained_model_name_or_path = "Lightricks/LTX-2"
    subfolder = "audio_vae"
    
    tensors = load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, "cpu")
    pt_keys = list(tensors.keys())
    
    print("\nDecoder Up Block Keys:")
    up_keys = [k for k in pt_keys if "decoder.up" in k]
    for k in up_keys:
        print(k)
        
    print("\nSample Encoder Keys:")
    enc_keys = [k for k in pt_keys if "encoder" in k]
    for k in enc_keys[:20]:
        print(k)
        
    # Check specific encoder key shape if possible? 
    # Can't easily check shape here without loading tensor, but load_sharded_checkpoint loads all.
    # tensors is already loaded.
    
    print("\nChecking Encoder Down Block 0 shape:")
    if "encoder.down.0.block.0.conv1.conv.weight" in tensors:
        print("encoder.down.0.block.0.conv1.conv.weight:", tensors["encoder.down.0.block.0.conv1.conv.weight"].shape)
    if "encoder.down.0.block.1.conv1.conv.weight" in tensors:
        print("encoder.down.0.block.1.conv1.conv.weight:", tensors["encoder.down.0.block.1.conv1.conv.weight"].shape)
    if "encoder.down.1.block.0.conv1.conv.weight" in tensors:
         print("encoder.down.1.block.0.conv1.conv.weight:", tensors["encoder.down.1.block.0.conv1.conv.weight"].shape)

    print("\nChecking Decoder Up Block 0 shape:")
    if "decoder.up.0.block.0.conv1.conv.weight" in tensors:
        print("decoder.up.0.block.0.conv1.conv.weight:", tensors["decoder.up.0.block.0.conv1.conv.weight"].shape)

        
    print("\nTesting Renaming Logic...")
    renamed_keys = []
    for k in pt_keys:
        renamed = rename_for_ltx2_audio_vae(k)
        renamed_keys.append(renamed)
        if "mid_block.resnets.0.conv1.weight" in k:
             print(f"Renaming check: {k} -> {renamed}")
             
    # Check for misaligned expected keys
    # specific missing ones
    targets = [
        ('decoder', 'mid_block1', 'conv1', 'conv', 'bias'),
        ('decoder', 'mid_block1', 'conv1', 'conv', 'kernel'),
    ]
    
    print("\nSearching for targets in RENAMED keys:")
    for t in targets:
        t_str = ".".join([str(x) for x in t])
        found = False
        for rk in renamed_keys:
             # We need to simulate the structure mapping logic too?
             # rename_for_ltx2_audio_vae only does string replacement, 
             # load_audio_vae_weights does structural mapping (mid_block -> mid_block1)
             pass
             
    # Let's verify specific renaming for mid_block1
    # PyTorch: decoder.mid_block.resnets.0.conv1.weight
    # My rename: decoder.mid_block.resnets.0.conv1.conv.kernel
    # My logic in load_audio_vae_weights:
    #   if "mid_block.resnets.0" in k: replace with mid_block1
    #   -> decoder.mid_block1.conv1.conv.kernel
    # Flax expected: ('decoder', 'mid_block1', 'conv1', 'conv', 'kernel')
    
    # Is it possible that 'mid_block.resnets.0' is NOT in the key?
    # Maybe it's 'mid_block.resnets.0.conv1.weight'? Yes.
    
    # We will print all RENAMED and STRUCTURED keys produced by our logic
    print("\nGenerating final Flax keys from PyTorch keys using current logic...")
    final_keys = set()
    
    for pt_key in pt_keys:
        key = rename_for_ltx2_audio_vae(pt_key)
        
        # Determine conversion to tuple (Same logic as in ltx2_utils.py)
        parts = key.split(".")
        flax_key_parts = []
        for part in parts:
            if part.isdigit():
                flax_key_parts.append(int(part))
            else:
                flax_key_parts.append(part)
        flax_key = tuple(flax_key_parts)
        
        if "mid_block" in pt_key:
            if "mid_block.resnets.0" in pt_key:
                flax_key_str = ".".join([str(x) for x in flax_key])
                flax_key_str = flax_key_str.replace("mid_block.resnets.0", "mid_block1")
            elif "mid_block.resnets.1" in pt_key:
                flax_key_str = ".".join([str(x) for x in flax_key])
                flax_key_str = flax_key_str.replace("mid_block.resnets.1", "mid_block2")
            elif "mid_block.attentions.0" in pt_key:
                flax_key_str = ".".join([str(x) for x in flax_key])
                flax_key_str = flax_key_str.replace("mid_block.attentions.0", "mid_attn")
            else:
                flax_key_str = ".".join([str(x) for x in flax_key])
            
            parts = flax_key_str.split(".")
            flax_key_parts = []
            for part in parts:
                if part.isdigit():
                    flax_key_parts.append(int(part))
                else:
                    flax_key_parts.append(part)
            flax_key = tuple(flax_key_parts)

        if "down_blocks" in key:
             key_str = ".".join([str(x) for x in flax_key])
             if "resnets" in key_str:
                 key_str = key_str.replace("down_blocks", "down_stages")
                 key_str = key_str.replace("resnets", "blocks")
             elif "attentions" in key_str:
                 key_str = key_str.replace("down_blocks", "down_stages")
                 key_str = key_str.replace("attentions", "attns")
             elif "downsamplers" in key_str:
                 key_str = key_str.replace("down_blocks", "down_stages")
                 key_str = key_str.replace("downsamplers.0", "downsample")
             
             parts = key_str.split(".")
             flax_key_parts = []
             for part in parts:
                if part.isdigit():
                    flax_key_parts.append(int(part))
                else:
                    flax_key_parts.append(part)
             flax_key = tuple(flax_key_parts)

        if "up_blocks" in key:
             key_str = ".".join([str(x) for x in flax_key])
             if "resnets" in key_str:
                 key_str = key_str.replace("up_blocks", "up_stages")
                 key_str = key_str.replace("resnets", "blocks")
             elif "attentions" in key_str:
                 key_str = key_str.replace("up_blocks", "up_stages")
                 key_str = key_str.replace("attentions", "attns")
             elif "upsamplers" in key_str:
                 key_str = key_str.replace("up_blocks", "up_stages")
                 key_str = key_str.replace("upsamplers.0", "upsample")
                 
             parts = key_str.split(".")
             flax_key_parts = []
             for part in parts:
                if part.isdigit():
                    flax_key_parts.append(int(part))
                else:
                    flax_key_parts.append(part)
             flax_key = tuple(flax_key_parts)
             
        final_keys.add(flax_key)
        
    print("\nComparing Final Keys vs Expected Keys...")
    flax_keys_set = set(flax_keys)
    missing = flax_keys_set - final_keys
    
    # Filter stats logic check
    print("\nDebugging Filtering Logic...")
    filtered_missing = []
    skipped_count = 0
    for k in missing:
         k_str = [str(x) for x in k]
         is_stat = False
         for ks in k_str:
             if "dropout" in ks or "rngs" in ks:
                 is_stat = True
                 break
         if is_stat:
             skipped_count += 1
             continue
         filtered_missing.append(k)
         
    print(f"Skipped {skipped_count} keys due to dropout/rngs filtering.")
    print(f"Remaining Missing Keys (Count: {len(filtered_missing)}):")
    for k in sorted(filtered_missing):
        print(k)

    # Also check if validation function itself is behaving as expected
    from flax.traverse_util import unflatten_dict, flatten_dict
    from maxdiffusion.models.modeling_flax_pytorch_utils import validate_flax_state_dict
    
    # Construct a dummy flax_state_dict with only the keys we found
    # We need to map our final_keys back to a dict
    # This is hard because we don't have the values here.
    # But we can check if the filtering removes the keys from eval_shapes
    
    print("\nChecking if eval_shapes still has dropout keys after filtering:")
    filtered_eval_shapes = {}
    for k, v in eval_shapes.items(): # eval_shapes is already flattened if from to_pure_dict()?
        # Wait, to_pure_dict returns a nested dict or flat?
        # nnx.state(model).to_pure_dict() returns a nested dict structure usually compatible with unflatten_dict?
        # Let's check type of eval_shapes
        pass
        
    # flatten_dict(eval_shapes)
    flat_eval = flatten_dict(eval_shapes)
    filtered_flat = {}
    for k, v in flat_eval.items():
          k_str = [str(x) for x in k]
          is_stat = False
          for ks in k_str:
              if "dropout" in ks or "rngs" in ks:
                  is_stat = True
                  break
          if is_stat:
              continue
          filtered_flat[k] = v
          
    # Now check if the missing keys are in filtered_flat
    print(f"Filtered Flat Eval Shapes Count: {len(filtered_flat)}")
    print(f"Original Flat Eval Shapes Count: {len(flat_eval)}")
    
    # Check if 'rngs' keys are in filtered_flat
    rngs_keys = [k for k in filtered_flat.keys() if "rngs" in str(k) or "dropout" in str(k)]
    print(f"Keys with 'rngs' or 'dropout' remaining in filtered dict: {len(rngs_keys)}")
    for k in rngs_keys[:10]:
        print(k)

if __name__ == "__main__":
    debug_keys()
