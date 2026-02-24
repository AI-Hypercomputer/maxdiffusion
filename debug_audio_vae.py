
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
    
    print("\nSample PyTorch Keys (Original):")
    for k in pt_keys[:20]:
        print(k)
        
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
    
    # Filter stats
    filtered_missing = []
    for k in missing:
         k_str = [str(x) for x in k]
         if "dropout" in k_str or "rngs" in k_str:
             continue
         filtered_missing.append(k)
         
    print(f"Missing Keys (Count: {len(filtered_missing)}):")
    for k in sorted(filtered_missing)[:20]:
        print(k)

    print("\nExtra Keys (Count: {len(final_keys - flax_keys_set)}):")
    for k in sorted(list(final_keys - flax_keys_set))[:20]:
        print(k)

if __name__ == "__main__":
    debug_keys()
