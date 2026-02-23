
import os
from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch

def inspect_checkpoint():
    try:
        # Allow looking in the user's cache
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"Scanning cache dir: {cache_dir}")
        
        # We know the model is Lightricks/LTX-Video
        # We look for snapshots
        repo_id = "Lightricks/LTX-Video"
        
        # Try to find it
        # We can use hf_hub_download to find the path without downloading if it exists
        from huggingface_hub import hf_hub_download
        
        try:
            vae_path = hf_hub_download(repo_id, subfolder="vae", filename="diffusion_pytorch_model.safetensors")
            print(f"Found VAE checkpoint at: {vae_path}")
            
            with safe_open(vae_path, framework="pt") as f:
                keys = f.keys()
                print(f"Total keys in VAE checkpoint: {len(keys)}")
                print("Sample keys:")
                for i, k in enumerate(keys):
                    if i < 20: 
                        print(k)
                    if "resnets" in k and "up_blocks" in k and i % 10 == 0:
                        print(f"Resnet key sample: {k}")
                        
        except Exception as e:
            print(f"Could not find VAE checkpoint via hf_hub_download: {e}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_checkpoint()
