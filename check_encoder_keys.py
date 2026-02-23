
from safetensors import safe_open
from huggingface_hub import snapshot_download
import os

def check_encoder():
    resume_from_checkpoint = "Lightricks/LTX-Video"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub")
    
    print(f"Scanning cache dir: {cache_dir}")
    
    vae_path = None
    # Try to find the specific file
    search_path = os.path.join(cache_dir, "models--Lightricks--LTX-Video/snapshots")
    if os.path.exists(search_path):
        for root, dirs, files in os.walk(search_path):
            if "vae" in root and "diffusion_pytorch_model.safetensors" in files:
                vae_path = os.path.join(root, "diffusion_pytorch_model.safetensors")
                break
    
    if not vae_path:
        print("VAE checkpoint not found in cache. Downloading...")
        # Fallback to downloading if not found (though user seems to have it)
        try:
            download_path = snapshot_download(repo_id=resume_from_checkpoint, allow_patterns=["vae/*"])
            vae_path = os.path.join(download_path, "vae", "diffusion_pytorch_model.safetensors")
        except Exception as e:
            print(f"Failed to download: {e}")
            return

    print(f"Found VAE checkpoint at: {vae_path}")
    
    try:
        with safe_open(vae_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            encoder_keys = [k for k in keys if "encoder" in k]
            decoder_keys = [k for k in keys if "decoder" in k]
            
            print(f"Total keys: {len(keys)}")
            print(f"Encoder keys count: {len(encoder_keys)}")
            print(f"Decoder keys count: {len(decoder_keys)}")
            
            if len(encoder_keys) > 0:
                print("First 5 encoder keys:")
                for k in encoder_keys[:5]:
                    print(k)
            else:
                print("NO ENCODER KEYS FOUND.")

    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    check_encoder()
