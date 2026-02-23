
from safetensors import safe_open
from huggingface_hub import snapshot_download
import os

def inspect_structure():
    resume_from_checkpoint = "Lightricks/LTX-Video"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub")
    
    vae_path = None
    search_path = os.path.join(cache_dir, "models--Lightricks--LTX-Video/snapshots")
    if os.path.exists(search_path):
        for root, dirs, files in os.walk(search_path):
            if "vae" in root and "diffusion_pytorch_model.safetensors" in files:
                vae_path = os.path.join(root, "diffusion_pytorch_model.safetensors")
                break
    
    if not vae_path:
        print("VAE checkpoint not found.")
        return

    print(f"Analyzing checkpoint: {vae_path}")
    
    structure = {
        "encoder": {"down_blocks": {}, "mid_block": 0},
        "decoder": {"up_blocks": {}, "mid_block": 0}
    }
    
    try:
        with safe_open(vae_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            print(f"Total keys: {len(keys)}")
            
            for key in keys:
                parts = key.split(".")
                if "resnets" not in parts:
                    continue
                    
                try:
                    resnets_idx = parts.index("resnets")
                    # The next part should be the index
                    if len(parts) > resnets_idx + 1 and parts[resnets_idx + 1].isdigit():
                        block_idx = int(parts[resnets_idx + 1])
                        
                        if parts[0] == "encoder":
                            if "down_blocks" in parts:
                                down_idx_loc = parts.index("down_blocks") + 1
                                down_idx = int(parts[down_idx_loc])
                                if down_idx not in structure["encoder"]["down_blocks"]:
                                    structure["encoder"]["down_blocks"][down_idx] = 0
                                structure["encoder"]["down_blocks"][down_idx] = max(structure["encoder"]["down_blocks"][down_idx], block_idx + 1)
                            elif "mid_block" in parts:
                                 structure["encoder"]["mid_block"] = max(structure["encoder"]["mid_block"], block_idx + 1)
                                 
                        elif parts[0] == "decoder":
                            if "up_blocks" in parts:
                                up_idx_loc = parts.index("up_blocks") + 1
                                up_idx = int(parts[up_idx_loc])
                                if up_idx not in structure["decoder"]["up_blocks"]:
                                    structure["decoder"]["up_blocks"][up_idx] = 0
                                structure["decoder"]["up_blocks"][up_idx] = max(structure["decoder"]["up_blocks"][up_idx], block_idx + 1)
                            elif "mid_block" in parts:
                                 structure["decoder"]["mid_block"] = max(structure["decoder"]["mid_block"], block_idx + 1)

                except (ValueError, IndexError) as e:
                    # print(f"Skipping key {key}: {e}")
                    continue

        print("\nDuced VAE Structure (Layers per block):")
        print("Encoder:")
        for i in sorted(structure["encoder"]["down_blocks"].keys()):
            print(f"  Down Block {i}: {structure['encoder']['down_blocks'][i]} layers")
        print(f"  Mid Block: {structure['encoder']['mid_block']} layers")
        
        print("Decoder:")
        for i in sorted(structure["decoder"]["up_blocks"].keys()):
            print(f"  Up Block {i}: {structure['decoder']['up_blocks'][i]} layers")
        print(f"  Mid Block: {structure['decoder']['mid_block']} layers")

    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    inspect_structure()
