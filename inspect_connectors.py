
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
import os

def inspect_connectors():
    repo_id = "Lightricks/LTX-2"
    subfolder = "connectors"
    
    print(f"Downloading {subfolder} from {repo_id}...")
    try:
        path = snapshot_download(repo_id, allow_patterns=[f"{subfolder}/*"])
        print(f"Downloaded to {path}")
        
        folder_path = os.path.join(path, subfolder)
        files = [f for f in os.listdir(folder_path) if f.endswith(".safetensors")]
        
        for f in files:
            print(f"\nScanning {f}...")
            file_path = os.path.join(folder_path, f)
            state_dict = load_file(file_path)
            for key, value in state_dict.items():
                print(f"{key}: {value.shape}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_connectors()
