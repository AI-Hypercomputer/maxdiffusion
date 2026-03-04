from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

safetensor_path = hf_hub_download("Lightricks/LTX-2", "audio_vae/diffusion_pytorch_model.safetensors")
state_dict = load_file(safetensor_path)

print("latents_mean shape:", state_dict["latents_mean"].shape)
print("latents_std shape:", state_dict["latents_std"].shape)
print("latents_mean:", state_dict["latents_mean"])
