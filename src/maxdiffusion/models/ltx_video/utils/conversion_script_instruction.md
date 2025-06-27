### Transformer Pytorch Weight Downloading and Jax Weight Loading Instructions:
1. Create new tansformers_pytorch folder under models/ltx_video.
2. Move files from LTX repo, specifically, attention.py, embeddings.py, symmetric_patchifier.py, and transformer3d.py into the newly created folder. See here: https://github.com/Lightricks/LTX-Video/tree/main/ltx_video/models/transformers 
3. Rename transformer3d.py to transformer_pt.py to distinguish from the pytorch version. Change classname to Transformer3DModel_PT. Also change classname in line "transformer = Transformer3DModel.from_config(transformer_config)"
4. Weight Downloading and Conversion
    - If first time running (no local safetensors): \
    In the src/maxdiffusion/models/ltx_video/utils folder, run python convert_torch_weights_to_jax.py --download_ckpt_path [location to download safetensors] --output_dir [location to save jax ckpt] --transformer_config_path ../xora_v1.2-13B-balanced-128.json.
    - If already have local pytorch checkpoint: \
    Replace the --download_ckpt_path with --local_ckpt_path and add corresponding location
5. Restoring Jax Weights into transformer:
    - Replace the "ckpt_path" in src/maxdiffusion/models/ltx_video/xora_v1.2-13B-balanced-128.json with jax ckpt path.
    - Run python src/maxdiffusion/generate_ltx_video.py src/maxdiffusion/configs/ltx_video.yml in the outer repo folder.

