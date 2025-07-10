import numpy as np
from absl import app
from typing import Sequence
from maxdiffusion.pipelines.ltx_video.ltx_video_pipeline import LTXVideoPipeline
from maxdiffusion.pipelines.ltx_video.ltx_video_pipeline import LTXMultiScalePipeline
from maxdiffusion import pyconfig
import jax.numpy as jnp
from maxdiffusion.models.ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from maxdiffusion.models.ltx_video.autoencoders.latent_upsampler import LatentUpsampler
from huggingface_hub import hf_hub_download
import imageio
from datetime import datetime
from maxdiffusion.utils import export_to_video

import os
import json
import torch
from pathlib import Path


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)

def create_latent_upsampler(latent_upsampler_model_path: str, device: str):
    latent_upsampler = LatentUpsampler.from_pretrained(latent_upsampler_model_path)
    latent_upsampler.to(device)
    latent_upsampler.eval()
    return latent_upsampler

def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / \
            f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )


def run(config):
    height_padded = ((config.height - 1) // 32 + 1) * 32
    width_padded = ((config.width - 1) // 32 + 1) * 32
    num_frames_padded = ((config.num_frames - 2) // 8 + 1) * 8 + 1
    padding = calculate_padding(
        config.height, config.width, height_padded, width_padded)
    # prompt_enhancement_words_threshold = config.prompt_enhancement_words_threshold
    # prompt_word_count = len(config.prompt.split())
    # enhance_prompt = (
    #     prompt_enhancement_words_threshold > 0 and prompt_word_count < prompt_enhancement_words_threshold
    # )

    seed = 10  # change this, generator in pytorch, used in prepare_latents
    generator = torch.Generator().manual_seed(seed)
    pipeline = LTXVideoPipeline.from_pretrained(config, enhance_prompt = False)
    if config.pipeline_type == "multi-scale":   #move this to pipeline file??
        spatial_upscaler_model_name_or_path = config.spatial_upscaler_model_path
    
        if spatial_upscaler_model_name_or_path and not os.path.isfile(
            spatial_upscaler_model_name_or_path
        ):
            spatial_upscaler_model_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=spatial_upscaler_model_name_or_path,
                local_dir= "/mnt/disks/diffusionproj",
                repo_type="model",
            )
        else:
            spatial_upscaler_model_path = spatial_upscaler_model_name_or_path
        if not config.spatial_upscaler_model_path:
            raise ValueError(
                "spatial upscaler model path is missing from pipeline config file and is required for multi-scale rendering"
            )
        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, "cpu"  #device set to cpu for now
        )
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)
    stg_mode = config.stg_mode
    if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
        skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")
    # images = pipeline(height=height_padded, width=width_padded, num_frames=num_frames_padded,
    #                   is_video=True, output_type='pt', generator=generator, guidance_scale = config.first_pass.guidance_scale, stg_scale = config.stg_scale, rescaling_scale = config.rescaling_scale, skip_initial_inference_steps= config.skip_initial_inference_steps, skip_final_inference_steps= config.skip_final_inference_steps, num_inference_steps = config.num_inference_steps,
    #                   guidance_timesteps = config.guidance_timesteps, cfg_star_rescale = config.cfg_star_rescale, skip_layer_strategy = None, skip_block_list=config.skip_block_list).images
    images = pipeline(height=height_padded, width=width_padded, num_frames=num_frames_padded, is_video=True, output_type='pt', generator=generator, config = config)
    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, :config.num_frames,
                    pad_top:pad_bottom, pad_left:pad_right]
    output_dir = Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(images.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = images[i].permute(1, 2, 3, 0).detach().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = config.frame_rate
        height, width = video_np.shape[1:3]
        # In case a single image is generated
        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=config.prompt,
                seed=seed,
                resolution=(height, width, config.num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            output_filename = get_unique_filename(
                f"video_output_{i}",
                ".mp4",
                prompt=config.prompt,
                seed=seed,
                resolution=(height, width, config.num_frames),
                dir=output_dir,
            )
            print(output_filename)
            # Write video
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)


def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    run(pyconfig.config)


if __name__ == "__main__":
    app.run(main)
