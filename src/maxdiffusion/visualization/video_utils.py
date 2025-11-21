# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Video generation utilities for WAN visualization.
Creates videos from timestamped visualization images using imageio (same as WAN 2.1).
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from .. import max_logging


def find_visualization_images(input_dir: str, image_type: str = "current_image") -> List[Tuple[int, str]]:
    """
    Find visualization images and extract timestep information.

    Args:
        input_dir: Directory containing visualization images
        image_type: Type of images to process ("current_image" or "noise")

    Returns:
        List of (timestep, filepath) tuples sorted by timestep (descending)
    """
    images = []

    # Search for images recursively
    input_path = Path(input_dir)

    # Pattern to match: current_image_t{timestep}_frame{frame}.png or noise_t{timestep}_frame{frame}.png
    pattern = rf"{image_type}_t(\d+)_frame(\d+)\.png"

    for img_path in input_path.rglob("*.png"):
        match = re.match(pattern, img_path.name)
        if match:
            timestep = int(match.group(1))
            frame_idx = int(match.group(2))
            # For now, only process frame 0
            if frame_idx == 0:
                images.append((timestep, str(img_path)))

    # Sort by timestep (descending - from high noise to low noise)
    images.sort(key=lambda x: x[0], reverse=True)

    return images


def export_visualization_video(
    image_list: List[Tuple[int, str]],
    output_path: str,
    fps: float = 4.0
) -> bool:
    """
    Create video using imageio (same method as WAN 2.1) from timestamped images.

    Args:
        image_list: List of (timestep, filepath) tuples
        output_path: Output video file path
        fps: Frames per second (default 4.0 for 0.25 second per frame)

    Returns:
        True if successful, False otherwise
    """
    try:
        import imageio
    except ImportError:
        max_logging.log("imageio not available. Please install: pip install imageio[ffmpeg]")
        return False

    if not image_list:
        max_logging.log("No images found to process")
        return False

    max_logging.log(f"Creating video from {len(image_list)} visualization frames...")

    # Load all images and convert to the format expected by imageio
    video_frames = []
    target_size = None

    for i, (timestep, img_path) in enumerate(image_list):
        # Load image using PIL (same way WAN processes images)
        img = Image.open(img_path).convert('RGB')

        # Determine target size from first image
        if target_size is None:
            target_size = img.size
            max_logging.log(f"Video dimensions: {target_size[0]}x{target_size[1]}")

        # Resize image to ensure all frames have same size
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array and ensure proper format
        frame = np.array(img)

        # Ensure values are in 0-255 range (imageio expects uint8)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                # Normalized to 0-1, scale to 0-255
                frame = (frame * 255).astype(np.uint8)
            else:
                # Already in 0-255 range, just convert type
                frame = frame.astype(np.uint8)

        video_frames.append(frame)

    try:
        # Use imageio to create video (same as WAN 2.1)
        with imageio.get_writer(
            output_path,
            fps=fps,
            quality=8,  # High quality (WAN default)
            macro_block_size=16  # Standard macroblock size
        ) as writer:
            for frame in video_frames:
                writer.append_data(frame)

        max_logging.log(f"Video created successfully: {output_path}")
        return True

    except Exception as e:
        max_logging.log(f"Video creation error: {e}")
        return False


def create_visualization_videos(
    viz_dir: str,
    output_prefix: str = "wan_visualization",
    fps: float = 4.0
) -> None:
    """
    Create both denoising and noise evolution videos from visualization directory.

    Args:
        viz_dir: Directory containing visualization images
        output_prefix: Prefix for output video files
        fps: Frames per second (default 4.0 for 0.25 second per frame)
    """
    if not os.path.exists(viz_dir):
        max_logging.log(f"Visualization directory not found: {viz_dir}")
        return

    # Create denoising process video (decoded images)
    current_images = find_visualization_images(viz_dir, "current_image")
    if current_images:
        denoising_video_path = os.path.join(viz_dir, f"{output_prefix}_denoising_process.mp4")
        max_logging.log(f"Creating denoising process video: {denoising_video_path}")
        success = export_visualization_video(current_images, denoising_video_path, fps)
        if success:
            timestep_range = f"t={current_images[0][0]} → t={current_images[-1][0]}"
            duration = len(current_images) / fps
            max_logging.log(f"Denoising video: {duration:.1f}s, {timestep_range}")
    else:
        max_logging.log("No current_image files found for denoising video")

    # Create noise evolution video (latent channels)
    noise_images = find_visualization_images(viz_dir, "noise")
    if noise_images:
        noise_video_path = os.path.join(viz_dir, f"{output_prefix}_noise_evolution.mp4")
        max_logging.log(f"Creating noise evolution video: {noise_video_path}")
        success = export_visualization_video(noise_images, noise_video_path, fps)
        if success:
            timestep_range = f"t={noise_images[0][0]} → t={noise_images[-1][0]}"
            duration = len(noise_images) / fps
            max_logging.log(f"Noise evolution video: {duration:.1f}s, {timestep_range}")
    else:
        max_logging.log("No noise files found for noise evolution video")
