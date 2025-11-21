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
Base visualization mixin providing common utilities for model pipeline visualization.
"""

import os
from typing import Optional, List
import numpy as np
from .. import max_logging


class VisualizationMixin:
    """
    Mixin class providing common visualization utilities for diffusion model pipelines.

    This class provides shared functionality like file I/O, directory management,
    and basic plotting utilities. Model-specific visualization logic should be
    implemented in the concrete pipeline classes.
    """

    def _should_visualize(self, stage_name: str = "default") -> bool:
        """
        Check if visualization should be enabled for a given stage.

        Args:
            stage_name: Name of the visualization stage

        Returns:
            True if visualization should be performed
        """
        config_attr = f"visualize_{stage_name}"
        return getattr(self.config, config_attr, False) or getattr(self.config, "visualize_all", False)

    def _get_visualization_dir(self, stage_name: str = "default") -> str:
        """
        Get the visualization output directory for a given stage.

        Args:
            stage_name: Name of the visualization stage

        Returns:
            Path to the visualization directory
        """
        base_dir = getattr(self.config, "visualization_output_dir", f"visualization_{self.config.seed}")
        stage_dir = os.path.join(base_dir, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        return stage_dir

    def _save_tensor_stats(self, tensor: np.ndarray, name: str, stage_name: str = "default") -> None:
        """
        Save tensor statistics to log and optionally to file.

        Args:
            tensor: Tensor to analyze
            name: Name of the tensor
            stage_name: Stage name for organization
        """
        stats = {
            "shape": list(tensor.shape),  # Convert to list for JSON serialization
            "dtype": str(tensor.dtype),   # Convert dtype to string
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "nan_count": int(np.isnan(tensor).sum()),
            "inf_count": int(np.isinf(tensor).sum()),
        }

        max_logging.log(f"[{stage_name}] {name} stats: {stats}")

        # Optionally save to JSON file
        if getattr(self.config, "save_tensor_stats", False):
            import json
            stats_dir = self._get_visualization_dir(stage_name)
            stats_file = os.path.join(stats_dir, f"{name}_stats.json")
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

    def _create_grid_plot(
        self,
        images: np.ndarray,
        titles: Optional[List[str]] = None,
        output_path: str = None,
        figsize: tuple = (16, 12),
        cmap: str = "gray",
        max_images: int = 16
    ) -> str:
        """
        Create a grid plot of images using matplotlib.

        Args:
            images: Array of images with shape (N, H, W) or (N, H, W, C)
            titles: Optional titles for each image
            output_path: Path to save the plot
            figsize: Figure size for matplotlib
            cmap: Colormap for grayscale images
            max_images: Maximum number of images to display

        Returns:
            Path where the plot was saved
        """
        try:
            import matplotlib.pyplot as plt

            num_images = min(len(images), max_images)
            grid_size = int(np.ceil(np.sqrt(num_images)))

            fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
            if grid_size == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i in range(num_images):
                img = images[i]

                # Handle different image formats
                if len(img.shape) == 3 and img.shape[-1] == 3:
                    # RGB image
                    axes[i].imshow(np.clip(img, 0, 1))
                elif len(img.shape) == 3 and img.shape[-1] == 1:
                    # Single channel
                    axes[i].imshow(img.squeeze(), cmap=cmap)
                else:
                    # Grayscale or unknown format
                    if len(img.shape) == 3:
                        img = img.mean(axis=-1)
                    axes[i].imshow(img, cmap=cmap)

                if titles and i < len(titles):
                    axes[i].set_title(titles[i])
                axes[i].axis('off')

            # Hide unused subplots
            for i in range(num_images, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                max_logging.log(f"Saved visualization to {output_path}")

            plt.close()
            return output_path

        except ImportError:
            max_logging.log("matplotlib not available, skipping grid plot visualization")
            return None
        except Exception as e:
            max_logging.log(f"Error creating grid plot: {e}")
            return None

    def _save_tensor_as_numpy(self, tensor: np.ndarray, name: str, stage_name: str = "default") -> str:
        """
        Save tensor as numpy file.

        Args:
            tensor: Tensor to save
            name: Name for the file
            stage_name: Stage name for organization

        Returns:
            Path where the tensor was saved
        """
        viz_dir = self._get_visualization_dir(stage_name)
        output_path = os.path.join(viz_dir, f"{name}.npy")
        np.save(output_path, tensor)
        max_logging.log(f"Saved tensor {name} to {output_path}")
        return output_path

    def _create_histogram_plot(
        self,
        data: np.ndarray,
        title: str = "Distribution",
        output_path: str = None,
        bins: int = 100
    ) -> str:
        """
        Create histogram plot of data distribution.

        Args:
            data: Data to plot
            title: Plot title
            output_path: Path to save the plot
            bins: Number of histogram bins

        Returns:
            Path where the plot was saved
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.hist(data.flatten(), bins=bins, alpha=0.7, edgecolor='black')
            plt.title(title)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f'Mean: {data.mean():.4f}\nStd: {data.std():.4f}\nMin: {data.min():.4f}\nMax: {data.max():.4f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8})

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                max_logging.log(f"Saved histogram to {output_path}")

            plt.close()
            return output_path

        except ImportError:
            max_logging.log("matplotlib not available, skipping histogram visualization")
            return None
        except Exception as e:
            max_logging.log(f"Error creating histogram plot: {e}")
            return None
