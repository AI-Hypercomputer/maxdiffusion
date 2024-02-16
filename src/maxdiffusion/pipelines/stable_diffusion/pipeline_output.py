# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import List

import numpy as np

import flax

from ...utils import BaseOutput


@flax.struct.dataclass
class FlaxStableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Flax-based Stable Diffusion pipelines.

    Args:
        images (`np.ndarray`):
            Denoised images of array shape of `(batch_size, height, width, num_channels)`.
        nsfw_content_detected (`List[bool]`):
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content
            or `None` if safety checking could not be performed.
    """

    images: np.ndarray
    nsfw_content_detected: List[bool]
