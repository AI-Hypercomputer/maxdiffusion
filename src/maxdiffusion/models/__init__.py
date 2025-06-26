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

from typing import TYPE_CHECKING
from maxdiffusion.utils import DIFFUSERS_SLOW_IMPORT, _LazyModule, is_flax_available, is_torch_available

_import_structure = {}


_import_structure["controlnet_flax"] = ["FlaxControlNetModel"]
_import_structure["unet_2d_condition_flax"] = ["FlaxUNet2DConditionModel"]
_import_structure["vae_flax"] = ["FlaxAutoencoderKL"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:

    from .controlnet_flax import FlaxControlNetModel
    from .unet_2d_condition_flax import FlaxUNet2DConditionModel
    from .vae_flax import FlaxAutoencoderKL
    from .lora import *
    from .flux.transformers.transformer_flux_flax import FluxTransformer2DModel
    from .ltx_video.transformers.transformer3d import Transformer3DModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__)
