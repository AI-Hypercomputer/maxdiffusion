"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_torch_available,
)


# These modules contain pipelines from multiple libraries/frameworks
_dummy_objects = {}
_import_structure = {"stable_diffusion": [], "stable_diffusion_xl": [], "latent_diffusion": [], "controlnet": []}

try:
  if not is_onnx_available():
    raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
  from ..utils import dummy_onnx_objects  # noqa F403

  _dummy_objects.update(get_objects_from_module(dummy_onnx_objects))
else:
  _import_structure["onnx_utils"] = ["OnnxRuntimeModel"]
try:
  if not (is_torch_available() and is_onnx_available()):
    raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
  from ..utils import dummy_torch_and_transformers_and_onnx_objects  # noqa F403

  _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_onnx_objects))
else:
  _import_structure["stable_diffusion"].extend(
      [
          "OnnxStableDiffusionImg2ImgPipeline",
          "OnnxStableDiffusionInpaintPipeline",
          "OnnxStableDiffusionInpaintPipelineLegacy",
          "OnnxStableDiffusionPipeline",
          "OnnxStableDiffusionUpscalePipeline",
          "StableDiffusionOnnxPipeline",
      ]
  )

try:
  if not is_flax_available():
    raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
  from ..utils import dummy_flax_objects  # noqa F403

  _dummy_objects.update(get_objects_from_module(dummy_flax_objects))
else:
  _import_structure["pipeline_flax_utils"] = ["FlaxDiffusionPipeline"]
try:
  if not (is_flax_available()):
    raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
  from ..utils import dummy_flax_and_transformers_objects  # noqa F403

  _dummy_objects.update(get_objects_from_module(dummy_flax_and_transformers_objects))
else:
  _import_structure["controlnet"].extend(
      ["FlaxStableDiffusionControlNetPipeline", "FlaxStableDiffusionXLControlNetPipeline"]
  )
  _import_structure["stable_diffusion"].extend(
      [
          "FlaxStableDiffusionImg2ImgPipeline",
          "FlaxStableDiffusionInpaintPipeline",
          "FlaxStableDiffusionPipeline",
      ]
  )
  _import_structure["stable_diffusion_xl"].extend(
      [
          "FlaxStableDiffusionXLPipeline",
      ]
  )
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
  try:
    if not is_onnx_available():
      raise OptionalDependencyNotAvailable()
  except OptionalDependencyNotAvailable:
    from ..utils.dummy_onnx_objects import *  # noqa F403

  else:
    from .onnx_utils import OnnxRuntimeModel

  try:
    if not (is_torch_available() and is_onnx_available()):
      raise OptionalDependencyNotAvailable()
  except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_and_onnx_objects import *
  else:
    from .stable_diffusion import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
        OnnxStableDiffusionInpaintPipelineLegacy,
        OnnxStableDiffusionPipeline,
        OnnxStableDiffusionUpscalePipeline,
        StableDiffusionOnnxPipeline,
    )
  try:
    if not is_flax_available():
      raise OptionalDependencyNotAvailable()
  except OptionalDependencyNotAvailable:
    from ..utils.dummy_flax_objects import *  # noqa F403
  else:
    from .pipeline_flax_utils import FlaxDiffusionPipeline

  try:
    if not (is_flax_available()):
      raise OptionalDependencyNotAvailable()
  except OptionalDependencyNotAvailable:
    from ..utils.dummy_flax_and_transformers_objects import *
  else:
    from .controlnet import (FlaxStableDiffusionControlNetPipeline, FlaxStableDiffusionXLControlNetPipeline)
    from .stable_diffusion import (
        FlaxStableDiffusionImg2ImgPipeline,
        FlaxStableDiffusionInpaintPipeline,
        FlaxStableDiffusionPipeline,
    )
    from .stable_diffusion_xl import FlaxStableDiffusionXLPipeline

  try:
    if not (is_torch_available() and is_note_seq_available()):
      raise OptionalDependencyNotAvailable()
  except OptionalDependencyNotAvailable:
    from ..utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403

  else:
    from .spectrogram_diffusion import MidiProcessor, SpectrogramDiffusionPipeline

else:
  import sys

  sys.modules[__name__] = _LazyModule(
      __name__,
      globals()["__file__"],
      _import_structure,
      module_spec=__spec__,
  )
  for name, value in _dummy_objects.items():
    setattr(sys.modules[__name__], name, value)
