from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_flax_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_flax_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_flax_and_transformers_objects))
else:
    _import_structure["pipeline_flax_controlnet"] = ["FlaxStableDiffusionControlNetPipeline"]
    _import_structure["pipeline_flax_controlnet_sdxl"] = ["FlaxStableDiffusionXLControlNetPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_flax_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_flax_and_transformers_objects import *  # noqa F403
    else:
        from .pipeline_flax_controlnet import FlaxStableDiffusionControlNetPipeline
        from .pipeline_flax_controlnet_sdxl import FlaxStableDiffusionXLControlNetPipeline


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
