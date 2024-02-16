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

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_scipy_available,
    is_torch_available,
    is_torchsde_available,
)


_dummy_modules = {}
_import_structure = {}


try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_flax_objects  # noqa F403

    _dummy_modules.update(get_objects_from_module(dummy_flax_objects))

else:
    _import_structure["scheduling_ddim_flax"] = ["FlaxDDIMScheduler"]
    _import_structure["scheduling_euler_discrete_flax"] = ["FlaxEulerDiscreteScheduler"]
    _import_structure["scheduling_ddpm_flax"] = ["FlaxDDPMScheduler"]
    _import_structure["scheduling_dpmsolver_multistep_flax"] = ["FlaxDPMSolverMultistepScheduler"]
    _import_structure["scheduling_euler_discrete_flax"] = ["FlaxEulerDiscreteScheduler"]
    _import_structure["scheduling_karras_ve_flax"] = ["FlaxKarrasVeScheduler"]
    _import_structure["scheduling_lms_discrete_flax"] = ["FlaxLMSDiscreteScheduler"]
    _import_structure["scheduling_pndm_flax"] = ["FlaxPNDMScheduler"]
    _import_structure["scheduling_sde_ve_flax"] = ["FlaxScoreSdeVeScheduler"]
    _import_structure["scheduling_utils_flax"] = [
        "FlaxKarrasDiffusionSchedulers",
        "FlaxSchedulerMixin",
        "FlaxSchedulerOutput",
        "broadcast_to_shape_from_left",
    ]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from ..utils import OptionalDependencyNotAvailable, is_flax_available
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_flax_objects import *  # noqa F403
    else:
        from .scheduling_ddim_flax import FlaxDDIMScheduler
        from .scheduling_ddpm_flax import FlaxDDPMScheduler
        from .scheduling_dpmsolver_multistep_flax import FlaxDPMSolverMultistepScheduler
        from .scheduling_euler_discrete_flax import FlaxEulerDiscreteScheduler
        from .scheduling_karras_ve_flax import FlaxKarrasVeScheduler
        from .scheduling_lms_discrete_flax import FlaxLMSDiscreteScheduler
        from .scheduling_pndm_flax import FlaxPNDMScheduler
        from .scheduling_sde_ve_flax import FlaxScoreSdeVeScheduler
        from .scheduling_utils_flax import (
            FlaxKarrasDiffusionSchedulers,
            FlaxSchedulerMixin,
            FlaxSchedulerOutput,
            broadcast_to_shape_from_left,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    for name, value in _dummy_modules.items():
        setattr(sys.modules[__name__], name, value)
