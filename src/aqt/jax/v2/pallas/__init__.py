# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AQT APIs for pallas."""

import aqt.jax.v2.pallas.dot_general as _aqt_dot_general
import aqt.jax.v2.pallas.pallas_call as _aqt_pallas_call
import aqt.jax.v2.pallas.quantizer as _aqt_quantizer

pallas_call = _aqt_pallas_call.pallas_call
quant = _aqt_quantizer.quant
load_qtensor = _aqt_dot_general.load_qtensor
dot_general = _aqt_dot_general.dot_general
DequantMode = _aqt_dot_general.DequantMode
