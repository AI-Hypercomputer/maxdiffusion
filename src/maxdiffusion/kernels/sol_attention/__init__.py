# Copyright 2026 Google LLC
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
"""Sol-Attn: training-free sparse attention with threshold routing and
approximate correction (arXiv:2607.24027)."""

from maxdiffusion.kernels.sol_attention.routing import SolBlockSizes
from maxdiffusion.kernels.sol_attention.sol_attention_kernel import sol_attention

__all__ = ["SolBlockSizes", "sol_attention"]
