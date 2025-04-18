"""
 Copyright 2025 Google LLC

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

from typing import Callable, List, Union, Sequence
from flax import nnx
from absl import app
from maxdiffusion import pyconfig, max_logging
from maxdiffusion.models.wan.transformers.transformer_flux_wan_nnx import WanModel
from maxdiffusion.pipelines.wan.pipeline_wan import WanPipeline

def run(config):
  max_logging.log("Wan 2.1 inference script")

  pipeline, params = WanPipeline.from_pretrained(
    config.pretrained_model_name_or_path,
    vae=None,
    transformer=None
  )
  breakpoint()

  #wan_transformer = WanModel(rngs=nnx.Rngs(config.seed))


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)

if __name__ == "__main__":
  app.run(main)
