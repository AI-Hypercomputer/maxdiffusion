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

from typing import Sequence
from absl import app
import time
from maxdiffusion import pyconfig, max_logging
from maxdiffusion.inference.loader import InferenceLoader
from maxdiffusion.inference.runner import DiffusionRunner

def run(config):
    # 1. Load Model
    max_logging.log("Initializing InferenceLoader...")
    loaded_model = InferenceLoader.load(config)
    
    # 2. Initialize Runner
    max_logging.log("Initializing DiffusionRunner...")
    runner = DiffusionRunner(loaded_model, config)
    
    # 3. Run Inference
    max_logging.log("Starting Inference...")
    t0 = time.perf_counter()
    pil_images = runner.run()
    t1 = time.perf_counter()
    max_logging.log(f"Inference time: {t1 - t0:.2f}s")
    
    # 4. Save Images
    for i, image in enumerate(pil_images):
        save_path = f"image_sdxl_{i}.png"
        image.save(save_path)
        max_logging.log(f"Saved image to {save_path}")
        
    return pil_images

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)

if __name__ == "__main__":
  app.run(main)