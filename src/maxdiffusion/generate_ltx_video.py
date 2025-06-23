
from absl import app
from typing import Sequence
from maxdiffusion.pipelines.ltx_video.ltx_video_pipeline import LTXVideoPipeline
from maxdiffusion import pyconfig
import jax.numpy as jnp
from datetime import datetime
import os
import json
import torch
from pathlib import Path

def run(config):
  
  height_padded = ((config.height - 1) // 32 + 1) * 32
  width_padded = ((config.width - 1) // 32 + 1) * 32
  num_frames_padded = ((config.num_frames - 2) // 8 + 1) * 8 + 1
  prompt_enhancement_words_threshold = config.prompt_enhancement_words_threshold
  prompt_word_count = len(config.prompt.split())
  enhance_prompt = (
    prompt_enhancement_words_threshold > 0 and prompt_word_count < prompt_enhancement_words_threshold
  )
  

  pipeline = LTXVideoPipeline.from_pretrained(config, enhance_prompt)
  images = pipeline()
  
  
  
  output_dir = (
    Path(config.output_path)
    if config.output_path
    else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  
  
  


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)