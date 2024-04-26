import os
from maxdiffusion import FlaxStableDiffusionXLPipeline


pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
os.environ['MODEL_ID'], revision="refs/pr/95", split_head_dim=True
)
