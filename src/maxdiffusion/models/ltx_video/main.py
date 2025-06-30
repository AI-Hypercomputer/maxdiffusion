import json
import os

from maxdiffusion.models.ltx_video.transformers_pytorch.transformer_pt import Transformer3DModel_PT

base_dir = os.path.dirname(__file__)
config_path = os.path.join(base_dir, "xora_v1.2-13B-balanced-128.json")
with open(config_path, "r") as f:
  model_config = json.load(f)
transformer = Transformer3DModel_PT.from_config(model_config)
