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

import json
import jax
import numpy as np
from typing import Optional, Tuple
from maxdiffusion.pipelines.ltx2.ltx2_pipeline import LTX2Pipeline
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2 import LTX2VideoAutoencoderKL
from maxdiffusion.models.ltx2.autoencoder_kl_ltx2_audio import FlaxAutoencoderKLLTX2Audio
from maxdiffusion.models.ltx2.text_encoders.text_encoders_ltx2 import LTX2AudioVideoGemmaTextEncoder
from maxdiffusion.models.ltx2.vocoder_ltx2 import LTX2Vocoder
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from maxdiffusion import max_logging, max_utils
from maxdiffusion.checkpointing.checkpointing_utils import create_orbax_checkpoint_manager
import orbax.checkpoint as ocp
from etils import epath
import torch

LTX2_CHECKPOINT = "LTX2_CHECKPOINT"

class LTX2Checkpointer:

  def __init__(self, config, checkpoint_type: str = LTX2_CHECKPOINT):
    self.config = config
    self.checkpoint_type = checkpoint_type
    self.opt_state = None

    self.checkpoint_manager: ocp.CheckpointManager = create_orbax_checkpoint_manager(
        self.config.checkpoint_dir,
        enable_checkpointing=True,
        save_interval_steps=1,
        checkpoint_type=checkpoint_type,
        dataset_type=getattr(config, "dataset_type", None),
    )

  def load_ltx2_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    if step is None:
      step = self.checkpoint_manager.latest_step()
      max_logging.log(f"Latest LTX2 checkpoint step: {step}")
      if step is None:
        max_logging.log("No LTX2 checkpoint found.")
        return None, None
    max_logging.log(f"Loading LTX2 checkpoint from step {step}")
    metadatas = self.checkpoint_manager.item_metadata(step)
    transformer_metadata = metadatas.ltx2_state
    abstract_tree_structure_params = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, transformer_metadata)
    params_restore = ocp.args.PyTreeRestore(
        restore_args=jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
            abstract_tree_structure_params,
        )
    )

    max_logging.log("Restoring LTX2 checkpoint")
    restored_checkpoint = self.checkpoint_manager.restore(
        directory=epath.Path(self.config.checkpoint_dir),
        step=step,
        args=ocp.args.Composite(
            ltx2_state=params_restore,
            ltx2_config=ocp.args.JsonRestore(),
        ),
    )
    max_logging.log(f"restored checkpoint {restored_checkpoint.keys()}")
    max_logging.log(f"restored checkpoint ltx2_state {restored_checkpoint.ltx2_state.keys()}")
    max_logging.log(f"optimizer found in checkpoint {'opt_state' in restored_checkpoint.ltx2_state.keys()}")
    return restored_checkpoint, step

  def load_diffusers_checkpoint(self):
    config = self.config
    max_logging.log("Loading LTX2 components from Hugging Face base models.")
    
    # 1. Tokenizer
    max_logging.log("Loading Gemma Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    
    # 2. Text Encoder (PyTorch)
    max_logging.log("Loading Gemma3 Text Encoder...")
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )
    text_encoder.eval()
    
    # 3. Connectors
    max_logging.log("Loading Connectors...")
    connectors = LTX2AudioVideoGemmaTextEncoder.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="connectors",
    )
    
    # 4. Video VAE
    max_logging.log("Loading Video VAE...")
    vae = LTX2VideoAutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
    )

    # 5. Audio VAE
    max_logging.log("Loading Audio VAE...")
    audio_vae = FlaxAutoencoderKLLTX2Audio.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="audio_vae",
    )

    # 6. Transformer
    max_logging.log("Loading Transformer...")
    # NOTE: Transformer weights are usually sharded and loaded separately in generation scripts
    # This just instantiates the architecture wrapper or loads full weights.
    # In MaxDiffusion we typically let the pipeline or generation script handle sharded loading
    # but we load the raw config/eval shape here.
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
         config.pretrained_model_name_or_path,
         subfolder="transformer",
    )

    # 7. Vocoder
    max_logging.log("Loading Vocoder...")
    vocoder = LTX2Vocoder.from_pretrained(
         config.pretrained_model_name_or_path,
         subfolder="vocoder",
    )

    # 8. Scheduler
    max_logging.log("Loading Scheduler...")
    scheduler = FlaxFlowMatchScheduler.from_pretrained(
         config.pretrained_model_name_or_path,
         subfolder="scheduler",
    )

    pipeline = LTX2Pipeline(
        scheduler=scheduler,
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        transformer=transformer,
        vocoder=vocoder,
    )
    
    return pipeline

  def load_checkpoint(self, step=None) -> Tuple[LTX2Pipeline, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_ltx2_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      max_logging.log("Loading LTX2 pipeline from checkpoint (TODO: implement fully if needed)")
      # pipeline = LTX2Pipeline.from_checkpoint(self.config, restored_checkpoint)
      # if "opt_state" in restored_checkpoint.ltx2_state.keys():
      #   opt_state = restored_checkpoint.ltx2_state["opt_state"]
      pipeline = self.load_diffusers_checkpoint() # Fallback for now 
    else:
      max_logging.log("No checkpoint found, loading default pipeline.")
      pipeline = self.load_diffusers_checkpoint()

    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: LTX2Pipeline, train_states: dict):
    """Saves the training state and model configurations."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    items = {
        "ltx2_config": ocp.args.JsonSave(config_to_json(pipeline.transformer)),
    }

    items["ltx2_state"] = ocp.args.PyTreeSave(train_states)

    # Save the checkpoint
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")

