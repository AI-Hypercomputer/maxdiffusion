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

from maxdiffusion.loaders.flux_loader import FluxLoader
from maxdiffusion.loaders.sdxl_loader import SDXLLoader
from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p1 import WanCheckpointerI2V_2_1
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
from maxdiffusion import max_utils, max_logging
from maxdiffusion.common_types import WAN2_1, WAN2_2
import jax
from jax.sharding import Mesh

class InferenceLoader:
    """Unified facade for loading models for inference."""

    @staticmethod
    def load(config):
        """
        Loads the pipeline and resources based on config.model_name.
        
        Returns:
            A dictionary containing:
            - 'pipeline': The loaded pipeline object (or dict of components for Flux).
            - 'params': The parameters (weights).
            - 'states': TrainStates/InferenceStates (if applicable).
            - 'mesh': The JAX mesh.
            - 'shardings': Sharding specs (if applicable).
            - 'lora_interceptors': List of LoRA interceptors (optional).
        """
        
        # Create Mesh
        devices_array = max_utils.create_device_mesh(config)
        mesh = Mesh(devices_array, config.mesh_axes)
        max_logging.log(f"Created mesh with axes: {config.mesh_axes}")

        model_name = config.model_name.lower()
        
        if "wan" in model_name:
            return InferenceLoader._load_wan(config, mesh)
        elif "flux" in model_name:
            return InferenceLoader._load_flux(config, mesh)
        elif "sdxl" in model_name:
            return InferenceLoader._load_sdxl(config, mesh)
        else:
            raise ValueError(f"Unsupported model name for InferenceLoader: {config.model_name}")

    @staticmethod
    def _load_flux(config, mesh):
        max_logging.log("Loading Flux model...")
        components, states, shardings = FluxLoader.load(config, mesh)
        # FluxLoader returns (components, states, shardings)
        # We wrap it in a dict standard format
        return {
            "pipeline": components, # Flux components are passed as dict
            "params": {}, # Params are inside states for Flux in our Loader design
            "states": states,
            "mesh": mesh,
            "shardings": shardings,
            "lora_interceptors": components.get("lora_interceptors", [])
        }

    @staticmethod
    def _load_sdxl(config, mesh):
        max_logging.log("Loading SDXL model...")
        pipeline, params, states, state_shardings, lora_interceptors = SDXLLoader.load(config, mesh)
        return {
            "pipeline": pipeline,
            "params": params,
            "states": states,
            "mesh": mesh,
            "shardings": state_shardings,
            "lora_interceptors": lora_interceptors
        }

    @staticmethod
    def _load_wan(config, mesh):
        max_logging.log(f"Loading Wan model: {config.model_name}...")
        model_key = config.model_name
        model_type = config.model_type
        
        if model_key == WAN2_1:
            if model_type == "I2V":
                loader = WanCheckpointerI2V_2_1(config=config)
            else:
                loader = WanCheckpointer2_1(config=config)
        elif model_key == WAN2_2:
            if model_type == "I2V":
                loader = WanCheckpointerI2V_2_2(config=config)
            else:
                loader = WanCheckpointer2_2(config=config)
        else:
             raise ValueError(f"Unknown Wan model: {model_key}")

        # WanCheckpointer loads pipeline, params, train_states
        # It creates its own mesh inside if not passed? 
        # Actually WanCheckpointer usually inherits BaseStableDiffusionCheckpointer which creates mesh.
        # But we passed `config`. 
        # NOTE: WanCheckpointer creates its own mesh instance. We might have double mesh creation.
        # Ideally we refactor WanCheckpointer to accept a mesh, but for now we accept it.
        
        pipeline, params, states = loader.load_checkpoint()
        
        return {
            "pipeline": pipeline,
            "params": params,
            "states": states,
            "mesh": loader.mesh, # Use loader's mesh
            "shardings": None, # Wan pipeline handles sharding internally via NNX
            "lora_interceptors": [] 
        }
