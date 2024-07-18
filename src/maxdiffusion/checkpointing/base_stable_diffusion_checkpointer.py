

import jax
from jax.sharding import Mesh
from maxdiffusion import (
    max_utils,
    FlaxStableDiffusionPipeline,
    FlaxStableDiffusionXLPipeline,
    FlaxUNet2DConditionModel,
    FlaxAutoencoderKL,
)

from transformers import (
  CLIPTokenizer,
  FlaxCLIPTextModel,
  PretrainedConfig
)

from maxdiffusion.checkpointing.checkpointing_utils import (
    create_orbax_checkpoint_manager,
    load_stable_diffusion_configs,
)

STABLE_DIFFUSION_CHECKPOINT = "STABLE_DIFFUSION_CHECKPOINT"
STABLE_DIFFUSION_XL_CHECKPOINT = "STABLE_DIFUSSION_XL_CHECKPOINT"
_CHECKPOINT_FORMAT_DIFFUSERS = "CHECKPOINT_FORMAT_DIFFUSERS"
_CHECKPOINT_FORMAT_ORBAX = "CHECKPOINT_FORMAT_ORBAX"

class BaseStableDiffusionCheckpointer:
    def __init__(self, config, checkpoint_type):
        self.config = config
        self.checkpoint_type = checkpoint_type
        if len(config.cache_dir) > 0:
            jax.config.update("jax_compilation_cache_dir", config.cache_dir)
        
        self.rng = jax.random.PRNGKey(self.config.seed)

        devices_array = max_utils.create_device_mesh(config)
        self.mesh = Mesh(devices_array, self.config.mesh_axes)

        self.pipeline = None
        self.params = {}

        self.checkpoint_manager = create_orbax_checkpoint_manager(
            self.config.checkpoint_dir,
            enable_checkpointing=True,
            save_interval_steps=1,
            checkpoint_type=checkpoint_type
        )
    
    def _get_pipeline_class(self):
        if self.checkpoint_type == STABLE_DIFFUSION_CHECKPOINT:
            pipeline_class = FlaxStableDiffusionPipeline
        else:
            pipeline_class = FlaxStableDiffusionXLPipeline
        
        return pipeline_class

    def _set_checkpoint_format(self, checkpoint_format):
        self.checkpoint_format = checkpoint_format
    
    def load_diffusers_checkpoint(self):
        pipeline_class = self._get_pipeline_class()

        precision = max_utils.get_precision(self.config)
        flash_block_sizes = max_utils.get_flash_block_sizes(self.config)

        pipeline, params = pipeline_class.from_pretrained(
            self.config.pretrained_model_name_or_path,
            revision=self.config.revision,
            dtype=self.config.activations_dtype,
            safety_checker=None,
            feature_extractor=None,
            from_pt=self.config.from_pt,
            split_head_dim=self.config.split_head_dim,
            norm_num_groups=self.config.norm_num_groups,
            attention_kernel=self.config.attention,
            flash_block_sizes=flash_block_sizes,
            mesh=self.mesh,
            precision=precision
        )

        params = jax.tree_util.tree_map(lambda x: x.astype(self.config.weights_dtype), params)

        self.pipeline = pipeline
        self.params = params

    def load_checkpoint(self, step = None, scheduler_class = None):

        pipeline_class = self._get_pipeline_class()
        
        precision = max_utils.get_precision(self.config)
        flash_block_sizes = max_utils.get_flash_block_sizes(self.config)
        # try loading using orbax, if not, use diffusers loading
        model_configs = load_stable_diffusion_configs(
            self.checkpoint_manager,
            self.checkpoint_type, step)

        if model_configs:
            unet = FlaxUNet2DConditionModel.from_config(
                model_configs[0]["unet_config"],
                dtype=self.config.activations_dtype,
                from_pt=self.config.from_pt,
                split_head_dim=self.config.split_head_dim,
                norm_num_groups=self.config.norm_num_groups,
                attention_kernel=self.config.attention,
                flash_block_sizes=flash_block_sizes,
                mesh=self.mesh,
                precision=precision
            )
            
            vae = FlaxAutoencoderKL.from_config(
                model_configs[0]["vae_config"],
                dtype=self.config.activations_dtype,
                from_pt=self.config.from_pt
            )
            
            te_pretrained_config = PretrainedConfig.from_dict(model_configs[0]["text_encoder_config"])
            text_encoder = FlaxCLIPTextModel(
                te_pretrained_config,
                seed=self.config.seed,
                dtype=self.config.activations_dtype
            )
            
            tokenizer = CLIPTokenizer.from_pretrained(
                self.config.tokenizer_model_name_or_path,
                subfolder="tokenizer",
                dtype=self.config.activations_dtype,
            )

            scheduler = None
            if scheduler_class:
                scheduler = scheduler_class.from_config(model_configs[0]["scheduler_config"])
            
            pipeline_kwargs = {
            "unet" : unet,
            "vae" : vae,
            "text_encoder" : text_encoder,
            "scheduler" : scheduler,
            "tokenizer" : tokenizer,
            }

            if self.checkpoint_type == STABLE_DIFFUSION_CHECKPOINT:
                pipeline_kwargs["safety_checker"] = None
                pipeline_kwargs["feature_extractor"] = None
            else:
                te_pretrained_2_config = PretrainedConfig.from_dict(model_configs[0]["text_encoder_2_config"])
                text_encoder_2 = FlaxCLIPTextModel(
                    te_pretrained_2_config,
                    seed=self.config.seed,
                    dtype=self.config.activations_dtype
                )
            pipeline_kwargs["text_encoder_2"] = text_encoder_2
            pipeline_kwargs["tokenizer_2"] = tokenizer

            pipeline = pipeline_class(
            **pipeline_kwargs
            )
            self.pipeline = pipeline

            (unet_state,
            unet_state_mesh_shardings,
            vae_state,
            vae_state_shardings) = max_utils.get_states(pipeline) 
            self.train_states["unet_state"] = unet_state
            self.train_states["vae_state"] = vae_state
            self.state_shardings["unet_state_shardings"] = unet_state_mesh_shardings
            self.state_shardings["vae_state_shardings"] = vae_state_shardings
        else:
            self.load_diffusers_checkpoint()