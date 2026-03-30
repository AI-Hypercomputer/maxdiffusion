import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx

from maxdiffusion import pyconfig
from maxdiffusion.utils import logging
from maxdiffusion import max_utils
from maxdiffusion.models.ltx2.transformer_ltx2 import LTX2VideoTransformer3DModel
from maxdiffusion.max_utils import get_precision

logger = logging.get_logger(__name__)

def get_dummy_ltx2_inputs(batch_size, dtype):
    rng = jax.random.key(0)
    # LTX-2 121 frames 512x768 -> latent 16x16x24
    latents = jax.random.normal(rng, (batch_size, 128, 16, 16, 24), dtype=dtype)
    audio_latents = None
    timestep = jnp.array(500.0, dtype=jnp.float32)
    # Gemma dim=3072, sequence=128
    prompt_embeds = jax.random.normal(rng, (batch_size, 128, 3072), dtype=dtype)
    audio_prompt_embeds = None
    encoder_attention_mask = jnp.ones((batch_size, 128), dtype=jnp.int32)
    audio_encoder_attention_mask = None

    return latents, audio_latents, timestep, prompt_embeds, audio_prompt_embeds, encoder_attention_mask, audio_encoder_attention_mask

def calibrate_fbs(config):
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    
    rng = jax.random.key(config.seed)
    rngs = nnx.Rngs(rng)
    
    # 1. Load config
    ltx2_config_dict = LTX2VideoTransformer3DModel.load_config(config.pretrained_model_name_or_path, subfolder="transformer")
    if ltx2_config_dict.get("activation_fn") == "gelu-approximate":
        ltx2_config_dict["activation_fn"] = "gelu"
    
    ltx2_config_dict["scan_layers"] = getattr(config, "scan_layers", True)
    ltx2_config_dict["mesh"] = mesh
    ltx2_config_dict["dtype"] = config.activations_dtype
    ltx2_config_dict["weights_dtype"] = config.weights_dtype
    ltx2_config_dict["attention_kernel"] = config.attention
    ltx2_config_dict["precision"] = get_precision(config)
    ltx2_config_dict["flash_block_sizes"] = max_utils.get_flash_block_sizes(config)
    ltx2_config_dict["remat_policy"] = config.remat_policy
    ltx2_config_dict["names_which_can_be_saved"] = config.names_which_can_be_saved
    ltx2_config_dict["names_which_can_be_offloaded"] = config.names_which_can_be_offloaded
    
    print(f"Creating model with flash_block_sizes: {ltx2_config_dict['flash_block_sizes']}")
    
    with mesh:
        # Standard initialization
        transformer = LTX2VideoTransformer3DModel(**ltx2_config_dict, rngs=rngs)
        
        # Shard the model
        graphdef, state, rest_of_state = nnx.split(transformer, nnx.Param, ...)
        def _add_sharding_rule(vs: nnx.VariableState, logical_axis_rules):
            vs.sharding_rules = logical_axis_rules
            return vs
        
        p_add_sharding_rule = partial(_add_sharding_rule, logical_axis_rules=config.logical_axis_rules)
        state_sharded = jax.tree.map(p_add_sharding_rule, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))
        pspecs = nnx.get_partition_spec(state_sharded)
        sharded_state = jax.lax.with_sharding_constraint(state_sharded, pspecs)

        from maxdiffusion.pipelines.ltx2.ltx2_pipeline import transformer_forward_pass
        
        # Define forward_pass strictly bounded by parameters, just like pipeline does
        
        # Batch size handling
        batch_size = config.global_batch_size_to_train_on
        latents, audio_latents, timestep, prompt_embeds, audio_prompt_embeds, encoder_attention_mask, audio_encoder_attention_mask = get_dummy_ltx2_inputs(batch_size, config.activations_dtype)
        
        data_sharding = NamedSharding(mesh, P())
        if config.global_batch_size_to_train_on // config.per_device_batch_size == 0:
             data_sharding = jax.sharding.NamedSharding(mesh, P(*config.data_sharding))
        
        # Add unconditional latents for CFG
        double_latents = jnp.concatenate([latents, latents], axis=0)
        double_prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds], axis=0)
        double_encoder_attention_mask = jnp.concatenate([encoder_attention_mask, encoder_attention_mask], axis=0)

        double_latents = jax.device_put(double_latents, data_sharding)
        timestep = jax.device_put(timestep, data_sharding)
        double_prompt_embeds = jax.device_put(double_prompt_embeds, data_sharding)
        double_encoder_attention_mask = jax.device_put(double_encoder_attention_mask, data_sharding)
        
        print("Compiling transformer forward pass...")
        start_compile = time.perf_counter()
        
        # Using 50 runs to ensure XLA completely settles
        num_runs = 50
        
        # Provide exactly what transformer_forward_pass needs
        latent_num_frames = 16
        latent_height = 16
        latent_width = 24
        audio_num_frames = 0
        fps = 24.0

        _ = transformer_forward_pass(
            graphdef, sharded_state, double_latents,
            None, # audio_latents
            timestep, double_prompt_embeds,
            None, # audio_encoder_hidden_states
            double_encoder_attention_mask,
            None, # audio_encoder_attention_mask
            do_classifier_free_guidance=True,
            guidance_scale=1.5,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            audio_num_frames=audio_num_frames,
            fps=fps
        )
        
        # Ensure compiled
        import jax.tree_util as jtu
        jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, _)
        
        compile_time = time.perf_counter() - start_compile
        print(f"Compilation finished. Time: {compile_time:.4f}s")
        
        # Benchmarking
        print(f"Starting Benchmarking ({num_runs} runs)...")
        total_time = 0.0
        
        for i in range(num_runs):
            start = time.perf_counter()
            _ = transformer_forward_pass(
                graphdef, sharded_state, double_latents,
                None, 
                timestep, double_prompt_embeds,
                None, 
                double_encoder_attention_mask,
                None, 
                do_classifier_free_guidance=True,
                guidance_scale=1.5,
                latent_num_frames=latent_num_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                audio_num_frames=audio_num_frames,
                fps=fps
            )
            # block until ready
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, _)
            
            step_time = time.perf_counter() - start
            if i > 5: # Ignore first few runs for warmup
                total_time += step_time
            print(f"[Tuning] Run {i+1}/{num_runs} - E2E Step time: {step_time*1000:.2f} ms")
            
        print(f"Average pure diffusion cycle (after warmup): {(total_time/(num_runs-6))*1000:.2f} ms")
        
if __name__ == "__main__":
    config = pyconfig.initialize(sys.argv)
    calibrate_fbs(config)
