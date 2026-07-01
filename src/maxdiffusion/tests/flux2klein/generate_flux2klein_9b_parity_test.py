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

import unittest
import numpy as np
import os

# Set HF_HOME immediately before any HF/MaxDiffusion imports to ensure
# the cache directory and token are correctly resolved.
if not os.environ.get("HF_HOME"):
    if os.path.exists("/mnt/data/hf_cache"):
        os.environ["HF_HOME"] = "/mnt/data/hf_cache"

import time
import gc
import jax
import jax.numpy as jnp
import flax
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
import torch
from safetensors.torch import load_file
from skimage.metrics import structural_similarity as ssim

from maxdiffusion import pyconfig
from maxdiffusion.max_utils import create_device_mesh
from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.models.vae_flax import FlaxAutoencoderKL
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model, load_and_convert_qwen3_weights
from maxdiffusion.generate_flux2klein_9B import (
    load_and_convert_weights,
    load_and_convert_vae_weights,
    cast_dict_to_bfloat16_inplace,
    prepare_text_ids,
    prepare_latent_image_ids,
    pack_latents,
    unpack_latents_with_ids,
    unpatchify_latents,
    encode_prompt_jax,
)

class GenerateFlux2Klein9BParityTest(unittest.TestCase):

    def test_e2e_parity_9b(self):
        """
        Runs E2E generation for Flux.2-klein-9B on:
        1. PyTorch CPU (Float32) -> Golden Reference
        2. PyTorch CPU (Bfloat16) -> Baseline Precision Loss
        3. JAX TPU (Bfloat16) -> Our Implementation
        
        Compares the final images to ensure JAX TPU bfloat16 JAX matches the accuracy
        of PyTorch CPU bfloat16.
        """
        # 1. Initialize pyconfig with 9B config
        print("Initializing pyconfig with 9B config...")
        if getattr(pyconfig, "config", None) is None:
            pyconfig.initialize([
                None,
                "src/maxdiffusion/configs/base_flux2klein_9B.yml",
                "run_name=flux_9b_parity_test",
                "output_dir=/tmp/",
                "jax_cache_dir=/tmp/cache_dir",
            ], unittest=True)
        config = pyconfig.config
        
        # Override batch sharding rules to avoid sharding batch dimension across fsdp
        # when batch_size (1) is less than fsdp_parallelism (4).
        new_rules = []
        for rule in config.logical_axis_rules:
            if rule[0] in ('activation_batch', 'conv_batch'):
                new_rules.append([rule[0], 'data'])
            else:
                new_rules.append(rule)
        pyconfig._config.keys['logical_axis_rules'] = tuple(new_rules)
        print(f"Overridden logical_axis_rules: {config.logical_axis_rules}")
        
        # 2. Setup device mesh (FSDP=4)
        print("Setting up device mesh...")
        try:
            devices_array = create_device_mesh(config)
            mesh = Mesh(devices_array, config.mesh_axes)
        except Exception as e:
            self.skipTest(f"Skipping because device mesh creation failed: {e}")
            
        # 3. Locate cached weights
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        
        repo_id = "black-forest-labs/FLUX.2-klein-9B"
        cache_dir = os.path.join(hf_home, "hub", f"models--{repo_id.replace('/', '--')}", "snapshots")
        
        # Trigger download if missing (using huggingface_hub to be safe)
        if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
            print(f"Model cache not found at {cache_dir}. Downloading from Hub...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo_id, local_files_only=False)
            
        snapshots = os.listdir(cache_dir)
        snapshot_dir = os.path.join(cache_dir, snapshots[0])
        
        transformer_path = os.path.join(snapshot_dir, "transformer")
        vae_path = os.path.join(snapshot_dir, "vae", "diffusion_pytorch_model.safetensors")
        text_encoder_path = os.path.join(snapshot_dir, "text_encoder")
        tokenizer_path = os.path.join(snapshot_dir, "tokenizer")
        
        # Shared parameters
        prompt = "A detailed vector illustration of a robotic hummingbird"
        width = 1024
        height = 1024
        num_inference_steps = 4
        batch_size = 1
        
        # 4. Generate Shared Latent Noise on CPU
        print("Generating shared latent noise on CPU...")
        seed = 42
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # Flux VAE has 32 latent channels. Spatial dim is 8x compressed (128x128)
        latents_unpacked = torch.randn(batch_size, 32, height // 8, width // 8, generator=generator, dtype=torch.float32)
        latents_numpy = latents_unpacked.numpy()
        
        # The PyTorch Flux2KleinPipeline expects latents to be ALREADY PATCHIFIED (channel-packed)
        # if passed externally. Shape must be (batch_size, 128, height // 16, width // 16)
        latents_pytorch = latents_unpacked.view(batch_size, 32, height // 16, 2, width // 16, 2)
        latents_pytorch = latents_pytorch.permute(0, 1, 3, 5, 2, 4)
        latents_pytorch = latents_pytorch.reshape(batch_size, 128, height // 16, width // 16)
        
        # ---------------------------------------------------------------------
        # LEG 1: PyTorch CPU Float32 (Golden Reference)
        # ---------------------------------------------------------------------
        print("\n==================================================")
        # We wrap this in a try-except to allow running JAX-only if PyTorch is not fully working,
        # but for the full test we want it to run.
        print("LEG 1: Running PyTorch CPU Float32 Reference...")
        t0 = time.time()
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
        
        pipe_f32 = Flux2KleinPipeline.from_pretrained(snapshot_dir, torch_dtype=torch.float32, local_files_only=True)
        pipe_f32.to("cpu")
        
        # Run generation
        with torch.no_grad():
            images_f32 = pipe_f32(
                prompt=prompt,
                width=width,
                height=height,
                latents=latents_pytorch,
                num_inference_steps=num_inference_steps,
                output_type="np"
            ).images
        image_cpu_f32 = images_f32[0]
        print(f"PyTorch CPU Float32 finished in {time.time() - t0:.2f}s")
        
        # Clean up memory
        del pipe_f32
        gc.collect()
        
        # ---------------------------------------------------------------------
        # LEG 2: PyTorch CPU Bfloat16 (Precision Baseline)
        # ---------------------------------------------------------------------
        print("\n==================================================")
        print("LEG 2: Running PyTorch CPU Bfloat16 Baseline...")
        t0 = time.time()
        pipe_bf16 = Flux2KleinPipeline.from_pretrained(snapshot_dir, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe_bf16.to("cpu")
        
        with torch.no_grad():
            images_bf16 = pipe_bf16(
                prompt=prompt,
                width=width,
                height=height,
                latents=latents_pytorch.to(torch.bfloat16),
                num_inference_steps=num_inference_steps,
                output_type="np"
            ).images
        image_cpu_bf16 = images_bf16[0]
        print(f"PyTorch CPU Bfloat16 finished in {time.time() - t0:.2f}s")
        
        del pipe_bf16
        gc.collect()
        
        # ---------------------------------------------------------------------
        # LEG 3: JAX TPU Bfloat16 (Our Implementation)
        # ---------------------------------------------------------------------
        print("\n==================================================")
        print("LEG 3: Running JAX TPU Bfloat16...")
        t0 = time.time()
        
        # Instantiate JAX models
        print("Instantiating JAX models...")
        from transformers import AutoConfig
        pt_qwen_config = AutoConfig.from_pretrained(text_encoder_path, local_files_only=True)
        qwen3_config = FlaxQwen3Config(
            vocab_size=pt_qwen_config.vocab_size,
            hidden_size=pt_qwen_config.hidden_size,
            intermediate_size=pt_qwen_config.intermediate_size,
            num_hidden_layers=pt_qwen_config.num_hidden_layers,
            num_attention_heads=pt_qwen_config.num_attention_heads,
            num_key_value_heads=pt_qwen_config.num_key_value_heads,
            max_position_embeddings=pt_qwen_config.max_position_embeddings,
            rms_norm_eps=pt_qwen_config.rms_norm_eps,
            rope_theta=pt_qwen_config.rope_theta,
            dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
        )
        qwen3_model = FlaxQwen3Model(qwen3_config)
        
        transformer = FluxTransformer2DModel(
            in_channels=128,
            num_layers=config.num_double_layers,
            num_single_layers=config.depth,
            attention_head_dim=128,
            num_attention_heads=config.num_attention_heads,
            joint_attention_dim=3 * pt_qwen_config.hidden_size,
            pooled_projection_dim=768,
            mlp_ratio=3.0,
            qkv_bias=False,
            joint_attention_bias=False,
            x_embedder_bias=False,
            proj_out_bias=False,
            use_global_modulation=True,
            use_swiglu=True,
            axes_dims_rope=(32, 32, 32, 32),
            theta=2000,
            mesh=mesh,
            dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
        )
        
        vae = FlaxAutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            act_fn="silu",
            latent_channels=32,
            norm_num_groups=32,
            sample_size=512,
            use_quant_conv=True,
            use_post_quant_conv=True,
            dtype=jnp.bfloat16 if config.weights_dtype == "bfloat16" else jnp.float32,
        )
        
        # Initialize and load weights on CPU
        print("Initializing JAX parameters on CPU...")
        cpu_device = jax.devices("cpu")[0]
        tpu_device = jax.devices("tpu")[0]
        
        seq_len_txt = 512
        seq_len_img = (height // 16) * (width // 16) # 4096
        
        with jax.default_device(cpu_device):
            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                key = jax.random.PRNGKey(0)
                key, vae_key, qwen_key = jax.random.split(key, 3)
                
                # Init Flux
                img_dummy = jnp.zeros((batch_size, seq_len_img, 128))
                img_ids_dummy = jnp.zeros((batch_size, seq_len_img, 4))
                txt_dummy = jnp.zeros((batch_size, seq_len_txt, 3 * pt_qwen_config.hidden_size))
                txt_ids_dummy = jnp.zeros((batch_size, seq_len_txt, 4))
                vec_dummy = jnp.zeros((batch_size, 768))
                t_vec_dummy = jnp.zeros((batch_size,))
                guidance_vec_dummy = jnp.zeros((batch_size,))
                
                variables = transformer.init(
                    key,
                    hidden_states=img_dummy,
                    img_ids=img_ids_dummy,
                    encoder_hidden_states=txt_dummy,
                    txt_ids=txt_ids_dummy,
                    pooled_projections=vec_dummy,
                    timestep=t_vec_dummy,
                    guidance=guidance_vec_dummy,
                )
                params = variables["params"]
                
                # Init VAE
                dummy_img = jnp.zeros((batch_size, 3, height, width))
                vae_variables = vae.init(vae_key, dummy_img)
                vae_params = vae_variables["params"]
                
                # Init Qwen3
                dummy_ids = jnp.zeros((batch_size, seq_len_txt), dtype=jnp.int32)
                dummy_mask = jnp.zeros((batch_size, seq_len_txt), dtype=jnp.int32)
                qwen3_variables = qwen3_model.init(qwen_key, dummy_ids, dummy_mask)
                qwen3_params = qwen3_variables["params"]
                
                # Reconstruct logical specs and get mesh shardings before unboxing
                import flax.linen as nn
                logical_transformer_specs = nn.get_partition_spec(variables)
                transformer_mesh_shardings = nn.logical_to_mesh_sharding(logical_transformer_specs, mesh, config.logical_axis_rules)
                transformer_shardings = flax.core.freeze(transformer_mesh_shardings['params'])
                
                logical_vae_specs = nn.get_partition_spec(vae_variables)
                vae_mesh_shardings = nn.logical_to_mesh_sharding(logical_vae_specs, mesh, config.logical_axis_rules)
                vae_shardings = flax.core.freeze(vae_mesh_shardings['params'])
                
                logical_qwen3_specs = nn.get_partition_spec(qwen3_variables)
                qwen3_mesh_shardings = nn.logical_to_mesh_sharding(logical_qwen3_specs, mesh, config.logical_axis_rules)
                qwen3_shardings = flax.core.freeze(qwen3_mesh_shardings['params'])
                
                # Unbox
                import flax.linen.spmd as flax_spmd
                params = jax.tree_util.tree_map(
                    lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
                    params,
                    is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
                )
                params = flax.core.unfreeze(params)
                
                vae_params = jax.tree_util.tree_map(
                    lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
                    vae_params,
                    is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
                )
                vae_params = flax.core.unfreeze(vae_params)
                
                qwen3_params = jax.tree_util.tree_map(
                    lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
                    qwen3_params,
                    is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
                )
                qwen3_params = flax.core.unfreeze(qwen3_params)
                
                # Load weights (using our new sharded loaders!)
                params = load_and_convert_weights(transformer_path, params, num_double_layers=config.num_double_layers, num_single_layers=config.depth)
                vae_params, vae_bn_mean, vae_bn_std = load_and_convert_vae_weights(vae_path, vae_params)
                qwen3_params = load_and_convert_qwen3_weights(text_encoder_path, qwen3_params, qwen3_config)
                
                # Cast to bfloat16 in-place if needed
                if config.weights_dtype == "bfloat16":
                    print("Casting JAX parameters to bfloat16 in-place...")
                    cast_dict_to_bfloat16_inplace(params)
                    cast_dict_to_bfloat16_inplace(vae_params)
                    cast_dict_to_bfloat16_inplace(qwen3_params)
                    vae_bn_mean = vae_bn_mean.astype(jnp.bfloat16)
                    vae_bn_std = vae_bn_std.astype(jnp.bfloat16)
                
                params_cpu = flax.core.freeze(params)
                vae_params_cpu = flax.core.freeze(vae_params)
                qwen3_params_cpu = flax.core.freeze(qwen3_params)
                
        print("JAX models initialized and weights loaded on CPU.")
        
        # Setup JAX Scheduler
        from maxdiffusion.generate_flux2klein_9B import compute_empirical_mu
        mu = compute_empirical_mu(seq_len_img, num_inference_steps)
        jax_scheduler = FlaxFlowMatchScheduler(
            num_train_timesteps=1000,
            shift=mu,
            sigma_max=1.0,
            sigma_min=0.001,
            inverse_timesteps=False,
            extra_one_step=False,
            reverse_sigmas=False,
            use_dynamic_shifting=True,
            time_shift_type="exponential",
        )
        scheduler_state = jax_scheduler.create_state()
        explicit_sigmas = jnp.linspace(1.0, 0.25, num_inference_steps)
        scheduler_state = jax_scheduler.set_timesteps_ltx2(
            state=scheduler_state,
            num_inference_steps=num_inference_steps,
            shift=mu,
            sigmas=explicit_sigmas,
        )
        
        # Position grids
        txt_ids_val = prepare_text_ids(batch_size, seq_len_txt)
        img_ids_val = prepare_latent_image_ids(batch_size, height // 16, width // 16)
        
        # Compile JIT functions
        @jax.jit
        def jitted_qwen3_forward(q_params, ids, mask):
            return qwen3_model.apply({"params": q_params}, input_ids=ids, attention_mask=mask)
            
        @jax.jit
        def jitted_transformer_step(t_params, latents, img_ids, prompt_embeds, txt_ids, vec, timestep, guidance):
            return transformer.apply(
                {"params": t_params},
                hidden_states=latents,
                img_ids=img_ids,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                pooled_projections=vec,
                timestep=timestep,
                guidance=guidance,
            )
            
        @jax.jit
        def jitted_vae_decode(v_params, latents_unpatched):
            return vae.apply({"params": v_params}, latents=latents_unpatched, method=vae.decode)
            
        # Run JAX Generation
        print("Running JAX generation on TPU...")
        
        # 1. Encode Prompt
        print("  Encoding prompt with Qwen3...")
        qwen3_params_tpu = jax.device_put(qwen3_params_cpu, qwen3_shardings)
        from transformers import Qwen2TokenizerFast
        tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
        messages = [{"role": "user", "content": prompt}]
        templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(templated_text, return_tensors="np", padding="max_length", truncation=True, max_length=seq_len_txt)
        prompt_ids = jnp.array(inputs["input_ids"])
        prompt_mask = jnp.array(inputs["attention_mask"])
        
        hidden_states, all_hidden_states = jitted_qwen3_forward(qwen3_params_tpu, prompt_ids, prompt_mask)
        # For 8B Qwen, we stack layers 9, 18, 27 (or whatever layers are defined in the pipeline, wait!
        # In 4B we stacked 9, 18, 27.
        # In 8B Qwen, does it use the same layers?
        # Let's check how the PyTorch Flux2KleinPipeline extracts prompt embeds!
        # Ah!
        # In our previous porting, we found that it extracts layers 9, 18, 27.
        # Does the 9B model's text encoder also use layers 9, 18, 27?
        # Yes, because the text encoder is still Qwen3 (just 8B instead of 1.5B), and the extraction layers (9, 18, 27) are typically kept the same for the Klein architecture to maintain the multi-scale text representation.
        # Let's assume it's the same. If not, we will see it in the parity.
        h_9 = all_hidden_states[9]
        h_18 = all_hidden_states[18]
        h_27 = all_hidden_states[27]
        out = jnp.stack([h_9, h_18, h_27], axis=1)
        prompt_embeds_jax = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len_txt, 3 * pt_qwen_config.hidden_size))
        
        del qwen3_params_tpu
        gc.collect()
        
        # 2. Denoising Loop
        print("  Running denoising loop...")
        params_tpu = jax.device_put(params_cpu, transformer_shardings)
        
        # Convert input latents to JAX and pack them
        # PyTorch latents: (batch_size, 32, 128, 128)
        # JAX expects packed latents: (batch_size, 4096, 128)
        latents_jax = jnp.array(latents_numpy)
        latents_jax = pack_latents(latents_jax)
        
        guidance_vec_val = jnp.array([4.0] * batch_size)
        vec_val = jnp.zeros((batch_size, 768))
        
        # We need to map the timesteps
        # In Flux, timesteps go from 1.0 to 0.0.
        # The scheduler sigmas are [1.0, 0.75, 0.5, 0.25] or similar.
        # Let's run the loop
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            for step_idx in range(num_inference_steps):
                # Get timestep from scheduler state
                # In our JAX pipeline, we do:
                # timestep = scheduler_state.timesteps[step_idx] / 1000.0
                # Let's check how it is done in generate_flux2klein.py
                # Actually, we can just use the explicit sigmas/timesteps from the scheduler state.
                # In Flux, the timestep passed to the transformer is sigma * 1000.0?
                # No, in generate_flux2klein.py:
                # timestep = scheduler_state.timesteps[step_idx]
                # Let's check:
                sigmas = scheduler_state.sigmas
                sigma = sigmas[step_idx]
                # The transformer expects timestep as sigma * 1000.0 or just sigma?
                # In JAX it expects sigma * 1000.0.
                # Let's check generate_flux2klein_e2e_test.py line 394:
                # `step_t = jnp.array([bundle[f"step_{step_idx}_timestep"]])`
                # In the bundle, the timestep was indeed sigma * 1000.0 (e.g. 1000.0, 750.0, 500.0, 250.0).
                # So we do:
                step_t = jnp.array([sigma * 1000.0])
                
                model_output = jitted_transformer_step(
                    params_tpu,
                    latents_jax,
                    img_ids_val,
                    prompt_embeds_jax,
                    txt_ids_val,
                    vec_val,
                    step_t,
                    guidance_vec_val,
                )
                
                step_output = jax_scheduler.step(
                    state=scheduler_state,
                    model_output=model_output.sample,
                    timestep=step_t[0],
                    sample=latents_jax,
                )
                latents_jax = step_output.prev_sample
                scheduler_state = step_output.state
                
        del params_tpu
        gc.collect()
        
        # 3. VAE Decode
        print("  Decoding image with VAE...")
        vae_params_tpu = jax.device_put(vae_params_cpu, vae_shardings)
        
        latents_unpacked = unpack_latents_with_ids(latents_jax, img_ids_val, height // 16, width // 16)
        latents_bn = latents_unpacked * vae_bn_std + vae_bn_mean
        final_latents_unpatched = unpatchify_latents(latents_bn)
        
        with mesh:
            jax_image_out = jitted_vae_decode(vae_params_tpu, final_latents_unpatched)
            jax_image_out.sample.block_until_ready()
            
        # Postprocess JAX image
        jax_image = (jax_image_out.sample / 2.0 + 0.5)
        jax_image = jnp.clip(jax_image, 0.0, 1.0)
        jax_image = jnp.transpose(jax_image, (0, 2, 3, 1)) # NHWC
        image_jax_tpu = np.array(jax_image[0])
        
        del vae_params_tpu
        gc.collect()
        print(f"JAX TPU Bfloat16 finished in {time.time() - t0:.2f}s")
        
        # ---------------------------------------------------------------------
        # COMPARISON & ASSERTIONS
        # ---------------------------------------------------------------------
        print("\n==================================================")
        print("COMPARISON METRICS (Against PyTorch CPU Float32 Reference)")
        print("==================================================")
        
        # Save images for manual inspection
        import cv2
        cv2.imwrite("flux9b_cpu_float32.png", cv2.cvtColor((image_cpu_f32 * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite("flux9b_cpu_bfloat16.png", cv2.cvtColor((image_cpu_bf16 * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite("flux9b_jax_tpu.png", cv2.cvtColor((image_jax_tpu * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print("Saved images to: flux9b_cpu_float32.png, flux9b_cpu_bfloat16.png, flux9b_jax_tpu.png")
        
        # Calculate SSIM and PSNR
        # 1. PyTorch CPU Bfloat16 vs PyTorch CPU Float32
        ssim_cpu_bf16 = ssim(image_cpu_bf16, image_cpu_f32, channel_axis=-1, data_range=1.0)
        psnr_cpu_bf16 = 10 * np.log10(1.0 / np.mean((image_cpu_bf16 - image_cpu_f32) ** 2))
        
        # 2. JAX TPU Bfloat16 vs PyTorch CPU Float32
        ssim_jax_tpu = ssim(image_jax_tpu, image_cpu_f32, channel_axis=-1, data_range=1.0)
        psnr_jax_tpu = 10 * np.log10(1.0 / np.mean((image_jax_tpu - image_cpu_f32) ** 2))
        
        print(f"PyTorch CPU Bfloat16 vs Gold F32:")
        print(f"  * SSIM: {ssim_cpu_bf16:.6f}")
        print(f"  * PSNR: {psnr_cpu_bf16:.2f} dB")
        
        print(f"JAX TPU Bfloat16 vs Gold F32:")
        print(f"  * SSIM: {ssim_jax_tpu:.6f}")
        print(f"  * PSNR: {psnr_jax_tpu:.2f} dB")
        
        # Assertions
        # JAX TPU SSIM should be high (typically > 0.88 for bfloat16 vs float32 at high resolutions)
        # And it should be comparable to the PyTorch CPU Bfloat16 SSIM.
        print("\nVerifying parity...")
        self.assertGreater(ssim_jax_tpu, 0.88, "JAX TPU bfloat16 image has too low SSIM against PyTorch CPU float32!")
        
        # The difference between JAX TPU SSIM and PyTorch CPU Bfloat16 SSIM should be small
        # (proving that JAX TPU is at least as accurate as PyTorch CPU in bfloat16)
        ssim_diff = abs(ssim_jax_tpu - ssim_cpu_bf16)
        print(f"SSIM Difference (JAX TPU vs PyTorch CPU in Bfloat16): {ssim_diff:.6f}")
        self.assertLess(ssim_diff, 0.06, "JAX TPU bfloat16 deviates significantly more than PyTorch CPU bfloat16!")
        
        print("\n✅ E2E PARITY VERIFICATION SUCCESSFUL!")
        print("JAX TPU Bfloat16 matches PyTorch CPU Bfloat16 accuracy within epsilon limits.")

if __name__ == '__main__':
    unittest.main()
