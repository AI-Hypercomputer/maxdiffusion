class MockPipeline:
    def __init__(self):
        pass

    def run_logic(self, prompt, num_videos_per_prompt=1, guidance_scale=3.0):
        # 2. Encode inputs (Text)
        if isinstance(prompt, str):
            _bs0 = 1
        elif isinstance(prompt, list):
            _bs0 = len(prompt)
        
        # Simulate encode_prompt output shapes
        prompt_embeds_shape = (_bs0 * num_videos_per_prompt, 10)
        negative_prompt_embeds_shape = (_bs0 * num_videos_per_prompt, 10)

        # 3. Prepare latents
        _bs = prompt_embeds_shape[0]
        batch_size = _bs // 2 if guidance_scale > 1.0 else _bs
        print(f"Evaluated true batch_size: {batch_size}")

        # 6. Prepare JAX State
        latents_jax_shape = (batch_size, 5)
        prompt_embeds_jax_shape = prompt_embeds_shape
        negative_prompt_embeds_jax_shape = negative_prompt_embeds_shape

        if guidance_scale > 1.0:
            prompt_embeds_jax_shape = (negative_prompt_embeds_jax_shape[0] + prompt_embeds_jax_shape[0], 10)
            latents_jax_shape = (latents_jax_shape[0] * 2, 5)
            
        print(f"latents_jax shape during generation: {latents_jax_shape}")
        print(f"prompt_embeds_jax shape during generation: {prompt_embeds_jax_shape}")
        print(f"Videos Decoded: {latents_jax_shape[0] - batch_size}")

p = MockPipeline()
p.run_logic(["Prompt"] * 8)
