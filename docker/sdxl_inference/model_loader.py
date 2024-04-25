from maxdiffusion import FlaxStableDiffusionXLPipeline

pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", revision="refs/pr/95", split_head_dim=True
)