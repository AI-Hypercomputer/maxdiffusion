"""
 Copyright 2024 Google LLC

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

import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from maxdiffusion.utils import load_image, make_image_grid
from PIL import Image
from maxdiffusion import FlaxStableDiffusionXLControlNetPipeline, FlaxControlNetModel
import cv2

def create_key(seed=0):
  return jax.random.PRNGKey(seed)

rng = create_key(0)

prompts = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompts = 'low quality, bad quality, sketches'

image = load_image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/1024px-Google_%22G%22_logo.svg.png")

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
  "diffusers/controlnet-canny-sdxl-1.0", from_pt=True, dtype=jnp.float32
)

pipe, params = FlaxStableDiffusionXLControlNetPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, revision="refs/pr/95", dtype=jnp.bfloat16
)

scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state

params["controlnet"] = controlnet_params

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())

prompt_ids = pipe.prepare_text_inputs([prompts] * num_samples)
negative_prompt_ids = pipe.prepare_text_inputs([negative_prompts] * num_samples)
processed_image = pipe.prepare_image_inputs([image] * num_samples)

p_params = replicate(params)
prompt_ids = shard(prompt_ids)
negative_prompt_ids = shard(negative_prompt_ids)
processed_image = shard(processed_image)

output = pipe(
    prompt_ids=prompt_ids,
    image=processed_image,
    params=p_params,
    prng_seed=rng,
    num_inference_steps=50,
    neg_prompt_ids=negative_prompt_ids,
    controlnet_conditioning_scale=1.0,
    jit=True,
).images

output_images = pipe.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
output_images = make_image_grid(output_images, num_samples // 4, 4)
output_images.save("generated_image.png")
