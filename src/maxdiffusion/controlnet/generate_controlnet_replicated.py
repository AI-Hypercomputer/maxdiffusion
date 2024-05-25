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
from maxdiffusion import FlaxStableDiffusionControlNetPipeline, FlaxControlNetModel
def create_key(seed=0):
  return jax.random.PRNGKey(seed)


rng = create_key(0)

# get canny image
canny_image = load_image(
  "https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_10_output_0.jpeg"
)

prompts = "best quality, extremely detailed"
negative_prompts = "monochrome, lowres, bad anatomy, worst quality, low quality"

# load control net and stable diffusion v1-5
controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
  "lllyasviel/sd-controlnet-canny", from_pt=True, dtype=jnp.float32
)
pipe, params = FlaxStableDiffusionControlNetPipeline.from_pretrained(
  "runwayml/stable-diffusion-v1-5", controlnet=controlnet, revision="flax", dtype=jnp.float32
)
params["controlnet"] = controlnet_params

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())

prompt_ids = pipe.prepare_text_inputs([prompts] * num_samples)
negative_prompt_ids = pipe.prepare_text_inputs([negative_prompts] * num_samples)
processed_image = pipe.prepare_image_inputs([canny_image] * num_samples)

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
    jit=True,
).images

output_images = pipe.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
output_images = make_image_grid(output_images, num_samples // 4, 4)
output_images.save("generated_image.png")
