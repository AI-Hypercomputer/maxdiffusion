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

from maxdiffusion import generate
import jax
import numpy as np
from maxdiffusion.metrics.fid import fid_score
from typing import Sequence
from absl import app
from maxdiffusion import pyconfig
import torch
import pandas as pd
from tempfile import TemporaryFile
import pathlib




def load_captions(file_path):
    captions_df = pd.read_csv(file_path, delimiter='\t', header=0, names=['image_id','id', 'caption'])
    return captions_df

def load_stats(file_path):
    images_data = np.load(file_path)
    sigma = images_data['sigma']
    mu = images_data['mu']
    return sigma, mu

# # File paths
captions_file = '/home/shahrokhi/coco2014/val2014_30k.tsv'
# stat_file = '/home/shahrokhi/coco2014/val2014_30k_stats.npz'

# # Load captions
# captions_df = load_captions(captions_file)
# # Load images
# mu, sigma = load_stats(stat_file)


def generate_images(config):
    images = generate.run(config)
    #numpy_images = np.array(images)
    return images

#generate_images(pyconfig.config)

def eval(config):
    captions_df = load_captions(captions_file)
    batch_size = config.per_device_batch_size * jax.device_count()

    #inference happenning here: 

    while(not captions_df.empty):
        if (captions_df.size >= batch_size):
            captions_df['caption'][:batch_size].to_csv('prompts.txt', header=False, index=False)
            captions_df['id'][:batch_size].to_csv('ids.txt', header=False,index=False)
            captions_df = captions_df.iloc[batch_size:]
        else:
            captions_df['caption'].to_csv('prompts.txt', header=False, index=False)
            captions_df['id'][:batch_size].to_csv('ids.txt', header=False,index=False)
            captions_df = pd.DataFrame()
        images = generate_images(config)
        device = torch.device('cpu')
        breakpoint()
        paths = ["/home/shahrokhi/maxdiffusion","/home/shahrokhi/maxdiffusion/outputfile"]
        
        fid_score.save_fid_stats(paths, batch_size, device, 2048)
        breakpoint()
    #     with open(config.image_ids, "r") as ids_file:
    #         image_ids = ids_file.readlines()
    #         for image_id in image_ids:
    #             pathlib.Path(f"image_{image_id}.png").unlink()

    
    device = torch.device('cpu')
    paths = ["/home/shahrokhi/maxdiffusion/outputfile.npz", "/home/shahrokhi/maxdiffusion/outputfile.npz"]
    fid = fid_score.calculate_fid_given_paths(paths, batch_size, device, 2048 )
    print("fid is : " + str(fid))





# def generate_images(unet_state, vae_state, params, rng, config, batch_size, pipeline):
#     images = generate.run_inference(unet_state, vae_state, params, rng, config, batch_size, pipeline)
#     numpy_images = np.array(images)
#     return numpy_images

# def eval(config):
#     rng = jax.random.PRNGKey(config.seed)
    
# def eval_step(batch, eval_rng, cache_latents_text_encoder_outputs, config):
#     _, gen_dummy_rng = jax.random.split(eval_rng)
#     sample_rng, new_eval_rng = jax.random.split(gen_dummy_rng)
#     devices_array = create_device_mesh(config)
#     mesh = Mesh(devices_array, config.mesh_axes)
#     pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
#         config.pretrained_model_name_or_path,revision=config.revision,
#         dtype=weight_dtype,
#         safety_checker=None,
#         feature_extractor=None,
#         from_pt=config.from_pt,
#         split_head_dim=config.split_head_dim,
#         attention_kernel=config.attention,
#         flash_block_sizes=flash_block_sizes,
#         mesh=mesh,
#     )
#     (unet_state,
#     unet_state_mesh_shardings,
#     vae_state, vae_state_mesh_shardings) = max_utils.get_states(mesh,
#                                                                 None, sample_rng, config,
#                                                                 pipeline, params["unet"], params["vae"], training=False)
#     del params["vae"]
#     del params["unet"]

#     generated_images = generate_images(unet_state, vae_state, params, eval_rng, config, batch_size, pipeline)
#     breakpoint()
#     # Calculate FID score
#     fid_score = calculate_fid_score(batch["pixel_values"], generated_images)

#     return fid_score


def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    eval(config)
if __name__ == "__main__":
    app.run(main)