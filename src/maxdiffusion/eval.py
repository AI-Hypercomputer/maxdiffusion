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
from maxdiffusion.metrics.fid import inception
from maxdiffusion.metrics.fid import fid_score
from maxdiffusion.metrics.clip.clip_encoder import CLIPEncoder
from typing import Sequence
from absl import app
from maxdiffusion import pyconfig
import torch
import pandas as pd
from tempfile import TemporaryFile
import pathlib
import os

import jax.numpy as jnp
import flax
import functools

from transformers import CLIPProcessor, CLIPModel
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from PIL import Image

# if self.validation_run_clip:
#             if self.clip_encoder is None:
#                 self.clip_encoder = CLIPEncoder(clip_version=self.clip_version, cache_dir=self.clip_cache_dir)
#             for prompt, x_sampler in zip(prompts, x_samples):
#                 # TODO(ahmadki): this is not efficient but clip model expects a PIL image,
#                 # modify the clip encoder so we can use raw tensors instead
#                 img = self.to_pil_image(x_sampler)
#                 score = self.clip_encoder.get_clip_score(prompt, img)
#                 self.validation_clip_scores.append(score)

def load_captions(file_path):
    captions_df = pd.read_csv(file_path, delimiter='\t', header=0, names=['image_id','id', 'caption'])
    return captions_df

def load_stats(file_path):
    images_data = np.load(file_path)
    sigma = images_data['sigma']
    mu = images_data['mu']
    return sigma, mu

def calculate_clip(images, prompts, config):
    clip_encoder = CLIPEncoder(cache_dir=config.clip_cache_dir)
    assert len(images) == len(prompts)
    clip_scores = []
    for i in range(0,len(images)):
        score = clip_encoder.get_clip_score(prompts[i], images[i])
        clip_scores.append(score)
    clip_scores = torch.cat(clip_scores, 0)
    clip_score = np.mean(clip_scores.detach().cpu().numpy())
    print("clip score is" + str(clip_score))
    return clip_score

def load_images(path):
    images = []
    for f in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, f))
        images.append(img)
    return images

def eval(config):
    batch_size = config.per_device_batch_size * jax.device_count()

    #inference happenning here: first generate the images
    generate.run(config)

    # calculating CLIP:
    images = load_images(config.images_directory)
    prompts = load_captions(config.caption_coco_file)['caption']
    calculate_clip(images, prompts, config)

    # calculating FID:
    rng = jax.random.PRNGKey(0)
    
    model = inception.InceptionV3(pretrained=True)
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))

    apply_fn = jax.jit(functools.partial(model.apply, train=False))
    mu, sigma = fid_score.compute_statistics_with_mmap(config.images_directory, "/tmp/temp.dat", params, apply_fn, batch_size, (config.resolution, config.resolution))
    os.makedirs(config.stat_output_directory, exist_ok=True)
    np.savez(os.path.join(config.stat_output_directory, 'stats'), mu=mu, sigma=sigma)

    mu1, sigma1 = fid_score.compute_statistics(config.stat_output_file, params, apply_fn, batch_size,)
    mu2, sigma2 = fid_score.compute_statistics(config.stat_coco_file, params, apply_fn, batch_size,)

    fid = fid_score.compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6)
    print("fid score is : " + str(fid))


def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    eval(config)
if __name__ == "__main__":
    app.run(main)