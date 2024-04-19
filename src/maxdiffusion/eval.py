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
from maxdiffusion.metrics.clip.clip_encoder import CLIPEncoderFlax
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

from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from PIL import Image


def load_captions(file_path):
    captions_df = pd.read_csv(file_path, delimiter='\t', header=0, names=['image_id','id', 'caption'])
    return captions_df

def load_stats(file_path):
    images_data = np.load(file_path)
    sigma = images_data['sigma']
    mu = images_data['mu']
    return sigma, mu

def calculate_clip(images, prompts):
    clip_encoder = CLIPEncoderFlax()
    
    clip_scores = []
    for i in tqdm(range(0, len(images))):
        score = clip_encoder.get_clip_score(prompts[i], images[i])
        clip_scores.append(score)
        
    overall_clip_score = jnp.mean(jnp.stack(clip_scores))
    return np.array(overall_clip_score)

def load_images(path, captions_df):
    images = []
    prompts = []
    for f in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, f))
        img_id = f[6:len(f)-4]
        pmt = captions_df.query(f'image_id== {img_id}')['caption'].to_string(index=False)
        images.append(img)
        prompts.append(pmt)
     
    return images, prompts

def eval_scores(config, images_directory=None):
    batch_size = config.per_device_batch_size * jax.device_count() * 10

    #inference happenning here: first generate the images
    if images_directory is None:
        generate.run(config)
        images_directory = config.images_directory

    # calculating CLIP:
    captions_df = load_captions(config.caption_coco_file)
    images, prompts = load_images(images_directory, captions_df)
    
    clip_score = calculate_clip(images, prompts)

    # calculating FID:
    rng = jax.random.PRNGKey(0)
    
    model = inception.InceptionV3(pretrained=True, transform_input=False)
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))

    apply_fn = jax.jit(functools.partial(model.apply, train=False))
    dataloader_images_directory="/".join(images_directory.split("/")[:-2])
    mu, sigma = fid_score.compute_statistics_with_mmap(dataloader_images_directory, "/tmp/temp.dat", params, apply_fn, batch_size, (299, 299))
    os.makedirs(config.stat_output_directory, exist_ok=True)
    np.savez(os.path.join(config.stat_output_directory, 'stats'), mu=mu, sigma=sigma)
    mu1, sigma1 = fid_score.compute_statistics(config.stat_output_file, params, apply_fn, batch_size,)
    mu2, sigma2 = fid_score.compute_statistics(config.stat_coco_file, params, apply_fn, batch_size,)

    fid = fid_score.compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6)
    return clip_score, fid

def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    clip, fid = eval_scores(config)
    print("clip score is " + str(clip))
    print("fid score is : " + str(fid))

if __name__ == "__main__":
    app.run(main)