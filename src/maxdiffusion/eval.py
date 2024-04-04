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

def load_captions(file_path):
    captions_df = pd.read_csv(file_path, delimiter='\t', header=0, names=['image_id','id', 'caption'])
    return captions_df

def load_stats(file_path):
    images_data = np.load(file_path)
    sigma = images_data['sigma']
    mu = images_data['mu']
    return sigma, mu

def eval(config):
    captions_df = load_captions(config.caption_coco_file)
    batch_size = config.per_device_batch_size * jax.device_count()

    #inference happenning here: 

    # while(not captions_df.empty):
    #     if (captions_df.size > batch_size):
    #         captions_df['caption'][:batch_size].to_csv('prompts.txt', header=False, index=False)
    #         captions_df['id'][:batch_size].to_csv('ids.txt', header=False,index=False)
    #         captions_df = captions_df.iloc[batch_size:]
    #     else:
    #         captions_df['caption'].to_csv('prompts.txt', header=False, index=False)
    #         captions_df['id'].to_csv('ids.txt', header=False,index=False)
    #         captions_df = pd.DataFrame()
    #     generate.run(config)

    rng = jax.random.PRNGKey(0)
    
    model = inception.InceptionV3(pretrained=True)
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))

    apply_fn = jax.jit(functools.partial(model.apply, train=False))
    mu, sigma = fid_score.compute_statistics(config.images_directory, params, apply_fn, batch_size)
    os.makedirs(config.stat_output_directory, exist_ok=True)
    np.savez(os.path.join(config.stat_output_directory, 'stats'), mu=mu, sigma=sigma)

    #     with open(config.image_ids, "r") as ids_file:
    #         image_ids = ids_file.readlines()
    #         for image_id in image_ids:
    #             pathlib.Path(f"image_{image_id}.png").unlink()

    mu1, sigma1 = fid_score.compute_statistics(config.stat_output_file, params, apply_fn, batch_size)
    mu2, sigma2 = fid_score.compute_statistics(config.stat_coco_file, params, apply_fn, batch_size)

    fid = fid_score.compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6)
    print("fid is : " + str(fidscore))



def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    eval(config)
if __name__ == "__main__":
    app.run(main)