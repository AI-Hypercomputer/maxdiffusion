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
from jax.experimental.compilation_cache import compilation_cache as cc
from maxdiffusion.metrics.fid import inception
from maxdiffusion.metrics.fid import fid_score

from maxdiffusion.metrics.clip.clip_encoder import CLIPEncoderFlax, CLIPEncoderTorch
from typing import Sequence
from absl import app
from maxdiffusion import (
    generate,
    pyconfig,
    mllog_utils,
)
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
import tensorflow as tf
from PIL import Image

def load_captions(file_path):
    with tf.io.gfile.GFile(file_path, 'r') as f:
        captions_df = pd.read_csv(f, delimiter='\t', header=0, names=['image_id','id', 'caption'])
    return captions_df

def load_stats(file_path):
    with tf.io.gfile.GFile(file_path, 'rb') as f:
        images_data = np.load(f)
    sigma = images_data['sigma']
    mu = images_data['mu']
    return sigma, mu


def calculate_clip(images, prompts):
    clip_encoder = CLIPEncoderFlax()
    clip_encoder_torch = CLIPEncoderTorch()
    
    clip_scores = []
    torch_scores = []
    not_matched = 0
    for i in tqdm(range(0, len(images))):
        with jax.default_device(jax.devices('tpu')[0]):
            score = clip_encoder.get_clip_score(prompts[i], images[i])
            clip_scores.append(score)

        torch_score = clip_encoder_torch.get_clip_score(prompts[i], images[i])
        torch_scores.append(torch_score)

        if not np.allclose(np.array(score), torch_score.numpy(), atol=1e-3):
            not_matched += 1
            print(f"Scores did not match. Jax {score}. Torch {torch_score}. Diff {np.abs(np.array(score) - torch_score.numpy())}")
    
    print(f'{not_matched} scores did not match')
    overall_clip_score = np.array(jnp.mean(jnp.stack(clip_scores)))
    overall_torch_score = torch.mean(torch.stack(torch_scores)).numpy()

    if not np.allclose(overall_clip_score, overall_torch_score, atol=1e-3):
        print(f"Overall scores did not match. Jax {overall_clip_score}. Torch {overall_torch_score}. Diff {np.abs(overall_clip_score - overall_torch_score)}")

    return overall_clip_score


def load_images(path, captions_df):
    images = []
    prompts = []
    captions_df = captions_df
    for f in tqdm(tf.io.gfile.listdir(path)):
        img = Image.open(tf.io.gfile.GFile(os.path.join(path, f), 'rb'))
        img_id = f[6:len(f)-4]
        pmt = captions_df.query(f'image_id== {img_id}')['caption'].to_string(index=False)
        images.append(img)
        prompts.append(pmt)
    print(f'Evaluation on {len(images)} images')
    return images, prompts


def write_eval_metrics(config, clip_score: float, fid: float):
    if jax.process_index() == 0:
        eval_metrics_path = os.path.join(config.base_output_directory, "eval_metrics.csv")
        metrics = {
            "step_num": mllog_utils.extract_info_from_ckpt_name(config.pretrained_model_name_or_path, "step_num"),
            "samples_count": mllog_utils.extract_info_from_ckpt_name(config.pretrained_model_name_or_path, "samples_count"),
            "clip": clip_score,
            "fid": fid,
        }
        df = pd.DataFrame.from_dict([metrics])
        if not tf.io.gfile.exists(eval_metrics_path):
            with tf.io.gfile.GFile(eval_metrics_path, 'w') as f:
                df.to_csv(f, index=False)
        else:
            with tf.io.gfile.GFile(eval_metrics_path, 'a') as f:
                df.to_csv(f, index=False, header=False)

def eval_scores(config, images_directory=None):
    batch_size = config.per_device_batch_size * jax.device_count() * 10

    # mllog_utils.eval_start(config)
    # #inference happenning here: first generate the images
    # if images_directory is None:
    #     generate.run(config)
    #     images_directory = config.images_directory
    # mllog_utils.eval_end(config)

    # calculating CLIP:

    captions_df = load_captions(config.caption_coco_file)
    images, prompts = load_images(config.images_directory, captions_df)
    
    clip_score = calculate_clip(images, prompts)
    print('All matched')
    return clip_score, None
    mllog_utils.eval_clip(config, clip_score)

    # calculating FID:
    rng = jax.random.PRNGKey(0)
    
    model = inception.InceptionV3(pretrained=True, transform_input=False)
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))

    apply_fn = jax.jit(functools.partial(model.apply, train=False))

    dataloader_images_directory = os.path.dirname(images_directory.rstrip("/"))
    mu, sigma = fid_score.compute_statistics_with_mmap(dataloader_images_directory, "/tmp/temp.dat", params, apply_fn, batch_size, (299, 299))

    os.makedirs(config.stat_output_directory, exist_ok=True)
    np.savez(os.path.join(config.stat_output_directory, 'stats'), mu=mu, sigma=sigma)

    mu1, sigma1 = fid_score.compute_statistics(config.stat_output_file, params, apply_fn, batch_size,)
    mu2, sigma2 = fid_score.compute_statistics(config.stat_coco_file, params, apply_fn, batch_size,)
    fid = fid_score.compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6)
    mllog_utils.eval_fid(config, fid)
    write_eval_metrics(config, clip_score, fid)
    return clip_score, fid


def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    clip, fid = eval_scores(config)
    print(f"clip score is {clip}")
    print(f"fid score is : {fid}")

if __name__ == "__main__":
    app.run(main)