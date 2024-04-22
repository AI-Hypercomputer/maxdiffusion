import random
import time
import torch
import torch.nn as nn

from transformers import FlaxCLIPModel, AutoProcessor

import open_clip
import jax
import datasets
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

from google.cloud import storage

class CLIPEncoderTorch(nn.Module):
    """
        PyTorch implementation. See Flax version below    
    """
    def __init__(self, clip_version='ViT-H-14', pretrained='', cache_dir=None, device='cpu'):
        super().__init__()

        self.clip_version = clip_version
        if not pretrained:
            if self.clip_version == 'ViT-H-14':
                self.pretrained = 'laion2b_s32b_b79k'
            elif self.clip_version == 'ViT-g-14':
                self.pretrained = 'laion2b_s12b_b42k'
            else:
                self.pretrained = 'openai'

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.clip_version,
                                                                               pretrained=self.pretrained,
                                                                               cache_dir=cache_dir)
        self.model.eval()
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def get_clip_score(self, text, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if not isinstance(text, (list, tuple)):
            text = [text]
        text = open_clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        return similarity

class CLIPEncoderFlax:
    """
        Flax implementation. See PyTorch version above    
    """

    def __init__(self, pretrained="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        assert pretrained is not None

        self.model = FlaxCLIPModel.from_pretrained(pretrained)
        self.logit_scale = jnp.exp(self.model.params['logit_scale'])
        self.get_image_features = jax.jit(self.model.get_image_features)
        self.get_text_features = jax.jit(self.model.get_text_features)
        self.model = jax.jit(self.model)
        self.processor = AutoProcessor.from_pretrained(pretrained)
        
    def get_clip_score(self, text, image):
        # See https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/models/clip/modeling_flax_clip.py#L1214
        inputs = self.processor(images=image, return_tensors="jax")
        image_embeddings = self.get_image_features(**inputs)
        inputs = self.processor(text=text, return_tensors="jax", padding="max_length", truncation=True)
        text_embeddings = self.get_text_features(**inputs)

        return CLIPEncoderFlax._calculate_clip_score(text_embeddings, image_embeddings, self.logit_scale)

    @staticmethod
    @jax.jit
    def _calculate_clip_score(text_embeddings, image_embeddings, logit_scale):
        image_embeddings /= jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings /= jnp.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        image_embeddings = jnp.expand_dims(image_embeddings, 1)
        text_embeddings = jnp.expand_dims(text_embeddings, 2)

        return jnp.squeeze(jax.lax.batch_matmul(image_embeddings, text_embeddings), (2,)) * logit_scale / 100


    def get_clip_score_batched(self, prompts, images, batch_size):
        dataset = datasets.Dataset.from_dict({"images": images, "texts": prompts})

        clip_scores = []

        for batch in tqdm(dataset.iter(batch_size=batch_size)):
            batch_texts, batch_images = batch["texts"], batch["images"]
            batch_clip_scores = self.get_clip_score(batch_texts, batch_images)
            clip_scores.append(batch_clip_scores)
            
        overall_clip_score = jnp.mean(jnp.concatenate(clip_scores, axis=0))
        print("clip score is" + str(overall_clip_score))
        return np.array(overall_clip_score)
