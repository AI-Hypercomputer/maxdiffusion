import random
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

        # TODO: jit this again
        self.model = FlaxCLIPModel.from_pretrained(pretrained)
        self.processor = AutoProcessor.from_pretrained(pretrained)
    
    def get_clip_score(self, text, image):

        inputs = self.processor(text=text, images=image, return_tensors="jax", padding="max_length", truncation=True)
        outputs = self.model(**inputs)

        return outputs.logits_per_image / 100
    
    def get_clip_score_manual(self, text, image):
        # See https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/models/clip/modeling_flax_clip.py#L1214
        inputs = self.processor(images=image, return_tensors="jax")
        image_embeddings = self.model.get_image_features(**inputs)
        inputs = self.processor(text=text, return_tensors="jax", padding="max_length", truncation=True)
        text_embeddings = self.model.get_text_features(**inputs)

        image_embeddings /= jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings /= jnp.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        image_embeddings = jnp.expand_dims(image_embeddings, 1)
        text_embeddings = jnp.expand_dims(text_embeddings, 2)

        logit_scale = jnp.exp(self.model.params['logit_scale'])
        return jax.lax.squeeze(jax.lax.batch_matmul(image_embeddings, text_embeddings), (2,)) * logit_scale / 100

        #return jnp.sum(image_embeddings * text_embeddings, axis=1, keepdims=True) * logit_scale / 100


    def get_clip_score_batched(self, prompts, images, batch_size = 4):
        dataset = datasets.Dataset.from_dict({"images": images, "texts": prompts})

        clip_scores = []

        for batch in tqdm(dataset.iter(batch_size=batch_size)):
            batch_texts, batch_images = batch["texts"], batch["images"]
            batch_clip_scores = self.get_clip_score_manual(batch_texts, batch_images)
            clip_scores.append(batch_clip_scores)
            
        overall_clip_score = jnp.mean(jnp.concatenate(clip_scores, axis=0))
        print("clip score is" + str(overall_clip_score))
        return np.array(overall_clip_score), jnp.concatenate(clip_scores, axis=0)
        
            


        



# TODO: delete all this stuff when i merge
def load_random_images_from_gcs(bucket_name, folder_path, max_images=10):
    """Loads a specified number of random images from a folder in a GCS bucket.

    Args:
        bucket_name (str): Name of the GCS bucket.
        folder_path (str): The path to the folder within the bucket.
        max_images (int): The maximum number of images to load. Defaults to 10.

    Returns:
        list: A list of PIL.Image objects.
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get a list of image blobs in the specified folder
    blobs = bucket.list_blobs(prefix=folder_path)
    image_blobs = [blob for blob in blobs if blob.name.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Select random images (up to max_images)
    num_images_to_load = min(max_images, len(image_blobs))
    random_blobs = random.sample(image_blobs, num_images_to_load)

    images = []
    for blob in random_blobs:
        image_bytes = blob.download_as_bytes()
        from PIL import Image
        from io import BytesIO
        images.append((blob.name, Image.open(BytesIO(image_bytes))))

    return images

def get_random_caption():
    sentences = [
        "The early bird might get the worm, but the second mouse gets the cheese.",
        "Don't count your chickens before they hatch... or your omelet will be disappointing.",
        "If at first you don't succeed, try hiding all evidence that you ever tried.",
        "Experience is a great teacher, but she gives really tough exams.",
        "My imaginary friends think I'm the best listener.",
        "A clear conscience is often a sign of a bad memory.",
        "Today was a total waste of makeup.",
        "My level of sarcasm has gotten to the point where I don't even know if I'm kidding or not.",
        "If you think nobody cares if you're alive, try missing a couple of payments.",
        "Apparently, rock bottom has a basement." 
    ]

    return random.sample(sentences, 1)

def calculate_clip(images, prompts):
    clip_encoder = CLIPEncoderFlax()

    score_batched, all_scores_batched = clip_encoder.get_clip_score_batched(prompts, images)
    
    clip_scores = []
    for i in tqdm(range(0, len(images))):
        score = clip_encoder.get_clip_score(prompts[i], images[i])
        clip_scores.append(score)
        
    overall_clip_score = jnp.mean(jnp.stack(clip_scores))
    print("clip score is" + str(overall_clip_score))
    assert np.allclose(np.array(overall_clip_score), score_batched)
    print("They matched")

def verify_correctness_single_image(images, prompts):
    # should be exactly the same
    clip_encoder_flax = CLIPEncoderFlax()
    clip_encoder = CLIPEncoderTorch()

    for image, prompt in zip(images, prompts):
        score = clip_encoder.get_clip_score(prompt, image)

        manual_score = clip_encoder_flax.get_clip_score_manual(prompt, image)

        assert np.allclose(score, manual_score, atol=1e-3)

        print('Matched')


def batch_playgroud(device='tpu'):
    my_bucket_name = "jfacevedo-maxdiffusion-v5p"
    my_folder_path = "checkpoints/ckpt_generated_images/512000"
    random_images = [image for blob, image in load_random_images_from_gcs(my_bucket_name, my_folder_path, max_images=40)]
    random_prompts = [get_random_caption()[0] for _ in range(len(random_images))]

    #calculate_clip(random_images, random_prompts)
    verify_correctness_single_image(random_images, random_prompts)



    # TODO: verify correctness of batched impl
    # TODO: get the jit implementation working
    # TODO: time batched impl vs manual impl 
    # TODO: 

if __name__ == "__main__":
    batch_playgroud()