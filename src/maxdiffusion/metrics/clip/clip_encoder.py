import torch
import torch.nn as nn
from transformers import AutoTokenizer, FlaxCLIPModel, AutoProcessor
import numpy as np

import requests

from google.cloud import storage
import random


import open_clip
from PIL import Image
import jax



class CLIPEncoder(nn.Module):
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

        return similarity.numpy()

class CLIPEncoderFlax:

    def __init__(self, pretrained="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        assert pretrained is not None

        self.model = FlaxCLIPModel.from_pretrained(pretrained)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def get_clip_score(self, text, image):

        inputs = self.processor(text=text, images=image, return_tensors="np")
        outputs = self.model(**inputs)

        return np.array(outputs.logits_per_image) / 100
    
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

    
def verify_models_match(device='cpu'):
    my_bucket_name = "jfacevedo-maxdiffusion-v5p"
    my_folder_path = "checkpoints/ckpt_generated_images/512000"
    random_images = load_random_images_from_gcs(my_bucket_name, my_folder_path)

    pytorch_encoder = CLIPEncoder()
    flax_encoder = CLIPEncoderFlax()

    some_mismatch = False
    for blob, image in random_images:
        caption = get_random_caption()
        torch_score = pytorch_encoder.get_clip_score(caption, image)
        with jax.default_device(jax.devices(device)[0]):
            flax_score = flax_encoder.get_clip_score(caption, image)
        if not np.allclose(torch_score, flax_score, atol=1e-3):
            print(f"The scores did not match for blob {blob}. Torch Score was {torch_score} and Flax Score was {flax_score}")
            some_mismatch = True
        else:
            print(f"Blob {blob} matched")
    
    if not some_mismatch:
        print("All matched")
    return True

if __name__ == "__main__":
    for i in range(4):
        matched = verify_models_match('tpu')
        if not matched:
            print('Batch did not match')



    








    








        