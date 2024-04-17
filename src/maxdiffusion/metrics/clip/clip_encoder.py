import torch
import torch.nn as nn
from transformers import AutoTokenizer, FlaxCLIPModel, AutoProcessor
import numpy as np

import requests

from google.cloud import storage
import random


import open_clip
from PIL import Image



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
        similarity = 100 * image_features @ text_features.T

        return similarity.numpy()

class CLIPEncoderFlax:

    def __init__(self, pretrained="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        assert pretrained is not None

        self.model = FlaxCLIPModel.from_pretrained(pretrained)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def get_clip_score(self, text, image):

        inputs = self.processor(text=text, images=image, return_tensors="np")
        outputs = self.model(**inputs)

        return np.array(outputs.logits_per_image)
    
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
    
if __name__ == "__main__":

    my_bucket_name = "jfacevedo-maxdiffusion-v5p"
    my_folder_path = "checkpoints/ckpt_generated_images/512000"
    random_images = load_random_images_from_gcs(my_bucket_name, my_folder_path)

    pytorch_encoder = CLIPEncoder()
    flax_encoder = CLIPEncoderFlax()

    some_mismatch = False
    for blob, image in random_images:
        caption = "The struggle is real, but so is the coffee."
        torch_score = pytorch_encoder.get_clip_score(caption, image)
        flax_score = flax_encoder.get_clip_score(caption, image)
        if not np.allclose(torch_score, flax_score):
            print(f"The scores did not match for blob {blob}. Torch Score was {torch_score} and Flax Score was {flax_score}")
            some_mismatch = True
        else:
            print(f"Blob {blob} matched")
    
    if not some_mismatch:
        print("All matched")


    








    








        