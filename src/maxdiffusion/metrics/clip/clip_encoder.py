import torch
import torch.nn as nn

from transformers import FlaxCLIPModel, AutoProcessor

import open_clip
import jax

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

        self.model = jax.jit(FlaxCLIPModel.from_pretrained(pretrained))
        self.processor = AutoProcessor.from_pretrained(pretrained)
    
    def get_clip_score(self, text, image):

        inputs = self.processor(text=text, images=image, return_tensors="jax", padding="max_length", truncation=True)
        outputs = self.model(**inputs)

        return outputs.logits_per_image / 100
