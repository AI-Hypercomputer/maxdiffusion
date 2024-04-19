import unittest

import numpy as np
from PIL import Image
import os

from maxdiffusion.metrics.clip.clip_encoder import CLIPEncoderTorch, CLIPEncoderFlax

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestCLIPEncoders(unittest.TestCase):

    def test_encoder_consistency(self):
        encoder_torch = CLIPEncoderTorch()
        encoder_flax = CLIPEncoderFlax()

        text = 'a photo of a cat'
        image = Image.open(os.path.join(THIS_DIR,'images','test.png'))

        similarity_torch = encoder_torch.get_clip_score(text, image)
        similarity_flax = encoder_flax.get_clip_score(text, image)
        self.assertTrue(np.allclose(similarity_torch, similarity_flax, atol=1e-3))


if __name__ == '__main__':
    unittest.main()
