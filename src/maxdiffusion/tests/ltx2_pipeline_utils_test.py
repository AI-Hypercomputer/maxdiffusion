# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import tempfile
import os
import torch
import numpy as np

from maxdiffusion.pipelines.ltx2 import ltx2_pipeline_utils
from maxdiffusion.utils import import_utils


class LTX2PipelineUtilsTest(unittest.TestCase):
    
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.output_dir, "test_output.mp4")

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.rmdir(self.output_dir)

    @unittest.skipIf(not import_utils.is_av_available(), "av not available")
    def test_encode_video_silent(self):
        # Create dummy video frames: 10 frames, 64x64, RGB
        frames = 10
        height = 64
        width = 64
        video = torch.randint(0, 255, (frames, height, width, 3), dtype=torch.uint8)
        fps = 24
        
        ltx2_pipeline_utils.encode_video(
            video=video,
            fps=fps,
            audio=None,
            audio_sample_rate=None,
            output_path=self.output_path
        )
        
        self.assertTrue(os.path.exists(self.output_path))
        self.assertGreater(os.path.getsize(self.output_path), 0)

    @unittest.skipIf(not import_utils.is_av_available(), "av not available")
    def test_encode_video_with_audio(self):
        # Create dummy video frames
        frames = 10
        height = 64
        width = 64
        video = torch.randint(0, 255, (frames, height, width, 3), dtype=torch.uint8)
        fps = 24

        # Create dummy audio: 1 second of stereo noise at 16kHz
        # encode_video expects audio as [samples, channels] usually based on logic:
        # if samples.shape[1] != 2 and samples.shape[0] == 2: samples = samples.T
        audio_sample_rate = 16000
        duration = frames / fps
        num_samples = int(duration * audio_sample_rate)
        # Create [2, num_samples] to test transpose logic
        audio = torch.linspace(-1, 1, num_samples).unsqueeze(0).repeat(2, 1) # [2, N]
        
        ltx2_pipeline_utils.encode_video(
            video=video,
            fps=fps,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            output_path=self.output_path
        )
        
        self.assertTrue(os.path.exists(self.output_path))
        self.assertGreater(os.path.getsize(self.output_path), 0)

if __name__ == "__main__":
    unittest.main()
