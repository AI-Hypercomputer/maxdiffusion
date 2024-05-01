import os
import unittest
from maxdiffusion import max_utils
from maxdiffusion import mllog_utils

class MaxUtils(unittest.TestCase):
  def test_download_blobs(self):
    fid_file = "gs://jfacevedo-maxdiffusion-v5p/inception_checkpoints/inception_v3/inception_v3_weights_fid.pickle"
    clip_file =  "gs://jfacevedo-maxdiffusion-v5p/CLIP_checkpoints/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    main_ckpt_file =  "gs://jfacevedo-maxdiffusion-v5p/stable_diffusion_checkpoints/models--stabilityai--stable-diffusion-2-base"
    saved_ckpt_file =  "gs://jfacevedo-maxdiffusion-v5p/training_results/v5p-256-xpk-moments--lr-1.75e-4-v-pred-no-grad-norm-2-ip-snr/checkpoints/step_num=1250-samples_count=2560000"
    local_destination="/tmp"

    local_path = max_utils.download_blobs(fid_file, local_destination)
    _, prefix_name = max_utils.parse_gcs_bucket_and_prefix(fid_file)
    assert local_path == os.path.join(local_destination, prefix_name)
    step_num = mllog_utils.extract_info_from_ckpt_name(local_path, "step_num")
    samples_count = mllog_utils.extract_info_from_ckpt_name(local_path, "samples_count")
    assert step_num == -1
    assert samples_count == -1

    local_path = max_utils.download_blobs(clip_file, local_destination)
    _, prefix_name = max_utils.parse_gcs_bucket_and_prefix(clip_file)
    assert local_path == os.path.join(local_destination, prefix_name)
    step_num = mllog_utils.extract_info_from_ckpt_name(local_path, "step_num")
    samples_count = mllog_utils.extract_info_from_ckpt_name(local_path, "samples_count")
    assert step_num == -1
    assert samples_count == -1

    local_path = max_utils.download_blobs(main_ckpt_file, local_destination)
    _, prefix_name = max_utils.parse_gcs_bucket_and_prefix(main_ckpt_file)
    assert local_path == os.path.join(local_destination, prefix_name)
    step_num = mllog_utils.extract_info_from_ckpt_name(local_path, "step_num")
    samples_count = mllog_utils.extract_info_from_ckpt_name(local_path, "samples_count")
    assert step_num == -1
    assert samples_count == -1

    local_path = max_utils.download_blobs(saved_ckpt_file, local_destination)
    _, prefix_name = max_utils.parse_gcs_bucket_and_prefix(saved_ckpt_file)
    assert local_path == os.path.join(local_destination, prefix_name)
    step_num = mllog_utils.extract_info_from_ckpt_name(local_path, "step_num")
    samples_count = mllog_utils.extract_info_from_ckpt_name(local_path, "samples_count")
    assert step_num == 1250
    assert samples_count == 2560000

    
