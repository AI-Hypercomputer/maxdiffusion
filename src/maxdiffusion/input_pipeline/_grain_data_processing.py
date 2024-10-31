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

import dataclasses
import glob
import tensorflow as tf
import numpy as np
import grain.python as grain

from maxdiffusion import multihost_dataloading


def make_grain_iterator(
    config,
    dataloading_host_index,
    dataloading_host_count,
    mesh,
    global_batch_size,
):
  """Use Grain data input pipeline with ArrayRecord data format"""
  data_files = glob.glob(config.grain_train_files)
  data_source = grain.ArrayRecordDataSource(data_files)

  operations = []
  operations.append(ParseFeatures())
  operations.append(grain.Batch(batch_size=global_batch_size // dataloading_host_count, drop_remainder=True))

  index_sampler = grain.IndexSampler(
      num_records=len(data_source),
      num_epochs=None,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=True
      ),
      shuffle=True,
      seed=config.seed,
  )

  dataloader = grain.DataLoader(
      data_source=data_source,
      operations=operations,
      sampler=index_sampler,
      worker_count=config.grain_worker_count,
  )

  data_iter = multihost_dataloading.MultiHostDataLoadIterator(dataloader, mesh)
  return data_iter


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized example"""

  def __init__(self):
    self.feature_description = {
        "moments": tf.io.FixedLenFeature([], tf.string),
        "clip_embeddings": tf.io.FixedLenFeature([], tf.string),
    }

  def map(self, example):
    def _parse(example):
      features = tf.io.parse_single_example(example, self.feature_description)
      moments = tf.io.parse_tensor(np.asarray(features["moments"]), out_type=tf.float32)
      clip_embeddings = tf.io.parse_tensor(np.asarray(features["clip_embeddings"]), out_type=tf.float32)
      return {"pixel_values": moments, "input_ids": clip_embeddings}

    return _parse(example)
