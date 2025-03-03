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

import warnings
import datasets
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import grain.python as grain

from maxdiffusion import max_logging
from maxdiffusion import multihost_dataloading


def make_hf_streaming_iterator(
    config,
    dataloading_host_index,
    dataloading_host_count,
    mesh,
    global_batch_size,
    tokenize_fn=None,
    image_transforms_fn=None,
    hf_batch_factor=4,
):
  """Streaming data from HF Hub or GCS buckect.
  No download regardless of config.cache_latents_text_encoder_outputs"""
  ds = load_dataset(
      config.dataset_name,
      split=config.train_split,
      data_dir=config.hf_data_dir,
      data_files=config.hf_train_files,
      streaming=True,
      token=config.hf_access_token,
  )

  ds = ds.shuffle(seed=config.seed)
  ds = ds.select_columns([config.caption_column, config.image_column])

  if tokenize_fn:
    ds = ds.map(
        function=tokenize_fn,
        batched=True,
        batch_size=hf_batch_factor * config.total_train_batch_size,
        remove_columns=[config.caption_column],
    )

  if image_transforms_fn:
    ds = ds.map(
        function=image_transforms_fn,
        batched=True,
        batch_size=hf_batch_factor * config.total_train_batch_size,
        remove_columns=[config.image_column],
    )

  ds = HFDataSource(
      ds,
      dataloading_host_index,
      dataloading_host_count,
  )
  dummy_index_sampler = grain.IndexSampler(
      num_records=len(ds),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
      ),
      shuffle=False,
      seed=0,
  )
  operations = [grain.Batch(batch_size=global_batch_size // dataloading_host_count, drop_remainder=True)]
  dataloader = grain.DataLoader(
      data_source=ds,
      operations=operations,
      sampler=dummy_index_sampler,
      worker_count=1,  # only supports one worker for now, more workers results in duplicated data
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=hf_batch_factor * config.total_train_batch_size),
  )
  train_iter = multihost_dataloading.MultiHostDataLoadIterator(dataloader, mesh)
  return train_iter


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(
      self,
      dataset: datasets.IterableDataset,
      dataloading_host_index: int,
      dataloading_host_count: int,
  ):
    self.dataset = dataset
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.n_shards = dataset.n_shards
    self._check_shard_count()
    self.current_shard = dataloading_host_index
    self.dataset_shard = split_dataset_by_node(dataset, world_size=self.n_shards, rank=self.current_shard)
    self.data_iter = None
    self.out_of_data = False

  def _check_shard_count(self):
    if self.n_shards < self.dataloading_host_count:
      warnings.warn(
          f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
          "smaller than number of host loading data. This is known to lead to inefficient dataloading. "
          "see https://github.com/AI-Hypercomputer/maxdiffusion/blob/main/docs/data_README.md#best-practice"
      )
      self.n_shards = self.dataloading_host_count

  def _update_shard(self):
    new_shard = self.current_shard + self.dataloading_host_count
    if new_shard < self.n_shards:
      max_logging.log(f"Updating host {self.dataloading_host_index} dataset from shard {self.current_shard} to {new_shard}")
      self.current_shard = new_shard
      self.dataset_shard = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.current_shard)
      self.data_iter = iter(self.dataset_shard)
    else:
      max_logging.log(f"Run out of shards on host {self.dataloading_host_index}, shard {new_shard} is not available")
      self.out_of_data = True

  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if not self.data_iter:
      self.data_iter = iter(self.dataset_shard)

    while True:
      try:
        if self.out_of_data:
          return None
        data = next(self.data_iter)
        return data
      except StopIteration:
        self._update_shard()
