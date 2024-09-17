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
import warnings
import datasets
from datasets.distributed import split_dataset_by_node
import grain.python as grain
import numpy as np
from maxdiffusion import max_logging

class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""
  def __init__(self,
              dataset: datasets.IterableDataset,
              dataloading_host_index: int,
              dataloading_host_count: int,
              ):
    self.dataset = dataset
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.n_shards = dataset.n_shards
    self.current_shard = dataloading_host_index
    self.dataset_shard = split_dataset_by_node(dataset, world_size=self.n_shards, rank=self.current_shard)
    self.data_iter = None

  def _check_shard_count(self):
    if self.n_shards < self.dataloading_host_count:
      warnings.warn(f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
                      "smaller than number of host loading data. This is known to lead to inefficient dataloading. " 
                      "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
                      )
      self.n_shards = self.dataloading_host_count

  def _update_shard(self):
    new_shard = (self.current_shard + self.dataloading_host_count) % self.n_shards
    max_logging.log(f"Updating host {self.dataloading_host_index} dataset, was on shard {self.current_shard}")
    max_logging.log(f"New shard is {new_shard}")
    self.current_shard = new_shard
    self.dataset_shard = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.current_shard)
    self.data_iter = iter(self.dataset_shard)


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
        data = next(self.data_iter)
        return data
      except StopIteration:
        self._update_shard()
