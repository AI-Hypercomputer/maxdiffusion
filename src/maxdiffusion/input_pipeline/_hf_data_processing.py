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

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import grain.python as grain
#import jax

from maxdiffusion.input_pipeline import _input_pipeline_utils
#from maxdiffusion import max_logging
from maxdiffusion import multihost_dataloading

def make_pokemon_train_iterator(
    config,
    dataloading_host_index,
    dataloading_host_count,
    mesh,
    global_batch_size,
    tokenize_fn,
    image_transforms_fn):
  
  train_ds = load_dataset(config.dataset_name,split="train", streaming=True)

  captions_column = config.caption_column
  image_column = config.image_column
  train_ds = train_ds.shuffle(seed=config.seed)
  train_ds = train_ds.map(
    function=tokenize_fn,
    batched=True,
    remove_columns=[captions_column],
    #num_proc=1 if cache_latents_text_encoder_outputs else 4,
    #desc="Running tokenizer on train dataset",
  )
  # need to do it before load_as_tf_dataset
  # since raw images are different sizes
  # will break from_tensor_slices
  train_ds = train_ds.map(
    function=image_transforms_fn,
    batched=True,
    remove_columns=[image_column],
    #num_proc=1 if cache_latents_text_encoder_outputs else config.transform_images_num_proc,
    #desc="Transforming images",
  )
    # train_ds.save_to_disk(dataset_save_location)
    # train_ds.cleanup_cache_files()

  # train_ds = load_as_tf_dataset(
  #   train_ds, global_batch_size, True
  # )
  #train_ds = split_dataset_by_node(train_ds, world_size=jax.process_count(), rank=dataloading_host_index)
  #train_ds = train_ds.batch(batch_size=global_batch_size // jax.process_count(), drop_last_batch=True)
  #train_ds = train_ds.with_format("np")
  train_ds = _input_pipeline_utils.HFDataSource(train_ds,
                                                dataloading_host_index,
                                                dataloading_host_count,
                                               )
  dummy_index_sampler = grain.IndexSampler(
      num_records=len(train_ds),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
      ),
      shuffle=False,
      seed=0,
  )
  operations =  [grain.Batch(batch_size=global_batch_size // dataloading_host_count, drop_remainder=True)]
  dataloader = grain.DataLoader(
      data_source=train_ds,
      operations=operations,
      sampler=dummy_index_sampler,
      worker_count=1,  # only supports one worker for now, more workers results in duplicated data
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=8),
  )
  train_iter = multihost_dataloading.MultiHostDataLoadIterator(dataloader, mesh)
  return train_iter
