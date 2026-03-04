# Data Input Guide

## Overview
Currently MaxDiffusion supports 3 data input pipelines, controlled by the flag `dataset_type`
| Pipeline | Dataset Location | Dataset formats | Features or limitations |
| -------- | ---------------- | --------------- | ----------------------- |
| HuggingFace (hf) | datasets in HuggingFace Hub or local/Cloud Storage | Formats supported in HF Hub: parquet, arrow, json, csv, txt | data are not loaded in memory but streamed from the saved location, good for big dataset | 
| tf | dataset will be downloaded form HuggingFace Hub to disk | Formats supported in HF Hub: parquet, arrow, json, csv, txt | Will read the whole dataset into memory, works for small dataset |
| tfrecord | local/Cloud Storage | TFRecord | data are not loaded in memory but streamed from the saved location, good for big dataset |
| Grain | local/Cloud Storage | ArrayRecord (or any random access format) | data are not loaded in memory but streamed from the saved location, good for big dataset, supports global shuffle and data iterator checkpoint for determinism (see details in [doc](https://github.com/AI-Hypercomputer/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#grain-pipeline---for-determinism)) |

## Usage examples

### HuggingFace Streaming (dataset_type=hf)
#### Example config for streaming from HuggingFace Hub (no download needed):
```
dataset_type: hf 
dataset_name: BleachNick/UltraEdit_500k  # for using https://huggingface.co/datasets/BleachNick/UltraEdit_500k
image_column: source_image 
caption_column: source_caption 
train_split: FreeForm
hf_access_token: ''  # provide token if using gated dataset or tokenizer
```
#### Example config for streaming from downloaded data in a GCS bucket:
```
dataset_type: hf 
dataset_name: parquet  # or json, arrow, etc.
hf_train_files: gs://<bucket>/<folder>/*-train-*.parquet  # match the train files
```

### tf.data in-memory dataset (dataset_type=tf)
#### Example config
```
dataset_type: tf
dataset_name: diffusers/pokemon-gpt4-captions  # will download https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions
dataset_save_location: '/tmp/pokemon-gpt4-captions_xl'
# If cache_latents_text_encoder_outputs=True, apply vae to images and encode text when downloading dataset,
# the saved dataset contains latents and text encoder outputs.
cache_latents_text_encoder_outputs: True
```

### tf.data.TFRecordDataset (dataset_type=tfrecord)
#### Example config
```
dataset_type: tfrecord
train_data_dir: gs://<bucket>/<folder>  # will use all TFRecord files under the directory
```

### Grain (dataset_type=grain)
```
dataset_type: grain
grain_train_files: gs://<bucket>/<folder>/*.arrayrecord  # match the file pattern
```

## Best Practice
### Multihost Dataloading
In multihost environment, if use a streaming type of input pipeline and the data format only supports sequential reads (dataset_type in (hf, tfrecord in MaxDiffusion)), the most performant way is to have each data file only accessed by one host, and each host access a subset of data files (shuffle is within the subset of files). This requires (# of data files) > (# of hosts loading data). We recommand users to reshard the dataset if this requirement is not met.
#### HuggingFace pipeline when streaming from Hub
* When (# of data files) >= (# of hosts loading data), assign files to each host as evenly as possible, some host may ended up with 1 file more than the others. When a host run out of data, it will automatically start another epoch. Since each host run out of data at different speed, different host come to next epoch at different time.
* When (# of data files) < (# of hosts loading data), files are read sequentially with multiple hosts accessing each file, perf can degrade quickly as # of host increases.
