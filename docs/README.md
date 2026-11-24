# MaxDiffusion Documentation

This folder contains documentation for getting started with and using MaxDiffusion.

## Getting Started

* **[First Run](getting_started/first_run.md)** - Provides instructions for setting up and running MaxDiffusion for the first time.
* **[Running MaxDiffusion via XPK](getting_started/run_maxdiffusion_via_xpk.md)** - Explains how to run MaxDiffusion on GKE using XPK.
* **[NVIDIA DGX Spark](dgx_spark.md)** - Explains how to run MaxDiffusion on an NVIDIA DGX Spark.

## Contributing & Community

* **[Code of Conduct](code-of-conduct.md)** - Outlines the expected behavior for contributors to the project.
* **[Contributing](contributing.md)** - Provides guidelines for contributing to the MaxDiffusion project.

## Training

* **[Training Guide](../README.md#training)** - Per-model training walkthroughs (Wan 2.1 / 2.2, Flux, SDXL, SD 2 base, SD 1.4, Dreambooth) live in the main README.

## Data Input

* **[Common Data Input Guide](data_README.md)** - Provides a comprehensive guide to data input pipelines.

## Observability

* **[Profiling](profiling.md)** - How to enable ML Diagnostics and XProf profiling for your runs.
* **[Metrics](metrics.md)** - How to enable ML Diagnostics metrics tracking for your runs.

## Internals

* **[Attention Block Sizes](attention_blocks_flowchart.md)** - Explains the flash-attention `block_*` tiling parameters and how they relate to each other.
