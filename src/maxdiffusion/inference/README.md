# MaxDiffusion Inference Framework

This module provides a unified, production-ready inference stack for MaxDiffusion models (Wan, Flux, SDXL). It decouples inference from training dependencies and provides a consistent interface for both offline generation scripts and online serving.

## Components

*   **`loader`**: Unified Model Loader. Handles loading weights from `orbax` (MaxDiffusion checkpoints) or HuggingFace/Diffusers (Safetensors) without instantiating Training Trainers.
*   **`runner`**: Core Inference Runner. Encapsulates the JAX/TPU mesh, JIT compilation of inference steps, and the denoising loop.
*   **`server`**: A high-performance decoupled serving stack.
    *   **Frontend**: FastAPI server handling HTTP requests.
    *   **Backend**: ZeroMQ-based Scheduler and TPU Worker.

## Usage

### 1. Offline Generation (Scripts)

The root level scripts `generate_flux.py` and `generate_sdxl.py` have been refactored to use this framework.

```bash
python src/maxdiffusion/generate_flux.py src/maxdiffusion/configs/base_flux_dev.yml
```

### 2. Online Serving

To start the serving stack:

**Start Scheduler (Backend)**
```bash
python -m maxdiffusion.inference.server.scheduler src/maxdiffusion/configs/base_flux_dev.yml
```

**Start API (Frontend)**
```bash
python -m maxdiffusion.inference.server.api
```

**Send Request**
```bash
curl -X POST http://localhost:8000/generate -d '{"prompt": "A photo of a cat", "num_inference_steps": 20}'
```

## Architecture

```
InferenceLoader -> [Pipeline, Params, State] -> DiffusionRunner -> [Images]
                                                      ^
                                                      |
                                                 TPU Worker
                                                      ^
                                                      | ZMQ
                                                      v
                                                  Scheduler <-> API <-> User
```
