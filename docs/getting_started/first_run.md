# Getting Started

We recommend starting with a single host first and then moving to multihost.

## Getting Started: Local Development for single host

#### Running on Cloud TPUs
Local development is a convenient way to run MaxDiffusion on a single host. It doesn't scale to
multiple hosts.

1. [Create and SSH to a single-host TPU (v4-8). ](https://cloud.google.com/tpu/docs/users-guide-tpu-vm#creating_a_cloud_tpu_vm_with_gcloud)
1. Clone MaxDiffusion in your TPU VM.
1. Within the root directory of the MaxDiffusion `git` repo, install dependencies by running:
```bash
pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -r requirements.txt
pip3 install .
```

## Getting Starting: Multihost development

[GKE, recommended] [Running MaxDiffusion with xpk](run_maxdiffusion_via_xpk.md) - Quick Experimentation and Production support
