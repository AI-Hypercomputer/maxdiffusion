# Getting Started

We recommend starting with a single host first and then moving to multihost.

## Getting Started: Local Development for single host

#### Running on Cloud TPUs
Local development is a convenient way to run MaxDiffusion on a single host. It doesn't scale to
multiple hosts.

1. [Create and SSH to a single-host TPU (v6-8). ](https://cloud.google.com/tpu/docs/users-guide-tpu-vm#creating_a_cloud_tpu_vm_with_gcloud)
* You can find here [here](https://cloud.google.com/tpu/docs/regions-zones) the list of zones that support the v6(Trillium) TPUs
* We recommend using the base VM image "v2-alpha-tpuv6e", which meets the version requirements: Ubuntu Version 22.04, Python 3.12 and Tensorflow >= 2.12.0
   
1. Clone MaxDiffusion in your TPU VM.
```
git clone https://github.com/AI-Hypercomputer/maxdiffusion.git
cd maxdiffusion
```

1. Within the root directory of the MaxDiffusion `git` repo, install dependencies by running:
```
# If a Python 3.12+ virtual environment doesn't already exist, you'll need to run the install command twice.
bash setup.sh MODE=stable DEVICE=tpu
```

1. Active your virtual environment:
```
# Replace with your virtual environment name if not using this default name
venv_name="maxdiffusion_venv"
source ~/$venv_name/bin/activate
```

## Getting Starting: Multihost development

[GKE, recommended] [Running MaxDiffusion with xpk](run_maxdiffusion_via_xpk.md) - Quick Experimentation and Production support

