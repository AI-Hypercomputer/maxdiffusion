<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# How to run MaxDiffusion at scale with Cluster Toolkit (`gcluster`)

This guide describes the recommended workflow for running MaxDiffusion on Google Kubernetes Engine (GKE) using
**Cluster Toolkit's `gcluster` CLI**. It replaces the previous [XPK guide](run_maxdiffusion_via_xpk.md):
[XPK is deprecated](https://github.com/AI-Hypercomputer/xpk) (see the notice at the top of its README) and new TPU/GPU
generations are only supported through Cluster Toolkit.

For a complete reference see the [Cluster Toolkit repository](https://github.com/GoogleCloudPlatform/cluster-toolkit),
the [Google Cloud documentation](https://docs.cloud.google.com/cluster-toolkit/docs/overview), the
[`gcluster` job submission guide](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/docs/gcluster_job_guide.md)
and the official [XPK → Cluster Toolkit migration guide](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/docs/migration/xpk_to_clustertoolkit.md).

> [!IMPORTANT]
> The `gcluster` commands in this guide (and the matching ones in the main [README](../../README.md)) were translated
> from the previous XPK commands using the official migration guide's 1:1 flag mapping. They have **not yet been
> validated end-to-end on a cluster** by the MaxDiffusion team. If something does not work as written, please
> [open an issue](https://github.com/AI-Hypercomputer/maxdiffusion/issues) so we can correct it.

## Overview of the workflow

1. **Package MaxDiffusion into a container image.** Build the dependency image once with
   `docker_build_dependency_image.sh`, layer your current checkout on top of it with `maxdiffusion_runner.Dockerfile`
   (takes seconds) and push the result to Artifact Registry.
2. **Submit the workload.** `gcluster job submit --image ...` inspects your GKE cluster, generates the Kubernetes
   resources (a `JobSet` admitted through Kueue) and launches the multi-host job.

```none
+--------------------------+      +--------------------+      +-------------------+
| Your Development Machine +------>  Artifact Registry +------>  GKE Cluster      |
| (docker + gcluster CLI)  |      | (stores images)    |      | (with TPUs/GPUs)  |
|                          |      |                    |      |                   |
| 1. Build & push image    |      | 2. Nodes pull the  |      | 3. gcluster runs  |
|    from local checkout   |      |    image           |      |    multi-host job |
+--------------------------+      +--------------------+      +-------------------+
```

## 1. Prerequisites

### Required tools

* **Google Cloud CLI (`gcloud`)** – install from [here](https://docs.cloud.google.com/sdk/docs/install-sdk) and run `gcloud init`.
* **kubectl and the GKE auth plugin**:
  ```bash
  gcloud components install kubectl gke-gcloud-auth-plugin
  ```
  If `gcloud` was installed through a package manager (apt or snap, which is the case on TPU VMs), `gcloud components`
  is disabled; install `kubectl` and `google-cloud-cli-gke-gcloud-auth-plugin` with that package manager instead (the
  [XPK guide](run_maxdiffusion_via_xpk.md#steps-to-setup-xpk-on-tpu-vm) lists the apt commands).
* **`gcluster` CLI** (v1.103.0 or later, see the Cluster Toolkit
  [security bulletins](https://docs.cloud.google.com/cluster-toolkit/docs/security-bulletins)) – follow the
  [Cluster Toolkit setup guide](https://docs.cloud.google.com/cluster-toolkit/docs/setup/configure-environment)
  or download a release bundle from the [releases page](https://github.com/GoogleCloudPlatform/cluster-toolkit/releases)
  and make sure `gcluster` is on your `$PATH`:
  ```bash
  TAG=vX.Y.Z   # see https://github.com/GoogleCloudPlatform/cluster-toolkit/releases
  # Bundles are also published for linux_arm64, mac_amd64 and mac_arm64.
  mkdir -p cluster-toolkit && \
    curl -fL https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/download/${TAG?}/gcluster_bundle_linux_amd64.tgz \
    | tar -xz -C cluster-toolkit
  export PATH="$PWD/cluster-toolkit:$PATH"
  ```
* **Docker** – used to build the MaxDiffusion images. Configure credentials for your Artifact Registry region:
  ```bash
  gcloud auth configure-docker <REGION>-docker.pkg.dev --quiet
  ```

### Google Cloud APIs and permissions

```bash
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com
```

Your account needs at least the following IAM roles in the target project:

* Artifact Registry Writer
* Kubernetes Engine Admin (or Developer on an existing cluster)
* Storage Admin (GCS buckets for datasets, checkpoints and outputs)
* Logging / Monitoring Viewer

### A GKE cluster with accelerators

This guide assumes you already have a GKE cluster with TPU (or GPU) node pools and the JobSet and Kueue controllers
installed. Clusters are provisioned with declarative blueprints via `gcluster deploy`; see the
[Cluster Toolkit GKE examples](https://github.com/GoogleCloudPlatform/cluster-toolkit/tree/main/examples) and the
*Cluster Infrastructure Migration* section of the
[migration guide](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/docs/migration/xpk_to_clustertoolkit.md)
for a worked example. Clusters created earlier with `xpk cluster create` keep running, but in-place migration is not
supported, so plan to recreate them with `gcluster deploy`.

## 2. Environment configuration

```bash
export PROJECT_ID=<PROJECT_ID>
export LOCATION=<ZONE_OR_REGION>   # e.g. us-east5-a (zonal cluster) or europe-west4 (regional cluster)
export CLUSTER_NAME=<CLUSTER_NAME>
export REGION=<REGION>             # Artifact Registry region, e.g. us-east5
export AR_REPO=maxdiffusion-images # Artifact Registry repository name

# Workload name: max 28 characters, lowercase alphanumerics and hyphens only (adjust if ${USER} does not comply).
export RUN_NAME=${USER}-first-job

gcloud config set project ${PROJECT_ID?}
gcloud container clusters get-credentials ${CLUSTER_NAME?} --location ${LOCATION?} --project ${PROJECT_ID?}

gcluster job config set project ${PROJECT_ID?}
gcluster job config set cluster ${CLUSTER_NAME?}
gcluster job config set location ${LOCATION?}
```

Before submitting, `gcluster` runs prerequisite checks (gcloud authentication and Application Default Credentials,
`kubectl`, the GKE auth plugin, the Docker credential helper and the Artifact Registry API) and prints remediation
commands for anything that is missing. It then verifies (and installs if needed) the JobSet CRD on the cluster and
auto-discovers the Kueue `LocalQueue` to submit to (override it with `--queue`).

## 3. Build and push the MaxDiffusion image

MaxDiffusion does not publish a public base image, so build one from this repository. Run everything in this section
from the **root of your MaxDiffusion checkout**.

```bash
# One-time: create the Artifact Registry repository.
gcloud artifacts repositories create ${AR_REPO?} \
  --repository-format=docker \
  --location=${REGION?} \
  --description="MaxDiffusion container images"

# 1. Dependency image (slow, minutes). Produces the local tag maxdiffusion_base_image and only needs to be
#    rebuilt when dependencies change. MODE=stable (default) or MODE=nightly; see the script header for all options.
bash docker_build_dependency_image.sh MODE=stable

# 2. Runner image (fast, seconds): copies your current checkout into /deps on top of the dependency image.
docker build --build-arg BASEIMAGE=maxdiffusion_base_image -f maxdiffusion_runner.Dockerfile -t maxdiffusion_runner .

# 3. Push it to Artifact Registry.
export IMAGE=${REGION?}-docker.pkg.dev/${PROJECT_ID?}/${AR_REPO?}/maxdiffusion_runner:latest
docker tag maxdiffusion_runner ${IMAGE?}
docker push ${IMAGE?}
```

Repeat steps 2–3 whenever your local code changes (`docker_upload_runner.sh` automates them, but pushes to
`gcr.io/<project>` rather than Artifact Registry).

> [!NOTE]
> The MaxText guide and the migration guide use `gcluster`'s on-the-fly image build (`--base-image` + `--build-context .`)
> instead of a prebuilt image. That mode appends the build context at the **root** of the image filesystem and keeps the
> base image's working directory, whereas MaxDiffusion's images set `WORKDIR /deps` and already contain a copy of the
> source tree there. A command such as `python src/maxdiffusion/train.py` would therefore silently run the copy baked
> into the base image rather than your local changes, which is why this guide uses `--image` with a runner image.

## 4. Submit your first workload

`${IMAGE}` contains your checkout under `/deps`, the container's working directory, so paths in `--command` are
relative to the repository root. The dependency image also installs `maxdiffusion` into site-packages at build time,
while the runner image only refreshes the source tree; prefix the command with `pip install --no-deps . &&` (the XPK
guide used `pip install . &&`) so that your latest changes are the ones imported. You can omit it if you rebuilt the
dependency image from the same checkout.

```bash
export COMPUTE_TYPE=v6e-8   # TPU shorthand; see "Choosing --compute-type" below
export OUTPUT_DIR=gs://<your-bucket>/

gcluster job submit \
  --name=${RUN_NAME?} \
  --compute-type=${COMPUTE_TYPE?} \
  --num-slices=1 \
  --image=${IMAGE?} \
  --command="pip install --no-deps . && python src/maxdiffusion/train.py src/maxdiffusion/configs/base_2_base.yml run_name=${RUN_NAME?} output_dir=${OUTPUT_DIR?}"
```

`--cluster`, `--project` and `--location` can be omitted because they were stored with `gcluster job config set`
above; pass them explicitly to target a different cluster.

For full-scale examples (Wan 2.1 / Wan 2.2 training on 128-device slices, including the recommended `LIBTPU_INIT_ARGS`
and sharding flags) see the [Deploying with Cluster Toolkit](../../README.md#deploying-with-cluster-toolkit) and
[Multi-Host Training with Cluster Toolkit](../../README.md#multi-host-training-with-cluster-toolkit) sections of the
main README.

### Choosing `--compute-type`

* Common shorthands such as `v4-8`, `v6e-8`, `v6e-16`, `l4-8` or `h100-80gb-8` can be passed directly; `gcluster`
  resolves the machine type and topology for you.
* For shapes that are not in the shorthand map (for example `v5e-*`), for multi-host slices where you want to pin the
  shape, and **always for TPU7x**, pass the GCE machine type plus an explicit `--topology`, e.g.
  `--compute-type=ct5p-hightpu-4t --topology=4x4x8` (v5p-256, 128 chips),
  `--compute-type=ct6e-standard-4t --topology=8x16` (v6e-128, 128 chips) or
  `--compute-type=tpu7x-standard-4t --topology=4x4x4` (64 chips; TPU7x exposes two JAX devices per chip, so this is
  also 128 devices).
* `--num-nodes` is for GPU/CPU jobs only; omit it for TPU jobs.

### Environment variables

Environment variables from your shell are **not** forwarded to the job. Pass them explicitly with `--env`, e.g.
`--env HF_HUB_ENABLE_HF_TRANSFER=1` or `--env "LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS}"`. Alternatively prefix them inside
`--command` as the README examples do (`HF_HUB_CACHE=... python ...`).

### Mounting storage

`xpk storage attach` is replaced by the inline `--mount` flag, which accepts GCS buckets, Filestore instances or
PVC claim names, e.g. `--mount 'gs://<bucket>/datasets;/mnt/data;ro'` (add `;options=implicit-dirs` for GCS Fuse
options). MaxDiffusion reads `gs://` paths directly, so mounting is optional.

## 5. Monitoring and managing workloads

```bash
# List active or queued workloads.
gcluster job list

# Stream logs (or use the Cloud Logging link printed by gcluster job submit).
gcluster job logs ${RUN_NAME?}

# Pod status and logs of an individual pod.
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${RUN_NAME?}
kubectl logs -f <POD_NAME>

# Cancel a running workload / clean up a finished JobSet.
gcluster job cancel ${RUN_NAME?}
```

## 6. Mapping from the previous XPK commands

The flags used by MaxDiffusion's old XPK commands map to `gcluster job submit` as follows (full table in the
[migration guide](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/docs/migration/xpk_to_clustertoolkit.md#7-xpk--ct-command-mappings-table)):

| `xpk workload create` | `gcluster job submit` | Notes |
| :--- | :--- | :--- |
| `--workload NAME` | `--name NAME` | Max 28 characters |
| `--cluster` / `--project` | same | Optional once set via `gcluster job config set` |
| `--zone ZONE` | `--location ZONE_OR_REGION` | |
| `--tpu-type` / `--device-type TYPE` | `--compute-type TYPE [--topology T]` | Shorthand or machine type + topology |
| `--num-slices N` | `--num-slices N` | |
| `--base-docker-image IMG` | `--image IMG` | `IMG` must be a registry image, not a local Docker tag such as `maxdiffusion_base_image`: push a runner image built from your checkout (section 3). `--base-image IMG --build-context .` is not suitable for MaxDiffusion images, see the note in section 3 |
| `--docker-image IMG` | `--image IMG` | Pre-built image containing the code |
| `--command "..."` | `--command "..."` | |
| `--env KEY=VAL` | `--env KEY=VAL` | |
| `--priority P` | `--priority P` | `low`, `medium`, `high` |
| `--max-restarts N` | `--restarts N` | |
| `--enable-debug-logs` | `--verbose` | Both set `TPU_STDERR_LOG_LEVEL=0`, `TPU_MIN_LOG_LEVEL=0`, `TF_CPP_MIN_LOG_LEVEL=0` and `TPU_VMODULE=real_program_continuator=1` in the containers |
| `xpk workload list` / `delete` | `gcluster job list` / `gcluster job cancel NAME` | |
| `xpk inspector` | `gcluster job logs NAME` | |

When `run_name` is empty, MaxDiffusion ([pyconfig.py](../../src/maxdiffusion/pyconfig.py)) falls back to the
`JOBSET_NAME` environment variable. XPK injected that variable into every pod; `gcluster` does not, so always pass
`run_name=...` explicitly (as all examples here do) or add `--env JOBSET_NAME=${RUN_NAME?}`.
