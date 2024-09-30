# How to run MaxDiffusion with XPK?

This document focuses on steps required to setup XPK on TPU VM and assumes you have gone through the [README](https://github.com/google/xpk/blob/main/README.md) to understand XPK basics.

## Steps to setup XPK on TPU VM

* Verify you have these permissions for your account or service account

    Storage Admin \
    Kubernetes Engine Admin

* gcloud is installed on TPUVMs using the snap distribution package. Install kubectl using snap
```shell
sudo apt-get update
sudo apt install snapd
sudo snap install kubectl --classic
```
* Install `gke-gcloud-auth-plugin`
```shell
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

sudo apt update && sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin
```

* Authenticate gcloud installation by running this command and following the prompt
```
gcloud auth login
```

* Run this command to configure docker to use docker-credential-gcloud for GCR registries:
```
gcloud auth configure-docker
```

* Test the installation by running
```
docker run hello-world
```

* If getting a permission error, try running
```
sudo usermod -aG docker $USER
```
after which log out and log back in to the machine.

## Build Docker Image for MaxDiffusion

1. Git clone MaxDiffusion locally

    ```shell
    git clone https://github.com/google/MaxDiffusion.git
    cd MaxDiffusion
    ```
2. Build local MaxDiffusion docker image

    This only needs to be rerun when you want to change your dependencies. This image may expire which would require you to rerun the below command

    ```shell
    # Default will pick stable versions of dependencies
    bash docker_build_dependency_image.sh
    ```
3. After building the dependency image `maxdiffusion_base_image`, xpk can handle updates to the working directory when running `xpk workload create` and using `--base-docker-image`.

    See details on docker images in xpk here: https://github.com/google/xpk/blob/main/README.md#how-to-add-docker-images-to-a-xpk-workload

    **Note:** When using the XPK command, ensure you include `pip install .` to install the package from the current directory. This is necessary because the container is created from a copy of your local directory, and `pip install .` ensures any local changes you've made are applied within the container. 

    __Using xpk to upload image to your gcp project and run MaxDiffusion__

      ```shell
      gcloud config set project $PROJECT_ID
      gcloud config set compute/zone $ZONE

      # See instructions in README.me to create below buckets.
      BASE_OUTPUT_DIR=gs://output_bucket/
      DATASET_PATH=gs://dataset_bucket/

      # Install xpk
      pip install xpk

      # Make sure you are still in the MaxDiffusion github root directory when running this command
      xpk workload create \
      --cluster ${CLUSTER_NAME} \
      --base-docker-image maxDiffusion_base_image \
      --workload ${USER}-first-job \
      --tpu-type=v4-8 \
      --num-slices=1  \
      --command "pip install . && python src/maxdiffusion/train.py src/maxdiffusion/configs/base_2_base.yml run_name="my_run" output_dir="gs://your-bucket/""
      ```

      __Using [xpk github repo](https://github.com/google/xpk.git)__

      ```shell
      git clone https://github.com/google/xpk.git

      # Make sure you are still in the MaxDiffusion github root directory when running this command
      python3 xpk/xpk.py workload create \
      --cluster ${CLUSTER_NAME} \
      --base-docker-image maxDiffusion_base_image \
      --workload ${USER}-first-job \
      --tpu-type=v4-8 \
      --num-slices=1  \
      --command "pip install . && python src/maxdiffusion/train.py src/maxdiffusion/configs/base_2_base.yml run_name="my_run" output_dir="gs://your-bucket/""
      ```