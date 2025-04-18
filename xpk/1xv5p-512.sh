#!/bin/bash
export PROJECT_ID=cloud-tpu-multipod-dev
export ZONE=europe-west4-b
# export PROJECT_ID=tpu-prod-env-multipod
# export ZONE=us-west1-c
TPU_TYPE=v5p-512

CLUSTER_NAME=mohitkhatwani-v5p-512
# CLUSTER_NAME=bodaborg-v6e-256-ts

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

uuid=$(uuidgen)

# GBS=256
# MAX_TRAIN_STEPS=8000
# PER_DEVICE_BATCH_SIZE=1
# LR=7.56e-05
# WARM_UP=0.25
# SEED=4498
# INIT_LR=0

# GBS=512
MAX_TRAIN_STEPS=4000
PER_DEVICE_BATCH_SIZE=2
LR=0.0001762
WARM_UP=0.25
SEED=1854
INIT_LR=0

# GBS=1024
# MAX_TRAIN_STEPS=3000
# PER_DEVICE_BATCH_SIZE=4
# LR=0.00032740000000000004
# WARM_UP=0.3
# SEED=4258
# INIT_LR=0

# GBS=1024
# MAX_TRAIN_STEPS=3000
# PER_DEVICE_BATCH_SIZE=4
# LR=0.0004024
# WARM_UP=0.33
# SEED=1002
# INIT_LR=0

python3 ~/dev/xpk/xpk.py  workload create --cluster  $CLUSTER_NAME  --workload "$USER"-maxdiffusion-"${uuid:0:8}"  --command "USER=$USER MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} LR=${LR} WARM_UP=${WARM_UP} SEED=${SEED} INIT_LR=${INIT_LR} \
RUN_NAME=mlperf_${TPU_TYPE}_${uuid:0:8} METRICS_INTERVAL=500 bash xpk/run.sh"  \
--base-docker-image=maxdiffusion_base_image \
--tpu-type=${TPU_TYPE} --num-slices=1 --zone=$ZONE --project=$PROJECT_ID --priority=high
