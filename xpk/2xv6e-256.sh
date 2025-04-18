#!/bin/bash
export PROJECT_ID=tpu-prod-env-one-vm
export ZONE=southamerica-west1-a
# export PROJECT_ID=tpu-prod-env-multipod
# export ZONE=us-west1-c
TPU_TYPE=v6e-256

CLUSTER_NAME=bodaborg-v6e-256-lcscld-c
# CLUSTER_NAME=bodaborg-v6e-256-ts

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

uuid=$(uuidgen)

# GBS=512
MAX_TRAIN_STEPS=5000
PER_DEVICE_BATCH_SIZE=1
LR=0.0001762
WARM_UP=0.2
SEED=4590
INIT_LR=0

# GBS=1024
# MAX_TRAIN_STEPS=2500
# PER_DEVICE_BATCH_SIZE=2
# LR=0.0003274
# WARM_UP=0.36
# SEED=5330
# INIT_LR=0

# GBS=2048
# MAX_TRAIN_STEPS=2000
# PER_DEVICE_BATCH_SIZE=4
# LR=0.0002548
# WARM_UP=0.3375
# SEED=5330
# INIT_LR=0

# GBS=2048
# MAX_TRAIN_STEPS=2000
# PER_DEVICE_BATCH_SIZE=4
# LR=0.0003548
# WARM_UP=0.3375
# SEED=8642
# INIT_LR=0

python3 ~/dev/xpk/xpk.py  workload create --cluster  $CLUSTER_NAME  --workload "$USER"-maxdiffusion-"${uuid:0:8}"  --command "USER=$USER MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} LR=${LR} WARM_UP=${WARM_UP} SEED=${SEED} INIT_LR=${INIT_LR} \
RUN_NAME=mlperf_${TPU_TYPE}_${uuid:0:8} METRICS_INTERVAL=500 bash xpk/run.sh"  \
--base-docker-image=maxdiffusion_base_image \
--tpu-type=${TPU_TYPE} --num-slices=2 --zone=$ZONE --project=$PROJECT_ID --priority=high