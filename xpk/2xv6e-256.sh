#!/bin/bash
export PROJECT_ID=tpu-prod-env-one-vm
export ZONE=us-east5-b
TPU_TYPE=v6e-256

CLUSTER_NAME=bodaborg-v6e-256-dnd-yucmhab-new

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

uuid=$(uuidgen)

MAX_TRAIN_STEPS=2500
PER_DEVICE_BATCH_SIZE=2
LR=0.00032740000000000004
WARM_UP=0.36
SEED=1002
INIT_LR=0

python3 ~/dev/xpk/xpk.py  workload create --cluster  $CLUSTER_NAME  --workload "$USER"-maxdiffusion-"${uuid:0:8}"  --command "USER=$USER MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} LR=${LR} WARM_UP=${WARM_UP} SEED=${SEED} INIT_LR=${INIT_LR} \
RUN_NAME=mlperf_${TPU_TYPE}_${uuid:0:8} METRICS_INTERVAL=500 bash xpk/run.sh"  \
--base-docker-image=maxdiffusion_base_image \
--tpu-type=${TPU_TYPE} --num-slices=2 --zone=$ZONE --project=$PROJECT_ID