#!/usr/bin/env bash

set -x

JOB_NAME=$1
GPUS=$2

PARTITION=a2584704-17c1-47e9-aeb5-bf500faf7d0f
WORK_SPACE=fe790a91-4552-4cfc-bffe-8d7a657aee7b
RESOURCE=N1lS.Ia.I20.${GPUS}
CONTAINER_IMAGE=registry.st-sh-01.sensecore.cn/zoetrope/shenwenhao-image:20240126-19h20m44s


GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}
MASTER_PROT=23452
master_addr=127.0.0.1


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    -f pt \
    --workspace-id ${WORK_SPACE} \
    --resource ${RESOURCE} \
    --job-name ${JOB_NAME} \
    --container-image ${CONTAINER_IMAGE} \
    -o /mnt/AFS_shenwenhao/ScoreHypo/debug.log \
    ${SRUN_ARGS} \
    sudo -E -u shenwenhao bash -c " 
    cd /mnt/AFS_shenwenhao/ScoreHypo &&
    sleep 1d"


# torchrun --nproc_per_node=2 --master_port=23452 main/main.py --config config/train/hyponet/3dpw.yaml --exp experiment/hyponet --doc 3dpw