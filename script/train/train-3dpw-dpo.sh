#!/usr/bin/env bash

set -x

JOB_NAME=$1
GPUS=$2

# 内容生成一区
# PARTITION=a2584704-17c1-47e9-aeb5-bf500faf7d0f
# RESOURCE=N1lS.Ia.I20.${GPUS}
# 视频生成二区
PARTITION=vigen-2
RESOURCE=N1lS.Ib.I20.${GPUS}


WORK_SPACE=fe790a91-4552-4cfc-bffe-8d7a657aee7b
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
    -o /mnt/AFS_shenwenhao/ScoreHypo/train-3dpw-dpo.log \
    ${SRUN_ARGS} \
    sudo -E -u shenwenhao bash -c " 
    cd /mnt/AFS_shenwenhao/ScoreHypo &&
    /mnt/AFS_shenwenhao/.conda/envs/py38/bin/python \
        -m torch.distributed.launch --nnodes 1 --node_rank 0 --master_addr $master_addr --nproc_per_node=$GPUS_PER_NODE --master_port $MASTER_PROT \
        main/main.py \
        --config config/train/hyponet/3dpw-dpo.yaml \
        --exp experiment/hyponet \
        --doc 3dpw \
        --batch_size 64"


# torchrun --nproc_per_node=2 --master_port=23452 main/main.py --config config/train/hyponet/3dpw.yaml --exp experiment/hyponet --doc 3dpw