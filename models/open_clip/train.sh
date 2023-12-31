#!/bin/bash
set -x

eval "$(source $PYTHONENV)"
export MASTER_PORT=12802
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODE)
export LOCAL_RANK=$NODE_RANK
export WORLD_SIZE=$(($NTASK_PER_NODE*$NNODE))
export RANK=$SLURM_PROCID

cd /home/zanzong/workspace/flexpipe/models/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

exec python -u src/training/main.py \
    --deepspeed_config=ds_configs/ds_config.json \
    --save-frequency 1 \
    --train-data="/mnt/zoltan/zanzong/CC3M/cc3m/{00000..00331}.tar" \
    --train-num-samples 3000000 \
    --epochs=1 \
    --model ViT-H-16 \
    --name "ViT-H-16-"$TIMESTAMP \
    --seed 0 \
    --force-patch-dropout 0. \
    --gather-with-grad \
    --enable-deepspeed \
    --enable-flexpipe