#!/bin/bash
set -x

eval "$(source $PYTHONENV)"
export MASTER_PORT=12802
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODE)
export LOCAL_RANK=$NODE_RANK
export WORLD_SIZE=$(($NTASK_PER_NODE*$NNODE))
export RANK=$SLURM_PROCID

cd /home/zanzong/workspace/flexpipe/models/swin_transformer
export PYTHONPATH="$PYTHONPATH:$PWD/src"

exec python -u main.py \
	--master_port 12345 \
	--local_rank $LOCAL_RANK \
	--cfg configs/swin/swin_large_patch4_window7_224_22k.yaml \
	--data-path /mnt/znvme/zms/imagenet22k \
	--output train-output \
	--tag pretrain \
	--enable_deepspeed \
	--enable_pipeline \
	--deepspeed_config ds_configs/ds_config.json
