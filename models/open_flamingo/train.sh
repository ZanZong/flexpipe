#!/bin/bash
set -x

eval "$(source $PYTHONENV)"
export MASTER_PORT=12802
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODE)
export LOCAL_RANK=$NODE_RANK
export WORLD_SIZE=$(($NTASK_PER_NODE*$NNODE))
export RANK=$SLURM_PROCID

exec python -u open_flamingo/train/train.py \
	--vision_encoder_path "ViT-L-14" \
    --vision_encoder_pretrained "openai" \
	--lm_path "cache/mpt-1b-redpajama-200b" \
	--tokenizer_path "cache/mpt-1b-redpajama-200b" \
	--cross_attn_every_n_layers 1 \
	--batch_size_mmc4 1 \
	--batch_size_laion 1 \
    --train_num_samples_mmc4 2 \
    --train_num_samples_laion 2 \
	--num_epochs 100 \
    --mmc4_textsim_threshold 0.24 \
    --mmc4_shards "mmc4-data/00000.tar" \
    --laion_shards "laion2b-en-data/00000.tar" \
	--offline \
	--freeze_lm_embeddings \
    --gradient_checkpointing \
    --run_name OpenFlamingo-3B-vitl-mpt1b