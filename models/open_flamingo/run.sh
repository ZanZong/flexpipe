#!/bin/bash
set -x
# if [ "$#" -ne 8 ]
# then
#     echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist"
#     exit 1
# fi

export NNODE=1
export NODELIST="nico3"
export GPU="v132p"
export NTASK_PER_NODE=1
export TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")

srun --exclusive=user \
    -N $NNODE \
    -w $NODELIST \
    -K \
    --time 30:00 \
    --job-name="clip" \
    --gres=gpu:$GPU:$NTASK_PER_NODE \
    --export=ALL \
    bash train.sh
