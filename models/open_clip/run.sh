#!/bin/bash
set -x
# if [ "$#" -ne 8 ]
# then
#     echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist"
#     exit 1
# fi

export NNODE=1
export NODELIST="nico3"
export NTASK_PER_NODE=2
export PYTHONENV="/home/zanzong/workspace/deepspeed-env/bin/activate"
export TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")

# --gres=gpu:[v116p,v132p]:$NTASK_PER_NODE # for allocate specific 32GB/16GB cards
# --gres=gpu:$NTASK_PER_NODE # no specific
srun --exclusive=user \
    -p V100 \
    -N $NNODE \
    -w $NODELIST \
    -K \
    --time 30:00 \
    --job-name="clip" \
    --ntasks-per-node=$NTASK_PER_NODE \
    --gres=gpu:v132p:$NTASK_PER_NODE \
    --export=ALL \
    bash train.sh
