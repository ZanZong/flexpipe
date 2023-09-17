#!/bin/bash

a=$(echo $HOSTNAME | cut  -c12-16)

CONFIG="configs/BERT_L12_H192_experiments/4tasks_training_small_datasets.yaml"
JOB_NAME="uni_perceiver_toy"
GPUS_PER_NODE=4
partition=V100
NNODE=1
NODELIST=nico4


WORK_DIR=${CONFIG//configs/work_dirs}
WORK_DIR=${WORK_DIR//.yaml//$JOB_NAME}
echo $WORK_DIR
mkdir  -p $WORK_DIR
mkdir -p data/temp

# please change DATA_PATH where you put the training data
export DATA_PATH='/home/zanzong/datasets/uni_perceiver'

srun --exclusive=user \
    --partition=${partition} \
    -N $NNODE \
    -w $NODELIST \
    -K \
    --time 30:00 \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --export=ALL \
    python -u main.py --num-gpus $GPUS_PER_NODE \
    --config-file ${CONFIG} --init_method slurm --resume \
    OUTPUT_DIR $WORK_DIR 

