#!/bin/bash

a=$(echo $HOSTNAME | cut  -c12-16)

CONFIG="configs/BERT_L12_H768_experiments/training_basedense_h192_flexpipe.yaml"
JOB_NAME="perceiver_6tasks"
GPUS_PER_NODE=8
partition=Big
NNODE=1
NODELIST=nico2


WORK_DIR=${CONFIG//configs/work_dirs}
WORK_DIR=${WORK_DIR//.yaml//$JOB_NAME}
echo $WORK_DIR
mkdir -p $WORK_DIR
mkdir -p data/temp

# please change DATA_PATH where you put the training data
export DATA_PATH='/mnt/zoltan/zanzong/uniperceiver_data'

srun --exclusive=user \
    --partition=${partition} \
    -N $NNODE \
    -w $NODELIST \
    -K \
    --time 30:00 \
    --job-name=${JOB_NAME} \
    --gres=gpu:v132p:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --export=ALL \
    python -u main.py --num-gpus $GPUS_PER_NODE \
    --config-file ${CONFIG} --init_method slurm --resume \
    OUTPUT_DIR $WORK_DIR 

