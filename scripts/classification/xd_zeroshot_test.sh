#!/bin/bash


DATA=/data/dataset/
TRAINER=ZeroshotCLIP

DATASET=$1
CFG=$2 # rn50, rn101, vit_b32 or vit_b16
SHOTS=$3


DIR=output/xd/xd_test/${TRAINER}/${CFG}/${DATASET}/seeds
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/ZS/${CFG}.yaml \
    --output-dir ${DIR} \
    --eval-only
    DATASET.NUM_SHOTS ${SHOTS}
fi