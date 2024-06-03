#!/bin/bash

#cd ../..

# custom config
DATA=/data/dataset/
TRAINER=CoOp

SOURCE_DATASET=$1
TARGET_DATASET=$2
SEED=$3
SHOTS=$4
CFG=$5
LOADEP=$6


DIR=output/xd/xd_test/${TRAINER}/${CFG}_${SHOTS}shots/${TARGET_DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${TARGET_DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/xd/xd_train/${SOURCE_DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only
fi