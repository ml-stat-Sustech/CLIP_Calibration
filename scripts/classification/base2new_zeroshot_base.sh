#!/bin/bash

# custom config

# trainer
TRAINER=$1
CFG=$2

# dataset
DATASET=$3
DATA=$4
SHOTS=$5
SUB=base

SEED=$6

# calirbation config
CALIBRATION_CFG=$7

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--calibration-config "${CALIBRATION_CFG}" \
--output-dir output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES ${SUB} \
DATASET.NUM_SHOTS ${SHOTS} \
