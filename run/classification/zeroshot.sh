#!/bin/bash

# CUDA
export CUDA_VISIBLE_DEVICES=$1

# dataset
DATA_DIR=/mnt/sharedata/ssd/common/datasets/
new_class_datasets=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101" "imagenet")
seed=1
SHOTS=16

# model
BACKBONE=vit_b16 # ("rn50" "vit_b32" "vit_b16" "vit_l14")

# trainer
TRAINER=ZeroshotCLIP

# keywords for evaluation
KEYWORDS=('accuracy' 'confidence' 'ece' 'ace' 'mce' 'piece')


CFG=$BACKBONE

for dataset in "${new_class_datasets[@]}"; do

    # evaluates on base classes
    bash scripts/classification/base2new_zeroshot_base.sh ${TRAINER} ${CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed}
    # evaluates on novel classes
    bash scripts/classification/base2new_zeroshot_new.sh ${TRAINER} ${CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed}

    for keyword in "${KEYWORDS[@]}"; do
        # # prints averaged results for base classes
        python parse_test_res.py output/base2new/train_base/${dataset}/shots_${SHOTS}/ZeroshotCLIP/${CFG} --test-log --keyword ${keyword}
        # # averaged results for novel classes
        python parse_test_res.py output/base2new/test_new/${dataset}/shots_${SHOTS}/ZeroshotCLIP/${CFG}  --test-log --keyword ${keyword}
    done
done
