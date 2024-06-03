#!/bin/bash

# CUDA
export CUDA_VISIBLE_DEVICES=$1

# dataset
DATA_DIR=/mnt/sharedata/ssd/common/datasets/
new_class_datasets=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101" "imagenet")
seed=1
SHOTS=16


# model
BACKBONE=vit_b16 # ("rn50" "rn101" "vit_b32" "vit_b16")

# trainer
TRAINER=ZeroshotCLIP

# calibrator
CALIBRATION=histogram_binning # "histogram_binning", "isotonic_regression", "multi_isotonic_regression"
DAC=false
PROCAL=false

# keywords for evaluation
KEYWORDS=('accuracy' 'confidence' 'ece' 'mce' 'ace' 'piece')


# temp training
# new_class_datasets=("imagenet")

CFG=$BACKBONE

# calibration config
cal_cfgs='{
    "BASE_CALIBRATION_MODE": "bin_based",
    "SCALING_CALIBRATOR_NAME": null,
    "SCALING_CONFIG": null,
    "BIN_CALIBRATOR_NAME": "'${CALIBRATION}'",
    "IF_DAC": '${DAC}',
    "IF_PROCAL": '${PROCAL}'
}'


for dataset in "${new_class_datasets[@]}"; do

    # evaluates on base classes
    bash scripts/classification/base2new_zeroshot_base.sh ${TRAINER} ${CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed} "${cal_cfgs}"

    # # evaluates on novel classes
    # bash scripts/classification/base2new_zeroshot_new.sh ${TRAINER} ${CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed} "${cal_cfgs}"

    for keyword in "${KEYWORDS[@]}"; do
        # # prints averaged results for base classes
        python parse_test_res.py output/base2new/train_base/${dataset}/shots_${SHOTS}/ZeroshotCLIP/${CFG} --test-log --keyword ${keyword} --calibration-config "${cal_cfgs}"
        # # averaged results for novel classes
        # python parse_test_res.py output/base2new/test_new/${dataset}/shots_${SHOTS}/ZeroshotCLIP/${CFG}  --test-log --keyword ${keyword} --calibration-config "${cal_cfgs}"
    done

done
