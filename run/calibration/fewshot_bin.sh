#!/bin/bash

# CUDA
export CUDA_VISIBLE_DEVICES=$1

# dataset
DATA_DIR=/mnt/sharedata/ssd/common/datasets/
new_class_datasets=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101" "imagenet")
seeds=(1 2 3)
SHOTS=16

# model
BACKBONE=vit_b16 # ("rn50" "rn101" "vit_b32" "vit_b16")

# trainer
TRAINERS=('CoOp' 'CoCoOp' 'KgCoOp' 'MaPLe' 'ProDA' 'ProGrad' 'PromptSRC')

# calibrator
CALIBRATION=histogram_binning # "histogram_binning", "isotonic_regression", "multi_isotonic_regression"
DAC=false
PROCAL=false

# keywords for evaluation
KEYWORDS=('accuracy' 'confidence' 'ece' 'mce' 'ace' 'piece')


for TRAINER in "${TRAINERS[@]}"; do

    # train cfg
    if [ "${TRAINER}" == "CoOp" ]; then
        EPOCH=200
        BATCH_SIZE=32
        N_CTX=16
    elif [ "${TRAINER}" == "CoCoOp" ]; then
        EPOCH=10
        BATCH_SIZE=1
        N_CTX=4
    elif [ "${TRAINER}" == "KgCoOp" ]; then
        EPOCH=200
        BATCH_SIZE=32
        N_CTX=16
    elif [ "${TRAINER}" == "MaPLe" ]; then
        EPOCH=5
        BATCH_SIZE=4
        N_CTX=2
    elif [ "${TRAINER}" == "ProDA" ]; then
        EPOCH=100
        BATCH_SIZE=4
        N_CTX=16
    elif [ "${TRAINER}" == "ProGrad" ]; then
        EPOCH=100
        BATCH_SIZE=32
        N_CTX=16
    elif [ "${TRAINER}" == "PromptSRC" ]; then
        EPOCH=50
        BATCH_SIZE=4
        N_CTX=4
    else
    echo "Unknown trainer: ${TRAINER}"
    exit 1 
    fi


    LOADEP=${EPOCH} # use last epoch
    TRAINER_CFG=${BACKBONE}_c${N_CTX}_ep${EPOCH}_batch${BATCH_SIZE} # build trainer cfg

    # calibration config
    cal_cfgs='{
        "BASE_CALIBRATION_MODE": "bin_based",
        "SCALING_CALIBRATOR_NAME": null,
        "SCALING_CONFIG": null,
        "BIN_CALIBRATOR_NAME": "'${CALIBRATION}'",
        "IF_DAC": '${DAC}',
        "IF_PROCAL": '${PROCAL}'
    }'
    

    # few-shot on datasets
    for dataset in "${new_class_datasets[@]}"; do

        for seed in "${seeds[@]}"; do
            # trains and evaluates on base classes
            bash scripts/classification/base2new_fewshot_train.sh ${TRAINER} ${TRAINER_CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed} "${cal_cfgs}"
            # evaluates on novel classes
            # bash scripts/classification/base2new_fewshot_test.sh ${TRAINER} ${TRAINER_CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed} ${LOADEP} "${cal_cfgs}"
        done

        for keyword in "${KEYWORDS[@]}"; do
            # # prints averaged results for base classes
            python parse_test_res.py output/base2new/train_base/${dataset}/shots_${SHOTS}/${TRAINER}/${TRAINER_CFG} --test-log --keyword ${keyword} --calibration-config "${cal_cfgs}"
            # # averaged results for novel classes
            # python parse_test_res.py output/base2new/test_new/${dataset}/shots_${SHOTS}/${TRAINER}/${TRAINER_CFG} --test-log --keyword ${keyword} --calibration-config "${cal_cfgs}"
        done

    done

done


