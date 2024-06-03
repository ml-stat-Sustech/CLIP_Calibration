#!/bin/bash

# CUDA
export CUDA_VISIBLE_DEVICES=$1

# dataset
DATA_DIR=/mnt/sharedata/ssd/common/datasets/
new_class_datasets=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101" "imagenet")
seeds=(1)
SHOTS=16

# model
BACKBONE=vit_b16

# base trainer
TRAINERS=('ZeroshotCLIP')

# calibrator
CALIBRATION=TempScaling # None, TempScaling
DAC=false # true, false
PROCAL=false # true, false


KEYWORDS=('accuracy' 'confidence' 'ece' 'mce'  'ace' 'piece')


for TRAINER in "${TRAINERS[@]}"; do

    # caliration cfg
    if [ "${CALIBRATION}" == "TempScaling" ]; then
        CALIBRATION_EPOCH=20
        CALIBRATION_LR='5e-2'
    elif [ "${CALIBRATION}" == "None" ]; then
        CALIBRATION_EPOCH=0 # only use dac and procal, do not use other scaling
        CALIBRATION_LR=0
    else
    echo "Unknown calibration: ${CALIBRATION}"
    exit 1 
    fi
    
    # build trainer config
    TRAINER_CFG=${BACKBONE}

     # build calibrtor config
    CALIBRATION_CFG=ep${CALIBRATION_EPOCH}_lr${CALIBRATION_LR}
    LOADEP=${CALIBRATION_EPOCH}

    # calibration config
    if [ "${CALIBRATION}" == "None" ]; then
        cal_cfgs='{
            "BASE_CALIBRATION_MODE": "scaling_based",
            "SCALING_CALIBRATOR_NAME": null,
            "SCALING_CONFIG": null,
            "BIN_CALIBRATOR_NAME": null,
            "IF_DAC": '${DAC}',
            "IF_PROCAL": '${PROCAL}'
        }'
    else
        cal_cfgs='{
            "BASE_CALIBRATION_MODE": "scaling_based",
            "SCALING_CALIBRATOR_NAME": "'"${CALIBRATION}"'",
            "SCALING_CONFIG": "'"configs/calibration/${CALIBRATION}/${CALIBRATION_CFG}.yaml"'",
            "BIN_CALIBRATOR_NAME": null,
            "IF_DAC": '${DAC}',
            "IF_PROCAL": '${PROCAL}'
        }'
    fi


    # calibration on datasets
    for dataset in "${new_class_datasets[@]}"; do
        for seed in "${seeds[@]}"; do

            # zero-shot inference or scaling training & inference
            if [ "${CALIBRATION}" == "None" ]; then
                # evaluates on base classes
                bash scripts/classification/base2new_zeroshot_base.sh ${TRAINER} ${TRAINER_CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed} "${cal_cfgs}"
                # evaluates on novel classes
                bash scripts/classification/base2new_zeroshot_new.sh ${TRAINER} ${TRAINER_CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed} "${cal_cfgs}"
            else
                bash scripts/calibration/base2new_scaling_train.sh ${TRAINER} ${TRAINER_CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed} "${cal_cfgs}"
                # evaluates on novel classes
                bash scripts/calibration/base2new_scaling_test.sh ${TRAINER} ${TRAINER_CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed} ${LOADEP} "${cal_cfgs}"
            fi

        done

        for keyword in "${KEYWORDS[@]}"; do
            # # prints averaged results for base classes
            python parse_test_res.py output/base2new/train_base/${dataset}/shots_${SHOTS}/${TRAINER}/${TRAINER_CFG} --test-log --keyword ${keyword} --calibration-config "${cal_cfgs}"
            # # averaged results for novel classes
            python parse_test_res.py output/base2new/test_new/${dataset}/shots_${SHOTS}/${TRAINER}/${TRAINER_CFG} --test-log --keyword ${keyword} --calibration-config "${cal_cfgs}"
        done


    done

done