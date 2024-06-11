import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from trainers.calibration.base_model.coop import CustomCLIP as CoOpModel
from trainers.calibration.base_model.cocoop import CustomCLIP as CoCoOpModel
from trainers.calibration.base_model.kgcoop import CustomCLIP as KgCoOpModel
from trainers.calibration.base_model.maple import CustomCLIP as MaPLeModel
from trainers.calibration.base_model.proda import CustomCLIP as ProDAModel
from trainers.calibration.base_model.prograd import CustomCLIP as ProgradModel
from trainers.calibration.base_model.clip_adapter import CustomCLIP as CLIPAdapterModel
from trainers.calibration.base_model.zsclip import CustomCLIP as ZeroShotModel
from trainers.calibration.base_model.promptsrc import CustomCLIP as PromptSRCModel


def get_base_model(cfg, classnames):

    model_name = cfg.CALIBRATION.SCALING.BASE_LEARNER

    models = {
        'coop': CoOpModel,
        'cocoop': CoCoOpModel,
        'kgcoop': KgCoOpModel,
        'maple': MaPLeModel,
        'proda': ProDAModel,
        'prograd': ProgradModel,
        'promptsrc': PromptSRCModel,
        'clip_adapter': CLIPAdapterModel,
        'zeroshotclip': ZeroShotModel
    }

    model = models.get(model_name.lower())
    
    if model:
        return model(cfg, classnames)
    else:
        raise ValueError(f'Unknown model: {model_name}')




def count_unique_labels_in_dataloader(dataloader):

    unique_labels = set()


    for batch in dataloader:
        labels = batch[1] 
        unique_labels.update(labels.tolist())

    num_unique_labels = len(unique_labels)

    return num_unique_labels