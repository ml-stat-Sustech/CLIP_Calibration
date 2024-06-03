import os.path as osp
from copy import deepcopy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from trainers.classification.base_learner import VLBaseLearner
from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.calibration.basemodel_loader import get_base_model


class ScaleLearner(nn.Module):
    def __init__(self, cfg, dtype):
        super().__init__()
        logit_scale = torch.tensor(4.6052, dtype=dtype)
        self.logit_scale = nn.Parameter(logit_scale)

    def forward(self, ):

        logit_scale = self.logit_scale.exp()

        return logit_scale


class CustomCLIPCalibration(nn.Module):
    def __init__(self, cfg, base_model):
        super().__init__()
        self.logits_encoder = base_model
        self.dtype = base_model.dtype
        self.scale_learner = ScaleLearner(cfg, self.dtype)

    def forward(self, image, label=None):

        _, image_features, text_features = self.logits_encoder(image)

        logit_scale = self.scale_learner()
        logits = logit_scale * image_features @ text_features.t()

        
        return logits, image_features, text_features




@TRAINER_REGISTRY.register()
class TempScaling(VLBaseLearner):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]



    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames


        print("Building base custom CLIP for calibration")
        base_model = get_base_model(cfg, classnames)


        # load tuning stats from origin V-L learner
        base_model = self.load_base_stat(cfg, base_model)

        self.model = CustomCLIPCalibration(cfg, base_model)


        print("Turning off all the gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
                if "scale_learner" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)

        # NOTE: only give scale_learner to the optimizer
        self.optim = build_optimizer(self.model.scale_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("tempscaling", self.model.scale_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            # self.model.text_encoder = nn.DataParallel(self.model.text_encoder)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x # train loader
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

        # NOTE:Use val loader for calibration 
        self.train_loader_x = dm.val_loader


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COOP.PREC

        logits, image_features, text_features = model(image, label)
        loss = F.cross_entropy(logits, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_summary = {
            "loss": loss.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    # load base stats from the tuning model
    def load_base_stat(self, cfg, base_model):

        directory = cfg.CALIBRATION.SCALING.BASE_DIR

        # zero shot do not need stat
        if cfg.CALIBRATION.SCALING.BASE_LEARNER == 'ZeroshotCLIP':
            return base_model

        if cfg.CALIBRATION.SCALING.BASE_LEARNER == 'MaPLe':
            names = ['MultiModalPromptLearner']
        elif cfg.CALIBRATION.SCALING.BASE_LEARNER == 'CLIP_Adapter':
            names = ['adapter']
        else:
            names = ['prompt_learner']

        epoch = cfg.CALIBRATION.SCALING.BASE_EPOCH

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
        

        if cfg.CALIBRATION.SCALING.BASE_LEARNER == 'MaPLe':
            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
        elif cfg.CALIBRATION.SCALING.BASE_LEARNER == 'PromptSRC':
            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
        else:
            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

        print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
        # set strict=False

        
        if cfg.CALIBRATION.SCALING.BASE_LEARNER == 'MaPLe': # load the whole model
            base_model.load_state_dict(state_dict, strict=False)
        elif cfg.CALIBRATION.SCALING.BASE_LEARNER == 'PromptSRC':
            base_model.load_state_dict(state_dict, strict=False)
        else:
            state_dict_ = {}   # only load the prompt_learner
            for key, value in state_dict.items():
                print(key,'keys in base model')
                new_key = f'prompt_learner.{key}' 
                state_dict_[new_key] = value

            base_model.load_state_dict(state_dict_, strict=False)


        # set classifier for ProDA
        if cfg.CALIBRATION.SCALING.BASE_LEARNER == 'ProDA': 
            base_model.set_classifier()


        return base_model

    # load the scale learner
    def load_model(self, directory, epoch=None):
        # load logit_scale
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()


        # By default, the best model is loaded
        model_file = "model-calibrated-best.pth.tar"

        if epoch is not None:
            model_file = "model-calibrated.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]


            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            

            state_dict_ = {}
            for key, value in state_dict.items():
                print(key,'keys in calibration model')
                new_key = f'scale_learner.{key}'  # 在每个 key 前添加 'scale_learner'
                state_dict_[new_key] = value
                
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=True)
            # print(self._models[name].logit_scale, 'logit_scalelogit_scale')



    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-calibrated-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            model_name = "model-calibrated.pth.tar-" + str(self.epoch+1)
            self.save_model(self.epoch, self.output_dir, model_name=model_name)
