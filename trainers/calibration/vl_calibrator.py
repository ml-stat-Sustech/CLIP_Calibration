import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from clip import clip
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.special import softmax


from tools.zsclip_encoder import build_zsclip, build_clip_templates

# calibrator
from netcal.binning import HistogramBinning
from netcal.binning import IsotonicRegression
from trainers.calibration.distanse_aware_calibration import DistanseAwareCalibration

from trainers.calibration.multi_isotonic_regression import MultiIsotonicRegression
from trainers.calibration.density_ratio_calibration import DensityRatioCalibration
from trainers.calibration.multi_proximity_isotonic import BinMeanShift

class VLCalibration():

    def __init__(self, cfg, base_calibration_mode=None, base_bin_calibrator_name=None, dac_flag=False, procal_flag=False, val_dict=None, text_feature_dict=None):
        """
        Calibrator for vision-language models.

        Args:
            cfg (Config): Configuration object containing dataset and trainer settings.
            base_calibration_mode (str): Mode of base calibration. Option=[scaling_base, bin_base].
            base_bin_calibrator_name (str): Name of the base binary calibrator. Option=[histogram_binning, isotonic_regression, multi_isotonic_regression]
            dac_flag (bool): Flag indicating if Task Difficult-Aware Calibration is used.
            procal_flag (bool): Flag indicating if proximity-based methods are used.
            val_dict (dict): Dictionary containing validation set data like logits, image and text features, labels, etc.
            text_feature_dict (dict): Dictionary containing text feature of base class and current test class using both zero-shot CLIP and tuned CLIP
        """      
          
        self.cfg = cfg
        self.base_calibration_mode = base_calibration_mode
        self.base_bin_calibrator_name = base_bin_calibrator_name
        self.dac_flag = dac_flag
        self.procal_flag = procal_flag
        self.text_feature_dict = text_feature_dict


        # cfg
        self.dataset_name = cfg.DATASET.NAME
        self.trainer_name = self.cfg.TRAINER.NAME
        self.shots = cfg.DATASET.NUM_SHOTS
        self.k_dac = cfg.CALIBRATION.DAC.K

        # val(calibration) set
        self.val_logits = val_dict['val_logits']
        self.val_probs = softmax(self.val_logits, axis=1) # logits to prob [n, d] -> [n, d]
        self.val_preds = np.argmax(self.val_probs, axis=1) # prob to pred [n, d] -> [n, d]

        self.val_image_features = val_dict['val_image_features']
        self.val_text_features = val_dict['val_text_features']
        self.val_labels = val_dict['val_labels']
        
        self.val_image_knn_dists = val_dict['val_image_knn_dists']
        self.val_image_proximity = np.exp(-np.mean( self.val_image_knn_dists, axis=-1))  # distances to proximity


    def fit(self):
        self.dac_calibrator = None
        self.base_calibrator = None

        if self.dac_flag:
            self.dac_calibrator = self.build_dac_calibrator(self.text_feature_dict, self.k_dac)

        if self.base_calibration_mode is not None:
            self.base_calibrator = self.build_base_calibrator(self.base_bin_calibrator_name, self.val_image_proximity)



    def predict(self, logits, test_proximity):

        assert logits.shape[0] == test_proximity.shape[0], f"Shape mismatch: logits shape {logits.shape[0]} != test_proximity shape {test_proximity.shape[0]}"

        # task difficulty aware calibrator
        if self.dac_calibrator is not None: 
            logits = self.dac_calibrator.predict(logits)
        
        probs = softmax(logits, axis=-1)

        # 
        if self.base_calibrator is not None:

            if self.base_calibration_mode == 'scaling_based' and self.procal_flag:
                probs_calibrated = self.base_calibrator.predict(probs, test_proximity)

            elif self.base_calibration_mode == 'bin_based':
                if self.procal_flag:
                    probs_calibrated  = self.base_calibrator.transform(probs, test_proximity)
                else:
                    probs_calibrated  = self.base_calibrator.transform(probs)

        else:
            probs_calibrated = probs

        
        return probs_calibrated


    def build_base_calibrator(self, base_bin_calibrator_name, val_image_proximity):

        base_calibrator = None

        if self.base_calibration_mode == 'scaling_based':
            if self.procal_flag:
                base_calibrator = DensityRatioCalibration()
                base_calibrator.fit(self.val_probs, self.val_preds, self.val_labels, val_image_proximity)

        elif self.base_calibration_mode == 'bin_based':
            proximity_bin = 5
            if self.procal_flag:
                if base_bin_calibrator_name == 'histogram_binning':
                    base_calibrator = BinMeanShift('histogram_binning', HistogramBinning, bin_strategy='quantile', normalize_conf=False, proximity_bin=proximity_bin, bins=10)
                    probs_val = base_calibrator.fit_transform(self.val_probs, val_image_proximity, self.val_labels)

                elif base_bin_calibrator_name == 'isotonic_regression':
                    base_calibrator = BinMeanShift('isotonic_regression', IsotonicRegression, bin_strategy='quantile', normalize_conf=False, proximity_bin=proximity_bin)
                    probs_val = base_calibrator.fit_transform(self.val_probs, val_image_proximity, self.val_labels)

                elif base_bin_calibrator_name == 'multi_isotonic_regression':
                    base_calibrator = BinMeanShift('multi_isotonic_regression', MultiIsotonicRegression, bin_strategy='quantile', normalize_conf=False, proximity_bin=proximity_bin)
                    probs_val = base_calibrator.fit_transform(self.val_probs, val_image_proximity, self.val_labels)

            else:
                if base_bin_calibrator_name == 'histogram_binning':
                    base_calibrator = HistogramBinning(bins=10)
                    base_calibrator.fit(self.val_probs, self.val_labels)

                elif base_bin_calibrator_name == 'isotonic_regression':
                    base_calibrator = IsotonicRegression()
                    base_calibrator.fit(self.val_probs, self.val_labels)

                elif base_bin_calibrator_name == 'multi_isotonic_regression':
                    base_calibrator = MultiIsotonicRegression()
                    probs_val = base_calibrator.fit_transform(self.val_probs, self.val_labels)

        
        return base_calibrator
        


    
    def build_dac_calibrator(self, text_feature_dict, k_dac):

        """
        Task Difficulty Aware Calibrator

        Args:
        # base_text_features_zs: base class text feature generated by zero-shot CLIP
        # current_text_features_zs: new class text feature generated by zero-shot CLIP
        # base_text_features_tuned: base class text feature generated by few-shot CLIP
        # current_text_features_tuned: new class text feature generated by few-shot CLIP
        # k_dac: the number of top k nearest text features

        """

        print('build task difficulity aware calibrator for open-vocabulary classfication')

        base_text_features_zs = text_feature_dict['base_text_features_zs']
        current_text_features_zs = text_feature_dict['current_text_features_zs']
        base_text_features_tuned = text_feature_dict['base_text_features_tuned']
        current_text_features_tuned = text_feature_dict['current_text_features_tuned']

        dac_calibrator = DistanseAwareCalibration()
        dac_calibrator.fit(base_text_features_zs, current_text_features_zs,\
                                                        base_text_features_tuned, current_text_features_tuned, k=k_dac)
        
        return dac_calibrator
                                                        


    
     