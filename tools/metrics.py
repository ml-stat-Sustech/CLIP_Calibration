'''
    https://github.com/MiaoXiong2320/ProximityBias-Calibration/blob/main/utils/metrics.py
    https://github.com/markus93/NN_calibration/blob/eb235cdba006882d74a87114a3563a9efca691b7/scripts/utility/evaluation.py
    https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py
    
    This file contains the code for evaluation metrics:
    - ECE 
    - MCE
    - Dist-aware ECE
    - Adaptive ECE
    ...
'''

import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import pandas as pd
import time, pdb
from sklearn.metrics import log_loss, brier_score_loss
import sklearn.metrics as metrics
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import average_precision_score, roc_auc_score, auc
import sys
from os import path
# from KDEpy import FFTKDE

import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin



   
# def ECE(conf, pred, gt, conf_bin_num = 10):

#     """
#     Expected Calibration Error
    
#     Args:
#         conf (numpy.ndarray): list of confidences
#         pred (numpy.ndarray): list of predictions
#         true (numpy.ndarray): list of true labels
#         bin_size: (float): size of one bin (0,1)  
        
#     Returns:
#         ece: expected calibration error
#     """
#     df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
#     df['correct'] = (df.pred == df.ys).astype('int')
    

#     bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
#     df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
#     # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
#     # groupy by knn + conf
#     group_acc = df.groupby(['conf_bin'])['correct'].mean()
#     group_confs = df.groupby(['conf_bin'])['conf'].mean()
#     counts = df.groupby(['conf_bin'])['conf'].count()
#     ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
#     return ece

def ECE(conf, pred, gt, conf_bin_num = 10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    bins = np.linspace(0, 1, conf_bin_num+1)
    bin_indices = np.digitize(conf, bins) - 1

    bin_acc = []
    bin_confidences = []
    for i in range(conf_bin_num):

        in_bin = bin_indices == i

        if np.sum(in_bin) > 0:
            accuracy = np.mean(gt[in_bin] == pred[in_bin])
            mean_confidence = np.mean(conf[in_bin])
        else:
            accuracy = 0
            mean_confidence = 0
        bin_acc.append(accuracy)
        bin_confidences.append(mean_confidence)


    bin_acc = np.array(bin_acc)
    bin_confidences = np.array(bin_confidences)


    weights = np.histogram(conf, bins)[0] / len(conf)
    ece = np.sum(weights * np.abs(bin_confidences - bin_acc))
        
    return ece
     
def PIECE(conf, knndist, pred, gt, dist_bin_num =10, conf_bin_num = 10, knn_strategy='quantile'):

    """
    Proximity-Informed Expected Calibration Error 
    
    Args:
        conf (numpy.ndarray): list of confidences
        knndist (numpy.ndarray): list of distances of which a sample to its K nearest neighbors
        pred (numpy.ndarray): list of predictions
        gt (numpy.ndarray): list of true labels
        dist_bin_num: (float): the number of bins for knndist
        conf_bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    
    
    df = pd.DataFrame({'ys':gt, 'knndist':knndist, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')
    df['knn_bin'] = KBinsDiscretizer(n_bins=dist_bin_num, encode='ordinal',strategy=knn_strategy).fit_transform(knndist[:, np.newaxis])
    
    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['knn_bin', 'conf_bin'])['correct'].mean()
    group_confs = df.groupby(['knn_bin', 'conf_bin'])['conf'].mean()
    counts = df.groupby(['knn_bin', 'conf_bin'])['conf'].count()
    ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    
    # group by only knn
    # group_acc = df.groupby(['knn_bin'])['correct'].mean()
    # group_confs = df.groupby(['knn_bin'])['conf'].mean()
    # counts = df.groupby(['knn_bin'])['conf'].count()
    # ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    
    
    # n = len(conf)
    # ece = 0  # Starting error
    # upper_bounds = np.arange(conf_bin_size, 1+conf_bin_size, conf_bin_size)  # Get bounds of bins
    # for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
    #     acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-conf_bin_size, conf_thresh, conf, pred, gt)        
    #     ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece


def MCE(conf, pred, gt, conf_bin_num = 10):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        mce: maximum calibration error
    """
    df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')

    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    mce = (np.abs(group_acc - group_confs) * counts / len(df)).max()
        
    return mce



def AdaptiveECE(conf, pred, gt, conf_bin_num=10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ace: expected calibration error
    """
    df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')
    df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='quantile').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    ace = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
    return ace