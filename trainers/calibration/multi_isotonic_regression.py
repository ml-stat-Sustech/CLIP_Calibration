from sklearn.isotonic import IsotonicRegression
import numpy as np
from sklearn.preprocessing import label_binarize


class MultiIsotonicRegression():
    """multi-class isotonic regression adopted from https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/e41afbaf8181a0bd2fb194f9e9d30bcbe5b7f6c3/util_calibration.py"""
    
    def __init__(self) -> None:
        self.__name__ = 'MultiIsotonicRegression'
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        
    def fit_transform(self, logit, label):
        # logit: [samples, classes]
        # label: [samples, classes]
        
        n_classes = logit.shape[1]
        if len(label.shape) == 1:
            if n_classes == 2:
                one_hot_encoded_labels = np.zeros((len(label), n_classes))
                one_hot_encoded_labels[np.arange(len(label)), label.flatten()] = 1
                label = one_hot_encoded_labels
            elif n_classes > 2:
                label = label_binarize(label, classes=np.arange(n_classes))
        
        p = np.exp(logit)/np.sum(np.exp(logit),1)[:,None] 
        y_ = self.calibrator.fit_transform(p.flatten(), (label.flatten()))
        p = y_.reshape(logit.shape) + 1e-9 * p
        
        return p
    
    def transform(self, logit):
        p = np.exp(logit)/np.sum(np.exp(logit),1)[:,None] 
        y_ = self.calibrator.predict(p.flatten())
        p = y_.reshape(logit.shape) + 1e-9 * p
        return p