import torch
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, f1_score, cohen_kappa_score
from scipy.stats import spearmanr

def compute_metrics(preds, targets):
    acc = accuracy_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    macro_f1 = f1_score(targets, preds, average='macro')
    qwk = cohen_kappa_score(targets, preds, weights='quadratic')
    
    if len(np.unique(preds)) > 1 and len(np.unique(targets)) > 1:
        rho, _ = spearmanr(targets, preds)
    else:
        rho = 0.0
        
    diffs = np.abs(np.array(targets) - np.array(preds))
    cs1 = np.mean(diffs <= 1)
    
    return {
        'acc': acc,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'macro_f1': macro_f1,
        'qwk': qwk,
        'spearman': rho,
        'cs1': cs1
    }

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
