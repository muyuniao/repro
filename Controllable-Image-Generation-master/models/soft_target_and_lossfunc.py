import torch
import torch.nn.functional as F

def l2_loss(pred, target):
    """
    计算 L2 损失 (均方误差)，对应论文中要求的欧氏距离平方
    """
    return F.mse_loss(pred, target, reduction='mean')