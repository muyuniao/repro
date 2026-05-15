import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSFDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, p_hat, target_pi):
        # target_pi can be actual ranks
        B, K, _ = p_hat.shape
        j = torch.arange(K, device=p_hat.device, dtype=p_hat.dtype) # [K]
        
        expected_ranks = torch.sum(p_hat * j.unsqueeze(0).unsqueeze(0), dim=-1)
        
        loss = torch.abs(expected_ranks - target_pi.float()).mean()
        return loss

class CORALLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, logits, target):
        B = logits.shape[0]
        k = torch.arange(self.num_classes - 1, device=logits.device).unsqueeze(0) # [1, K-1]
        target_bin = (target.unsqueeze(1) > k).float() # [B, K-1]
        
        loss = F.binary_cross_entropy_with_logits(logits, target_bin, reduction='mean')
        return loss
