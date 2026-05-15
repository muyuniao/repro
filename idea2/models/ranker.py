import torch
import torch.nn as nn

def sinkhorn(scores, tau=1.0, n_iters=20):
    B, K = scores.shape
    j = torch.arange(1, K + 1, device=scores.device, dtype=scores.dtype)
    c = K + 1 - 2 * j  # [K]
    
    A = scores.unsqueeze(-1) * c.unsqueeze(0).unsqueeze(0)  # [B, K, K]
    
    log_alpha = A / tau
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        
    return torch.exp(log_alpha)

class Ranker(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, tau=1.0):
        # x: [B, K, D]
        B, K, D = x.shape
        x_flat = x.view(B * K, D)
        scores = self.mlp(x_flat).view(B, K) # [B, K]
        
        p_hat = sinkhorn(scores, tau=tau) # [B, K, K]
        return scores, p_hat
