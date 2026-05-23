import torch
import torch.nn as nn
import torch.nn.functional as F

class StressAwareLoss(nn.Module):
    def __init__(self, penalty_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.penalty_weight = penalty_weight

    def forward(self, preds, targets, stress_mask):
        """
        Computes MSE + Stress Penalty.
        
        Args:
            preds: Predicted heatmap (N, 1, H, W)
            targets: Ground truth heatmap (N, 1, H, W)
            stress_mask: Mask where 1.0 indicates a critical area (N, 1, H, W)
        """
        # Standard MSE loss for heatmap prediction
        base_loss = self.mse(preds, targets)
        
        # Stress penalty: heavily penalize the model if it predicts high safety (preds > 0)
        # in regions where stress_mask is high.
        # We compute the squared prediction in critical areas to penalize confident wrong predictions.
        stress_penalty = torch.mean((preds ** 2) * stress_mask)
        
        total_loss = base_loss + self.penalty_weight * stress_penalty
        
        return total_loss, base_loss, stress_penalty
