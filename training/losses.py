import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: class weights (num_classes,)
        gamma: focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes)
        targets: (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, contrastive_weight=0.5):
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.contrastive_weight = contrastive_weight
    
    def forward(self, outputs, targets, contrastive_loss=None):
        focal = self.focal_loss(outputs, targets)
        
        if contrastive_loss is not None:
            total_loss = focal + self.contrastive_weight * contrastive_loss
        else:
            total_loss = focal
        
        return total_loss