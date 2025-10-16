import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLearningModule(nn.Module):
    def __init__(self, feature_dim=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 128)  # Projection dimension
        )
    
    def forward(self, features):
        """Project features for contrastive learning"""
        return F.normalize(self.projection_head(features), dim=1)
    
    def contrastive_loss(self, z_i, z_j, labels):
        """
        Supervised Contrastive Loss
        z_i, z_j: projected features (batch_size, projection_dim)
        labels: class labels (batch_size,)
        """
        batch_size = z_i.shape[0]
        
        # Concatenate z_i and z_j
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.repeat(2)
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask.fill_diagonal_(False)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        loss = -mean_log_prob_pos.mean()
        
        return loss