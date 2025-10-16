import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, swin_dim=768, gnn_dim=512, meta_dim=128, hidden_dim=256):
        super().__init__()
        
        # Projection layers
        self.swin_proj = nn.Linear(swin_dim, hidden_dim)
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.meta_proj = nn.Linear(meta_dim, hidden_dim)
        
        # Attention scoring
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(0.15)
    
    def forward(self, swin_features, gnn_features, meta_features):
        """
        Attention-based fusion of multimodal features
        """
        # Project all features to same dimension
        f_swin = self.swin_proj(swin_features)
        f_gnn = self.gnn_proj(gnn_features)
        f_meta = self.meta_proj(meta_features)
        
        # Stack features
        features = torch.stack([f_swin, f_gnn, f_meta], dim=1)  # (B, 3, hidden_dim)
        
        # Calculate attention weights
        attention_scores = self.attention(features)  # (B, 3, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, 3, 1)
        
        # Weighted sum
        fused_features = (features * attention_weights).sum(dim=1)  # (B, hidden_dim)
        fused_features = self.dropout(fused_features)
        
        return fused_features, attention_weights