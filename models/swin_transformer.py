import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinTransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_rate=0.1
    ):
        super().__init__()
        
        # ใช้ Swin Transformer จาก timm
        self.swin = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate
        )
        
        # Output dimension
        self.feature_dim = embed_dim * 8  # 768 for base model
    
    def forward(self, x):
        """
        x: (batch_size, channels, height, width)
        return: (batch_size, feature_dim)
        """
        features = self.swin.forward_features(x)
        return features