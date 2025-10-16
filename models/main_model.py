import torch
import torch.nn as nn
from .swin_transformer import SwinTransformerEncoder
from .gnn import GNN
from .contrastive_learning import ContrastiveLearningModule
from .multimodal_fusion import AttentionFusion

class SpinalAbnormalityDetector(nn.Module):
    def __init__(
        self,
        num_classes=5,
        swin_dim=768,
        gnn_dim=512,
        meta_dim=128,
        hidden_dim=256,
        dropout=0.2
    ):
        super().__init__()
        
        # Swin Transformer for image feature extraction
        self.swin_transformer = SwinTransformerEncoder(
            img_size=224,
            patch_size=4,
            embed_dim=96,
            window_size=7,
            drop_rate=0.1
        )
        
        # GNN for spatial relationship modeling
        self.gnn = GNN(
            input_dim=512,
            hidden_dim=256,
            output_dim=gnn_dim,
            num_layers=3,
            dropout=0.2
        )
        
        # Contrastive Learning Module
        self.contrastive = ContrastiveLearningModule(
            feature_dim=hidden_dim,
            temperature=0.07
        )
        
        # Metadata processing
        self.meta_processor = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, meta_dim)
        )
        
        # Multimodal Fusion
        self.fusion = AttentionFusion(
            swin_dim=swin_dim,
            gnn_dim=gnn_dim,
            meta_dim=meta_dim,
            hidden_dim=hidden_dim
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Graph construction layer
        self.graph_constructor = nn.Sequential(
            nn.Linear(swin_dim, 512),
            nn.ReLU()
        )
    
    def construct_graph(self, features, batch_size, num_nodes=5):
        """
        Construct adjacency matrix from image features
        Assuming we have 5 vertebrae (L1-L5)
        """
        # Reshape features to nodes
        node_features = self.graph_constructor(features)
        node_features = node_features.view(batch_size, num_nodes, -1)
        
        # Simple adjacency: connect adjacent vertebrae
        adj = torch.zeros(batch_size, num_nodes, num_nodes).to(features.device)
        for i in range(num_nodes - 1):
            adj[:, i, i+1] = 1
            adj[:, i+1, i] = 1
        
        # Self-connections
        for i in range(num_nodes):
            adj[:, i, i] = 1
        
        # Normalize adjacency matrix
        adj = self.normalize_adjacency(adj)
        
        return node_features, adj
    
    def normalize_adjacency(self, adj):
        """Normalize adjacency matrix"""
        # Add self-loops and normalize
        degree = adj.sum(dim=2, keepdim=True)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        adj_normalized = degree_inv_sqrt * adj * degree_inv_sqrt.transpose(1, 2)
        return adj_normalized
    
    def forward(self, images, metadata, return_features=False):
        """
        images: (batch_size, 3, 224, 224)
        metadata: (batch_size, meta_dim)
        """
        batch_size = images.shape[0]
        
        # Extract image features with Swin Transformer
        swin_features = self.swin_transformer(images)  # (B, swin_dim)
        
        # Construct graph and apply GNN
        node_features, adj_matrix = self.construct_graph(
            swin_features, batch_size
        )
        gnn_features = self.gnn(node_features, adj_matrix)  # (B, gnn_dim)
        
        # Process metadata
        meta_features = self.meta_processor(metadata)  # (B, meta_dim)
        
        # Multimodal fusion
        fused_features, attention_weights = self.fusion(
            swin_features, gnn_features, meta_features
        )
        
        # Classification
        logits = self.classifier(fused_features)
        
        if return_features:
            return logits, {
                'swin_features': swin_features,
                'gnn_features': gnn_features,
                'meta_features': meta_features,
                'fused_features': fused_features,
                'attention_weights': attention_weights
            }
        
        return logits
    
    def get_contrastive_features(self, images, metadata):
        """Get features for contrastive learning"""
        swin_features = self.swin_transformer(images)
        node_features, adj_matrix = self.construct_graph(
            swin_features, images.shape[0]
        )
        gnn_features = self.gnn(node_features, adj_matrix)
        meta_features = self.meta_processor(metadata)
        fused_features, _ = self.fusion(
            swin_features, gnn_features, meta_features
        )
        
        # Project for contrastive learning
        projected_features = self.contrastive(fused_features)
        return projected_features