import yaml
from pathlib import Path

class Config:
    """Configuration management"""
    
    # Default configuration
    DEFAULT_CONFIG = {
        # Data
        'data_dir': 'data/BUU-LSPINE',
        'train_csv': 'data/train.csv',
        'val_csv': 'data/val.csv',
        'test_csv': 'data/test.csv',
        'image_size': 224,
        'num_classes': 5,
        
        # Model architecture
        'swin_dim': 768,
        'gnn_dim': 512,
        'meta_dim': 128,
        'hidden_dim': 256,
        'dropout': 0.2,
        
        # Swin Transformer
        'patch_size': 4,
        'window_size': 7,
        'embed_dim': 96,
        'num_heads': [3, 6, 12, 24],
        'depths': [2, 2, 6, 2],
        
        # GNN
        'gnn_layers': 3,
        'gnn_hidden': 256,
        
        # Contrastive Learning
        'use_contrastive': True,
        'temperature': 0.07,
        'contrastive_weight': 0.5,
        
        # Training
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        
        # Loss
        'focal_gamma': 2.0,
        'use_balanced_sampling': True,
        
        # Optimizer & Scheduler
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealing',
        
        # Checkpointing
        'checkpoint_dir': 'checkpoints',
        'save_interval': 10,
        
        # Logging
        'use_wandb': True,
        'wandb_project': 'spinal-abnormality-detection',
        'wandb_entity': 'your-username',
        
        # Evaluation
        'eval_interval': 1,
        
        # Explainability
        'save_explanations': True,
        'explanation_dir': 'explanations',
    }
    
    def __init__(self, config_path=None):
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: path to YAML config file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self.config.update(user_config)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def save(self, path):
        """Save configuration to file"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def to_dict(self):
        """Convert to dictionary"""
        return self.config.copy()

# Example config.yaml file
"""
# config.yaml

# Data paths
data_dir: 'data/BUU-LSPINE'
train_csv: 'data/train.csv'
val_csv: 'data/val.csv'
test_csv: 'data/test.csv'

# Training hyperparameters
batch_size: 64
epochs: 100
learning_rate: 0.0001
weight_decay: 0.00001

# Model parameters
use_contrastive: true
temperature: 0.07
contrastive_weight: 0.5

# Logging
use_wandb: true
wandb_project: 'spinal-abnormality-detection'
"""