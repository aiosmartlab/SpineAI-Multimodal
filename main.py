import torch
import argparse
import wandb
from pathlib import Path

from utils.config import Config
from utils.dataset import create_dataloaders
from models.main_model import SpinalAbnormalityDetector
from training.train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Spinal Abnormality Detection Model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse arguments
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = Config(args.config)
    print("Configuration loaded:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            config=config.to_dict(),
            name=f"spinal_detection_{config['epochs']}epochs"
        )
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=config['train_csv'],
        val_csv=config['val_csv'],
        test_csv=config['test_csv'],
        image_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_balanced_sampling=config['use_balanced_sampling']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = SpinalAbnormalityDetector(
        num_classes=config['num_classes'],
        swin_dim=config['swin_dim'],
        gnn_dim=config['gnn_dim'],
        meta_dim=config['meta_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.to_dict(),
        device=args.device
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training")
    print("="*50)
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set")
    print("="*50)
    
    from evaluation.evaluation import evaluate_model
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=args.device,
        save_dir=Path(config['checkpoint_dir']) / 'test_results'
    )
    
    print("\nTest Results:")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    if config['use_wandb']:
        wandb.log({"test": test_metrics})
        wandb.finish()
    
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()