import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from pathlib import Path

from models.main_model import SpinalAbnormalityDetector
from training.losses import CombinedLoss
from evaluation.metrics import calculate_metrics

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = CombinedLoss(
            alpha=train_loader.dataset.class_weights.to(device),
            gamma=2.0,
            contrastive_weight=0.5
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
        
        # Best metrics tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images, metadata)
            
            # Calculate contrastive loss
            if self.config['use_contrastive']:
                # Create augmented view
                images_aug = self.augment_batch(images)
                
                # Get projected features
                z1 = self.model.get_contrastive_features(images, metadata)
                z2 = self.model.get_contrastive_features(images_aug, metadata)
                
                contrastive_loss = self.model.contrastive.contrastive_loss(
                    z1, z2, labels
                )
            else:
                contrastive_loss = None
            
            # Total loss
            loss = self.criterion(logits, labels, contrastive_loss)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to wandb
            if self.config['use_wandb']:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(images, metadata)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        
        avg_loss = total_loss / len(self.val_loader)
        
        print(f"\nValidation Results:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        
        # Log to wandb
        if self.config['use_wandb']:
            wandb.log({
                'val/loss': avg_loss,
                'val/accuracy': metrics['accuracy'],
                'val/precision': metrics['precision'],
                'val/recall': metrics['recall'],
                'val/f1': metrics['f1'],
                'val/mcc': metrics['mcc']
            })
        
        # Save best model
        if metrics['accuracy'] > self.best_val_acc:
            self.best_val_acc = metrics['accuracy']
            self.save_checkpoint(epoch, 'best_acc')
        
        if metrics['f1'] > self.best_val_f1:
            self.best_val_f1 = metrics['f1']
            self.save_checkpoint(epoch, 'best_f1')
        
        return metrics
    
    def augment_batch(self, images):
        """Apply random augmentation for contrastive learning"""
        # Simple augmentation: random horizontal flip and noise
        augmented = images.clone()
        
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            augmented = torch.flip(augmented, dims=[3])
        
        # Add Gaussian noise
        noise = torch.randn_like(augmented) * 0.01
        augmented = augmented + noise
        
        return augmented
    
    def save_checkpoint(self, epoch, name):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'config': self.config
        }
        
        path = self.checkpoint_dir / f'{name}_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def train(self):
        """Full training loop"""
        print("Starting training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint every N epochs
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, 'checkpoint')
        
        print("\nTraining completed!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        print(f"Best Validation F1-Score: {self.best_val_f1:.4f}")