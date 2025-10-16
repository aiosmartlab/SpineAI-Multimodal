import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from pathlib import Path

class SpinalDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None,
        is_train=True
    ):
        """
        csv_file: path to CSV with annotations
        image_dir: directory with images
        transform: data augmentation
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Class mapping
        self.class_mapping = {
            'Normal': 0,
            'Left Laterolisthesis': 1,
            'Right Laterolisthesis': 2,
            'Anterolisthesis': 3,
            'Retrolisthesis': 4
        }
        
        # Calculate class weights for focal loss
        self.class_counts = self.data['diagnosis'].value_counts()
        total = len(self.data)
        self.class_weights = torch.FloatTensor([
            total / self.class_counts[cls] 
            for cls in sorted(self.class_mapping.values())
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row['image_filename']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Get label
        diagnosis = row['diagnosis']
        label = self.class_mapping[diagnosis]
        
        # Get metadata
        metadata = torch.FloatTensor([
            row['age'] / 100.0,  # Normalize age
            1.0 if row['gender'] == 'Male' else 0.0,
            # Add more metadata features as needed
        ])
        
        # Pad metadata to fixed size (128)
        if metadata.shape[0] < 128:
            padding = torch.zeros(128 - metadata.shape[0])
            metadata = torch.cat([metadata, padding])
        
        return {
            'image': image,
            'metadata': metadata,
            'label': label,
            'image_id': row['image_filename']
        }

def create_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    image_dir,
    batch_size=64,
    num_workers=4,
    use_balanced_sampling=True
):
    """Create train, validation, and test dataloaders"""
    
    from preprocessing.data_augmentation import DataAugmentation
    
    # Create datasets
    train_dataset = SpinalDataset(
        train_csv,
        image_dir,
        transform=DataAugmentation().transform,
        is_train=True
    )
    
    val_dataset = SpinalDataset(
        val_csv,
        image_dir,
        transform=None,
        is_train=False
    )
    
    test_dataset = SpinalDataset(
        test_csv,
        image_dir,
        transform=None,
        is_train=False
    )
    
    # Create dataloaders
    if use_balanced_sampling:
        from preprocessing.class_balancing import ClassBalancer
        balancer = ClassBalancer()
        train_sampler = balancer.balanced_batch_sampler(
            train_dataset, batch_size
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader