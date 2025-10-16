import torch
from imblearn.over_sampling import SMOTE
import numpy as np

class ClassBalancer:
    def __init__(self):
        self.smote = SMOTE(random_state=42)
    
    def apply_smote(self, X, y):
        """ใช้ SMOTE สร้างข้อมูลสังเคราะห์"""
        # Flatten images for SMOTE
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        
        X_resampled, y_resampled = self.smote.fit_resample(X_flat, y)
        
        # Reshape back
        X_resampled = X_resampled.reshape(-1, *X.shape[1:])
        return X_resampled, y_resampled
    
    def balanced_batch_sampler(self, dataset, batch_size):
        """สร้าง balanced batch sampler"""
        class BalancedBatchSampler(torch.utils.data.Sampler):
            def __init__(self, dataset, batch_size):
                self.dataset = dataset
                self.batch_size = batch_size
                self.num_classes = len(np.unique(dataset.labels))
                
                # จัดกลุ่ม indices ตาม class
                self.class_indices = {}
                for idx, label in enumerate(dataset.labels):
                    if label not in self.class_indices:
                        self.class_indices[label] = []
                    self.class_indices[label].append(idx)
            
            def __iter__(self):
                batches = []
                samples_per_class = self.batch_size // self.num_classes
                
                # สุ่มตัวอย่างจากแต่ละ class
                for cls in self.class_indices:
                    indices = self.class_indices[cls].copy()
                    np.random.shuffle(indices)
                    batches.extend(indices[:samples_per_class])
                
                np.random.shuffle(batches)
                return iter(batches)
            
            def __len__(self):
                return len(self.dataset) // self.batch_size
        
        return BalancedBatchSampler(dataset, batch_size)