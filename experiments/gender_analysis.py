import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from evaluation.metrics import calculate_metrics
from evaluation.confusion_matrix import (
    plot_confusion_matrix,
    plot_gender_comparison_confusion_matrices
)

def split_by_gender(dataset):
    """Split dataset by gender"""
    male_indices = []
    female_indices = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        gender = sample['metadata'][1].item()  # Gender is at index 1
        
        if gender == 1.0:  # Male
            male_indices.append(idx)
        else:  # Female
            female_indices.append(idx)
    
    return male_indices, female_indices

def evaluate_gender_specific(
    model,
    dataset,
    gender_indices,
    gender_name,
    device='cuda',
    batch_size=64
):
    """
    Evaluate model on gender-specific subset
    
    Args:
        model: trained model
        dataset: full dataset
        gender_indices: indices for specific gender
        gender_name: 'Male' or 'Female'
        device: device to use
        batch_size: batch size for evaluation
    """
    from torch.utils.data import DataLoader, Subset
    
    # Create subset
    subset = Subset(dataset, gender_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    print(f"\nEvaluating {gender_name} subset ({len(gender_indices)} samples)...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images, metadata)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    
    print(f"\n{gender_name} Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    
    return metrics, all_labels, all_preds

def run_gender_analysis(
    model,
    test_dataset,
    device='cuda',
    save_dir='results/gender_analysis'
):
    """
    Run comprehensive gender-based analysis
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Split by gender
    male_indices, female_indices = split_by_gender(test_dataset)
    
    print(f"\nDataset distribution:")
    print(f"Male samples: {len(male_indices)}")
    print(f"Female samples: {len(female_indices)}")
    
    # Evaluate on male subset
    male_metrics, male_labels, male_preds = evaluate_gender_specific(
        model, test_dataset, male_indices, 'Male', device
    )
    
    # Evaluate on female subset
    female_metrics, female_labels, female_preds = evaluate_gender_specific(
        model, test_dataset, female_indices, 'Female', device
    )
    
    # Calculate differences
    print("\n" + "="*50)
    print("Gender Comparison")
    print("="*50)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
    
    print(f"{'Metric':<15} {'Male':>10} {'Female':>10} {'Difference':>12}")
    print("-"*50)
    
    for metric in metrics_to_compare:
        male_val = male_metrics[metric]
        female_val = female_metrics[metric]
        diff = abs(male_val - female_val)
        
        print(
            f"{metric.upper():<15} "
            f"{male_val:>10.4f} "
            f"{female_val:>10.4f} "
            f"{diff:>12.4f}"
        )
    
    # Plot confusion matrices
    class_names = [
        'Normal',
        'Left Latero.',
        'Right Latero.',
        'Anterolith.',
        'Retrolith.'
    ]
    
    male_cm = male_metrics['confusion_matrix']
    female_cm = female_metrics['confusion_matrix']
    
    # Individual confusion matrices
    plot_confusion_matrix(
        male_cm,
        class_names,
        save_path=save_dir / 'confusion_matrix_male.png',
        title='Confusion Matrix - Male Patients'
    )
    
    plot_confusion_matrix(
        female_cm,
        class_names,
        save_path=save_dir / 'confusion_matrix_female.png',
        title='Confusion Matrix - Female Patients'
    )
    
    # Side-by-side comparison
    plot_gender_comparison_confusion_matrices(
        male_cm,
        female_cm,
        class_names,
        save_path=save_dir / 'confusion_matrix_comparison.png'
    )
    
    # Save results
    with open(save_dir / 'gender_analysis.txt', 'w') as f:
        f.write("Gender-Based Performance Analysis\n")
        f.write("="*50 + "\n\n")
        
        f.write("Dataset Distribution:\n")
        f.write(f"Male samples: {len(male_indices)}\n")
        f.write(f"Female samples: {len(female_indices)}\n\n")
        
        f.write("Male Results:\n")
        for key, val in male_metrics.items():
            if isinstance(val, float):
                f.write(f"  {key}: {val:.4f}\n")
        
        f.write("\nFemale Results:\n")
        for key, val in female_metrics.items():
            if isinstance(val, float):
                f.write(f"  {key}: {val:.4f}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("Comparison\n")
        f.write("="*50 + "\n")
        f.write(f"{'Metric':<15} {'Male':>10} {'Female':>10} {'Difference':>12}\n")
        f.write("-"*50 + "\n")
        
        for metric in metrics_to_compare:
            male_val = male_metrics[metric]
            female_val = female_metrics[metric]
            diff = abs(male_val - female_val)
            
            f.write(
                f"{metric.upper():<15} "
                f"{male_val:>10.4f} "
                f"{female_val:>10.4f} "
                f"{diff:>12.4f}\n"
            )
    
    print(f"\nGender analysis results saved to {save_dir}")
    
    return {
        'male': male_metrics,
        'female': female_metrics
    }