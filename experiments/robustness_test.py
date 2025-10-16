import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from evaluation.metrics import calculate_metrics

def add_gaussian_noise(images, std=0.1):
    """Add Gaussian noise to images"""
    noise = torch.randn_like(images) * std
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)

def test_noise_robustness(
    model,
    test_loader,
    device='cuda',
    noise_levels=[0.0, 0.05, 0.1, 0.15, 0.2]
):
    """
    Test model robustness against different noise levels
    
    Args:
        model: trained model
        test_loader: test data loader
        device: device to use
        noise_levels: list of noise standard deviations
    """
    model.eval()
    model.to(device)
    
    results = {}
    
    for noise_std in noise_levels:
        print(f"\nTesting with noise level: {noise_std}")
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Noise {noise_std}'):
                images = batch['image'].to(device)
                metadata = batch['metadata'].to(device)
                labels = batch['label'].to(device)
                
                # Add noise
                if noise_std > 0:
                    images = add_gaussian_noise(images, noise_std)
                
                # Forward pass
                logits = model(images, metadata)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        results[f'noise_{noise_std}'] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
    
    return results

def test_cross_dataset_generalization(
    model,
    external_loader,
    device='cuda',
    dataset_name='External'
):
    """
    Test model on external dataset
    
    Args:
        model: trained model
        external_loader: data loader for external dataset
        device: device to use
        dataset_name: name of external dataset
    """
    model.eval()
    model.to(device)
    
    print(f"\nTesting on {dataset_name} dataset...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(external_loader):
            images = batch['image'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images, metadata)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    
    print(f"\n{dataset_name} Dataset Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    
    return metrics

def run_comprehensive_robustness_test(
    model,
    test_loader,
    device='cuda',
    save_dir='results/robustness'
):
    """
    Run comprehensive robustness testing
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Baseline (clean data)
    print("\n" + "="*50)
    print("Baseline Performance (Clean Data)")
    print("="*50)
    baseline_results = test_noise_robustness(
        model, test_loader, device, noise_levels=[0.0]
    )
    results['baseline'] = baseline_results['noise_0.0']
    
    # 2. Noise robustness
    print("\n" + "="*50)
    print("Noise Robustness Testing")
    print("="*50)
    noise_results = test_noise_robustness(
        model, test_loader, device,
        noise_levels=[0.05, 0.1, 0.15, 0.2]
    )
    results.update(noise_results)
    
    # Print summary
    print("\n" + "="*70)
    print("Robustness Test Summary")
    print("="*70)
    print(f"{'Test Condition':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}")
    print("-"*70)
    
    for condition, metrics in results.items():
        print(
            f"{condition:<20} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['mcc']:>10.4f}"
        )
    
    # Save results
    with open(save_dir / 'robustness_summary.txt', 'w') as f:
        f.write("Robustness Test Results\n")
        f.write("="*70 + "\n")
        f.write(f"{'Test Condition':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}\n")
        f.write("-"*70 + "\n")
        
        for condition, metrics in results.items():
            f.write(
                f"{condition:<20} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} "
                f"{metrics['f1']:>10.4f} "
                f"{metrics['mcc']:>10.4f}\n"
            )
    
    print(f"\nResults saved to {save_dir}")
    
    return results