import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics import calculate_metrics
from evaluation.confusion_matrix import plot_confusion_matrix

def evaluate_model(
    model,
    test_loader,
    device='cuda',
    save_dir=None
):
    """
    Comprehensive model evaluation
    
    Args:
        model: trained model
        test_loader: test data loader
        device: device to use
        save_dir: directory to save results
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(images, metadata)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_pred_proba=np.array(all_probs),
        num_classes=5
    )
    
    # Print results
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    if metrics['auc_roc'] is not None:
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    print("\nPer-Class Metrics:")
    class_names = [
        'Normal',
        'Left Laterolisthesis',
        'Right Laterolisthesis',
        'Anterolisthesis',
        'Retrolisthesis'
    ]
    for i, name in enumerate(class_names):
        print(f"\n{name}:")
        print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"  Recall: {metrics['recall_per_class'][i]:.4f}")
        print(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}")
    
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(metrics['classification_report'])
    
    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names,
            save_path=save_dir / 'confusion_matrix.png'
        )
        
        # Save metrics to file
        with open(save_dir / 'metrics.txt', 'w') as f:
            f.write("Test Results\n")
            f.write("="*50 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1']:.4f}\n")
            f.write(f"MCC: {metrics['mcc']:.4f}\n")
            if metrics['auc_roc'] is not None:
                f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n")
            f.write("\n" + "="*50 + "\n")
            f.write("Classification Report\n")
            f.write("="*50 + "\n")
            f.write(metrics['classification_report'])
        
        print(f"\nResults saved to {save_dir}")
    
    return metrics

def evaluate_with_explanations(
    model,
    test_loader,
    device='cuda',
    save_dir=None,
    num_samples=10
):
    """
    Evaluate model and generate explanations
    """
    from explainability.gradcam import generate_gradcam_visualization
    from explainability.lrp import visualize_lrp
    
    model.eval()
    model.to(device)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        explain_dir = save_dir / 'explanations'
        explain_dir.mkdir(exist_ok=True)
    
    # Get target layer for Grad-CAM
    target_layer = None
    for name, module in model.named_modules():
        if 'swin_transformer' in name and hasattr(module, 'norm'):
            target_layer = module
    
    class_names = [
        'Normal',
        'Left Laterolisthesis',
        'Right Laterolisthesis',
        'Anterolisthesis',
        'Retrolisthesis'
    ]
    
    # Generate explanations for sample images
    print(f"\nGenerating explanations for {num_samples} samples...")
    sample_count = 0
    
    for batch_idx, batch in enumerate(test_loader):
        if sample_count >= num_samples:
            break
        
        images = batch['image'].to(device)
        metadata = batch['metadata'].to(device)
        image_ids = batch['image_id']
        
        for i in range(images.shape[0]):
            if sample_count >= num_samples:
                break
            
            image = images[i:i+1]
            meta = metadata[i:i+1]
            img_id = image_ids[i]
            
            # Generate Grad-CAM++
            if target_layer:
                save_path = explain_dir / f'gradcam_{sample_count}_{img_id}.png'
                generate_gradcam_visualization(
                    model, image, meta, target_layer,
                    class_names, save_path
                )
            
            # Generate LRP
            save_path = explain_dir / f'lrp_{sample_count}_{img_id}.png'
            visualize_lrp(model, image, meta, save_path)
            
            sample_count += 1
            print(f"Generated explanations {sample_count}/{num_samples}")
    
    print(f"Explanations saved to {explain_dir}")