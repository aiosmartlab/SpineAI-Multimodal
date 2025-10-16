import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import torch

def calculate_metrics(y_true, y_pred, y_pred_proba=None, num_classes=5):
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        y_pred_proba: predicted probabilities (for AUC-ROC)
        num_classes: number of classes
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['recall'] = recall_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['f1'] = f1_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics['recall_per_class'] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics['f1_per_class'] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # AUC-ROC (if probabilities provided)
    if y_pred_proba is not None:
        try:
            # One-vs-rest AUC-ROC
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            metrics['auc_roc'] = roc_auc_score(
                y_true_bin, y_pred_proba, average='weighted', multi_class='ovr'
            )
        except:
            metrics['auc_roc'] = None
    
    # Classification report
    class_names = [
        'Normal',
        'Left Laterolisthesis',
        'Right Laterolisthesis',
        'Anterolisthesis',
        'Retrolisthesis'
    ]
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names
    )
    
    return metrics

def calculate_class_specific_metrics(y_true, y_pred, class_idx):
    """Calculate metrics for a specific class"""
    # Binary classification for this class
    y_true_binary = (np.array(y_true) == class_idx).astype(int)
    y_pred_binary = (np.array(y_pred) == class_idx).astype(int)
    
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }