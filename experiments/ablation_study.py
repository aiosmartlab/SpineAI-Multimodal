import torch
import copy
from pathlib import Path

from models.main_model import SpinalAbnormalityDetector
from evaluation.evaluation import evaluate_model

def run_ablation_study(
    base_model,
    test_loader,
    device='cuda',
    save_dir='results/ablation'
):
    """
    Run ablation study to evaluate component contributions
    
    Components to ablate:
    1. Contrastive Learning
    2. GNN
    3. Multimodal Fusion
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Full model (baseline)
    print("\n" + "="*50)
    print("Evaluating Full Model")
    print("="*50)
    full_metrics = evaluate_model(
        base_model,
        test_loader,
        device,
        save_dir=save_dir / 'full_model'
    )
    results['Full Model'] = full_metrics
    
    # 2. Without Contrastive Learning
    print("\n" + "="*50)
    print("Evaluating Without Contrastive Learning")
    print("="*50)
    model_no_cl = copy.deepcopy(base_model)
    # Disable contrastive learning
    model_no_cl.contrastive = None
    no_cl_metrics = evaluate_model(
        model_no_cl,
        test_loader,
        device,
        save_dir=save_dir / 'no_contrastive'
    )
    results['Without Contrastive Learning'] = no_cl_metrics
    
    # 3. Without GNN
    print("\n" + "="*50)
    print("Evaluating Without GNN")
    print("="*50)
    model_no_gnn = copy.deepcopy(base_model)
    # Replace GNN with identity mapping
    model_no_gnn.gnn = torch.nn.Identity()
    no_gnn_metrics = evaluate_model(
        model_no_gnn,
        test_loader,
        device,
        save_dir=save_dir / 'no_gnn'
    )
    results['Without GNN'] = no_gnn_metrics
    
    # 4. Without Multimodal Fusion (image only)
    print("\n" + "="*50)
    print("Evaluating Without Multimodal Fusion")
    print("="*50)
    model_no_fusion = copy.deepcopy(base_model)
    # Use only Swin features
    original_forward = model_no_fusion.forward
    
    def forward_no_fusion(images, metadata):
        swin_features = model_no_fusion.swin_transformer(images)
        logits = model_no_fusion.classifier(swin_features)
        return logits
    
    model_no_fusion.forward = forward_no_fusion
    no_fusion_metrics = evaluate_model(
        model_no_fusion,
        test_loader,
        device,
        save_dir=save_dir / 'no_fusion'
    )
    results['Without Multimodal Fusion'] = no_fusion_metrics
    
    # Print comparison table
    print("\n" + "="*70)
    print("Ablation Study Results Summary")
    print("="*70)
    print(f"{'Model Variant':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}")
    print("-"*70)
    
    for variant, metrics in results.items():
        print(
            f"{variant:<30} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['mcc']:>10.4f}"
        )
    
    # Save results to file
    with open(save_dir / 'ablation_summary.txt', 'w') as f:
        f.write("Ablation Study Results\n")
        f.write("="*70 + "\n")
        f.write(f"{'Model Variant':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}\n")
        f.write("-"*70 + "\n")
        
        for variant, metrics in results.items():
            f.write(
                f"{variant:<30} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} "
                f"{metrics['f1']:>10.4f} "
                f"{metrics['mcc']:>10.4f}\n"
            )
    
    return results