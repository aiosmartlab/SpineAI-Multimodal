import torch
import cv2
import numpy as np
from pathlib import Path
import argparse

from models.main_model import SpinalAbnormalityDetector
from explainability.gradcam import GradCAMPlusPlus
from explainability.lrp import LRP

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = SpinalAbnormalityDetector(
        num_classes=5,
        swin_dim=768,
        gnn_dim=512,
        meta_dim=128,
        hidden_dim=256,
        dropout=0.2
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path, target_size=224):
    """Preprocess input image"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size, target_size))
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def prepare_metadata(age, gender):
    """Prepare patient metadata"""
    metadata = torch.FloatTensor([
        age / 100.0,  # Normalize age
        1.0 if gender.lower() == 'male' else 0.0
    ])
    
    # Pad to 128 dimensions
    if metadata.shape[0] < 128:
        padding = torch.zeros(128 - metadata.shape[0])
        metadata = torch.cat([metadata, padding])
    
    return metadata.unsqueeze(0)

def predict(
    model,
    image_path,
    age,
    gender,
    device='cuda',
    return_probs=False
):
    """
    Make prediction for a single image
    
    Args:
        model: trained model
        image_path: path to X-ray image
        age: patient age
        gender: patient gender ('male' or 'female')
        device: device to use
        return_probs: whether to return probabilities
    """
    # Preprocess
    image = preprocess_image(image_path).to(device)
    metadata = prepare_metadata(age, gender).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(image, metadata)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
    
    class_names = [
        'Normal',
        'Left Laterolisthesis',
        'Right Laterolisthesis',
        'Anterolisthesis',
        'Retrolisthesis'
    ]
    
    pred_name = class_names[pred_class]
    confidence = probs[0, pred_class].item()
    
    result = {
        'prediction': pred_name,
        'class_index': pred_class,
        'confidence': confidence
    }
    
    if return_probs:
        result['probabilities'] = {
            class_names[i]: probs[0, i].item()
            for i in range(len(class_names))
        }
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description='Inference for Spinal Abnormality Detection'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to X-ray image'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--age',
        type=int,
        required=True,
        help='Patient age'
    )
    parser.add_argument(
        '--gender',
        type=str,
        required=True,
        choices=['male', 'female'],
        help='Patient gender'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--explain',
        action='store_true',
        help='Generate explanation visualizations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='inference_results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)
    
    # Make prediction
    print(f"\nProcessing image: {args.image}")
    print(f"Patient info: Age={args.age}, Gender={args.gender}")
    
    result = predict(
        model,
        args.image,
        args.age,
        args.gender,
        args.device,
        return_probs=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name:<25}: {prob:.2%}")
    
    # Save results
    result_file = output_dir / 'prediction.txt'
    with open(result_file, 'w') as f:
        f.write("Prediction Results\n")
        f.write("="*50 + "\n")
        f.write(f"Image: {args.image}\n")
        f.write(f"Age: {args.age}\n")
        f.write(f"Gender: {args.gender}\n")
        f.write(f"\nPrediction: {result['prediction']}\n")
        f.write(f"Confidence: {result['confidence']:.2%}\n")
        f.write("\nClass Probabilities:\n")
        for class_name, prob in result['probabilities'].items():
            f.write(f"  {class_name:<25}: {prob:.2%}\n")
    
    print(f"\nResults saved to {result_file}")
    
    # Generate explanations if requested
    if args.explain:
        print("\nGenerating explanations...")
        
        from explainability.gradcam import generate_gradcam_visualization
        from explainability.lrp import visualize_lrp
        
        image = preprocess_image(args.image).to(args.device)
        metadata = prepare_metadata(args.age, args.gender).to(args.device)
        
        # Get target layer for Grad-CAM
        target_layer = None
        for name, module in model.named_modules():
            if 'swin_transformer' in name and hasattr(module, 'layers'):
                # Get last layer
                if hasattr(module.layers, '__getitem__'):
                    target_layer = module.layers[-1]
                    break
        
        class_names = [
            'Normal',
            'Left Laterolisthesis',
            'Right Laterolisthesis',
            'Anterolisthesis',
            'Retrolisthesis'
        ]
        
        # Grad-CAM++
        if target_layer:
            gradcam_path = output_dir / 'gradcam_explanation.png'
            generate_gradcam_visualization(
                model, image, metadata, target_layer,
                class_names, gradcam_path
            )
            print(f"Grad-CAM++ saved to {gradcam_path}")
        
        # LRP
        lrp_path = output_dir / 'lrp_explanation.png'
        visualize_lrp(model, image, metadata, lrp_path)
        print(f"LRP saved to {lrp_path}")

if __name__ == '__main__':
    main()