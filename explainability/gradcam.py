import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        """
        model: trained model
        target_layer: layer to compute gradients (e.g., last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image, metadata, class_idx=None):
        """
        Generate Grad-CAM++ heatmap
        
        Args:
            image: input image (1, C, H, W)
            metadata: patient metadata
            class_idx: target class (if None, use predicted class)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(image, metadata)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (Grad-CAM++)
        # Alpha calculation
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + (activations * grad_3).sum(
            dim=(1, 2), keepdim=True
        )
        alpha_denom = torch.where(
            alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom)
        )
        alpha = alpha_num / alpha_denom
        
        # Weights
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
        
        # Generate CAM
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam, class_idx
    
    def overlay_heatmap(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            image: original image (H, W, C) in [0, 1]
            cam: CAM heatmap (H, W) in [0, 1]
            alpha: blending factor
        """
        # Convert image to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply colormap to CAM
        cam_uint8 = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = cv2.addWeighted(image_uint8, 1-alpha, heatmap, alpha, 0)
        
        return overlayed

def generate_gradcam_visualization(
    model,
    image,
    metadata,
    target_layer,
    class_names,
    save_path=None
):
    """
    Generate and save Grad-CAM++ visualization
    """
    import matplotlib.pyplot as plt
    
    # Create Grad-CAM++ object
    gradcam = GradCAMPlusPlus(model, target_layer)
    
    # Generate heatmap
    cam, pred_class = gradcam.generate_cam(image, metadata)
    
    # Prepare image for visualization
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    
    # Overlay heatmap
    overlayed = gradcam.overlay_heatmap(img_np, cam)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM++ Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlayed)
    axes[2].set_title(f'Overlay\nPredicted: {class_names[pred_class]}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cam, pred_class