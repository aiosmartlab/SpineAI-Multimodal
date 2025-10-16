import torch
import torch.nn as nn
import numpy as np

class LRP:
    def __init__(self, model):
        """
        Layer-wise Relevance Propagation
        
        Args:
            model: trained neural network
        """
        self.model = model
        self.model.eval()
    
    def forward_hook(self, module, input, output):
        """Save forward pass activations"""
        self.activations[module] = output.detach()
    
    def register_hooks(self):
        """Register forward hooks to save activations"""
        self.activations = {}
        self.hooks = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(self.forward_hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def lrp_linear(self, layer, relevance_out, activation_in, epsilon=1e-9):
        """
        LRP rule for linear/dense layers
        
        Args:
            layer: linear layer
            relevance_out: relevance scores from next layer
            activation_in: input activations
            epsilon: stability constant
        """
        # Get weights
        weights = layer.weight.data
        
        # Calculate forward pass values
        z = torch.matmul(activation_in, weights.t()) + layer.bias.data
        
        # Avoid division by zero
        z = z + epsilon * torch.sign(z)
        
        # Calculate relevance
        s = relevance_out / z
        c = torch.matmul(s, weights)
        relevance_in = activation_in * c
        
        return relevance_in
    
    def lrp_conv2d(self, layer, relevance_out, activation_in, epsilon=1e-9):
        """
        LRP rule for convolutional layers
        
        Args:
            layer: conv2d layer
            relevance_out: relevance scores from next layer
            activation_in: input activations
            epsilon: stability constant
        """
        # Forward pass
        z = layer.forward(activation_in) + epsilon * torch.sign(
            layer.forward(activation_in)
        )
        
        # Backward pass for relevance
        s = relevance_out / z
        
        # Gradient computation
        (z * s).sum().backward()
        c = activation_in.grad
        relevance_in = activation_in * c
        
        # Clear gradients
        activation_in.grad = None
        
        return relevance_in
    
    def explain(self, image, metadata, target_class=None):
        """
        Generate LRP explanation
        
        Args:
            image: input image (1, C, H, W)
            metadata: patient metadata
            target_class: target class to explain (if None, use prediction)
        """
        # Register hooks
        self.register_hooks()
        
        # Forward pass
        image.requires_grad = True
        output = self.model(image, metadata)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Initialize relevance with output
        relevance = torch.zeros_like(output)
        relevance[0, target_class] = output[0, target_class]
        
        # Backward relevance propagation through layers
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append((name, module))
        
        # Reverse order
        layers = layers[::-1]
        
        for name, layer in layers:
            if layer in self.activations:
                activation_in = self.activations[layer]
                
                if isinstance(layer, nn.Linear):
                    relevance = self.lrp_linear(
                        layer, relevance, activation_in
                    )
                elif isinstance(layer, nn.Conv2d):
                    relevance = self.lrp_conv2d(
                        layer, relevance, activation_in
                    )
        
        # Remove hooks
        self.remove_hooks()
        
        # Convert to heatmap
        relevance_map = relevance[0].sum(dim=0).abs()
        relevance_map = relevance_map.cpu().detach().numpy()
        
        # Normalize
        relevance_map = relevance_map - relevance_map.min()
        if relevance_map.max() > 0:
            relevance_map = relevance_map / relevance_map.max()
        
        return relevance_map, target_class

def visualize_lrp(model, image, metadata, save_path=None):
    """Visualize LRP explanation"""
    import matplotlib.pyplot as plt
    
    lrp = LRP(model)
    relevance_map, pred_class = lrp.explain(image, metadata)
    
    # Prepare image
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(relevance_map, cmap='hot')
    axes[1].set_title('LRP Relevance Map')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(relevance_map, cmap='hot', alpha=0.5)
    axes[2].set_title(f'Overlay (Class: {pred_class})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return relevance_map