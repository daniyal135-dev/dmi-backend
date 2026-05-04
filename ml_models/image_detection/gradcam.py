"""
==============================================================================
GRAD-CAM IMPLEMENTATION - Gradient-weighted Class Activation Mapping
==============================================================================

WHAT THIS FILE DOES:
Creates HEATMAPS showing which parts of an image the model looked at
to make its decision (Real vs Fake).

WHY GRAD-CAM IS IMPORTANT:
- Neural networks are "black boxes" - we can't see their reasoning
- Grad-CAM makes models EXPLAINABLE - shows what they're focusing on
- Builds trust - users can verify if the model is looking at the right things
- Debugging - helps identify if model learned the right features

HOW GRAD-CAM WORKS (SIMPLIFIED):
1. Image → Model → Prediction (e.g., "Fake")
2. Backpropagate to find: "Which pixels influenced this decision most?"
3. Create heatmap: Red = Important, Blue = Not important
4. Overlay heatmap on original image

EXAMPLE:
Input: Face image
Model predicts: FAKE
Grad-CAM shows: Red regions around eyes and mouth
→ Model detected artifacts in facial features (good!)

PAPER REFERENCE:
"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
Selvaraju et al., ICCV 2017

FILE STRUCTURE:
1. GradCAM class - Generate heatmaps
2. visualize_gradcam_comparison() - Create side-by-side visualizations
==============================================================================
"""

# ============== IMPORTS ==============
import torch                        # PyTorch: Deep learning framework
import torch.nn.functional as F     # Functional operations (ReLU, etc.)
import numpy as np                  # NumPy: Array operations
import cv2                          # OpenCV: Image processing
from PIL import Image               # PIL: Image handling
import matplotlib.pyplot as plt     # Matplotlib: Plotting
import matplotlib.cm as cm          # Colormaps (for heatmaps)


class GradCAM:
    """
    ==============================================================================
    GRAD-CAM (Gradient-weighted Class Activation Mapping)
    ==============================================================================
    
    Generates visual explanations showing which image regions influenced
    the model's decision.
    
    HOW IT WORKS (TECHNICAL):
    1. Forward pass: Image → Model → Output
    2. Backward pass: Compute gradients of target class w.r.t. feature maps
    3. Weight feature maps by gradients (importance)
    4. Sum weighted feature maps → Heatmap
    5. Apply ReLU (keep only positive influences)
    6. Normalize and resize to original image size
    
    MATHEMATICAL FORMULA:
    CAM = ReLU(Σ(α_k * A_k))
    
    Where:
    - A_k = Feature maps from layer4 (2048 channels)
    - α_k = Global average pooling of gradients (importance weights)
    - ReLU = Keep only positive influences
    
    USAGE EXAMPLE:
        gradcam = GradCAM(model)
        cam = gradcam.generate_cam(input_tensor, target_class=1)  # 1 = Fake
        gradcam.save_heatmap(original_image, cam, 'heatmap.jpg')
    """
    
    def __init__(self, model, target_layer=None):
        """
        INITIALIZE GRAD-CAM
        
        Args:
            model: The neural network model (DeepfakeImageDetector)
            target_layer: Layer to compute CAM from (usually last conv layer)
                - Default: model.feature_layer (ResNet50's layer4)
                - This layer has the most semantic information
        
        WHAT HAPPENS HERE:
        1. Set model to evaluation mode
        2. Identify target layer (where to capture activations)
        3. Register "hooks" to capture activations and gradients
        
        WHAT ARE HOOKS?
        Hooks are callbacks that PyTorch calls during forward/backward pass.
        They let us intercept and save intermediate values (activations, gradients).
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode (disable dropout, etc.)
        
        # ============== SET TARGET LAYER ==============
        # This is the layer we'll compute CAM from
        # For ResNet50, this is layer4 (last convolutional layer)
        if target_layer is None:
            self.target_layer = model.feature_layer  # ResNet layer4
        else:
            self.target_layer = target_layer
        
        # Storage for captured values
        self.gradients = None     # Will store gradients during backward pass
        self.activations = None   # Will store activations during forward pass
        
        # ============== REGISTER HOOKS ==============
        # Forward hook: Captures activations during forward pass
        # Backward hook: Captures gradients during backward pass
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
        print(f"✓ Grad-CAM initialized on layer: {type(target_layer).__name__ if target_layer else 'layer4'}")
    
    def save_activation(self, module, input, output):
        """
        FORWARD HOOK - Save activations during forward pass
        
        This function is called automatically by PyTorch when the target layer
        completes its forward pass.
        
        Args:
            module: The layer that triggered this hook (target_layer)
            input: Input to the layer (we don't use this)
            output: Output from the layer (these are the ACTIVATIONS!)
        
        WHAT ARE ACTIVATIONS?
        Activations are the feature maps output by the convolutional layer.
        For ResNet layer4: Shape (1, 2048, 7, 7)
        - 1: Batch size
        - 2048: Number of feature maps (channels)
        - 7x7: Spatial dimensions
        
        Each of the 2048 feature maps represents a different learned pattern.
        """
        # Save activations (detach to not interfere with gradients)
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        BACKWARD HOOK - Save gradients during backward pass
        
        This function is called automatically by PyTorch during backpropagation
        when gradients reach the target layer.
        
        Args:
            module: The layer that triggered this hook
            grad_input: Gradients w.r.t. layer inputs (we don't use this)
            grad_output: Gradients w.r.t. layer outputs (these are what we need!)
        
        WHAT ARE THESE GRADIENTS?
        They represent: "How much did each activation contribute to the prediction?"
        Large gradient = This feature map strongly influenced the prediction
        Small gradient = This feature map didn't matter much
        """
        # Save gradients (first element of tuple, detach to save memory)
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        GENERATE GRAD-CAM HEATMAP
        
        This is the main function that creates the heatmap!
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor
                Shape: (1, 3, 224, 224)
            
            target_class (int): Class to explain (0=Real, 1=Fake)
                - None: Explain the predicted class (automatic)
                - 0: Show what makes model think it's Real
                - 1: Show what makes model think it's Fake
        
        Returns:
            numpy.ndarray: Heatmap (values 0-1)
                Shape: (7, 7) - will be resized to image size later
        
        STEP-BY-STEP PROCESS:
        1. Forward pass → Get prediction
        2. Backward pass → Get gradients for target class
        3. Compute importance weights (average gradients)
        4. Weighted sum of activations
        5. Apply ReLU (remove negative values)
        6. Normalize to 0-1 range
        """
        # ============== STEP 1: FORWARD PASS ==============
        # Clear any existing gradients
        self.model.zero_grad()
        
        # Forward pass through model
        # The hooks we registered will automatically save activations
        output = self.model(input_tensor)
        # output shape: (1, 2) → [Real_score, Fake_score]
        
        # ============== STEP 2: GET TARGET CLASS ==============
        # If no target class specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()  # Index of highest score
        
        print(f"  Generating CAM for class: {['Real', 'Fake'][target_class]}")
        
        # ============== STEP 3: BACKWARD PASS ==============
        # Compute gradients of the target class score w.r.t. the model
        # This tells us: "Which features contributed to this prediction?"
        output[0, target_class].backward()
        # The backward hook will automatically save gradients
        
        # ============== STEP 4: GET CAPTURED VALUES ==============
        gradients = self.gradients      # Shape: (1, 2048, 7, 7)
        activations = self.activations  # Shape: (1, 2048, 7, 7)
        
        # ============== STEP 5: COMPUTE IMPORTANCE WEIGHTS ==============
        # Global Average Pooling of gradients across spatial dimensions
        # This gives us a weight for each of the 2048 feature maps
        # Shape: (1, 2048, 7, 7) → (1, 2048, 1, 1)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # weights[0, k, 0, 0] = Importance of feature map k
        # High weight = This feature map strongly influenced the decision
        
        # ============== STEP 6: WEIGHTED SUM OF ACTIVATIONS ==============
        # Multiply each activation map by its importance weight and sum
        # cam = Σ(weight_k * activation_k)
        # Shape: (1, 2048, 7, 7) → (1, 1, 7, 7)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ============== STEP 7: APPLY ReLU ==============
        # Keep only positive influences (negative means it worked against the prediction)
        cam = F.relu(cam)
        
        # ============== STEP 8: NORMALIZE ==============
        # Convert to numpy and remove batch/channel dimensions
        cam = cam.squeeze().cpu().numpy()  # Shape: (7, 7)
        
        # Normalize to [0, 1] range
        # Formula: (x - min) / (max - min)
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)  # +1e-8 to avoid division by zero
        
        return cam
    
    def generate_heatmap_overlay(self, input_image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        CREATE HEATMAP OVERLAY ON ORIGINAL IMAGE
        
        Takes the CAM heatmap and overlays it on the original image.
        
        Args:
            input_image (PIL.Image or numpy.ndarray): Original image
            cam (numpy.ndarray): CAM heatmap (7x7, values 0-1)
            alpha (float): Transparency of overlay (0-1)
                - 0.5: 50% original image, 50% heatmap (balanced)
                - Lower: More original image visible
                - Higher: More heatmap visible
            colormap (int): OpenCV colormap for heatmap colors
                - cv2.COLORMAP_JET: Blue (low) → Red (high) - DEFAULT
                - cv2.COLORMAP_HOT: Black → Red → Yellow → White
                - cv2.COLORMAP_VIRIDIS: Purple → Blue → Green → Yellow
        
        Returns:
            numpy.ndarray: Overlay image (RGB, uint8)
        
        VISUALIZATION EXPLANATION:
        - Blue regions: Not important for decision
        - Green regions: Moderately important
        - Yellow regions: Important
        - Red regions: Very important (model focused here!)
        """
        # ============== STEP 1: PREPARE ORIGINAL IMAGE ==============
        # Convert PIL Image to numpy array if needed
        if isinstance(input_image, Image.Image):
            img = np.array(input_image)
        else:
            img = input_image
        
        # Ensure image is RGB (3 channels)
        if len(img.shape) == 2:
            # Grayscale → RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            # RGBA → RGB (remove alpha channel)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # ============== STEP 2: RESIZE CAM TO IMAGE SIZE ==============
        # CAM is 7x7, but image might be 224x224 or any size
        h, w = img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))  # Resize to match image dimensions
        
        # ============== STEP 3: APPLY COLORMAP ==============
        # Convert CAM values (0-1) to colors
        # 0 → Blue, 0.5 → Green, 1.0 → Red (for JET colormap)
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        
        # OpenCV uses BGR, convert to RGB
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # ============== STEP 4: CREATE OVERLAY ==============
        # Blend original image with heatmap
        # Formula: result = (1-alpha)*img + alpha*heatmap
        overlay = cv2.addWeighted(img, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay
    
    def save_heatmap(self, input_image, cam, save_path, alpha=0.5):
        """
        SAVE HEATMAP TO FILE
        
        Convenience function to create and save heatmap in one step.
        
        Args:
            input_image: Original image
            cam: CAM heatmap
            save_path (str): Path to save heatmap image
            alpha (float): Transparency of overlay
        
        Returns:
            str: Path where heatmap was saved
        
        USAGE:
            gradcam.save_heatmap(image, cam, 'output/heatmap.jpg', alpha=0.5)
        """
        # Generate overlay
        overlay = self.generate_heatmap_overlay(input_image, cam, alpha)
        
        # Save using PIL (handles different image formats)
        overlay_pil = Image.fromarray(overlay)
        overlay_pil.save(save_path)
        
        print(f"✓ Heatmap saved to: {save_path}")
        return save_path
    
    def generate_cam_for_class(self, input_tensor, input_image, target_class, save_path=None):
        """
        COMPLETE GRAD-CAM PIPELINE
        
        All-in-one function: Generate CAM, create overlay, optionally save.
        
        Args:
            input_tensor: Preprocessed image tensor
            input_image: Original PIL image
            target_class: Class to explain (0 or 1)
            save_path: Path to save (None = don't save)
        
        Returns:
            dict: Dictionary containing:
                - 'cam': Raw CAM array
                - 'overlay': Overlay image
                - 'target_class': Target class
        
        USAGE:
            result = gradcam.generate_cam_for_class(
                tensor, image, target_class=1, save_path='heatmap.jpg'
            )
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # Create overlay
        overlay = self.generate_heatmap_overlay(input_image, cam)
        
        # Save if path provided
        if save_path:
            self.save_heatmap(input_image, cam, save_path)
        
        return {
            'cam': cam,
            'overlay': overlay,
            'target_class': target_class
        }
    
    def batch_generate_cams(self, input_tensors, input_images, target_classes=None):
        """
        GENERATE CAMS FOR MULTIPLE IMAGES
        
        Process multiple images in batch (more efficient than one-by-one).
        
        Args:
            input_tensors: Batch of tensors
            input_images: List of original images
            target_classes: List of target classes (or None for automatic)
        
        Returns:
            list: List of CAM result dictionaries
        """
        results = []
        
        for i, (tensor, image) in enumerate(zip(input_tensors, input_images)):
            target = target_classes[i] if target_classes else None
            cam_data = self.generate_cam_for_class(
                tensor.unsqueeze(0),  # Add batch dimension
                image, 
                target
            )
            results.append(cam_data)
        
        return results


def visualize_gradcam_comparison(original_image, cam, prediction, confidence, save_path=None):
    """
    ==============================================================================
    CREATE SIDE-BY-SIDE VISUALIZATION
    ==============================================================================
    
    Creates a nice figure showing:
    1. Original image
    2. Heatmap only
    3. Overlay with prediction
    
    Perfect for reports, presentations, or the web interface!
    
    Args:
        original_image: Original image
        cam: CAM heatmap
        prediction (str): Prediction label ("Real" or "Fake")
        confidence (float): Confidence score (0-1)
        save_path (str): Path to save figure (None = display only)
    
    Returns:
        matplotlib.figure.Figure: The created figure
    
    USAGE:
        fig = visualize_gradcam_comparison(
            image, cam, "Fake", 0.95, "report.png"
        )
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ============== SUBPLOT 1: ORIGINAL IMAGE ==============
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    else:
        img_np = original_image
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # ============== SUBPLOT 2: HEATMAP ONLY ==============
    axes[1].imshow(cam, cmap='jet')  # JET colormap: blue → red
    axes[1].set_title('Attention Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # ============== SUBPLOT 3: OVERLAY WITH PREDICTION ==============
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Create colored heatmap
    cam_colored = cm.jet(cam_resized)[:, :, :3]  # RGB only (no alpha)
    
    # Blend with original (60% image, 40% heatmap)
    overlay = 0.6 * (img_np / 255.0) + 0.4 * cam_colored
    
    axes[2].imshow(overlay)
    axes[2].set_title(
        f'Overlay\nPrediction: {prediction} ({confidence:.1%})',
        fontsize=14,
        fontweight='bold'
    )
    axes[2].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
    
    return fig


# ==============================================================================
# END OF FILE
# ==============================================================================
# 
# SUMMARY:
# This file implements Grad-CAM for visual explanations of model predictions.
# 
# MAIN FUNCTIONS:
# 1. GradCAM.generate_cam() - Generate heatmap
# 2. GradCAM.generate_heatmap_overlay() - Create overlay
# 3. GradCAM.save_heatmap() - Save to file
# 4. visualize_gradcam_comparison() - Create comparison figure
# 
# TYPICAL USAGE:
#     gradcam = GradCAM(model)
#     cam = gradcam.generate_cam(input_tensor, target_class=1)
#     gradcam.save_heatmap(original_image, cam, 'heatmap.jpg')
# 
# WHY GRAD-CAM MATTERS:
# - Makes AI explainable (not a black box)
# - Builds user trust (they can see the reasoning)
# - Helps debug model (is it looking at the right things?)
# - Required for responsible AI deployment
# ==============================================================================
