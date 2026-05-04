"""
==============================================================================
IMAGE DETECTION MODEL - DeepfakeImageDetector
==============================================================================
This file contains the neural network architecture for detecting deepfake images.

WHAT THIS FILE DOES:
- Defines the structure of our deepfake detection model
- Uses Transfer Learning with ResNet50 (a pre-trained CNN)
- Modifies ResNet50 to classify images as Real or Fake
- Supports Grad-CAM for explainability (heatmaps)

TRANSFER LEARNING EXPLAINED:
Instead of training from scratch (which takes weeks), we use ResNet50
that was already trained on ImageNet (1.4 million images). This model
already knows how to detect edges, textures, and patterns. We just
"fine-tune" it to detect deepfakes!

FILE STRUCTURE:
1. DeepfakeImageDetector - Main model class (ResNet50-based)
2. ConvNeXtDetector - Alternative model (more modern architecture)
==============================================================================
"""

# ============== IMPORTS ==============
import torch                          # PyTorch: Deep learning framework
import torch.nn as nn                 # Neural network building blocks (layers, loss functions)
import torchvision.models as models   # Pre-built models (ResNet, VGG, etc.)
from torchvision.models import ResNet50_Weights  # Pre-trained weights for ResNet50
import os                             # File operations


class DeepfakeImageDetector(nn.Module):
    """
    ==============================================================================
    DEEPFAKE IMAGE DETECTOR (ResNet50-based)
    ==============================================================================
    
    ARCHITECTURE OVERVIEW:
    - Base: ResNet50 (50-layer Convolutional Neural Network)
    - Input: 224x224 RGB image
    - Output: 2 values (logits for Real/Fake)
    
    WHY ResNet50?
    - It's proven, reliable, and widely used
    - Has 25 million parameters (powerful but not too heavy)
    - Works well with transfer learning
    - Good balance between accuracy and speed
    
    HOW IT WORKS:
    1. Takes an image (224x224)
    2. Passes through 50 layers of convolutions
    3. Extracts 2048 features
    4. Our custom head reduces to 2 outputs (Real/Fake)
    5. Softmax converts to probabilities
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        INITIALIZE THE MODEL
        
        This is called when you create the model: model = DeepfakeImageDetector()
        
        Args:
            num_classes (int): Number of output classes
                - For deepfake detection: 2 (Real, Fake)
                - Could be 3 if we add "Uncertain" class later
            
            pretrained (bool): Whether to use ImageNet pre-trained weights
                - True = Use weights from ImageNet (RECOMMENDED for transfer learning)
                - False = Random initialization (only if training from scratch)
        
        WHAT HAPPENS HERE:
        1. Load ResNet50 backbone
        2. Replace the final classification layer
        3. Set up hooks for Grad-CAM (explainability)
        """
        # Call parent class constructor (required for PyTorch models)
        super(DeepfakeImageDetector, self).__init__()
        
        # ============== LOAD RESNET50 BACKBONE ==============
        # TRANSFER LEARNING: We use a model pre-trained on ImageNet
        if pretrained:
            # Load ResNet50 with weights trained on ImageNet (1.4M images, 1000 classes)
            # These weights already know how to detect edges, textures, faces, etc.
            weights = ResNet50_Weights.IMAGENET1K_V2  # Latest ImageNet weights
            self.resnet = models.resnet50(weights=weights)
            print("✓ Loaded ResNet50 with ImageNet pre-trained weights")
        else:
            # Load ResNet50 with random weights (not recommended unless you have huge dataset)
            self.resnet = models.resnet50(weights=None)
            print("⚠ Loaded ResNet50 with RANDOM weights (no transfer learning)")
        
        # ============== MODIFY THE FINAL LAYER ==============
        # PROBLEM: ResNet50 was designed for ImageNet (1000 classes: cat, dog, car...)
        # SOLUTION: We replace the last layer to output only 2 classes (Real, Fake)
        
        # Get the number of input features to the final layer (2048 for ResNet50)
        num_features = self.resnet.fc.in_features  # fc = fully connected layer
        
        # Replace the final fully connected layer with our custom head
        # OLD: [...ResNet layers...] → fc → 1000 outputs (ImageNet classes)
        # NEW: [...ResNet layers...] → custom head → 2 outputs (Real, Fake)
        self.resnet.fc = nn.Sequential(
            # LAYER 1: Dropout (prevents overfitting)
            # Randomly turns off 50% of neurons during training
            # This forces the model to learn robust features, not memorize training data
            nn.Dropout(0.5),
            
            # LAYER 2: Fully connected layer (2048 → 512)
            # Reduces feature dimensionality from 2048 to 512
            nn.Linear(num_features, 512),
            
            # LAYER 3: ReLU activation (introduces non-linearity)
            # Converts negative values to 0, keeps positive values
            # Formula: f(x) = max(0, x)
            nn.ReLU(),
            
            # LAYER 4: Another dropout (30%)
            # Less aggressive than first dropout
            nn.Dropout(0.3),
            
            # LAYER 5: Final classification layer (512 → 2)
            # Outputs 2 "logits" (raw scores before softmax)
            # Example output: [-1.2, 2.5] → Real score, Fake score
            nn.Linear(512, num_classes)
        )
        
        # ============== SETUP FOR GRAD-CAM ==============
        # Grad-CAM creates heatmaps showing which image regions influenced the decision
        # We need to capture activations from the last convolutional layer (layer4)
        self.feature_layer = self.resnet.layer4  # Last conv layer (before fc)
        self.gradients = None      # Will store gradients during backward pass
        self.activations = None    # Will store activations during forward pass
    
    def forward(self, x, return_features=False):
        """
        ==============================================================================
        FORWARD PASS - How an image flows through the network
        ==============================================================================
        
        This function is called automatically when you do: output = model(image)
        
        Args:
            x (torch.Tensor): Input image tensor
                Shape: (batch_size, 3, 224, 224)
                - batch_size: Number of images processed together (e.g., 32)
                - 3: RGB channels (Red, Green, Blue)
                - 224x224: Image dimensions
            
            return_features (bool): Whether to return intermediate features
                - True: Return both output and features (for Grad-CAM)
                - False: Return only output (for normal prediction)
        
        Returns:
            torch.Tensor: Output logits (raw scores)
                Shape: (batch_size, 2) → [Real_score, Fake_score]
                Example: tensor([[-1.2, 2.5]]) means:
                    - Real score: -1.2 (low)
                    - Fake score: 2.5 (high) → Likely FAKE!
            
            OR tuple (output, features) if return_features=True
        
        FLOW EXPLANATION:
        Input Image (224x224x3)
            ↓
        [Conv1 + BN + ReLU + MaxPool] → Reduces to 56x56x64
            ↓
        [Layer1: 3 bottleneck blocks] → 56x56x256
            ↓
        [Layer2: 4 bottleneck blocks] → 28x28x512
            ↓
        [Layer3: 6 bottleneck blocks] → 14x14x1024
            ↓
        [Layer4: 3 bottleneck blocks] → 7x7x2048 (features for Grad-CAM)
            ↓
        [Average Pooling] → 1x1x2048 (summarize features)
            ↓
        [Flatten] → 2048 (convert to 1D vector)
            ↓
        [Custom FC layers] → 2 (Real, Fake logits)
        """
        
        # ============== INITIAL CONVOLUTION BLOCK ==============
        # First layer: Extract low-level features (edges, colors, textures)
        x = self.resnet.conv1(x)      # Conv: 224x224x3 → 112x112x64
        x = self.resnet.bn1(x)        # Batch Normalization (stabilizes training)
        x = self.resnet.relu(x)       # ReLU activation (non-linearity)
        x = self.resnet.maxpool(x)    # MaxPool: 112x112x64 → 56x56x64 (reduces size)
        
        # ============== RESNET LAYERS (Feature Extraction) ==============
        # Each layer has multiple "bottleneck blocks" that learn increasingly complex features
        x = self.resnet.layer1(x)     # Layer1: 56x56x64 → 56x56x256   (learns simple patterns)
        x = self.resnet.layer2(x)     # Layer2: 56x56x256 → 28x28x512  (learns object parts)
        x = self.resnet.layer3(x)     # Layer3: 28x28x512 → 14x14x1024 (learns objects)
        
        # Layer 4: Most important for Grad-CAM!
        # This layer captures high-level semantic features (faces, artifacts, inconsistencies)
        x = self.resnet.layer4(x)     # Layer4: 14x14x1024 → 7x7x2048 (learns complex concepts)
        
        # ============== SAVE FEATURES FOR GRAD-CAM ==============
        # If we need features for heatmap generation, save them here
        if return_features:
            features = x.clone()  # Clone to avoid gradient issues
        
        # ============== CLASSIFICATION HEAD ==============
        # Convert feature maps to class predictions
        x = self.resnet.avgpool(x)    # Average Pooling: 7x7x2048 → 1x1x2048
                                       # Reduces spatial dimensions while keeping features
        
        x = torch.flatten(x, 1)        # Flatten: 1x1x2048 → 2048 (convert to 1D vector)
                                       # Now we have 2048 numbers representing the image
        
        x = self.resnet.fc(x)          # Custom FC layers: 2048 → 512 → 2
                                       # Final output: 2 logits (Real score, Fake score)
        
        # ============== RETURN OUTPUT ==============
        if return_features:
            return x, features  # Return both output and layer4 features
        return x                # Return only output (normal case)
    
    def save_gradients(self, grad):
        """
        ==============================================================================
        HOOK FUNCTION - Save gradients for Grad-CAM
        ==============================================================================
        
        This is a "hook" - a callback function that PyTorch calls automatically
        during backpropagation to save gradients for Grad-CAM heatmap generation.
        
        Args:
            grad: Gradient tensor from backward pass
        """
        self.gradients = grad
    
    def forward_with_cam(self, x):
        """
        ==============================================================================
        FORWARD PASS WITH GRAD-CAM HOOKS
        ==============================================================================
        
        Similar to forward(), but with gradient tracking enabled for Grad-CAM.
        This method is used specifically when generating heatmaps.
        
        WHY SEPARATE METHOD?
        - Normal inference doesn't need gradients (faster)
        - Grad-CAM needs gradients to create heatmaps (slower but informative)
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Output logits
        """
        # Enable gradient computation for input
        x.requires_grad = True
        
        # Forward through conv layers (same as forward())
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        
        # Layer 4 - IMPORTANT: Register hook to capture gradients
        x = self.resnet.layer4(x)
        
        # Store activations (needed for Grad-CAM calculation)
        self.activations = x
        
        # Register backward hook to save gradients when backpropagation happens
        x.register_hook(self.save_gradients)
        
        # Continue forward pass
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.resnet.fc(x)
        
        return output
    
    def get_cam_weights(self):
        """
        ==============================================================================
        GET CAM WEIGHTS - Calculate importance of each feature map
        ==============================================================================
        
        Computes weights for each channel in layer4 by averaging gradients.
        These weights tell us which feature maps are most important for the decision.
        
        GRAD-CAM FORMULA:
        weights = GlobalAveragePool(gradients)  ← This function
        cam = ReLU(sum(weights * activations))  ← Done in gradcam.py
        
        Returns:
            torch.Tensor: Weights for each channel (shape: 1, 2048, 1, 1)
        """
        # Global average pooling of gradients across spatial dimensions
        # Shape: (batch, 2048, 7, 7) → (batch, 2048, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        return weights
    
    def load_pretrained(self, checkpoint_path):
        """
        ==============================================================================
        LOAD TRAINED MODEL WEIGHTS
        ==============================================================================
        
        After training, we save the model weights to a .pth file.
        This function loads those weights back into the model.
        
        USAGE:
            model = DeepfakeImageDetector()
            model.load_pretrained('weights/best_model.pth')
        
        Args:
            checkpoint_path (str): Path to saved model checkpoint (.pth file)
        
        CHECKPOINT STRUCTURE:
        The .pth file can be either:
        1. Just the model weights (state_dict)
        2. A dictionary containing:
           - 'model_state_dict': Model weights
           - 'optimizer_state_dict': Optimizer state (for resuming training)
           - 'epoch': Training epoch number
           - 'loss': Training loss
        """
        if os.path.exists(checkpoint_path):
            # Load checkpoint from file
            # map_location='cpu' ensures it works even without GPU
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if checkpoint is a dictionary with 'model_state_dict' key
            if 'model_state_dict' in checkpoint:
                # Load model weights from dictionary
                state_dict = checkpoint['model_state_dict']
                print(f"[DEBUG] Checkpoint contains 'model_state_dict'")
                print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")
                if 'val_acc' in checkpoint:
                    print(f"[DEBUG] Model was saved at epoch {checkpoint.get('epoch', '?')} with val_acc: {checkpoint.get('val_acc', '?'):.2f}%")
            else:
                # Checkpoint is just the state_dict itself
                state_dict = checkpoint
                print(f"[DEBUG] Checkpoint is direct state_dict")
            
            # Try to load with strict=False to see if there are mismatches
            try:
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"[WARNING] Missing keys when loading model: {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"[WARNING] Unexpected keys when loading model: {unexpected_keys[:5]}...")
            except Exception as e:
                print(f"[ERROR] Failed to load state_dict: {e}")
                raise
            
            print(f"✓ Loaded pretrained weights from {checkpoint_path}")
        else:
            print(f"✗ Warning: Checkpoint not found at {checkpoint_path}")
            print("  Model will use ImageNet pre-trained weights only.")


class ConvNeXtDetector(nn.Module):
    """
    ==============================================================================
    ALTERNATIVE DETECTOR - ConvNeXt Architecture (OPTIONAL)
    ==============================================================================
    
    ConvNeXt is a modern architecture (2022) that improves on ResNet.
    It's more accurate but also more computationally expensive.
    
    WHY ConvNeXt?
    - Better accuracy than ResNet50
    - More modern design principles
    - Inspired by Vision Transformers but still uses convolutions
    
    WHEN TO USE:
    - If you have good GPU (8GB+ VRAM)
    - If you want the best possible accuracy
    - For production deployment with powerful servers
    
    WHEN TO USE ResNet50 INSTEAD:
    - Limited GPU memory (4GB like Quadro T2000)
    - Faster inference needed
    - Good accuracy is enough
    
    NOTE: For this project, stick with ResNet50 unless you need the extra accuracy!
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        INITIALIZE ConvNeXt MODEL
        
        Args:
            num_classes (int): Number of output classes (default: 2)
            pretrained (bool): Use ImageNet pre-trained weights
        """
        super(ConvNeXtDetector, self).__init__()
        
        try:
            # Import ConvNeXt (only available in newer torchvision versions)
            from torchvision.models import convnext_base, ConvNeXt_Base_Weights
            
            # ============== LOAD CONVNEXT BACKBONE ==============
            if pretrained:
                # Load with ImageNet weights
                weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
                self.backbone = convnext_base(weights=weights)
                print("✓ Loaded ConvNeXt with ImageNet pre-trained weights")
            else:
                # Load with random weights
                self.backbone = convnext_base(weights=None)
                print("⚠ Loaded ConvNeXt with RANDOM weights")
            
            # ============== REPLACE CLASSIFIER ==============
            # ConvNeXt's classifier is at backbone.classifier[2]
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        except ImportError:
            # ConvNeXt not available in older torchvision versions
            raise ImportError(
                "ConvNeXt not available in your torchvision version.\n"
                "Please upgrade: pip install --upgrade torchvision\n"
                "Or use DeepfakeImageDetector (ResNet50) instead."
            )
    
    def forward(self, x):
        """
        FORWARD PASS
        
        Args:
            x (torch.Tensor): Input image (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits (batch_size, 2)
        """
        return self.backbone(x)


# ==============================================================================
# END OF FILE
# ==============================================================================
# 
# SUMMARY OF THIS FILE:
# 
# 1. DeepfakeImageDetector - Main model (ResNet50)
#    - Use this for training on Quadro T2000
#    - Good balance of speed and accuracy
#    - Well-tested and reliable
# 
# 2. ConvNeXtDetector - Alternative model (Optional)
#    - More accurate but heavier
#    - Requires more GPU memory
#    - Use only if ResNet50 accuracy isn't enough
# 
# NEXT STEPS:
# 1. Create training script (train.py)
# 2. Prepare dataset
# 3. Train the model
# 4. Evaluate performance
# 5. Save trained weights to weights/ folder
# ==============================================================================

