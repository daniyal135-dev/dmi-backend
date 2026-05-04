"""
==============================================================================
IMAGE DETECTION PACKAGE
==============================================================================

This package contains all components for image deepfake detection:

1. model.py - Neural network architectures (ResNet50, ConvNeXt)
2. inference.py - Prediction service (use this in Django!)
3. gradcam.py - Explainability through heatmaps

TYPICAL USAGE IN DJANGO:
    from ml_models.image_detection import ImageDetectionService
    
    service = ImageDetectionService(model_path='weights/trained_model.pth')
    result = service.predict_with_heatmap(image_path, output_dir)

==============================================================================
"""

from .model import DeepfakeImageDetector, ConvNeXtDetector
from .inference import ImageDetectionService
from .gradcam import GradCAM

__all__ = [
    'DeepfakeImageDetector',
    'ConvNeXtDetector',
    'ImageDetectionService',
    'GradCAM'
]
