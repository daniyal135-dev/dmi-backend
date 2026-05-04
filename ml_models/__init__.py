"""
==============================================================================
ML MODELS PACKAGE FOR DMI++
==============================================================================

This package contains all machine learning components for deepfake detection.

SUBPACKAGES:
- image_detection: Image deepfake detection (ResNet50)
- preprocessing: Data preparation (images, text, videos)
- text_detection: AI-generated text detection (to be implemented)
- video_detection: Video deepfake detection (to be implemented)

QUICK START:
    # Image detection
    from ml_models.image_detection import ImageDetectionService
    service = ImageDetectionService(model_path='weights/model.pth')
    result = service.predict_with_heatmap('image.jpg', 'output/')
    
    # Preprocessing
    from ml_models.preprocessing import ImagePreprocessor
    preprocessor = ImagePreprocessor()
    tensor = preprocessor.preprocess('image.jpg')

For detailed documentation, see ml_models/README.md
==============================================================================
"""

# Make key classes easily accessible
from .image_detection import ImageDetectionService, DeepfakeImageDetector, GradCAM
from .preprocessing import ImagePreprocessor, TextPreprocessor, VideoPreprocessor

__version__ = '1.0.0'

__all__ = [
    # Image Detection
    'ImageDetectionService',
    'DeepfakeImageDetector',
    'GradCAM',
    
    # Preprocessing
    'ImagePreprocessor',
    'TextPreprocessor',
    'VideoPreprocessor',
]
