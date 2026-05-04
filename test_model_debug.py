"""
Quick debug script to test model predictions
"""
import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.image_detection.inference import ImageDetectionService
from django.conf import settings

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dmi_project.settings')
import django
django.setup()

# Path to trained model
model_path = os.path.join(
    settings.BASE_DIR,
    'ml_models',
    'weights',
    'best_model.pth'
)

print("="*70)
print("MODEL DEBUG TEST")
print("="*70)
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")
print()

# Initialize service
print("Loading model...")
service = ImageDetectionService(model_path=model_path)
print()

# Test with a sample image (you'll need to provide a path)
if len(sys.argv) > 1:
    test_image = sys.argv[1]
    print(f"Testing with image: {test_image}")
    print()
    
    # Get prediction
    result = service.predict(test_image)
    
    print("="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities:")
    print(f"  - Fake: {result['probabilities']['fake']:.4f}")
    print(f"  - Real: {result['probabilities']['real']:.4f}")
    print(f"Predicted class index: {result['predicted_class']}")
    print()
    
    # Also check raw model output
    print("="*70)
    print("RAW MODEL OUTPUT (for debugging)")
    print("="*70)
    
    # Load and preprocess image
    from ml_models.preprocessing.image_preprocessor import ImagePreprocessor
    preprocessor = ImagePreprocessor()
    input_tensor = preprocessor.preprocess(test_image)
    input_tensor = input_tensor.to(service.device)
    
    # Get raw output
    with torch.no_grad():
        raw_output = service.model(input_tensor)
        probabilities = torch.softmax(raw_output, dim=1)
    
    print(f"Raw logits: {raw_output[0].cpu().numpy()}")
    print(f"Probabilities (raw): {probabilities[0].cpu().numpy()}")
    print(f"Class names mapping: {service.class_names}")
    print(f"  - Index 0 = {service.class_names[0]}")
    print(f"  - Index 1 = {service.class_names[1]}")
    print()
    
else:
    print("Usage: python test_model_debug.py <path_to_image>")
    print("Example: python test_model_debug.py media/uploads/test.jpg")

