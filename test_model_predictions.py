"""
Test script to verify model predictions
This will help us understand what the model is actually doing
"""
import os
import sys
import torch
import numpy as np

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
print("MODEL PREDICTION TEST")
print("="*70)
print()

# Initialize service
print("Loading model...")
service = ImageDetectionService(model_path=model_path)
print()

# Test with images from the test set to see if predictions match
test_fake_dir = os.path.join(settings.BASE_DIR, 'dataset', 'test', 'fake')
test_real_dir = os.path.join(settings.BASE_DIR, 'dataset', 'test', 'real')

print("Testing with images from test set...")
print()

# Test a few fake images
if os.path.exists(test_fake_dir):
    fake_images = [f for f in os.listdir(test_fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
    print(f"Testing {len(fake_images)} FAKE images (should predict 'fake'):")
    for img_name in fake_images:
        img_path = os.path.join(test_fake_dir, img_name)
        result = service.predict(img_path)
        print(f"  {img_name[:30]:30} → {result['verdict']:4} (conf: {result['confidence']:.4f}, fake_prob: {result['probabilities']['fake']:.4f}, real_prob: {result['probabilities']['real']:.4f})")
    print()

# Test a few real images
if os.path.exists(test_real_dir):
    real_images = [f for f in os.listdir(test_real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
    print(f"Testing {len(real_images)} REAL images (should predict 'real'):")
    for img_name in real_images:
        img_path = os.path.join(test_real_dir, img_name)
        result = service.predict(img_path)
        print(f"  {img_name[:30]:30} → {result['verdict']:4} (conf: {result['confidence']:.4f}, fake_prob: {result['probabilities']['fake']:.4f}, real_prob: {result['probabilities']['real']:.4f})")
    print()

print("="*70)
print("If FAKE images predict 'fake' and REAL images predict 'real', model is working!")
print("If everything predicts 'fake', there's still an issue.")
print("="*70)

