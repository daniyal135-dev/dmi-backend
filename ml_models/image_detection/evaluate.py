"""
==============================================================================
EVALUATION SCRIPT - Test Trained Model
==============================================================================

WHAT THIS SCRIPT DOES:
Tests your trained model on the test set and generates detailed metrics.

METRICS CALCULATED:
- Accuracy: % of correct predictions
- Precision: Of predicted fakes, how many are actually fake?
- Recall: Of actual fakes, how many did we detect?
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Visual breakdown of predictions

USAGE:
    python ml_models/image_detection/evaluate.py

REQUIREMENTS:
- Trained model (best_model.pth must exist)
- Test dataset (dataset/test/)

==============================================================================
"""

# ============== IMPORTS ==============
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from tqdm import tqdm
from PIL import ImageFile

# Import model
from model import DeepfakeImageDetector

# ============== FIX FOR CORRUPTED/TRUNCATED IMAGES ==============
# Some images in large datasets may be corrupted or incomplete
# This allows PIL to load truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============== CONFIGURATION ==============
DATASET_ROOT = Path("dataset")
WEIGHTS_PATH = Path("ml_models/weights/best_model.pth")
BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# DATA LOADING
# ==============================================================================

def get_test_loader():
    """
    CREATE TEST DATA LOADER
    
    Loads test images with same transformations as validation (no augmentation).
    """
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_dataset = datasets.ImageFolder(
        root=DATASET_ROOT / "test",
        transform=test_transforms
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return test_loader, test_dataset.classes


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

def evaluate_model(model, test_loader):
    """
    EVALUATE MODEL ON TEST SET
    
    Returns predictions and ground truth labels for metric calculation.
    """
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Evaluating model on test set...")
    print()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probabilities)
    )


# ==============================================================================
# METRICS CALCULATION
# ==============================================================================

def calculate_metrics(predictions, labels, class_names):
    """
    CALCULATE AND DISPLAY METRICS
    """
    
    print("\n" + "="*70)
    print("TEST SET EVALUATION RESULTS")
    print("="*70)
    print()
    
    # ============== CLASSIFICATION REPORT ==============
    print("Classification Report:")
    print("="*70)
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # ============== CONFUSION MATRIX ==============
    cm = confusion_matrix(labels, predictions)
    
    print("Confusion Matrix:")
    print("="*70)
    print(f"                 Predicted")
    print(f"                 Fake    Real")
    print(f"Actual  Fake     {cm[0][0]:<6}  {cm[0][1]:<6}")
    print(f"        Real     {cm[1][0]:<6}  {cm[1][1]:<6}")
    print()
    
    # Calculate accuracy
    accuracy = 100.0 * np.sum(predictions == labels) / len(labels)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("="*70)
    print()
    
    return cm


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_confusion_matrix(cm, class_names):
    """
    PLOT CONFUSION MATRIX HEATMAP
    """
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    output_path = Path("ml_models/weights/confusion_matrix.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    
    plt.close()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    MAIN EVALUATION FUNCTION
    """
    
    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MODEL EVALUATION SCRIPT" + " "*30 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Check if model exists
    if not WEIGHTS_PATH.exists():
        print(f"❌ Error: Trained model not found at {WEIGHTS_PATH}")
        print("   Please train the model first: python ml_models/image_detection/train.py")
        return
    
    # Load model
    print("Loading trained model...")
    model = DeepfakeImageDetector(num_classes=2, pretrained=False)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    print(f"✓ Model loaded from {WEIGHTS_PATH}")
    print(f"  Trained for {checkpoint['epoch']+1} epochs")
    print(f"  Best val accuracy: {checkpoint['val_acc']:.2f}%")
    print()
    
    # Load test data
    print("Loading test data...")
    test_loader, class_names = get_test_loader()
    print(f"✓ Test set loaded: {len(test_loader.dataset):,} images")
    print(f"  Classes: {class_names}")
    print()
    
    # Evaluate
    predictions, labels, probabilities = evaluate_model(model, test_loader)
    
    # Calculate metrics
    cm = calculate_metrics(predictions, labels, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    print()
    print("="*70)
    print("✓ Evaluation complete!")
    print("="*70)
    print()
    print("Your model is ready for production!")
    print("Next steps:")
    print("  1. Integrate with Django API")
    print("  2. Test with real-world images")
    print("  3. Deploy to production server")
    print()


if __name__ == "__main__":
    main()


# ==============================================================================
# END OF FILE
# ==============================================================================

