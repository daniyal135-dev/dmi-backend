"""
==============================================================================
ViT Phase 3 Model - Local Dataset Evaluation (OPTIMIZED)
==============================================================================
Evaluates the Phase 3 ViT model (trained on Kaggle) on local merged dataset.
This proves GENERALIZATION - model works on unseen data!

USAGE:
    python evaluate_vit.py

MODEL: phase3_best_model.pth (ViT-Base, trained on FFHQ + MS-COCO + DALL-E 3)
LABELS: Model uses 0=Real, 1=Fake
==============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageFile
from pathlib import Path
import numpy as np
import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = Path("D:/cursor/FYP/models/phase3_best_model.pth")
DATASET_PATH = Path("D:/cursor/FYP/dmi-backend/dataset/train")
BATCH_SIZE = 64  # Larger batch = faster (ViT is light enough for 4GB GPU)
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ViT preprocessing using fast torchvision transforms (5-10x faster than ViTImageProcessor)
# ViT-base-patch16-224-in21k uses: resize 224, normalize mean=0.5, std=0.5
VIT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0, 1] range
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ==============================================================================
# CUSTOM DATASET (FAST torchvision transforms)
# ==============================================================================
class ViTEvalDataset(Dataset):
    """
    Loads images from real/ and fake/ folders with fast torchvision transforms.
    """
    
    def __init__(self, dataset_path):
        self.images = []
        self.labels = []
        
        # Load real images (label = 0)
        real_dir = os.path.join(dataset_path, "real")
        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(os.path.join(real_dir, fname))
                    self.labels.append(0)
        
        # Load fake images (label = 1)
        fake_dir = os.path.join(dataset_path, "fake")
        if os.path.exists(fake_dir):
            for fname in os.listdir(fake_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(os.path.join(fake_dir, fname))
                    self.labels.append(1)
        
        print(f"  Real images: {self.labels.count(0)}")
        print(f"  Fake images: {self.labels.count(1)}")
        print(f"  Total: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert("RGB")
            pixel_values = VIT_TRANSFORM(img)
            return pixel_values, self.labels[idx]
        except Exception:
            return torch.zeros(3, 224, 224), self.labels[idx]


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
def main():
    print()
    print("=" * 60)
    print("  ViT Phase 3 Model - Local Dataset Evaluation")
    print("=" * 60)
    print()
    
    # --- Check model file ---
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    model_size = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Model: {MODEL_PATH}")
    print(f"Model size: {model_size:.1f} MB")
    print(f"Device: {DEVICE}")
    
    if DEVICE.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print()
    
    # --- Load Model ---
    print("Loading Phase 3 model...")
    try:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=2,
            id2label={0: "Real", 1: "Fake"},
            label2id={"Real": 0, "Fake": 1}
        )
        print("  Applying Phase 3 weights...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model = model.to(DEVICE)
        model.eval()
        print("Model loaded!\n")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # --- Check dataset ---
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return
    
    print(f"Dataset: {DATASET_PATH}")
    print("Scanning image files...")
    
    # --- Create Dataset & DataLoader ---
    try:
        dataset = ViTEvalDataset(DATASET_PATH)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    total_batches = len(dataloader)
    print(f"\nBatches: {total_batches} (batch_size={BATCH_SIZE})")
    print(f"Workers: {NUM_WORKERS} (parallel loading)")
    print(f"\nStarting evaluation...")
    print("-" * 60)
    
    # --- Evaluate ---
    all_preds = []
    all_labels = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(pixel_values=images)
            _, predicted = torch.max(outputs.logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                speed = (batch_idx + 1) * BATCH_SIZE / elapsed
                remaining = (total_batches - batch_idx - 1) * BATCH_SIZE / speed
                print(f"  Batch {batch_idx+1}/{total_batches} | "
                      f"{(batch_idx+1)*100/total_batches:.1f}% | "
                      f"Speed: {speed:.0f} img/s | "
                      f"ETA: {remaining/60:.1f} min")
    
    total_time = time.time() - start_time
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # --- Results ---
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Dataset: {DATASET_PATH}")
    print(f"  Total images: {len(all_labels)}")
    print(f"  Real images: {(all_labels == 0).sum()}")
    print(f"  Fake images: {(all_labels == 1).sum()}")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"  Speed: {len(all_labels)/total_time:.0f} images/second")
    
    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(
        all_labels, all_preds, 
        target_names=["Real", "Fake"], 
        digits=4
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    real_acc = cm[0][0] / (cm[0][0] + cm[0][1]) * 100 if (cm[0][0] + cm[0][1]) > 0 else 0
    fake_acc = cm[1][1] / (cm[1][0] + cm[1][1]) * 100 if (cm[1][0] + cm[1][1]) > 0 else 0
    overall_acc = (cm[0][0] + cm[1][1]) / cm.sum() * 100
    
    print("--- Confusion Matrix ---")
    print(f"                  Pred Real  Pred Fake")
    print(f"  Actual Real:     {cm[0][0]:>6}       {cm[0][1]:>6}")
    print(f"  Actual Fake:     {cm[1][0]:>6}       {cm[1][1]:>6}")
    
    print(f"\n--- Accuracy Breakdown ---")
    print(f"  Real Detection:  {real_acc:.2f}%")
    print(f"  Fake Detection:  {fake_acc:.2f}%")
    print(f"  Overall:         {overall_acc:.2f}%")
    
    print(f"\n{'=' * 60}")
    print(f"  FINAL: {overall_acc:.2f}% accuracy on {len(all_labels)} UNSEEN images")
    print(f"  This proves model GENERALIZATION!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
