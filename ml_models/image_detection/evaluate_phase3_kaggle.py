"""
==============================================================================
Phase 3 Best Model - Kaggle Evaluation (Two Runs to Avoid OOM)
==============================================================================

Phase 4 is NOT used. Only phase3_best_model.pth is used for evaluation.

RUN 1 (this script with --run 1):
  - Load Phase 1 (FFHQ) + Phase 2 (MS-COCO) datasets only.
  - Load Phase 3 best model, evaluate on Phase 1 and Phase 2.
  - Clear memory. Then RESTART KERNEL and run with --run 2.

RUN 2 (this script with --run 2, after kernel restart):
  - Load Phase 3 dataset only (DALL-E 3 fake + Kaggle Human real).
  - Load Phase 3 best model, evaluate on Phase 3.

Usage (Kaggle):
  Run 1: python evaluate_phase3_kaggle.py --run 1
  [Restart kernel]
  Run 2: python evaluate_phase3_kaggle.py --run 2

Model: phase3_best_model.pth (ViT-Base, 0=Real, 1=Fake)
==============================================================================
"""

import argparse
import gc
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# CONFIG (set your Kaggle paths here or via env)
# ==============================================================================
def _path(s):
    return Path(os.environ.get(s, s))

# Kaggle: add your dataset paths; local fallback for testing
MODEL_PATH = _path("PHASE3_MODEL_PATH") if os.environ.get("PHASE3_MODEL_PATH") else Path("/kaggle/working/checkpoints/phase3_best_model.pth")
PHASE1_ROOT = _path("PHASE1_ROOT") if os.environ.get("PHASE1_ROOT") else Path("/kaggle/input/")  # FFHQ real+fake
PHASE2_ROOT = _path("PHASE2_ROOT") if os.environ.get("PHASE2_ROOT") else Path("/kaggle/input/")  # MS-COCO real+fake
PHASE3_ROOT = _path("PHASE3_ROOT") if os.environ.get("PHASE3_ROOT") else Path("/kaggle/input/")  # DALL-E 3 + Human

BATCH_SIZE = 64
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ==============================================================================
# DATASET: single root with real/ and fake/ (or multiple roots combined)
# ==============================================================================
class RealFakeDataset(Dataset):
    """Images from one root: real/ and fake/ subdirs. Labels: 0=Real, 1=Fake."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform or VIT_TRANSFORM
        self.images = []
        self.labels = []
        for label, sub in enumerate(["real", "fake"]):
            d = self.root / sub
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    self.images.append(str(f))
                    self.labels.append(label)
        self.n_real = sum(1 for l in self.labels if l == 0)
        self.n_fake = sum(1 for l in self.labels if l == 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert("RGB")
            x = self.transform(img)
            return x, self.labels[idx]
        except Exception:
            return torch.zeros(3, 224, 224), self.labels[idx]


def _load_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        id2label={0: "Real", 1: "Fake"},
        label2id={"Real": 0, "Fake": 1}
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model = model.to(DEVICE)
    model.eval()
    return model


def _evaluate(model, dataloader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(DEVICE, non_blocking=True)
            out = model(pixel_values=pixel_values)
            _, pred = out.logits.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def _print_metrics(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"                 Pred Real  Pred Fake")
    print(f"  Actual Real:     {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"  Actual Fake:     {cm[1,0]:>6}       {cm[1,1]:>6}")
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
    print(f"  Overall Accuracy: {acc:.2f}%\n")
    return acc


def _clear_memory(model, loaders, datasets):
    del model, loaders, datasets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nMemory cleared. Restart kernel, then run with --run 2 (Phase 3 only).\n")


# ==============================================================================
# RUN 1: Phase 1 + Phase 2
# ==============================================================================
def run1(phase1_root, phase2_root):
    print("=" * 60)
    print("  RUN 1: Phase 3 Best Model on Phase 1 + Phase 2")
    print("=" * 60)
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    ds1 = RealFakeDataset(phase1_root)
    ds2 = RealFakeDataset(phase2_root)
    if len(ds1) == 0 and len(ds2) == 0:
        print("ERROR: No images in Phase 1 or Phase 2 roots.")
        return
    print(f"Phase 1: {len(ds1)} images (real={ds1.n_real}, fake={ds1.n_fake})")
    print(f"Phase 2: {len(ds2)} images (real={ds2.n_real}, fake={ds2.n_fake})")
    combined = ConcatDataset([ds1, ds2])
    loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print("Loading Phase 3 model...")
    model = _load_model()
    print("Evaluating...")
    t0 = time.time()
    y_pred, y_true = _evaluate(model, loader)
    elapsed = time.time() - t0
    print(f"Time: {elapsed/60:.1f} min, Speed: {len(y_true)/elapsed:.0f} img/s")
    _print_metrics("Phase 1 + Phase 2 Combined", y_true, y_pred)
    _clear_memory(model, loader, combined)


# ==============================================================================
# RUN 2: Phase 3 only
# ==============================================================================
def run2(phase3_root):
    print("=" * 60)
    print("  RUN 2: Phase 3 Best Model on Phase 3 Dataset")
    print("=" * 60)
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    ds3 = RealFakeDataset(phase3_root)
    if len(ds3) == 0:
        print("ERROR: No images in Phase 3 root (real/ and fake/).")
        return
    print(f"Phase 3: {len(ds3)} images (real={ds3.n_real}, fake={ds3.n_fake})")
    loader = DataLoader(ds3, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print("Loading Phase 3 model...")
    model = _load_model()
    print("Evaluating...")
    t0 = time.time()
    y_pred, y_true = _evaluate(model, loader)
    elapsed = time.time() - t0
    print(f"Time: {elapsed/60:.1f} min, Speed: {len(y_true)/elapsed:.0f} img/s")
    _print_metrics("Phase 3", y_true, y_pred)
    print("Evaluation complete.\n")


def main():
    p = argparse.ArgumentParser(description="Phase 3 model evaluation (two runs)")
    p.add_argument("--run", type=int, choices=[1, 2], required=True, help="1 = Phase 1+2, 2 = Phase 3 (run after kernel restart)")
    p.add_argument("--model", default=None, help="Override phase3_best_model path")
    p.add_argument("--phase1", default=None, help="Phase 1 dataset root (real/ fake/)")
    p.add_argument("--phase2", default=None, help="Phase 2 dataset root (real/ fake/)")
    p.add_argument("--phase3", default=None, help="Phase 3 dataset root (real/ fake/)")
    args = p.parse_args()
    global MODEL_PATH, PHASE1_ROOT, PHASE2_ROOT, PHASE3_ROOT
    if args.model:
        MODEL_PATH = Path(args.model)
    if args.phase1:
        PHASE1_ROOT = Path(args.phase1)
    if args.phase2:
        PHASE2_ROOT = Path(args.phase2)
    if args.phase3:
        PHASE3_ROOT = Path(args.phase3)
    if args.run == 1:
        run1(PHASE1_ROOT, PHASE2_ROOT)
    else:
        run2(PHASE3_ROOT)


if __name__ == "__main__":
    main()
