"""
Final Model (phase4_replay_best) - Evaluate on dataset/train.

Uses: dmi-backend/dataset/train/real and dataset/train/fake.
Same model & preprocessing as backend.

Run from dmi-backend:
    python ml_models/image_detection/evaluate_final_on_test.py
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pytree.*")

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageFile
from pathlib import Path
import numpy as np
import os
import sys
import time

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths: dmi-backend/dataset/train (has real/ and fake/)
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BACKEND_ROOT / "ml_models" / "weights" / "phase4_replay_best.pth"
TRAIN_FOLDER = BACKEND_ROOT / "dataset" / "train"

BATCH_SIZE = 32
# Use 0 on Windows to avoid DataLoader hang; use 4 on Linux
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same as inference.py (ViT 224, normalize 0.5)
VIT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class TestFolderDataset(Dataset):
    """Images from real/ and fake/ subfolders under root_path."""
    def __init__(self, root_path):
        self.images = []
        self.labels = []
        root = Path(root_path)
        for label, sub in [(0, "real"), (1, "fake")]:
            d = root / sub
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    self.images.append(str(f))
                    self.labels.append(label)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert("RGB")
            x = VIT_TRANSFORM(img)
            return x, self.labels[idx]
        except Exception:
            return torch.zeros(3, 224, 224), self.labels[idx]


def main():
    print()
    print("=" * 60)
    print("  Final Model (phase4_replay_best) - Evaluate on dataset/train")
    print("=" * 60)
    print()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    if not TRAIN_FOLDER.exists():
        print(f"ERROR: dataset/train not found at {TRAIN_FOLDER}")
        sys.exit(1)

    real_dir = TRAIN_FOLDER / "real"
    fake_dir = TRAIN_FOLDER / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        print(f"ERROR: dataset/train must have real/ and fake/ subfolders.")
        sys.exit(1)

    print(f"Model:   {MODEL_PATH}")
    print(f"Data:    {TRAIN_FOLDER}")
    print(f"Device:  {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    # Load model (same as inference.py)
    print("Loading ViT (phase4_replay_best)...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        id2label={0: "Real", 1: "Fake"},
        label2id={"Real": 0, "Fake": 1},
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(DEVICE)
    model.eval()
    print("Model loaded.\n")

    dataset = TestFolderDataset(TRAIN_FOLDER)
    n_real = (dataset.labels == 0).sum()
    n_fake = (dataset.labels == 1).sum()
    print(f"Test set: {len(dataset)} images (Real: {n_real}, Fake: {n_fake})\n")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    total_batches = len(loader)
    print(f"Evaluation started ({total_batches} batches, batch_size={BATCH_SIZE})...\n")
    all_preds = []
    all_labels = []
    start = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            logits = model(pixel_values=images).logits
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            if (batch_idx + 1) % 500 == 0:
                done = (batch_idx + 1) * BATCH_SIZE
                print(f"  Processed {min(done, len(dataset))}/{len(dataset)} images...")
    elapsed = time.time() - start

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    print("=" * 60)
    print(f"  RESULTS (dataset/train)")
    print("=" * 60)
    print(f"  Total: {len(all_labels)}  |  Time: {elapsed:.1f}s  |  Speed: {len(all_labels)/elapsed:.0f} img/s\n")

    print("--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"], digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("--- Confusion Matrix ---")
    print("                 Pred Real  Pred Fake")
    print(f"  Actual Real:    {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"  Actual Fake:    {cm[1,0]:>6}       {cm[1,1]:>6}")

    real_acc = 100.0 * cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0.0
    fake_acc = 100.0 * cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0.0
    overall = 100.0 * (cm[0,0] + cm[1,1]) / cm.sum()

    print("\n--- Accuracy ---")
    print(f"  Real:   {real_acc:.2f}%")
    print(f"  Fake:   {fake_acc:.2f}%")
    print(f"  Overall: {overall:.2f}%")
    print()
    print("=" * 60)
    print(f"  FINAL: {overall:.2f}% accuracy on TEST set ({len(all_labels)} images)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
