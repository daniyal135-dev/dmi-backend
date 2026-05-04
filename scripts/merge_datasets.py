"""
==============================================================================
MERGE ALL DATASETS SCRIPT
==============================================================================

WHAT THIS DOES:
- Merges Dataset 1, organized Dataset 2, and Dataset 3
- Creates final combined dataset structure
- Adds prefixes to avoid naming conflicts
- Creates train/val/test splits

USAGE:
    python scripts/merge_datasets.py

PREREQUISITES:
    - organize_dataset2.py must be run first!

INPUT:
    - Dataset 1: E:\Datasets\dataset1\
    - Dataset 2: dmi-backend/dataset/organized_dataset2/
    - Dataset 3: E:\Datasets\archive_2\Human Faces Dataset\

OUTPUT:
    dmi-backend/dataset/
    ├── train/
    │   ├── real/ (~47.7K images)
    │   └── fake/ (~47.7K images)
    └── val/
        ├── real/ (~5.3K images)
        └── fake/ (~5.3K images)

TIME: ~15-30 minutes
==============================================================================
"""

import shutil
import os
from pathlib import Path
from tqdm import tqdm
import random

# ============== CONFIGURATION ==============

# Source paths
DATASET1_ROOT = Path("E:/datasets/dataset 1")
DATASET2_ROOT = Path("dataset/organized_dataset2")
DATASET3_ROOT = Path("E:/datasets/dataset 3/Human Faces Datasets")

# Output paths
OUTPUT_ROOT = Path("dataset")
TRAIN_DIR = OUTPUT_ROOT / "train"
VAL_DIR = OUTPUT_ROOT / "val"

# Prefixes
PREFIX_D1 = "d1_"
PREFIX_D2 = "d2_"  # Already added in organize_dataset2.py
PREFIX_D3 = "d3_"

# Validation split ratio
VAL_SPLIT = 0.1  # 10% for validation

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def copy_with_prefix(source_dir, dest_dir, prefix, label=""):
    """
    Copy files from source to destination with prefix
    
    Returns: List of copied files
    """
    files = list(source_dir.glob("*"))
    copied = []
    
    for file in tqdm(files, desc=f"  Copying {label}", leave=False):
        if file.is_file():
            new_name = f"{prefix}{file.name}"
            dest_path = dest_dir / new_name
            try:
                shutil.copy2(file, dest_path)
                copied.append(str(dest_path))
            except Exception as e:
                print(f"    ⚠ Error copying {file.name}: {e}")
    
    return copied


def merge_datasets():
    """
    Main function to merge all datasets
    """
    print("=" * 70)
    print("MERGING ALL DATASETS")
    print("=" * 70)
    print()
    
    # ============== STEP 1: CHECK SOURCES ==============
    print("[Step 1/6] Checking source datasets...")
    
    # Check Dataset 1
    d1_train_real = DATASET1_ROOT / "train" / "real"
    d1_train_fake = DATASET1_ROOT / "train" / "fake"
    
    if not d1_train_real.exists() or not d1_train_fake.exists():
        print(f"❌ Dataset 1 not found at {DATASET1_ROOT}")
        return False
    
    # Check Dataset 2 (organized)
    d2_real = DATASET2_ROOT / "real"
    d2_fake = DATASET2_ROOT / "fake"
    
    if not d2_real.exists() or not d2_fake.exists():
        print(f"❌ Dataset 2 not organized yet!")
        print(f"   Please run: python scripts/organize_dataset2.py first")
        return False
    
    # Check Dataset 3
    d3_real = DATASET3_ROOT / "Real Images"
    d3_fake = DATASET3_ROOT / "AI-Generated Images"
    
    if not d3_real.exists() or not d3_fake.exists():
        print(f"❌ Dataset 3 not found at {DATASET3_ROOT}")
        return False
    
    print("✓ All source datasets found!")
    print(f"  Dataset 1: {DATASET1_ROOT}")
    print(f"  Dataset 2: {DATASET2_ROOT}")
    print(f"  Dataset 3: {DATASET3_ROOT}")
    print()
    
    # ============== STEP 2: CREATE TEMP FOLDERS ==============
    print("[Step 2/6] Creating temporary combined folder...")
    
    temp_real = OUTPUT_ROOT / "temp_combined" / "real"
    temp_fake = OUTPUT_ROOT / "temp_combined" / "fake"
    temp_real.mkdir(parents=True, exist_ok=True)
    temp_fake.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Temp folders created")
    print()
    
    # ============== STEP 3: COPY ALL DATASETS ==============
    print("[Step 3/6] Copying and merging datasets...")
    print()
    
    all_real = []
    all_fake = []
    
    # Dataset 1
    print("📁 Dataset 1 (MJ/DALL-E/SD):")
    real_files = copy_with_prefix(d1_train_real, temp_real, PREFIX_D1, "Real")
    fake_files = copy_with_prefix(d1_train_fake, temp_fake, PREFIX_D1, "Fake")
    all_real.extend(real_files)
    all_fake.extend(fake_files)
    print(f"  ✓ Copied: {len(real_files):,} real, {len(fake_files):,} fake")
    print()
    
    # Dataset 2 (already has d2_ prefix)
    print("📁 Dataset 2 (Kaggle Mixed):")
    real_files = list(d2_real.glob("*"))
    fake_files = list(d2_fake.glob("*"))
    
    for file in tqdm(real_files, desc="  Copying Real", leave=False):
        if file.is_file():
            shutil.copy2(file, temp_real / file.name)
            all_real.append(str(temp_real / file.name))
    
    for file in tqdm(fake_files, desc="  Copying Fake", leave=False):
        if file.is_file():
            shutil.copy2(file, temp_fake / file.name)
            all_fake.append(str(temp_fake / file.name))
    
    print(f"  ✓ Copied: {len(real_files):,} real, {len(fake_files):,} fake")
    print()
    
    # Dataset 3
    print("📁 Dataset 3 (AI Faces):")
    real_files = copy_with_prefix(d3_real, temp_real, PREFIX_D3, "Real")
    fake_files = copy_with_prefix(d3_fake, temp_fake, PREFIX_D3, "Fake")
    all_real.extend(real_files)
    all_fake.extend(fake_files)
    print(f"  ✓ Copied: {len(real_files):,} real, {len(fake_files):,} fake")
    print()
    
    # ============== STEP 4: SHUFFLE AND SPLIT ==============
    print("[Step 4/6] Creating train/val splits...")
    
    # Shuffle
    random.shuffle(all_real)
    random.shuffle(all_fake)
    
    # Calculate split
    val_real_count = int(len(all_real) * VAL_SPLIT)
    val_fake_count = int(len(all_fake) * VAL_SPLIT)
    
    train_real = all_real[val_real_count:]
    val_real = all_real[:val_real_count]
    train_fake = all_fake[val_fake_count:]
    val_fake = all_fake[:val_fake_count]
    
    print(f"  Train: {len(train_real):,} real, {len(train_fake):,} fake")
    print(f"  Val:   {len(val_real):,} real, {len(val_fake):,} fake")
    print()
    
    # ============== STEP 5: CREATE FINAL STRUCTURE ==============
    print("[Step 5/6] Creating final dataset structure...")
    
    # Create directories
    train_real_dir = TRAIN_DIR / "real"
    train_fake_dir = TRAIN_DIR / "fake"
    val_real_dir = VAL_DIR / "real"
    val_fake_dir = VAL_DIR / "fake"
    
    train_real_dir.mkdir(parents=True, exist_ok=True)
    train_fake_dir.mkdir(parents=True, exist_ok=True)
    val_real_dir.mkdir(parents=True, exist_ok=True)
    val_fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files to final locations
    for file_path in tqdm(train_real, desc="  Moving train/real", leave=False):
        shutil.move(file_path, train_real_dir / Path(file_path).name)
    
    for file_path in tqdm(train_fake, desc="  Moving train/fake", leave=False):
        shutil.move(file_path, train_fake_dir / Path(file_path).name)
    
    for file_path in tqdm(val_real, desc="  Moving val/real", leave=False):
        shutil.move(file_path, val_real_dir / Path(file_path).name)
    
    for file_path in tqdm(val_fake, desc="  Moving val/fake", leave=False):
        shutil.move(file_path, val_fake_dir / Path(file_path).name)
    
    print("✓ Final structure created")
    print()
    
    # ============== STEP 6: CLEANUP ==============
    print("[Step 6/6] Cleaning up temporary files...")
    
    try:
        shutil.rmtree(OUTPUT_ROOT / "temp_combined")
        print("✓ Temporary files deleted")
    except Exception as e:
        print(f"⚠ Could not delete temp files: {e}")
    
    print()
    
    # ============== FINAL REPORT ==============
    print("=" * 70)
    print("MERGE COMPLETE!")
    print("=" * 70)
    print()
    
    # Verify counts
    final_train_real = len(list(train_real_dir.glob("*")))
    final_train_fake = len(list(train_fake_dir.glob("*")))
    final_val_real = len(list(val_real_dir.glob("*")))
    final_val_fake = len(list(val_fake_dir.glob("*")))
    
    print("Final Dataset Structure:")
    print()
    print("📁 dataset/")
    print("   ├── train/")
    print(f"   │   ├── real/  ({final_train_real:,} images)")
    print(f"   │   └── fake/  ({final_train_fake:,} images)")
    print("   └── val/")
    print(f"       ├── real/  ({final_val_real:,} images)")
    print(f"       └── fake/  ({final_val_fake:,} images)")
    print()
    
    total = final_train_real + final_train_fake + final_val_real + final_val_fake
    print(f"Total Images: {total:,}")
    print(f"  Train: {final_train_real + final_train_fake:,} ({(final_train_real + final_train_fake)/total*100:.1f}%)")
    print(f"  Val:   {final_val_real + final_val_fake:,} ({(final_val_real + final_val_fake)/total*100:.1f}%)")
    print()
    
    real_total = final_train_real + final_val_real
    fake_total = final_train_fake + final_val_fake
    print(f"Balance:")
    print(f"  Real: {real_total:,} ({real_total/total*100:.1f}%)")
    print(f"  Fake: {fake_total:,} ({fake_total/total*100:.1f}%)")
    print()
    
    print("✅ Dataset ready for training!")
    return True


if __name__ == "__main__":
    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "MERGE DATASETS SCRIPT" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    print("This script will:")
    print("  1. Merge Dataset 1, 2, and 3")
    print("  2. Add prefixes (d1_, d2_, d3_)")
    print("  3. Create train/val splits (90/10)")
    print("  4. Create final organized structure")
    print()
    print("⚠ Make sure organize_dataset2.py was run first!")
    print()
    
    input("Press Enter to start...")
    print()
    
    success = merge_datasets()
    
    if success:
        print()
        print("=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("1. Verify: python scripts/verify_dataset.py")
        print("2. Start training with ViT!")
        print()
    else:
        print()
        print("⚠ Please fix the errors above and try again.")
        print()
