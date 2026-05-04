"""
==============================================================================
DATASET PREPARATION SCRIPT
==============================================================================

WHAT THIS SCRIPT DOES:
Splits your training data into training and validation sets.

WHY WE NEED THIS:
Your dataset has train/ and test/ folders, but we also need a validation set!

TRAIN vs VALIDATION vs TEST:
- TRAIN (80%): Model learns from this data
- VALIDATION (10%): Check performance during training, tune hyperparameters
- TEST (10%): Final evaluation (never seen during training!)

CURRENT STRUCTURE:
dataset/
├── train/
│   ├── fake/ (24,000)
│   └── real/ (24,000)
└── test/
    ├── fake/ (6,000)
    └── real/ (6,000)

TARGET STRUCTURE:
dataset/
├── train/
│   ├── fake/ (21,600 = 90% of 24,000)
│   └── real/ (21,600 = 90% of 24,000)
├── val/
│   ├── fake/ (2,400 = 10% of 24,000)
│   └── real/ (2,400 = 10% of 24,000)
└── test/
    ├── fake/ (6,000)
    └── real/ (6,000)

USAGE:
    python ml_models/image_detection/prepare_dataset.py

==============================================================================
"""

# ============== IMPORTS ==============
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm  # Progress bar library

# ============== CONFIGURATION ==============
# Paths
DATASET_ROOT = Path("dataset")           # Your dataset folder
TRAIN_DIR = DATASET_ROOT / "train"       # Original training folder
VAL_DIR = DATASET_ROOT / "val"           # Validation folder (we'll create this)

# Split ratio
VAL_SPLIT = 0.1  # 10% of training data goes to validation (90% stays in train)

# Random seed for reproducibility
RANDOM_SEED = 42


def create_validation_split():
    """
    ==============================================================================
    CREATE VALIDATION SET FROM TRAINING SET
    ==============================================================================
    
    This function:
    1. Creates val/fake/ and val/real/ folders
    2. Randomly selects 10% of images from train/
    3. Moves them to val/
    4. Leaves 90% in train/
    
    WHY MOVE instead of COPY?
    - Saves disk space (60GB dataset!)
    - Ensures clean split (no overlap)
    
    RANDOM SELECTION:
    - We shuffle the file list
    - Take first 10% for validation
    - Leave remaining 90% for training
    - Using random seed ensures reproducibility
    """
    
    print("="*70)
    print("DATASET PREPARATION - Creating Validation Split")
    print("="*70)
    print()
    
    # Set random seed for reproducibility
    # This ensures same split every time you run the script
    random.seed(RANDOM_SEED)
    
    # ============== CREATE VALIDATION FOLDERS ==============
    print("Step 1: Creating validation directory structure...")
    
    # Create val/fake/ and val/real/ folders
    val_fake_dir = VAL_DIR / "fake"
    val_real_dir = VAL_DIR / "real"
    
    val_fake_dir.mkdir(parents=True, exist_ok=True)
    val_real_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  ✓ Created: {val_fake_dir}")
    print(f"  ✓ Created: {val_real_dir}")
    print()
    
    # ============== PROCESS FAKE IMAGES ==============
    print("Step 2: Splitting FAKE images (90% train, 10% val)...")
    
    train_fake_dir = TRAIN_DIR / "fake"
    
    # Get list of all fake images
    fake_images = list(train_fake_dir.glob("*"))  # All files (jpg, png)
    total_fake = len(fake_images)
    
    # Calculate how many go to validation
    num_val_fake = int(total_fake * VAL_SPLIT)
    
    print(f"  Total fake images: {total_fake:,}")
    print(f"  Moving to validation: {num_val_fake:,} ({VAL_SPLIT*100:.0f}%)")
    print(f"  Remaining in train: {total_fake - num_val_fake:,} ({(1-VAL_SPLIT)*100:.0f}%)")
    
    # Shuffle and select random images for validation
    random.shuffle(fake_images)
    val_fake_images = fake_images[:num_val_fake]
    
    # Move selected images to validation folder
    for img_path in tqdm(val_fake_images, desc="  Moving fake images"):
        # Move file from train/fake/ to val/fake/
        dest_path = val_fake_dir / img_path.name
        shutil.move(str(img_path), str(dest_path))
    
    print(f"  ✓ Moved {num_val_fake:,} fake images to validation")
    print()
    
    # ============== PROCESS REAL IMAGES ==============
    print("Step 3: Splitting REAL images (90% train, 10% val)...")
    
    train_real_dir = TRAIN_DIR / "real"
    
    # Get list of all real images
    real_images = list(train_real_dir.glob("*"))
    total_real = len(real_images)
    
    # Calculate how many go to validation
    num_val_real = int(total_real * VAL_SPLIT)
    
    print(f"  Total real images: {total_real:,}")
    print(f"  Moving to validation: {num_val_real:,} ({VAL_SPLIT*100:.0f}%)")
    print(f"  Remaining in train: {total_real - num_val_real:,} ({(1-VAL_SPLIT)*100:.0f}%)")
    
    # Shuffle and select random images for validation
    random.shuffle(real_images)
    val_real_images = real_images[:num_val_real]
    
    # Move selected images to validation folder
    for img_path in tqdm(val_real_images, desc="  Moving real images"):
        # Move file from train/real/ to val/real/
        dest_path = val_real_dir / img_path.name
        shutil.move(str(img_path), str(dest_path))
    
    print(f"  ✓ Moved {num_val_real:,} real images to validation")
    print()
    
    # ============== SUMMARY ==============
    print("="*70)
    print("DATASET SPLIT COMPLETE!")
    print("="*70)
    print()
    print("Final Dataset Structure:")
    print()
    print(f"TRAINING SET (90%):")
    print(f"  ├── train/fake/  {len(list((TRAIN_DIR / 'fake').glob('*'))):,} images")
    print(f"  └── train/real/  {len(list((TRAIN_DIR / 'real').glob('*'))):,} images")
    print(f"  Total: {len(list((TRAIN_DIR / 'fake').glob('*'))) + len(list((TRAIN_DIR / 'real').glob('*'))):,} images")
    print()
    print(f"VALIDATION SET (10%):")
    print(f"  ├── val/fake/    {len(list((VAL_DIR / 'fake').glob('*'))):,} images")
    print(f"  └── val/real/    {len(list((VAL_DIR / 'real').glob('*'))):,} images")
    print(f"  Total: {len(list((VAL_DIR / 'fake').glob('*'))) + len(list((VAL_DIR / 'real').glob('*'))):,} images")
    print()
    print(f"TEST SET (Original):")
    print(f"  ├── test/fake/   {len(list((DATASET_ROOT / 'test' / 'fake').glob('*'))):,} images")
    print(f"  └── test/real/   {len(list((DATASET_ROOT / 'test' / 'real').glob('*'))):,} images")
    print(f"  Total: {len(list((DATASET_ROOT / 'test' / 'fake').glob('*'))) + len(list((DATASET_ROOT / 'test' / 'real').glob('*'))):,} images")
    print()
    print("="*70)
    print("✓ Dataset is ready for training!")
    print("="*70)


def check_dataset_exists():
    """
    CHECK IF DATASET EXISTS
    
    Verifies that the dataset folder structure is correct before proceeding.
    """
    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset folder not found at {DATASET_ROOT}")
        print("Please make sure your dataset is in the correct location!")
        return False
    
    if not TRAIN_DIR.exists():
        print(f"ERROR: Training folder not found at {TRAIN_DIR}")
        return False
    
    if not (TRAIN_DIR / "fake").exists() or not (TRAIN_DIR / "real").exists():
        print("ERROR: train/fake/ or train/real/ folders not found!")
        return False
    
    return True


def main():
    """
    MAIN FUNCTION
    
    Orchestrates the entire dataset preparation process.
    """
    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "DATASET PREPARATION SCRIPT" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Check if dataset exists
    if not check_dataset_exists():
        print("\n❌ Dataset check failed! Please fix the issues above.")
        return
    
    # Check if validation split already exists
    if VAL_DIR.exists() and len(list(VAL_DIR.glob("*"))) > 0:
        print("⚠️  WARNING: Validation folder already exists!")
        print(f"   Location: {VAL_DIR}")
        print()
        response = input("   Do you want to recreate it? This will move images again. (yes/no): ")
        
        if response.lower() not in ['yes', 'y']:
            print("\n✓ Keeping existing validation split. Dataset is ready!")
            return
        else:
            print("\n  Removing existing validation folder...")
            shutil.rmtree(VAL_DIR)
            print("  ✓ Removed")
            print()
    
    # Create validation split
    create_validation_split()
    
    print()
    print("🎉 All done! You can now run the training script:")
    print("   python ml_models/image_detection/train.py")
    print()


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()


# ==============================================================================
# END OF FILE
# ==============================================================================
#
# WHAT THIS SCRIPT DID:
# 1. Created validation folders (val/fake/, val/real/)
# 2. Moved 10% of training images to validation
# 3. Left 90% in training
# 4. Test set remains untouched
#
# WHY THIS SPLIT?
# - 90% train (43,200 images): Model learns from this
# - 10% val (4,800 images): Monitor progress, prevent overfitting
# - Test set (12,000 images): Final evaluation only!
#
# NEXT STEP:
# Run the training script: python ml_models/image_detection/train.py
# ==============================================================================

