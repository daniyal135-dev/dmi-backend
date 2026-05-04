"""
==============================================================================
TRAINING SCRIPT - DeepfakeImageDetector with ResNet50
==============================================================================

WHAT THIS SCRIPT DOES:
Trains a ResNet50 model to detect AI-generated images (Stable Diffusion, DALL-E, Midjourney)

WHAT YOU'LL LEARN:
- How to load and prepare image datasets
- How to initialize and configure a neural network
- What loss functions and optimizers do
- How the training loop works (forward pass, backward pass, optimization)
- How to track progress and save checkpoints
- How to prevent overfitting with validation monitoring

TRAINING PROCESS (High-level):
1. Load dataset (train/validation images)
2. Create model (ResNet50 with our custom head)
3. Define loss function (how we measure errors)
4. Define optimizer (how we update weights)
5. Training loop:
   - Forward pass: Image → Model → Prediction
   - Calculate loss: How wrong was the prediction?
   - Backward pass: Calculate gradients
   - Update weights: Make model better
   - Validate: Check performance on validation set
6. Save best model

USAGE:
    python ml_models/image_detection/train.py

REQUIREMENTS:
- Dataset prepared (run prepare_dataset.py first!)
- GPU recommended (Quadro T2000 or better)
- 8-12 GB RAM
- 2-6 hours training time

==============================================================================
"""

# ============== IMPORTS ==============
import torch                              # PyTorch: Deep learning framework
import torch.nn as nn                     # Neural network modules
import torch.optim as optim              # Optimizers (SGD, Adam, etc.)
from torch.utils.data import DataLoader  # Data loading utilities
from torchvision import datasets, transforms  # Image datasets and transformations
import os
import time
from pathlib import Path
from tqdm import tqdm                     # Progress bars
import matplotlib.pyplot as plt           # Plotting
import json                               # Save training history
from PIL import ImageFile                 # PIL image file handling

# Import our custom model
from model import DeepfakeImageDetector

# ============== FIX FOR CORRUPTED/TRUNCATED IMAGES ==============
# Some images in large datasets may be corrupted or incomplete
# This allows PIL to load truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============== CONFIGURATION ==============
# These are hyperparameters - values you can tune for better performance

# Paths
DATASET_ROOT = Path("dataset")
WEIGHTS_DIR = Path("ml_models/weights")
CHECKPOINTS_DIR = WEIGHTS_DIR / "checkpoints"

# Model name for saving
MODEL_NAME = "best_model.pth"

# Training hyperparameters
BATCH_SIZE = 32          # Number of images processed together
                        # Larger = faster but needs more GPU memory
                        # Your Quadro T2000 (4GB) can handle 32
                        
NUM_EPOCHS = 50          # Number of complete passes through dataset
                        # More epochs = more learning (but risk overfitting)
                        
LEARNING_RATE = 0.0001  # How big are weight updates
                        # Too high = unstable training
                        # Too low = slow learning
                        # 0.0001 is good for fine-tuning
                        
WEIGHT_DECAY = 0.0001   # L2 regularization (prevents overfitting)

# Early stopping
PATIENCE = 10            # Stop if no improvement for 10 epochs

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image size (ResNet50 expects 224x224)
IMG_SIZE = 224

# Number of worker processes for data loading
# Windows: Use 0 (multiprocessing issues)
# Linux/Mac: Use 4-8
NUM_WORKERS = 0


# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================

def get_data_transforms():
    """
    CREATE DATA TRANSFORMATIONS
    
    Transformations prepare images for the model:
    1. Resize to 224x224 (ResNet50 requirement)
    2. Convert to tensor
    3. Normalize using ImageNet statistics
    4. Data augmentation (training only)
    
    WHY DATA AUGMENTATION?
    Training images are randomly transformed:
    - Random horizontal flip (person facing left vs right)
    - Random rotation (slightly tilted photos)
    - Color jitter (different lighting conditions)
    
    This helps the model:
    - Learn robust features (not memorize specific images)
    - Generalize better to new images
    - Prevent overfitting
    
    Returns:
        dict: {'train': train_transforms, 'val': val_transforms}
    """
    
    # ============== TRAINING TRANSFORMATIONS ==============
    # More aggressive transformations for training (data augmentation)
    train_transforms = transforms.Compose([
        # Resize to 256 first (slightly larger than needed)
        transforms.Resize(256),
        
        # Random crop to 224x224 (adds variation)
        # Each epoch sees slightly different crops
        transforms.RandomCrop(IMG_SIZE),
        
        # Random horizontal flip (50% chance)
        # Augments dataset size (image + flipped = 2x data!)
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Random rotation (-10 to +10 degrees)
        # Helps with slightly tilted photos
        transforms.RandomRotation(degrees=10),
        
        # Color jitter (brightness, contrast, saturation, hue)
        # Simulates different lighting conditions
        transforms.ColorJitter(
            brightness=0.2,   # ±20% brightness
            contrast=0.2,     # ±20% contrast
            saturation=0.2,   # ±20% saturation
            hue=0.1          # ±10% hue
        ),
        
        # Convert PIL Image to PyTorch Tensor
        # Shape: (H, W, C) → (C, H, W)
        # Range: [0, 255] → [0.0, 1.0]
        transforms.ToTensor(),
        
        # Normalize using ImageNet mean and std
        # This standardizes pixel values for better training
        # WHY these numbers? ResNet50 was trained with them!
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean (R, G, B)
            std=[0.229, 0.224, 0.225]     # ImageNet std (R, G, B)
        )
    ])
    
    # ============== VALIDATION/TEST TRANSFORMATIONS ==============
    # No augmentation for validation (we want consistent evaluation)
    val_transforms = transforms.Compose([
        # Resize to 256
        transforms.Resize(256),
        
        # Center crop to 224x224 (not random!)
        # Always crops from center for consistency
        transforms.CenterCrop(IMG_SIZE),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize (same as training)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return {'train': train_transforms, 'val': val_transforms}


def get_dataloaders():
    """
    CREATE DATA LOADERS
    
    DataLoader wraps a dataset and provides:
    - Batching (group images into batches)
    - Shuffling (randomize order each epoch)
    - Parallel loading (load data in background)
    
    FOLDER STRUCTURE EXPECTED:
    dataset/
    ├── train/
    │   ├── fake/    (AI-generated images)
    │   └── real/    (Real images)
    ├── val/
    │   ├── fake/
    │   └── real/
    └── test/
        ├── fake/
        └── real/
    
    CLASS MAPPING:
    - fake/ → Label 0
    - real/ → Label 1
    (Alphabetical order by folder name)
    
    Returns:
        dict: {'train': train_loader, 'val': val_loader, 'test': test_loader}
    """
    
    print("="*70)
    print("LOADING DATASET")
    print("="*70)
    print()
    
    # Get transformations
    data_transforms = get_data_transforms()
    
    # ============== CREATE DATASETS ==============
    # ImageFolder automatically:
    # 1. Reads folder structure (fake/, real/)
    # 2. Assigns labels (0, 1) alphabetically
    # 3. Applies transformations
    
    print("Loading training data...")
    train_dataset = datasets.ImageFolder(
        root=DATASET_ROOT / "train",
        transform=data_transforms['train']
    )
    
    print("Loading validation data...")
    val_dataset = datasets.ImageFolder(
        root=DATASET_ROOT / "val",
        transform=data_transforms['val']
    )
    
    print("Loading test data...")
    test_dataset = datasets.ImageFolder(
        root=DATASET_ROOT / "test",
        transform=data_transforms['val']  # Use val transforms (no augmentation)
    )
    
    # Print class mapping
    print()
    print("Class mapping:")
    for idx, class_name in enumerate(train_dataset.classes):
        print(f"  {class_name} → {idx}")
    print()
    
    # ============== CREATE DATA LOADERS ==============
    # DataLoader provides batching and shuffling
    
    print("Creating data loaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,      # Process 32 images at once
        shuffle=True,                 # Randomize order each epoch
        num_workers=NUM_WORKERS,     # Number of parallel workers
        pin_memory=True              # Speed up GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,                # Don't shuffle validation (consistency)
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,                # Don't shuffle test
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Print dataset statistics
    print()
    print("Dataset statistics:")
    print(f"  Training samples:   {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Test samples:       {len(test_dataset):,}")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Training batches:   {len(train_loader):,}")
    print(f"  Validation batches: {len(val_loader):,}")
    print()
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


# ==============================================================================
# MODEL, LOSS, AND OPTIMIZER
# ==============================================================================

def create_model():
    """
    CREATE AND CONFIGURE MODEL
    
    Initializes ResNet50 with:
    - Pre-trained ImageNet weights (transfer learning!)
    - Custom classification head (2 classes: fake, real)
    - Moved to GPU if available
    
    Returns:
        model: DeepfakeImageDetector instance
    """
    
    print("="*70)
    print("CREATING MODEL")
    print("="*70)
    print()
    
    # Create model with pre-trained weights
    print("Initializing ResNet50 with ImageNet weights...")
    model = DeepfakeImageDetector(num_classes=2, pretrained=True)
    
    # Move model to GPU if available
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device:               {DEVICE}")
    print()
    
    return model


def create_criterion():
    """
    CREATE LOSS FUNCTION
    
    WHAT IS LOSS?
    Loss measures how wrong the model's predictions are.
    Goal: Minimize loss → Better predictions!
    
    CROSS-ENTROPY LOSS:
    Standard loss for classification tasks.
    
    How it works:
    1. Model outputs: [score_fake, score_real]
    2. Apply softmax: Convert to probabilities [prob_fake, prob_real]
    3. Compare to ground truth: 0 (fake) or 1 (real)
    4. Calculate penalty: Higher penalty for confident wrong predictions
    
    Example:
    - True label: Real (1)
    - Model predicts: [0.2, 0.8] (80% confident it's real)
    - Loss: LOW (good prediction!)
    
    - True label: Real (1)
    - Model predicts: [0.9, 0.1] (90% confident it's fake)
    - Loss: HIGH (bad prediction!)
    
    Returns:
        criterion: CrossEntropyLoss instance
    """
    
    print("Creating loss function (CrossEntropyLoss)...")
    
    # CrossEntropyLoss combines:
    # 1. Softmax (convert logits to probabilities)
    # 2. Negative log likelihood (calculate loss)
    criterion = nn.CrossEntropyLoss()
    
    print("  Loss function: CrossEntropyLoss")
    print("  → Measures classification error")
    print("  → Lower loss = Better predictions")
    print()
    
    return criterion


def create_optimizer(model):
    """
    CREATE OPTIMIZER
    
    WHAT IS AN OPTIMIZER?
    Optimizer updates model weights to minimize loss.
    
    Think of it like descending a hill:
    - Loss is height
    - Weights are position
    - Optimizer finds path down the hill (gradient descent)
    
    ADAM OPTIMIZER:
    - Adaptive Moment Estimation
    - Automatically adjusts learning rate
    - Works well for most tasks
    - Better than basic SGD
    
    LEARNING RATE:
    - How big are the steps down the hill?
    - Too big: Overshoot minimum (unstable)
    - Too small: Takes forever (slow)
    - 0.0001 is good for fine-tuning
    
    WEIGHT DECAY:
    - L2 regularization
    - Prevents weights from getting too large
    - Helps prevent overfitting
    
    Returns:
        optimizer: Adam optimizer instance
    """
    
    print("Creating optimizer (Adam)...")
    
    # Adam: Adaptive Moment Estimation
    # Combines best of multiple optimization methods
    optimizer = optim.Adam(
        model.parameters(),          # Parameters to optimize
        lr=LEARNING_RATE,            # Learning rate (step size)
        weight_decay=WEIGHT_DECAY    # L2 regularization
    )
    
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay:  {WEIGHT_DECAY}")
    print()
    
    return optimizer


def create_scheduler(optimizer):
    """
    CREATE LEARNING RATE SCHEDULER
    
    WHAT IS A SCHEDULER?
    Automatically adjusts learning rate during training.
    
    WHY REDUCE LEARNING RATE?
    - Early training: Large steps (explore quickly)
    - Late training: Small steps (fine-tune carefully)
    
    ReduceLROnPlateau:
    - Monitors validation loss
    - If loss stops improving → Reduce learning rate
    - Helps escape plateaus and improve convergence
    
    Parameters:
    - mode='min': We want to minimize loss
    - factor=0.5: Reduce LR by half
    - patience=5: Wait 5 epochs before reducing
    - verbose=True: Print when LR changes
    
    Returns:
        scheduler: ReduceLROnPlateau instance
    """
    
    print("Creating learning rate scheduler...")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # Minimize validation loss
        factor=0.5,          # Multiply LR by 0.5
        patience=5,          # Wait 5 epochs before reducing
        min_lr=1e-7         # Don't go below this
    )
    
    print("  Scheduler: ReduceLROnPlateau")
    print("  → Reduces LR when validation loss plateaus")
    print("  → Factor: 0.5 (halve LR)")
    print("  → Patience: 5 epochs")
    print()
    
    return scheduler


# ==============================================================================
# TRAINING AND VALIDATION
# ==============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """
    TRAIN FOR ONE EPOCH
    
    An epoch is one complete pass through the training dataset.
    
    TRAINING LOOP:
    For each batch of images:
    1. Forward pass: Image → Model → Prediction
    2. Calculate loss: Compare prediction to true label
    3. Backward pass: Calculate gradients (how to improve)
    4. Update weights: Apply gradients (make model better)
    
    Args:
        model: The neural network
        train_loader: DataLoader with training data
        criterion: Loss function
        optimizer: Weight optimizer
        epoch: Current epoch number
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    
    # Set model to training mode
    # This enables dropout and batch normalization training behavior
    model.train()
    
    # Track statistics
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar for visual feedback
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [TRAIN]')
    
    # ============== TRAINING LOOP ==============
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move data to GPU
        images = images.to(DEVICE)  # Shape: (batch_size, 3, 224, 224)
        labels = labels.to(DEVICE)  # Shape: (batch_size,)
        
        # ============== FORWARD PASS ==============
        # Pass images through model
        outputs = model(images)  # Shape: (batch_size, 2)
        # outputs[i] = [score_fake, score_real] for image i
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # ============== BACKWARD PASS ==============
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Compute gradients (backpropagation)
        # This calculates how to change each weight to reduce loss
        loss.backward()
        
        # ============== UPDATE WEIGHTS ==============
        # Apply gradients to weights
        optimizer.step()
        
        # ============== TRACK STATISTICS ==============
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)  # Get class with highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    # Calculate epoch averages
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """
    VALIDATE ON VALIDATION SET
    
    Validation checks how well the model performs on unseen data.
    
    WHY VALIDATION?
    - Monitor if model is overfitting (memorizing training data)
    - Compare different models/hyperparameters
    - Decide when to stop training (early stopping)
    
    DIFFERENCE FROM TRAINING:
    - No weight updates (evaluation only)
    - No dropout (want consistent predictions)
    - No gradient computation (saves memory and time)
    
    Args:
        model: The neural network
        val_loader: DataLoader with validation data
        criterion: Loss function
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    
    # Set model to evaluation mode
    # This disables dropout and uses batch norm in eval mode
    model.eval()
    
    # Track statistics
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(val_loader, desc='           [VAL]  ')
    
    # ============== VALIDATION LOOP ==============
    # Disable gradient computation (saves memory, faster)
    with torch.no_grad():
        for images, labels in pbar:
            # Move to GPU
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass only (no backward pass!)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    # Calculate averages
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


# ==============================================================================
# CHECKPOINTING
# ==============================================================================

def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, is_best=False):
    """
    SAVE MODEL CHECKPOINT
    
    Saves model weights and training state so you can:
    - Resume training later
    - Use the model for inference
    - Compare different training runs
    
    WHAT GETS SAVED:
    - model_state_dict: Model weights and biases
    - optimizer_state_dict: Optimizer state (for resuming)
    - epoch: Current epoch number
    - val_loss: Validation loss
    - val_acc: Validation accuracy
    
    Args:
        model: The neural network
        optimizer: The optimizer
        epoch: Current epoch
        val_loss: Validation loss
        val_acc: Validation accuracy
        is_best: Whether this is the best model so far
    """
    
    # Create checkpoint directory
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    
    # Save latest checkpoint
    latest_path = CHECKPOINTS_DIR / 'latest_checkpoint.pth'
    torch.save(checkpoint, latest_path)
    
    # Save best model separately
    if is_best:
        # Use MODEL_NAME setting (allows saving with different names)
        best_path = WEIGHTS_DIR / MODEL_NAME
        torch.save(checkpoint, best_path)
        print(f"  ✓ New best model saved as {MODEL_NAME}! (Val Acc: {val_acc:.2f}%)")


# ==============================================================================
# TRAINING HISTORY AND PLOTTING
# ==============================================================================

def save_training_history(history):
    """
    SAVE TRAINING HISTORY TO JSON
    
    Saves loss and accuracy curves for later analysis.
    """
    history_path = WEIGHTS_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ Training history saved to {history_path}")


def plot_training_curves(history):
    """
    PLOT TRAINING AND VALIDATION CURVES
    
    Creates plots showing:
    - Loss curves (training vs validation)
    - Accuracy curves (training vs validation)
    
    These plots help visualize:
    - Is the model learning? (loss should decrease)
    - Is it overfitting? (train loss << val loss)
    - Is it converging? (curves flatten out)
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = WEIGHTS_DIR / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Training curves saved to {plot_path}")
    
    plt.close()


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def main():
    """
    MAIN TRAINING FUNCTION
    
    Orchestrates the entire training process:
    1. Load data
    2. Create model, loss, optimizer
    3. Training loop
    4. Save checkpoints
    5. Plot results
    """
    
    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "DEEPFAKE IMAGE DETECTOR - TRAINING SCRIPT" + " "*17 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Create weights directory
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============== STEP 1: LOAD DATA ==============
    dataloaders = get_dataloaders()
    
    # ============== STEP 2: CREATE MODEL ==============
    model = create_model()
    criterion = create_criterion()
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    
    # ============== STEP 3: TRAINING SETUP ==============
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Epochs:          {NUM_EPOCHS}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  Device:          {DEVICE}")
    print(f"  Early stopping:  {PATIENCE} epochs patience")
    print("="*70)
    print()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # Start time
    start_time = time.time()
    
    # ============== STEP 4: TRAINING LOOP ==============
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*70}\n")
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, dataloaders['val'], criterion
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1} SUMMARY:")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.2f}%")
        
        # Check if best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, is_best)
        
        # Early stopping check
        if epochs_no_improve >= PATIENCE:
            print(f"\n⚠️  Early stopping triggered! No improvement for {PATIENCE} epochs.")
            print(f"  Best validation accuracy: {best_val_acc:.2f}%")
            break
        
        print(f"{'='*70}\n")
    
    # ============== STEP 5: TRAINING COMPLETE ==============
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"  Total time:        {total_time/3600:.2f} hours")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Final epoch:       {epoch+1}")
    print("="*70)
    print()
    
    # Save training history
    print("Saving training history...")
    save_training_history(history)
    
    # Plot training curves
    print("Generating training curves...")
    plot_training_curves(history)
    
    print()
    print("="*70)
    print("✓ All done! Your trained model is ready!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Evaluate on test set: python ml_models/image_detection/evaluate.py")
    print("  2. Integrate with Django API")
    print("  3. Test with real images!")
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
# CONGRATULATIONS! You've learned:
# ✓ How to load and preprocess image datasets
# ✓ How to use transfer learning (ResNet50 + ImageNet)
# ✓ What loss functions and optimizers do
# ✓ How the training loop works (forward, backward, optimize)
# ✓ How to track and visualize training progress
# ✓ How to save and load model checkpoints
# ✓ How to prevent overfitting (validation, early stopping)
#
# Your trained model can now detect AI-generated images from:
# - Stable Diffusion
# - DALL-E
# - Midjourney
#
# Expected performance: 85-95% accuracy on test set!
# ==============================================================================

