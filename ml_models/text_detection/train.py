"""
==============================================================================
TEXT DETECTION TRAINING SCRIPT
==============================================================================

WHAT THIS FILE DOES:
Trains the encoder-based classifier (default: DeBERTa-v3-base) for AI-generated text detection.

TRAINING WORKFLOW:
1. Load dataset (CSV with text and labels: 0=Human, 1=AI-generated)
2. Split into train/validation sets
3. Create data loaders (batches)
4. Initialize model (default microsoft/deberta-v3-base)
5. Train for multiple epochs
6. Save best model based on validation accuracy
7. Early stopping if validation doesn't improve

DATASET FORMAT:
CSV file with columns:
- text: The text content
- label: 0 (Human) or 1 (AI-generated)

Example:
text,label
"This is a human-written article about deepfakes.",0
"Artificial intelligence has revolutionized many industries.",1

USAGE:
    python train.py --dataset_path data/text_dataset.csv --epochs 10 --batch_size 16
==============================================================================
"""

# ============== IMPORTS ==============
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# Import our modules
from .model import AIGeneratedTextDetector, DEFAULT_TEXT_ENCODER
from ..preprocessing.text_preprocessor import TextPreprocessor


# ============== DATASET CLASS ==============
class TextDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    
    Loads text and labels from CSV file.
    """
    def __init__(self, texts, labels, preprocessor):
        """
        Initialize dataset.
        
        Args:
            texts (list): List of text strings
            labels (list): List of labels (0=Human, 1=AI-generated)
            preprocessor (TextPreprocessor): Preprocessor for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get one sample from dataset.
        
        Returns:
            dict: {
                'input_ids': tensor,
                'attention_mask': tensor,
                'label': tensor
            }
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoded = self.preprocessor.tokenize(text)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============== TRAINING FUNCTION ==============
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        float: Average loss for the epoch
        float: Accuracy for the epoch
    """
    model.train()  # Set to training mode
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy


# ============== VALIDATION FUNCTION ==============
def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Returns:
        float: Average loss
        float: Accuracy
        dict: Additional metrics (precision, recall, F1)
    """
    model.eval()  # Set to evaluation mode
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return avg_loss, accuracy, metrics


# ============== MAIN TRAINING SCRIPT ==============
def main():
    """
    Main training function.
    """
    # ============== ARGUMENT PARSING ==============
    parser = argparse.ArgumentParser(description='Train text encoder for AI-generated text detection')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to CSV dataset file')
    parser.add_argument('--output_dir', type=str, default='weights',
                        help='Directory to save model weights')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--model_name', type=str, default=DEFAULT_TEXT_ENCODER,
                        help='Hugging Face encoder id (must match tokenizer)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate (often 2e-5 or 1e-5 for encoder fine-tuning)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (0.0 to 1.0)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # ============== DEVICE SETUP ==============
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"✓ Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    
    # ============== LOAD DATASET ==============
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    
    df = pd.read_csv(args.dataset_path)
    print(f"✓ Loaded dataset: {len(df)} samples")
    
    # Check required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns")
    
    # Display label distribution
    label_counts = df['label'].value_counts()
    print(f"  Label distribution:")
    print(f"    Human (0): {label_counts.get(0, 0)}")
    print(f"    AI-generated (1): {label_counts.get(1, 0)}")
    
    # ============== SPLIT DATASET ==============
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    if args.val_split > 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=args.val_split, random_state=42, stratify=labels
        )
        print(f"✓ Split dataset: {len(train_texts)} train, {len(val_texts)} validation")
    else:
        train_texts, train_labels = texts, labels
        val_texts, val_labels = [], []
        print(f"⚠ No validation split (val_split=0). Training on all data.")
    
    # ============== INITIALIZE PREPROCESSOR ==============
    print("\n" + "="*60)
    print("INITIALIZING PREPROCESSOR")
    print("="*60)
    preprocessor = TextPreprocessor(model_name=args.model_name, max_length=512)
    
    # ============== CREATE DATALOADERS ==============
    print("\n" + "="*60)
    print("CREATING DATALOADERS")
    print("="*60)
    
    train_dataset = TextDataset(train_texts, train_labels, preprocessor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    print(f"✓ Train dataloader: {len(train_loader)} batches")
    
    if len(val_texts) > 0:
        val_dataset = TextDataset(val_texts, val_labels, preprocessor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"✓ Validation dataloader: {len(val_loader)} batches")
    else:
        val_loader = None
    
    # ============== INITIALIZE MODEL ==============
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    model = AIGeneratedTextDetector(num_classes=2, model_name=args.model_name)
    model.to(device)
    print(f"✓ Model initialized on {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ============== LOSS AND OPTIMIZER ==============
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # ============== TRAINING LOOP ==============
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Early stopping patience: {args.patience}")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_metrics': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, 'deberta-v3-bestmodel.pth')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_metrics'].append(val_metrics)
            
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model.save_model(best_model_path)
                print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
                
                # Early stopping
                if patience_counter >= args.patience:
                    print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                    print(f"  Best validation accuracy: {best_val_acc:.4f}")
                    break
    
    # ============== SAVE TRAINING HISTORY ==============
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to: {history_path}")
    
    # ============== FINAL SUMMARY ==============
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print("="*60)


if __name__ == '__main__':
    main()

