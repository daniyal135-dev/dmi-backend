# Text Detection Module

## Overview

This module implements AI-generated text detection using RoBERTa-base transformer model. It can distinguish between human-written and AI-generated text (e.g., from ChatGPT, GPT-4, Claude, etc.).

## Files

### 1. `model.py` - RoBERTa Model Architecture
- **AIGeneratedTextDetector**: Main model class
- Uses RoBERTa-base (125M parameters) as backbone
- Custom classification head: 768 → 2 (Human, AI-generated)
- Supports attention weights for explainability

### 2. `train.py` - Training Script
- Fine-tunes RoBERTa on AI-generated text dataset
- Supports CSV dataset format (text, label columns)
- Includes validation split, early stopping, learning rate scheduling
- Saves best model to `weights/best_text_model.pth`

### 3. `inference.py` - Inference Service
- **TextDetectionService**: Main service class for Django API
- Methods:
  - `predict()`: Basic prediction
  - `predict_with_explanation()`: Prediction + human-readable explanation
  - `predict_batch()`: Process multiple texts
  - `get_attention_weights()`: Get attention for explainability

## Dataset Format

CSV file with two columns:
```csv
text,label
"This is a human-written article about deepfakes.",0
"Artificial intelligence has revolutionized many industries.",1
```

- **text**: The text content to analyze
- **label**: 0 = Human-written, 1 = AI-generated

## Training

```bash
# From dmi-backend directory
python -m ml_models.text_detection.train \
    --dataset_path data/text_dataset.csv \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --val_split 0.2 \
    --patience 5 \
    --output_dir ml_models/weights
```

**Parameters:**
- `--dataset_path`: Path to CSV dataset file
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5, recommended for RoBERTa)
- `--val_split`: Validation split ratio (default: 0.2 = 20%)
- `--patience`: Early stopping patience (default: 5 epochs)
- `--output_dir`: Directory to save model weights (default: weights)

**Output:**
- `best_text_model.pth`: Best model weights (saved in output_dir)
- `training_history.json`: Training metrics history

## Usage in Django

The text detection service is integrated with Django API:

**Endpoint:** `POST /api/analysis/text/`

**Request:**
```json
{
  "text": "The text content to analyze..."
}
```

**Response:**
```json
{
  "message": "Text analyzed successfully",
  "result": {
    "id": 123,
    "verdict": "ai-generated",
    "confidence": 0.95,
    "explanation": "The text is highly likely to be AI-GENERATED...",
    "metadata": {
      "text_content": "...",
      "text_length": 150,
      "probabilities": {
        "human": 0.05,
        "ai-generated": 0.95
      }
    }
  }
}
```

## Configuration

Add to `dmi_project/settings.py`:
```python
TEXT_MODEL_NAME = 'best_text_model.pth'  # Model filename in ml_models/weights/
```

## Model Location

Trained models should be saved in:
```
dmi-backend/ml_models/weights/best_text_model.pth
```

If the model doesn't exist, the service will use base RoBERTa (not fine-tuned), which will still work but with lower accuracy.

## Next Steps

1. **Find/Prepare Dataset**: Get a dataset with human-written and AI-generated text pairs
2. **Train Model**: Run training script with your dataset
3. **Test**: Use the Django API endpoint to test predictions
4. **Evaluate**: Test on unseen data to measure accuracy

## Recommended Datasets

- **GPT-2 Output Dataset**: Human-written vs GPT-2 generated text
- **ChatGPT Output Dataset**: Human vs ChatGPT responses
- **Custom Dataset**: Collect your own human/AI text pairs

## Notes

- Text is tokenized to max 512 tokens (RoBERTa's limit)
- Longer texts are truncated to first 512 tokens
- Model uses transfer learning from RoBERTa-base (pre-trained on 160GB of text)
- Fine-tuning adapts the model specifically for AI text detection

