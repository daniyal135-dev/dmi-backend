# Step 6: Fine-tune RoBERTa-base for Text Detection - COMPLETE GUIDE

## 📋 Table of Contents
1. [Overview - What We Did](#overview)
2. [Files Created - Detailed Explanation](#files-created)
3. [Files Modified - Detailed Explanation](#files-modified)
4. [How Everything Works Together](#how-everything-works)
5. [Complete Workflow](#complete-workflow)

---

## 🎯 Overview - What We Did

**Step 6 Goal:** Create a complete text detection system that can identify AI-generated text (like ChatGPT, GPT-4, Claude) vs human-written text.

**What We Built:**
1. **Model Architecture** - RoBERTa-based neural network
2. **Training Script** - To fine-tune the model on your dataset
3. **Inference Service** - To make predictions (used by Django API)
4. **Django API Integration** - Endpoint to analyze text from frontend

**Why RoBERTa?**
- Pre-trained on 160GB of text data
- Understands language context and meaning
- Perfect for text classification tasks
- Transfer learning = faster training, better accuracy

---

## 📁 Files Created - Detailed Explanation

### 1. `model.py` - The Neural Network Architecture

**Location:** `dmi-backend/ml_models/text_detection/model.py`

**Purpose:**
This file defines the structure of our AI text detection model. It's like a blueprint for the neural network.

**What's Inside:**

#### **Class: `AIGeneratedTextDetector`**
This is the main model class. Think of it as a factory that creates the neural network.

**Key Components:**

1. **`__init__()` Method** - Initializes the Model
   - **What it does:**
     - Loads RoBERTa-base (pre-trained transformer model)
     - Adds a custom classification head (768 features → 2 classes)
     - Sets up dropout for regularization
   - **Why:**
     - RoBERTa knows language already (transfer learning)
     - We just need to teach it "Human vs AI" classification
     - Dropout prevents overfitting (model memorizing training data)

2. **`forward()` Method** - Makes Predictions
   - **What it does:**
     - Takes tokenized text (input_ids, attention_mask)
     - Passes through RoBERTa layers
     - Extracts [CLS] token (represents entire sentence)
     - Passes through classifier → outputs 2 logits (Human score, AI score)
   - **Why:**
     - This is the core prediction function
     - Called automatically when you do `model(input_ids, attention_mask)`
     - Returns raw scores (logits) that get converted to probabilities

3. **`get_attention_weights()` Method** - For Explainability
   - **What it does:**
     - Returns attention weights from all 12 RoBERTa layers
     - Shows which words the model focuses on
   - **Why:**
     - Helps explain WHY model thinks text is AI-generated
     - Can highlight suspicious words/phrases
     - Useful for generating explanations

4. **`save_model()` Method** - Saves Trained Weights
   - **What it does:**
     - Saves model state to `.pth` file
     - Includes model configuration
   - **Why:**
     - After training, save weights so we can use them later
     - Don't need to retrain every time

5. **`load_model()` Class Method** - Loads Saved Weights
   - **What it does:**
     - Loads previously saved model from `.pth` file
     - Restores model to exact trained state
   - **Why:**
     - Load trained model for inference (making predictions)
     - Don't need to retrain every time

**How It Works:**
```
Input Text → Tokenize → RoBERTa (12 layers) → [CLS] token → Classifier → Output (Human/AI)
```

**Example Usage:**
```python
# Create model
model = AIGeneratedTextDetector(num_classes=2)

# Make prediction
logits = model(input_ids, attention_mask)
# logits = [[-1.5, 2.1]]  # Human score: -1.5, AI score: 2.1
# AI score is higher → text is likely AI-generated
```

---

### 2. `train.py` - The Training Script

**Location:** `dmi-backend/ml_models/text_detection/train.py`

**Purpose:**
This script trains (fine-tunes) the RoBERTa model on your dataset. It's what you run to teach the model to detect AI-generated text.

**What's Inside:**

#### **Class: `TextDataset`**
- **Purpose:** PyTorch Dataset class for loading text data
- **What it does:**
  - Loads text and labels from CSV
  - Tokenizes text when requested
  - Returns batches of data for training
- **Why:**
  - PyTorch needs a Dataset class to load data efficiently
  - Handles tokenization automatically
  - Supports batching (process multiple texts at once)

#### **Function: `train_epoch()`**
- **Purpose:** Trains the model for one epoch (one pass through all training data)
- **What it does:**
  1. Sets model to training mode
  2. Loops through all batches
  3. For each batch:
     - Forward pass (get predictions)
     - Calculate loss (how wrong the predictions are)
     - Backward pass (update model weights)
     - Track accuracy
  4. Returns average loss and accuracy
- **Why:**
  - This is the core training loop
  - Updates model weights to improve accuracy
  - Tracks progress

#### **Function: `validate()`**
- **Purpose:** Tests model on validation data (data not seen during training)
- **What it does:**
  1. Sets model to evaluation mode (no weight updates)
  2. Loops through validation batches
  3. Makes predictions
  4. Calculates metrics: accuracy, precision, recall, F1-score
  5. Returns metrics
- **Why:**
  - Checks if model is learning (not just memorizing)
  - Prevents overfitting
  - Helps decide when to stop training

#### **Function: `main()`**
- **Purpose:** Main training function - orchestrates everything
- **What it does step-by-step:**

  1. **Parse Arguments**
     - Gets dataset path, epochs, batch size, etc. from command line
     - Sets up device (CPU or GPU)

  2. **Load Dataset**
     - Reads CSV file with text and labels
     - Checks for required columns
     - Shows label distribution

  3. **Split Dataset**
     - Splits into train/validation sets (e.g., 80% train, 20% validation)
     - Uses stratified split (keeps label balance)

  4. **Initialize Preprocessor**
     - Creates TextPreprocessor (handles tokenization)

  5. **Create DataLoaders**
     - Wraps datasets in DataLoaders (handles batching)
     - Shuffles training data

  6. **Initialize Model**
     - Creates AIGeneratedTextDetector
     - Moves to GPU if available
     - Counts parameters

  7. **Setup Training Components**
     - Loss function: CrossEntropyLoss (for classification)
     - Optimizer: AdamW (updates model weights)
     - Scheduler: Reduces learning rate if validation loss stops improving

  8. **Training Loop**
     - For each epoch:
       - Train on training data
       - Validate on validation data
       - Save best model (if validation accuracy improved)
       - Early stopping (if no improvement for N epochs)
     - Saves training history

  9. **Final Summary**
     - Prints best validation accuracy
     - Shows where model was saved

**Key Features:**
- **Early Stopping:** Stops training if validation accuracy doesn't improve
- **Learning Rate Scheduling:** Reduces learning rate automatically
- **Best Model Saving:** Saves model with highest validation accuracy
- **Training History:** Saves all metrics to JSON file

**Command Line Usage:**
```bash
python -m ml_models.text_detection.train \
    --dataset_path data/text_dataset.csv \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --val_split 0.2 \
    --patience 5
```

**What Each Parameter Does:**
- `--dataset_path`: Where your CSV dataset is located
- `--epochs`: How many times to go through all training data
- `--batch_size`: How many texts to process at once (larger = faster but needs more memory)
- `--learning_rate`: How fast model learns (2e-5 is recommended for RoBERTa)
- `--val_split`: Percentage of data to use for validation (0.2 = 20%)
- `--patience`: How many epochs to wait before early stopping

**Output Files:**
- `best_text_model.pth`: Trained model weights (saved in output_dir)
- `training_history.json`: All training metrics (loss, accuracy per epoch)

---

### 3. `inference.py` - The Inference Service

**Location:** `dmi-backend/ml_models/text_detection/inference.py`

**Purpose:**
This is the service class that Django API uses to make predictions. It's the bridge between your trained model and the web application.

**What's Inside:**

#### **Class: `TextDetectionService`**
This is the main service class. It's similar to `ImageDetectionService` but for text.

**Key Components:**

1. **`__init__()` Method** - Initializes Service
   - **What it does:**
     - Detects device (CPU or GPU)
     - Creates TextPreprocessor (for tokenization)
     - Loads model (trained or base RoBERTa)
     - Sets model to evaluation mode
   - **Why:**
     - Sets up everything needed for predictions
     - Only loads model once (singleton pattern)
     - Uses GPU if available (faster)

2. **`predict()` Method** - Basic Prediction
   - **What it does:**
     1. Tokenizes input text
     2. Moves to device (GPU/CPU)
     3. Runs model forward pass
     4. Converts logits to probabilities (softmax)
     5. Gets predicted class and confidence
     6. Returns result dictionary
   - **Returns:**
     ```python
     {
         'verdict': 'ai-generated',  # or 'human'
         'confidence': 0.95,          # 0.0 to 1.0
         'probabilities': {
             'human': 0.05,
             'ai-generated': 0.95
         },
         'predicted_class': 1  # 0=human, 1=ai-generated
     }
     ```
   - **Why:**
     - Core prediction function
     - Used by other methods
     - Returns structured results

3. **`predict_with_explanation()` Method** - Prediction + Explanation
   - **What it does:**
     1. Calls `predict()` to get basic results
     2. Generates human-readable explanation
     3. Returns complete results with explanation
   - **Returns:**
     ```python
     {
         'verdict': 'ai-generated',
         'confidence': 0.95,
         'probabilities': {...},
         'explanation': 'The text is highly likely to be AI-GENERATED (95% confidence)...'
     }
     ```
   - **Why:**
     - This is what Django API uses
     - Provides user-friendly explanations
     - Makes results understandable

4. **`predict_batch()` Method** - Batch Processing
   - **What it does:**
     - Processes multiple texts at once
     - Returns list of results (one per text)
   - **Why:**
     - More efficient than processing one-by-one
     - Useful for testing on datasets
     - Faster for bulk analysis

5. **`generate_explanation()` Method** - Creates Explanations
   - **What it does:**
     - Takes prediction results
     - Generates human-readable explanation
     - Includes confidence level, probabilities, reasoning
   - **Why:**
     - Makes AI decisions transparent
     - Helps users understand results
     - Builds trust in the system

6. **`get_attention_weights()` Method** - For Explainability
   - **What it does:**
     - Gets attention weights from model
     - Shows which words model focuses on
   - **Why:**
     - Can highlight suspicious words
     - Useful for detailed explanations
     - Helps debug model decisions

**How It Works:**
```
User Text → TextDetectionService.predict_with_explanation() 
         → Tokenize 
         → Model Prediction 
         → Generate Explanation 
         → Return Results
```

**Example Usage:**
```python
# Initialize service
service = TextDetectionService(model_path='weights/best_text_model.pth')

# Make prediction
result = service.predict_with_explanation("This is AI-generated text.")
print(result['verdict'])        # 'ai-generated'
print(result['confidence'])     # 0.95
print(result['explanation'])    # 'The text is highly likely...'
```

---

## 🔧 Files Modified - Detailed Explanation

### 1. `dmi-backend/analysis/views.py` - Django API Views

**What Was Changed:**

#### **Change 1: Added Import**
```python
# BEFORE:
from ml_models.image_detection.inference import ImageDetectionService

# AFTER:
from ml_models.image_detection.inference import ImageDetectionService
from ml_models.text_detection.inference import TextDetectionService
```
**Purpose:** Import the new TextDetectionService so we can use it in the API.

#### **Change 2: Implemented `analyze_text()` Function**
**BEFORE:**
```python
@api_view(['POST'])
def analyze_text(request):
    # TODO: Implement text analysis logic
    return Response({'message': 'Text analysis endpoint - to be implemented'}, status=status.HTTP_200_OK)
```

**AFTER:** Complete implementation (see full code in views.py)

**What the Function Does (Step-by-Step):**

1. **Validate Text Input**
   - Checks if text is provided
   - Validates length (min 10 chars, max 10,000 chars)
   - Returns error if invalid
   - **Why:** Prevents invalid requests, protects system

2. **Load Text Detection Service**
   - Gets model path from settings (`TEXT_MODEL_NAME`)
   - Creates TextDetectionService instance
   - If model doesn't exist, uses base RoBERTa (with warning)
   - **Why:** Reuses service instance (efficient), handles missing model gracefully

3. **Analyze Text**
   - Calls `service.predict_with_explanation(text_content)`
   - Gets verdict, confidence, probabilities, explanation
   - **Why:** This is the core analysis step

4. **Determine Verdict**
   - If confidence < 0.6, sets verdict to 'uncertain'
   - **Why:** Low confidence predictions should be marked uncertain

5. **Save to Database**
   - Creates AnalysisResult object
   - Stores: user, file_type='text', verdict, confidence, metadata, explanation
   - **Why:** Persists results for user to view later

6. **Return Response**
   - Serializes result
   - Returns JSON response with result and details
   - **Why:** Sends results back to frontend

**API Endpoint Details:**
- **URL:** `POST /api/analysis/text/`
- **Authentication:** Required (IsAuthenticated)
- **Request Body:**
  ```json
  {
    "text": "The text content to analyze..."
  }
  ```
- **Response:**
  ```json
  {
    "message": "Text analyzed successfully",
    "result": {
      "id": 123,
      "verdict": "ai-generated",
      "confidence": 0.95,
      "explanation": "...",
      "metadata": {...}
    },
    "details": {...}
  }
  ```

**Error Handling:**
- Missing text → 400 Bad Request
- Text too short/long → 400 Bad Request
- Model not found → 200 OK with warning (uses base RoBERTa)
- Other errors → 500 Internal Server Error

---

### 2. `dmi-backend/dmi_project/settings.py` - Django Settings

**What Was Changed:**

#### **Added: `TEXT_MODEL_NAME` Setting**
```python
# BEFORE:
MODEL_NAME = config('MODEL_NAME', default='best_model.pth')

# AFTER:
MODEL_NAME = config('MODEL_NAME', default='best_model.pth')
TEXT_MODEL_NAME = config('TEXT_MODEL_NAME', default='best_text_model.pth')
```

**Purpose:**
- Stores the filename of the trained text model
- Allows changing model without code changes
- Default: `best_text_model.pth` (in `ml_models/weights/`)

**Why:**
- Separates image model and text model settings
- Easy to switch models (just change setting)
- Follows same pattern as `MODEL_NAME` for images

**How It's Used:**
```python
# In views.py
text_model_name = getattr(settings, 'TEXT_MODEL_NAME', 'best_text_model.pth')
model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'weights', text_model_name)
```

---

## 🔄 How Everything Works Together

### Complete Flow Diagram:

```
1. USER SENDS TEXT
   ↓
2. FRONTEND → POST /api/analysis/text/ { "text": "..." }
   ↓
3. DJANGO views.py → analyze_text() function
   ↓
4. Validates text input
   ↓
5. Loads TextDetectionService (from inference.py)
   ↓
6. TextDetectionService.predict_with_explanation()
   ↓
7. TextPreprocessor.tokenize() → Converts text to tokens
   ↓
8. AIGeneratedTextDetector.forward() → Model prediction
   ↓
9. Converts logits to probabilities
   ↓
10. Generates explanation
   ↓
11. Saves to database (AnalysisResult)
   ↓
12. Returns JSON response to frontend
   ↓
13. FRONTEND displays results to user
```

### File Dependencies:

```
train.py
  ↓ (trains model)
  ↓ (saves to)
best_text_model.pth
  ↓ (loaded by)
inference.py (TextDetectionService)
  ↓ (used by)
views.py (analyze_text endpoint)
  ↓ (called by)
Frontend API call
```

### Data Flow:

```
CSV Dataset
  ↓
train.py (TextDataset class)
  ↓
Model Training (AIGeneratedTextDetector)
  ↓
Save Weights (best_text_model.pth)
  ↓
Load Weights (TextDetectionService)
  ↓
User Text Input
  ↓
Tokenization (TextPreprocessor)
  ↓
Model Prediction (AIGeneratedTextDetector)
  ↓
Results (verdict, confidence, explanation)
  ↓
Database (AnalysisResult)
  ↓
Frontend Display
```

---

## 📊 Complete Workflow

### Phase 1: Setup (Already Done ✅)
1. ✅ Created `model.py` - Model architecture
2. ✅ Created `train.py` - Training script
3. ✅ Created `inference.py` - Inference service
4. ✅ Modified `views.py` - Added API endpoint
5. ✅ Modified `settings.py` - Added TEXT_MODEL_NAME

### Phase 2: Training (Next Step)
1. **Prepare Dataset**
   - Get CSV file with `text` and `label` columns
   - Label: 0 = Human, 1 = AI-generated
   - Example sources: Kaggle, Hugging Face, custom collection

2. **Run Training**
   ```bash
   cd dmi-backend
   python -m ml_models.text_detection.train \
       --dataset_path data/text_dataset.csv \
       --epochs 10 \
       --batch_size 16 \
       --output_dir ml_models/weights
   ```

3. **Check Results**
   - Model saved to: `ml_models/weights/best_text_model.pth`
   - Training history: `ml_models/weights/training_history.json`
   - Check validation accuracy (should be > 80% for good model)

### Phase 3: Testing (After Training)
1. **Start Django Server**
   ```bash
   python manage.py runserver
   ```

2. **Test API Endpoint**
   ```bash
   # Using curl or Postman
   POST http://localhost:8000/api/analysis/text/
   Headers: Authorization: Bearer <token>
   Body: { "text": "This is a test text to analyze." }
   ```

3. **Check Response**
   - Should return verdict, confidence, explanation
   - Check database for saved result

### Phase 4: Integration (Frontend)
1. Frontend calls `/api/analysis/text/` endpoint
2. Displays results (verdict, confidence, explanation)
3. User can view analysis history

---

## 🎓 Key Concepts Explained

### Transfer Learning
- **What:** Using a pre-trained model (RoBERTa) and fine-tuning it for our task
- **Why:** Faster training, better accuracy, less data needed
- **How:** RoBERTa already knows language, we just teach it "Human vs AI"

### Tokenization
- **What:** Converting text to numbers (token IDs)
- **Why:** Models work with numbers, not text
- **How:** Each word/subword gets a unique ID from RoBERTa vocabulary

### Fine-tuning
- **What:** Training pre-trained model on new task
- **Why:** Adapts model to our specific problem (AI text detection)
- **How:** Update model weights using our dataset

### Early Stopping
- **What:** Stop training if validation accuracy stops improving
- **Why:** Prevents overfitting (model memorizing training data)
- **How:** Track validation accuracy, stop after N epochs without improvement

### Softmax
- **What:** Converts raw scores (logits) to probabilities
- **Why:** Makes outputs interpretable (0.0 to 1.0, sum to 1.0)
- **How:** `prob = exp(logit) / sum(exp(all_logits))`

---

## 📝 Summary

**What We Created:**
1. **model.py** - Neural network architecture (AIGeneratedTextDetector)
2. **train.py** - Training script to fine-tune model
3. **inference.py** - Service for making predictions (TextDetectionService)
4. **views.py** - Django API endpoint (analyze_text function)
5. **settings.py** - Configuration (TEXT_MODEL_NAME)

**What Each Does:**
- **model.py**: Defines the model structure
- **train.py**: Trains the model on your dataset
- **inference.py**: Makes predictions (used by API)
- **views.py**: Handles HTTP requests, calls inference service
- **settings.py**: Stores configuration

**How They Connect:**
- Training → Saves model → Inference loads model → API uses inference → Frontend calls API

**Next Steps:**
1. Find/prepare dataset (CSV with text and labels)
2. Run training script
3. Test API endpoint
4. Integrate with frontend

---

**Everything is now explained in detail!** 🎉

