# ML Models Directory - Complete Guide

## 📚 Overview

This directory contains all machine learning components for the DMI++ deepfake detection system. Each subdirectory has a specific purpose and contains well-commented code that teaches you ML concepts while building a production system.

---

## 📁 Directory Structure

```
ml_models/
├── README.md                  ← YOU ARE HERE! (Master guide)
│
├── image_detection/           ← IMAGE deepfake detection
│   ├── model.py              ← Neural network architecture (ResNet50)
│   ├── inference.py          ← Prediction service (use this in Django!)
│   ├── gradcam.py            ← Explainability (heatmap generation)
│   └── __init__.py
│
├── preprocessing/             ← Data preparation (BEFORE models)
│   ├── image_preprocessor.py ← Resize, normalize images
│   ├── text_preprocessor.py  ← Tokenize text for RoBERTa
│   ├── video_preprocessor.py ← Extract frames from videos
│   └── __init__.py
│
├── text_detection/            ← TEXT AI detection (RoBERTa)
│   └── __init__.py           ← (To be implemented in Phase 2)
│
├── video_detection/           ← VIDEO deepfake detection
│   └── __init__.py           ← (To be implemented in Phase 2)
│
├── utils/                     ← Helper functions
│   └── __init__.py           ← (Utilities as needed)
│
└── weights/                   ← Trained model files (.pth)
    └── (empty now - will have trained models after training)
```

---

## 🎯 What Each Folder Does

### 1. `image_detection/` - Image Deepfake Detection

**Purpose:** Detect if an image is Real or AI-generated/manipulated.

**Files:**
- **`model.py`** - The neural network architecture
  - DeepfakeImageDetector (ResNet50-based)
  - ConvNeXtDetector (alternative, more powerful)
  - Transfer learning from ImageNet
  
- **`inference.py`** - The prediction service (MOST IMPORTANT!)
  - ImageDetectionService class
  - This is what Django API calls!
  - Complete pipeline: image → prediction + heatmap
  
- **`gradcam.py`** - Visual explanations (explainability)
  - Generates heatmaps showing model's focus
  - Makes AI decisions transparent

**What You Need to Know:**
1. The model is defined in `model.py`
2. Training script will be in `train.py` (to be created)
3. After training, weights are saved to `weights/`
4. `inference.py` loads weights and makes predictions
5. Django views import `ImageDetectionService` from `inference.py`

---

### 2. `preprocessing/` - Data Preparation

**Purpose:** Prepare images, text, and videos BEFORE feeding to models.

**Files:**
- **`image_preprocessor.py`** - Image preparation
  - Resize to 224x224
  - Normalize using ImageNet statistics
  - Convert to PyTorch tensor
  - Face extraction (optional)
  - Noise removal (optional)

- **`text_preprocessor.py`** - Text preparation
  - Clean text (remove noise)
  - Tokenize using RoBERTa tokenizer
  - Pad/truncate to 512 tokens
  - Extract statistical features

- **`video_preprocessor.py`** - Video preparation
  - Extract frames using FFmpeg
  - Sample frames (1 per second)
  - Keyframe detection
  - Generate thumbnails

**Why Preprocessing Matters:**
- Models expect specific input formats
- Images must be 224x224 with normalized pixels
- Text must be tokenized into numbers
- Videos must be split into frames
- Preprocessing ensures consistency!

---

### 3. `text_detection/` - AI-Generated Text Detection

**Purpose:** Detect if text was written by AI (e.g., ChatGPT) or humans.

**Status:** 🚧 To be implemented

**What Will Go Here:**
- `model.py` - RoBERTa-based text classifier
- `inference.py` - Text prediction service
- `train.py` - Training script for text detection

**How It Will Work:**
1. User submits text through web interface
2. Text is tokenized using `TextPreprocessor`
3. RoBERTa model analyzes linguistic patterns
4. Returns: AI-generated or Human-written + confidence

---

### 4. `video_detection/` - Video Deepfake Detection

**Purpose:** Detect deepfake videos (face swaps, lip sync, etc.).

**Status:** 🚧 To be implemented

**What Will Go Here:**
- `video_pipeline.py` - Complete video analysis pipeline
- Frame extraction → Image detection → Temporal analysis → Aggregation

**How It Will Work:**
1. User uploads video
2. Extract frames (1 per second) using `VideoPreprocessor`
3. Analyze each frame with `ImageDetectionService`
4. Check temporal consistency (do faces change unrealistically?)
5. Aggregate results → Final verdict

---

### 5. `utils/` - Helper Functions

**Purpose:** Common utilities used across modules.

**Status:** Empty for now, will add as needed

**What Might Go Here:**
- Metadata extraction helpers
- Report generation (PDF)
- Performance metrics
- Logging utilities

---

### 6. `weights/` - Trained Model Files

**Purpose:** Store trained model weights (.pth files).

**Status:** Empty (you haven't trained models yet!)

**What Will Go Here After Training:**
```
weights/
├── image_detector_resnet50.pth     ← Trained image model
├── text_detector_roberta.pth       ← Trained text model
└── best_model_checkpoint.pth       ← Best model during training
```

**File Size:** Each .pth file is ~100MB (depends on model)

---

## 🚀 How Everything Fits Together

### Image Detection Workflow:

```
1. User uploads image via web interface
   ↓
2. Django view receives image file
   ↓
3. Django calls:
   from ml_models.image_detection.inference import ImageDetectionService
   service = ImageDetectionService(model_path='weights/trained_model.pth')
   result = service.predict_with_heatmap(image_path, output_dir)
   ↓
4. Inside predict_with_heatmap():
   a) ImagePreprocessor loads and preprocesses image
   b) Model predicts Real/Fake
   c) Grad-CAM generates heatmap
   d) Explanation is generated
   ↓
5. Result returned to Django:
   {
     'verdict': 'fake',
     'confidence': 0.95,
     'probabilities': {'real': 0.05, 'fake': 0.95},
     'heatmap_path': '/media/heatmaps/image_heatmap.jpg',
     'explanation': 'The image is highly likely to be FAKE...'
   }
   ↓
6. Django saves to database (AnalysisResult model)
   ↓
7. Frontend displays results to user with heatmap visualization
```

---

## 📖 Learning Path (How to Read the Code)

**If you want to understand image detection, read in this order:**

1. **Start Here:** `preprocessing/image_preprocessor.py`
   - Understand how images are prepared
   - Learn about normalization and tensors

2. **Then:** `image_detection/model.py`
   - See how ResNet50 is structured
   - Understand transfer learning
   - Learn about forward pass

3. **Next:** `image_detection/gradcam.py`
   - Learn how explainability works
   - Understand gradient-based methods

4. **Finally:** `image_detection/inference.py`
   - See how everything connects
   - This is what Django uses!

**Each file has extensive comments explaining:**
- What the code does
- Why it's needed
- How it works
- When to use it
- Examples

---

## 🛠️ Next Steps (What You'll Build)

### Phase 2A: Image Detection Training
1. Create `image_detection/train.py` - Training script
2. Prepare dataset (Real vs Fake images)
3. Train the model on your GPU
4. Save trained weights to `weights/`
5. Test inference with trained model

### Phase 2B: Text Detection
1. Create `text_detection/model.py` - RoBERTa classifier
2. Create `text_detection/train.py` - Training script
3. Create `text_detection/inference.py` - Prediction service
4. Train and integrate with Django

### Phase 2C: Video Detection
1. Create `video_detection/video_pipeline.py`
2. Implement frame-level analysis
3. Add temporal consistency checks
4. Aggregate results

---

## 💡 Key Concepts You'll Learn

### Machine Learning Concepts:
- **Transfer Learning** - Using pre-trained models (ResNet50)
- **Fine-tuning** - Adapting models to new tasks
- **Forward Pass** - How data flows through networks
- **Backward Pass** - How gradients update weights
- **Inference** - Using trained models for predictions

### Deep Learning Techniques:
- **Convolutional Neural Networks (CNNs)** - For images
- **Transformers (RoBERTa)** - For text
- **Grad-CAM** - For explainability
- **Batch Normalization** - For stable training
- **Dropout** - For preventing overfitting

### Practical Skills:
- **PyTorch** - Deep learning framework
- **Model Training** - Loss functions, optimizers
- **Model Evaluation** - Accuracy, precision, recall
- **Deployment** - Integrating ML with Django

---

## 🎓 Resources for Learning

### Recommended Reading:
1. **ResNet Paper:** "Deep Residual Learning for Image Recognition"
2. **Grad-CAM Paper:** "Grad-CAM: Visual Explanations from Deep Networks"
3. **PyTorch Tutorials:** https://pytorch.org/tutorials/

### Online Courses:
1. **PyTorch Basics:** FastAI Practical Deep Learning
2. **CNNs:** Stanford CS231n (YouTube)
3. **Transfer Learning:** Andrew Ng's Deep Learning Specialization

---

## 🤝 How to Use This in Your FYP

### For Your FYP Report:
1. **Architecture Diagrams** - Draw how data flows
2. **Methodology** - Explain ResNet50, transfer learning, Grad-CAM
3. **Implementation** - Show code snippets with explanations
4. **Results** - Accuracy, confusion matrices, example predictions

### For Your Presentation:
1. Show live demo (upload image → get prediction + heatmap)
2. Explain what Grad-CAM shows (trust building)
3. Discuss challenges (dataset quality, training time)
4. Future improvements (ensemble models, better datasets)

---

## ⚠️ Important Notes

1. **All code has detailed comments** - Read the code files to learn!
2. **No trained models yet** - You need to train them first
3. **GPU recommended** - Training will take hours/days on CPU
4. **Test thoroughly** - Validate models before deployment

---

## 🎯 Current Status

### ✅ Completed:
- [x] Image detection architecture (ResNet50)
- [x] Image preprocessing pipeline
- [x] Grad-CAM explainability
- [x] Inference service
- [x] Text preprocessing
- [x] Video preprocessing
- [x] Comprehensive documentation

### 🚧 Next Steps:
- [ ] Create training script
- [ ] Prepare/download dataset
- [ ] Train image detection model
- [ ] Integrate with Django API
- [ ] Test end-to-end pipeline

---

## 📞 Quick Reference

**To make a prediction:**
```python
from ml_models.image_detection.inference import ImageDetectionService

service = ImageDetectionService(model_path='weights/trained_model.pth')
result = service.predict_with_heatmap('image.jpg', 'output/')
print(result['verdict'], result['confidence'])
```

**To preprocess an image:**
```python
from ml_models.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor()
tensor = preprocessor.preprocess('image.jpg')
```

**To generate heatmap:**
```python
from ml_models.image_detection.gradcam import GradCAM

gradcam = GradCAM(model)
cam = gradcam.generate_cam(input_tensor, target_class=1)
gradcam.save_heatmap(image, cam, 'heatmap.jpg')
```

---

## 🎉 You're Ready!

Every file in this directory is heavily commented for learning. Start reading the code files in the suggested order above. Each file teaches you concepts while showing practical implementation.

**Happy Learning! 🚀**

---

*Last Updated: October 2025*
*Part of: DMI++ Final Year Project*
*Author: [Your Name]*

