# 🚀 Image Model Training Guide

## ✅ What You Have

You've successfully:
1. ✅ Downloaded modern AI dataset (60k images: Stable Diffusion, DALL-E, Midjourney)
2. ✅ Extracted to `dmi-backend/dataset/`
3. ✅ Have 3 training scripts ready with detailed educational comments

---

## 📂 Current Dataset Structure

```
dmi-backend/dataset/
├── train/
│   ├── fake/  (24,000 AI-generated images)
│   └── real/  (24,000 real images)
└── test/
    ├── fake/  (6,000 images)
    └── real/  (6,000 images)
```

---

## 🎯 Training Steps (Follow in Order!)

### **Step 1: Activate Virtual Environment**

Open PowerShell in `dmi-backend/` folder:

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### **Step 2: Install Required Packages**

```bash
# Make sure you have these packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm matplotlib seaborn scikit-learn
```

**Note:** This installs PyTorch with CUDA support for your Quadro T2000 GPU!

---

### **Step 3: Prepare Dataset (Split Train/Val)**

```bash
python ml_models/image_detection/prepare_dataset.py
```

**What this does:**
- Creates `dataset/val/` folder
- Moves 10% of training images to validation
- Leaves 90% in training

**Expected output:**
```
TRAINING SET (90%): 43,200 images
VALIDATION SET (10%): 4,800 images
TEST SET: 12,000 images
```

**Time:** 2-5 minutes

---

### **Step 4: Train the Model** 🏋️

```bash
python ml_models/image_detection/train.py
```

**What happens:**
- Loads datasets (train/val)
- Creates ResNet50 model
- Trains for up to 50 epochs
- Saves checkpoints
- Generates training curves

**Expected time:** 4-6 hours on Quadro T2000

**What you'll see:**
```
Epoch 1/50 [TRAIN]: 100%|████████| loss: 0.5234, acc: 75.23%
           [VAL]  : 100%|████████| loss: 0.4512, acc: 78.91%
✓ New best model saved! (Val Acc: 78.91%)
...
```

**Output files:**
- `ml_models/weights/best_model.pth` (best model)
- `ml_models/weights/checkpoints/latest_checkpoint.pth` (latest)
- `ml_models/weights/training_curves.png` (loss/accuracy plots)
- `ml_models/weights/training_history.json` (metrics)

---

### **Step 5: Evaluate the Model** 📊

```bash
python ml_models/image_detection/evaluate.py
```

**What this does:**
- Loads best trained model
- Tests on test set (12,000 images)
- Calculates accuracy, precision, recall, F1-score
- Generates confusion matrix

**Expected output:**
```
TEST SET EVALUATION RESULTS
==================================================================
Classification Report:
               precision    recall  f1-score   support
fake             0.9234    0.9156    0.9195      6000
real             0.9178    0.9252    0.9215      6000

accuracy                             0.9204     12000

Overall Accuracy: 92.04%
```

**Output files:**
- `ml_models/weights/confusion_matrix.png`

---

## 📊 Expected Results

### **Good Performance:**
- **Accuracy:** 88-95%
- **Training time:** 4-6 hours
- **Val loss steadily decreasing**
- **No overfitting** (train acc ≈ val acc)

### **If Results are Bad (<80%):**
- Check dataset is organized correctly
- Verify GPU is being used (`Device: cuda`)
- Try training longer (more epochs)
- Check training curves for issues

---

## 🎓 Learning from the Code

### **While Training Runs, Read:**

1. **`model.py`** - Understand ResNet50 architecture
2. **`train.py`** - See how training loop works
3. **`gradcam.py`** - Learn about explainability
4. **`inference.py`** - Understand prediction pipeline

**All files have DETAILED educational comments!**

---

## 💾 Saved Files After Training

```
ml_models/weights/
├── best_model.pth              ← Best model (use this!)
├── training_curves.png         ← Loss/accuracy plots
├── training_history.json       ← Training metrics
├── confusion_matrix.png        ← Test results visualization
└── checkpoints/
    └── latest_checkpoint.pth   ← Resume training from here
```

---

## ⚠️ Troubleshooting

### **Problem: CUDA out of memory**
**Solution:** Reduce batch size in `train.py`:
```python
BATCH_SIZE = 16  # Change from 32 to 16
```

### **Problem: Training very slow**
**Check:**
```python
print(DEVICE)  # Should show: cuda
```
If shows `cpu`, reinstall PyTorch with CUDA support.

### **Problem: Accuracy not improving**
**Try:**
- Train more epochs
- Check if dataset is balanced (should be 50% fake, 50% real)
- Verify images are loading correctly

---

## 🚀 Next Steps After Training

### **1. Test Individual Images:**

```python
from ml_models.image_detection.inference import ImageDetectionService

service = ImageDetectionService(model_path='ml_models/weights/best_model.pth')
result = service.predict_with_heatmap('test_image.jpg', 'output/')

print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Explanation: {result['explanation']}")
```

### **2. Integrate with Django API:**
- Update views.py to use ImageDetectionService
- Test through API endpoints
- Connect with frontend

### **3. Create Demo:**
- Upload test images
- Show predictions + heatmaps
- Demonstrate to supervisor

---

## 📈 Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Install dependencies | 10 min | ⏳ TODO |
| Prepare dataset | 5 min | ⏳ TODO |
| Train model | 4-6 hours | ⏳ TODO |
| Evaluate model | 5 min | ⏳ TODO |
| Test predictions | 10 min | ⏳ TODO |
| **Total** | **~6 hours** | |

---

## 🎉 You're Ready!

**Start with:**
```bash
# 1. Activate venv
.\venv\Scripts\Activate.ps1

# 2. Prepare dataset
python ml_models/image_detection/prepare_dataset.py

# 3. Start training (go have lunch/dinner!)
python ml_models/image_detection/train.py
```

**Good luck! You're about to train a state-of-the-art deepfake detector!** 🚀

---

*For questions or issues, refer to the detailed comments in each script file.*

