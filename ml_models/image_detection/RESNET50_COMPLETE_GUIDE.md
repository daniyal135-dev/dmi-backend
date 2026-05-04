# ResNet50 - Complete Guide (Urdu/English)

## 📋 Table of Contents
1. [ResNet50 Kya Hai?](#what-is-resnet50)
2. [ResNet50 Ki Architecture](#architecture)
3. [Pre-training (ImageNet)](#imagenet)
4. [Kaise Kaam Karta Hai?](#how-it-works)
5. [Humare Project Mein Use](#our-usage)
6. [Layers Ki Detail](#layers-detail)

---

## 🎯 ResNet50 Kya Hai? (What is ResNet50?)

### **Basic Definition:**
- **ResNet50** = **Residual Network** with **50 layers**
- **CNN (Convolutional Neural Network)** - Image classification ke liye
- **2015** mein Microsoft Research ne introduce kiya
- **ImageNet competition** mein winner (94% accuracy)

### **Key Features:**
- ✅ **50 layers** deep (bahut deep network)
- ✅ **25 million parameters** (model ka size)
- ✅ **Residual connections** (skip connections) - training ko easy banata hai
- ✅ **Pre-trained on ImageNet** - 1.4 million images par train hua
- ✅ **Transfer Learning** - hum isko deepfake detection ke liye fine-tune karte hain

### **Why ResNet50?**
- **Proven & Reliable** - industry mein widely used
- **Good Balance** - accuracy aur speed dono achhe
- **Transfer Learning** - ImageNet se learn kiya hua knowledge use karta hai
- **Not Too Heavy** - Quadro T2000 (4GB GPU) par easily run hota hai

---

## 🏗️ ResNet50 Ki Architecture (Architecture Details)

### **Total Layers: 50**

**Breakdown:**
```
1. Initial Convolution Block (1 layer)
   - conv1: 7x7 convolution
   - bn1: Batch Normalization
   - relu: ReLU activation
   - maxpool: Max Pooling

2. Layer 1: 3 bottleneck blocks (3 layers × 3 = 9 layers)
3. Layer 2: 4 bottleneck blocks (3 layers × 4 = 12 layers)
4. Layer 3: 6 bottleneck blocks (3 layers × 6 = 18 layers)
5. Layer 4: 3 bottleneck blocks (3 layers × 3 = 9 layers)

6. Average Pooling (1 layer)
7. Fully Connected Layer (1 layer)

Total: 1 + 9 + 12 + 18 + 9 + 1 + 1 = 50 layers
```

### **Layer-by-Layer Flow:**

#### **1. Input Layer:**
- **Input Size:** 224×224×3 (RGB image)
- **What it does:** Image ko receive karta hai

#### **2. Initial Convolution Block (conv1):**
```
Input:  224×224×3
  ↓
Conv1:  7×7 convolution, 64 filters
  ↓
Output: 112×112×64
```
- **Purpose:** Basic features extract karta hai (edges, colors, textures)
- **Stride:** 2 (size half ho jata hai)

#### **3. Batch Normalization + ReLU + MaxPool:**
```
112×112×64
  ↓
MaxPool: 3×3, stride=2
  ↓
Output: 56×56×64
```

#### **4. Layer 1 (3 Bottleneck Blocks):**
```
Input:  56×56×64
  ↓
3 Bottleneck Blocks
  ↓
Output: 56×56×256
```
- **Purpose:** Simple patterns learn karta hai (shapes, basic objects)
- **Spatial size same rehta hai** (56×56)

#### **5. Layer 2 (4 Bottleneck Blocks):**
```
Input:  56×56×256
  ↓
4 Bottleneck Blocks
  ↓
Output: 28×28×512
```
- **Purpose:** Object parts learn karta hai (eyes, nose, wheels, etc.)
- **Size half ho jata hai** (28×28)

#### **6. Layer 3 (6 Bottleneck Blocks):**
```
Input:  28×28×512
  ↓
6 Bottleneck Blocks
  ↓
Output: 14×14×1024
```
- **Purpose:** Complete objects learn karta hai (faces, cars, animals)
- **Size aur half** (14×14)

#### **7. Layer 4 (3 Bottleneck Blocks):**
```
Input:  14×14×1024
  ↓
3 Bottleneck Blocks
  ↓
Output: 7×7×2048
```
- **Purpose:** Complex concepts learn karta hai (deepfake artifacts, inconsistencies)
- **Most important layer** - Grad-CAM isi layer se heatmap banata hai
- **Size minimum** (7×7)

#### **8. Average Pooling:**
```
Input:  7×7×2048
  ↓
Average Pool
  ↓
Output: 1×1×2048
```
- **Purpose:** Spatial dimensions ko remove karta hai
- **Result:** 2048 features (numbers) jo image ko represent karte hain

#### **9. Fully Connected Layer (Original):**
```
Input:  2048 features
  ↓
FC Layer
  ↓
Output: 1000 classes (ImageNet)
```
- **Original ResNet50:** 1000 classes (cat, dog, car, etc.)
- **Humara Custom:** 2 classes (Real, Fake)

---

## 🖼️ Pre-training: ImageNet (What It's Pre-trained On)

### **ImageNet Dataset:**
- **Size:** 1.4 million images
- **Classes:** 1000 different categories
- **Categories:** Animals, objects, vehicles, food, etc.
- **Examples:** Cat, dog, car, airplane, pizza, etc.

### **ImageNet Statistics:**
```
Total Images:    1,431,167
Training Images: 1,281,167
Validation:       50,000
Test:             100,000
Classes:          1,000
```

### **What ResNet50 Learned from ImageNet:**
1. **Low-level Features:**
   - Edges (lines, curves)
   - Colors (RGB values)
   - Textures (smooth, rough, patterns)

2. **Mid-level Features:**
   - Shapes (circles, squares, triangles)
   - Object parts (eyes, wheels, wings)
   - Patterns (stripes, dots, grids)

3. **High-level Features:**
   - Complete objects (faces, cars, animals)
   - Scenes (indoor, outdoor, nature)
   - Relationships (object positions, context)

### **Why Pre-training Matters:**
- ✅ **Faster Training:** Scratch se train karne se 10x faster
- ✅ **Less Data Needed:** ImageNet se learn kiya hua knowledge use hota hai
- ✅ **Better Accuracy:** Pre-trained models zyada accurate hote hain
- ✅ **Generalization:** Different types ki images par better kaam karta hai

### **ImageNet Weights Version:**
- **IMAGENET1K_V2** - Latest version (best accuracy)
- **IMAGENET1K_V1** - Older version (still good)

**Humare Code Mein:**
```python
weights = ResNet50_Weights.IMAGENET1K_V2  # Latest weights
self.resnet = models.resnet50(weights=weights)
```

---

## ⚙️ Kaise Kaam Karta Hai? (How It Works)

### **Complete Flow:**

```
1. Image Input (224×224×3)
   ↓
2. Initial Conv Block → 56×56×64
   ↓
3. Layer 1 → 56×56×256 (simple patterns)
   ↓
4. Layer 2 → 28×28×512 (object parts)
   ↓
5. Layer 3 → 14×14×1024 (objects)
   ↓
6. Layer 4 → 7×7×2048 (complex concepts)
   ↓
7. Average Pool → 1×1×2048 (2048 features)
   ↓
8. Flatten → 2048 (1D vector)
   ↓
9. Custom FC Layers → 2 (Real, Fake)
   ↓
10. Softmax → Probabilities
```

### **Bottleneck Block (Residual Connection):**

**Key Innovation:** Residual (skip) connections

```
Input (x)
  ↓
┌─────────────────────┐
│ 1×1 Conv (reduce)  │
│ 3×3 Conv (main)    │
│ 1×1 Conv (expand)  │
└─────────────────────┘
  ↓
Output
  +
  ↓ (skip connection)
Input (x) ────────────┐
  ↓                   │
  └─── Add ───────────┘
  ↓
Final Output
```

**Why Skip Connections?**
- **Problem:** Deep networks mein gradient vanish ho jata hai (training slow/stop)
- **Solution:** Skip connection se gradient directly flow hota hai
- **Result:** 50 layers easily train ho sakte hain

### **Feature Extraction Process:**

1. **Layer 1:** "Yeh edge hai, yeh color hai"
2. **Layer 2:** "Yeh eye hai, yeh wheel hai"
3. **Layer 3:** "Yeh face hai, yeh car hai"
4. **Layer 4:** "Yeh deepfake artifact hai, inconsistency hai"

### **Classification Head (Humara Custom):**

**Original ResNet50:**
```
2048 → 1000 (ImageNet classes)
```

**Humara Custom:**
```
2048 → Dropout(0.5) → 512 → ReLU → Dropout(0.3) → 2 (Real/Fake)
```

**Why Custom Head?**
- ImageNet: 1000 classes (cat, dog, etc.)
- Humara task: 2 classes (Real, Fake)
- Custom head se model ko specifically Real/Fake sikhaya jata hai

---

## 🎯 Humare Project Mein Use (Our Usage)

### **Model Class:**
```python
class DeepfakeImageDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        # Load ResNet50 with ImageNet weights
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace final layer with custom head
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Real, Fake
        )
```

### **Training Process:**
1. **Load Pre-trained ResNet50** (ImageNet weights)
2. **Freeze Early Layers** (Layer 1, 2) - ImageNet knowledge preserve
3. **Fine-tune Later Layers** (Layer 3, 4) - Deepfake detection ke liye adapt
4. **Train Custom Head** - Real/Fake classification
5. **Save Trained Model** → `best_model.pth`

### **Inference (Prediction):**
```
User Image → Preprocess (224×224) → ResNet50 → Features (2048) 
→ Custom Head → Logits (2) → Softmax → Probabilities → Verdict
```

### **Grad-CAM (Heatmap):**
- **Layer 4** (7×7×2048) se activations capture
- **Gradients** calculate karke important regions highlight
- **Heatmap** generate karta hai (kahan manipulation hai)

---

## 📊 Layers Ki Complete Detail (Complete Layers Breakdown)

### **Total: 50 Layers**

#### **1. Initial Block (4 operations):**
- `conv1`: 7×7 conv, 64 filters, stride=2
- `bn1`: Batch Normalization
- `relu`: ReLU activation
- `maxpool`: 3×3 max pool, stride=2

#### **2. Layer 1: 3 Bottleneck Blocks**
- **Each Block:** 3 layers (1×1 conv, 3×3 conv, 1×1 conv)
- **Total:** 3 blocks × 3 layers = 9 layers
- **Input:** 56×56×64
- **Output:** 56×56×256

#### **3. Layer 2: 4 Bottleneck Blocks**
- **Each Block:** 3 layers
- **Total:** 4 blocks × 3 layers = 12 layers
- **Input:** 56×56×256
- **Output:** 28×28×512
- **Stride:** 2 (size half)

#### **4. Layer 3: 6 Bottleneck Blocks**
- **Each Block:** 3 layers
- **Total:** 6 blocks × 3 layers = 18 layers
- **Input:** 28×28×512
- **Output:** 14×14×1024
- **Stride:** 2 (size half)

#### **5. Layer 4: 3 Bottleneck Blocks**
- **Each Block:** 3 layers
- **Total:** 3 blocks × 3 layers = 9 layers
- **Input:** 14×14×1024
- **Output:** 7×7×2048
- **Stride:** 2 (size half)
- **Most Important:** Grad-CAM isi layer se kaam karta hai

#### **6. Average Pooling:**
- **Input:** 7×7×2048
- **Output:** 1×1×2048
- **Purpose:** Spatial dimensions remove

#### **7. Flatten:**
- **Input:** 1×1×2048
- **Output:** 2048 (1D vector)

#### **8. Custom FC Layers (Humara Addition):**
- **Dropout(0.5):** 50% neurons off (prevent overfitting)
- **Linear(2048→512):** Feature reduction
- **ReLU:** Activation
- **Dropout(0.3):** 30% neurons off
- **Linear(512→2):** Final classification (Real, Fake)

### **Layer Count Summary:**
```
Initial Block:     4 operations
Layer 1:           9 layers (3 blocks × 3)
Layer 2:          12 layers (4 blocks × 3)
Layer 3:          18 layers (6 blocks × 3)
Layer 4:           9 layers (3 blocks × 3)
Average Pool:      1 operation
Flatten:           1 operation
Custom FC:         5 layers (Dropout, Linear, ReLU, Dropout, Linear)
─────────────────────────────────────
Total:            59 operations (but counted as 50 "layers")
```

**Note:** "50 layers" typically refers to the main convolutional layers, not every operation.

---

## 🔢 Technical Specifications

### **Model Size:**
- **Parameters:** ~25 million
- **Model File Size:** ~100 MB (weights)
- **Memory (Inference):** ~400 MB GPU RAM
- **Memory (Training):** ~2-4 GB GPU RAM

### **Input/Output:**
- **Input Size:** 224×224×3 (RGB image)
- **Output Size:** 2 (Real, Fake logits)
- **Feature Vector:** 2048 dimensions

### **Performance:**
- **Inference Speed:** ~10-20 ms per image (GPU)
- **Training Speed:** ~100-200 images/second (GPU)
- **Accuracy (ImageNet):** 94% (top-5: 97%)
- **Accuracy (Our Task):** ~96% (deepfake detection)

---

## 📝 Summary

### **ResNet50 Key Points:**
1. ✅ **50 layers** deep convolutional network
2. ✅ **Pre-trained on ImageNet** (1.4M images, 1000 classes)
3. ✅ **25M parameters** - powerful but efficient
4. ✅ **Residual connections** - deep networks ko train karna easy
5. ✅ **Transfer Learning** - ImageNet knowledge use karke deepfake detection

### **Humare Project Mein:**
- ✅ ResNet50 backbone (ImageNet weights)
- ✅ Custom classification head (2 classes: Real/Fake)
- ✅ Fine-tuned on deepfake dataset
- ✅ Grad-CAM for explainability (heatmaps)
- ✅ ~96% accuracy on training/test data

### **Why ResNet50?**
- Proven & reliable
- Good balance (accuracy + speed)
- Works on Quadro T2000 (4GB GPU)
- Transfer learning se fast training
- Industry standard

---

**Complete Information!** 🎉

