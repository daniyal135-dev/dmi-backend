"""
==============================================================================
IMAGE PREPROCESSING MODULE
==============================================================================

WHAT THIS FILE DOES:
This file prepares images BEFORE feeding them to the neural network.
Think of it as a "photo editor" that standardizes all images.

WHY PREPROCESSING IS NEEDED:
1. Neural networks expect fixed input size (224x224 for ResNet50)
2. Pixel values need to be normalized (makes training stable)
3. Images come in different sizes, formats, and quality
4. We need consistency across all inputs

WHAT PREPROCESSING INCLUDES:
- Loading images from files
- Resizing to 224x224 pixels
- Converting to PyTorch tensor
- Normalizing pixel values using ImageNet statistics
- Optional: Face extraction, noise removal

FILE STRUCTURE:
1. ImagePreprocessor class - Main preprocessing pipeline
2. Helper methods for face extraction and noise removal
==============================================================================
"""

# ============== IMPORTS ==============
import cv2                    # OpenCV: Computer vision library (for face detection, filters)
import numpy as np            # NumPy: Array operations
from PIL import Image         # PIL: Python Image Library (for loading/saving images)
import torch                  # PyTorch: Deep learning framework
from torchvision import transforms  # Pre-built image transformations


class ImagePreprocessor:
    """
    ==============================================================================
    IMAGE PREPROCESSOR
    ==============================================================================
    
    Handles all image preprocessing tasks for deepfake detection.
    
    TYPICAL WORKFLOW:
    1. Load image from file → PIL Image
    2. Resize to 224x224 → Required by ResNet50
    3. Convert to tensor → PyTorch format
    4. Normalize → Using ImageNet statistics
    5. Add batch dimension → (1, 3, 224, 224)
    
    USAGE EXAMPLE:
        preprocessor = ImagePreprocessor()
        tensor = preprocessor.preprocess('image.jpg')
        output = model(tensor)  # Now ready for model!
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        INITIALIZE IMAGE PREPROCESSOR
        
        Args:
            target_size (tuple): Target dimensions for resized images
                - Default: (224, 224) for ResNet50
                - Must match what the model expects!
        
        WHAT HAPPENS HERE:
        We create a preprocessing pipeline using torchvision.transforms.
        This pipeline will be applied to every image automatically.
        """
        self.target_size = target_size
        
        # ============== DEFINE TRANSFORMATION PIPELINE ==============
        # IMPORTANT: Must match validation/test transforms from training!
        # Training uses: Resize(256) → CenterCrop(224) → ToTensor → Normalize
        # These transformations are applied IN ORDER:
        self.transform = transforms.Compose([
            
            # STEP 1: RESIZE TO 256
            # Resize image to 256 (slightly larger than needed)
            # This matches the validation/test preprocessing from training
            transforms.Resize(256),
            
            # STEP 2: CENTER CROP TO 224x224
            # Crop center 224x224 region (required by ResNet50)
            # This matches validation/test preprocessing exactly!
            transforms.CenterCrop(target_size),
            
            # STEP 3: CONVERT TO TENSOR
            # PIL Image → PyTorch Tensor
            # Shape: (Height, Width, Channels) → (Channels, Height, Width)
            # Pixel range: [0, 255] → [0.0, 1.0]
            transforms.ToTensor(),
            
            # STEP 4: NORMALIZE
            # Standardize pixel values using ImageNet statistics
            # WHY THESE NUMBERS?
            # - mean = [0.485, 0.456, 0.406]: Average RGB values across ImageNet
            # - std = [0.229, 0.224, 0.225]: Standard deviations across ImageNet
            # Formula: normalized = (pixel - mean) / std
            # This makes training more stable and faster!
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean (R, G, B)
                std=[0.229, 0.224, 0.225]     # ImageNet std (R, G, B)
            )
        ])
    
    def load_image(self, image_path):
        """
        LOAD IMAGE FROM FILE
        
        Args:
            image_path (str): Path to image file
                Examples: 'image.jpg', 'photos/face.png', etc.
        
        Returns:
            PIL.Image: RGB image (3 channels)
        
        WHY CONVERT TO RGB?
        - Some images are grayscale (1 channel)
        - Some images have alpha channel (4 channels: RGBA)
        - ResNet50 expects exactly 3 channels (RGB)
        - .convert('RGB') ensures consistency
        """
        try:
            # Open image file using PIL
            img = Image.open(image_path)
            
            # Convert to RGB (ensures 3 channels)
            img = img.convert('RGB')
            
            return img
            
        except Exception as e:
            # If loading fails, raise a clear error
            raise ValueError(f"Error loading image from {image_path}: {str(e)}")
    
    def preprocess(self, image_path):
        """
        COMPLETE PREPROCESSING PIPELINE
        
        This is the main function you'll use!
        It loads, resizes, and normalizes the image in one step.
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            torch.Tensor: Preprocessed image tensor
                Shape: (1, 3, 224, 224)
                - 1: Batch size (single image)
                - 3: RGB channels
                - 224x224: Image dimensions
        
        USAGE EXAMPLE:
            preprocessor = ImagePreprocessor()
            tensor = preprocessor.preprocess('photo.jpg')
            # tensor is now ready for model input!
        """
        # Load image from file
        img = self.load_image(image_path)
        
        # Apply transformation pipeline (resize, to_tensor, normalize)
        tensor = self.transform(img)
        
        # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
        # Models expect batches, even if it's just one image
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def preprocess_pil(self, pil_image):
        """
        PREPROCESS PIL IMAGE DIRECTLY
        
        Use this if you already have a PIL Image object (not a file path).
        
        Args:
            pil_image (PIL.Image): PIL Image object
        
        Returns:
            torch.Tensor: Preprocessed tensor (1, 3, 224, 224)
        
        WHEN TO USE:
        - If you're loading images from memory (not files)
        - If you've already done some preprocessing with PIL
        - If images come from a web upload (bytes → PIL Image)
        """
        # Apply transformations
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def denormalize(self, tensor):
        """
        DENORMALIZE TENSOR FOR VISUALIZATION
        
        Reverses the normalization to convert tensor back to viewable image.
        Useful for displaying preprocessed images or debugging.
        
        Args:
            tensor (torch.Tensor): Normalized tensor from model
        
        Returns:
            numpy.ndarray: Denormalized image (0-255, uint8)
                Shape: (Height, Width, Channels)
        
        WHY DENORMALIZE?
        After normalization, pixel values are no longer in [0, 255] range.
        To display the image, we need to reverse the normalization.
        
        FORMULA (reverse of normalize):
        pixel = (normalized * std) + mean
        """
        # ImageNet statistics (same as used in normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Convert tensor to numpy array
        img = tensor.squeeze().cpu().numpy()
        
        # Transpose from (C, H, W) to (H, W, C)
        img = img.transpose(1, 2, 0)
        
        # Denormalize: pixel = (normalized * std) + mean
        img = std * img + mean
        
        # Clip values to [0, 1] range (in case of numerical errors)
        img = np.clip(img, 0, 1)
        
        # Convert to [0, 255] range and uint8 type (standard image format)
        img = (img * 255).astype(np.uint8)
        
        return img
    
    def extract_faces(self, image_path):
        """
        EXTRACT FACE REGIONS FROM IMAGE
        
        Uses OpenCV's Haar Cascade classifier to detect faces.
        Useful for focusing on face regions (most deepfakes manipulate faces).
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            list: List of face images as PIL Images
                Empty list if no faces detected
        
        WHY FACE EXTRACTION?
        - Most deepfakes target faces
        - Analyzing faces gives better accuracy
        - Reduces background noise
        
        HOW IT WORKS:
        1. Load Haar Cascade classifier (pre-trained face detector)
        2. Convert image to grayscale (Haar works on grayscale)
        3. Detect faces (returns bounding boxes)
        4. Crop face regions
        5. Return as PIL Images
        
        NOTE: Haar Cascade is fast but not very accurate.
        For production, consider using MTCNN or RetinaFace.
        """
        # ============== LOAD FACE DETECTOR ==============
        # OpenCV provides pre-trained Haar Cascade models
        # haarcascade_frontalface_default.xml detects front-facing faces
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # ============== LOAD AND PREPARE IMAGE ==============
        # Read image using OpenCV
        img = cv2.imread(image_path)
        
        # Convert to grayscale (Haar Cascade works on grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ============== DETECT FACES ==============
        # detectMultiScale finds faces at different scales
        # Parameters:
        #   scaleFactor=1.1: How much to reduce image size at each scale
        #   minNeighbors=5: How many neighbors needed to keep a detection
        #   minSize=(30, 30): Minimum face size
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # ============== EXTRACT FACE REGIONS ==============
        face_images = []
        for (x, y, w, h) in faces:
            # Crop face region from original image
            face_roi = img[y:y+h, x:x+w]
            
            # Convert from BGR (OpenCV) to RGB (PIL)
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            face_pil = Image.fromarray(face_rgb)
            
            face_images.append(face_pil)
        
        return face_images
    
    def remove_noise(self, image_path, filter_type='gaussian'):
        """
        APPLY NOISE REMOVAL FILTER
        
        Removes noise from images to improve quality.
        Useful for low-quality images or camera artifacts.
        
        Args:
            image_path (str): Path to image file
            filter_type (str): Type of filter to apply
                - 'gaussian': Gaussian blur (fast, general purpose)
                - 'median': Median filter (good for salt-and-pepper noise)
                - 'bilateral': Bilateral filter (preserves edges)
        
        Returns:
            numpy.ndarray: Filtered image (BGR format)
        
        WHEN TO USE:
        - Images from low-quality cameras
        - Compressed images (JPEG artifacts)
        - Noisy social media images
        
        FILTER COMPARISON:
        - Gaussian: Fast, blurs everything (including edges)
        - Median: Good for removing outliers, preserves edges better
        - Bilateral: Best quality, preserves edges, but slower
        """
        # Read image using OpenCV
        img = cv2.imread(image_path)
        
        # Apply selected filter
        if filter_type == 'gaussian':
            # Gaussian blur: Weighted average of neighboring pixels
            # (5, 5) is kernel size, 0 lets OpenCV calculate sigma
            filtered = cv2.GaussianBlur(img, (5, 5), 0)
            
        elif filter_type == 'median':
            # Median filter: Replaces pixel with median of neighbors
            # 5 is kernel size (5x5 neighborhood)
            # Good for removing salt-and-pepper noise
            filtered = cv2.medianBlur(img, 5)
            
        elif filter_type == 'bilateral':
            # Bilateral filter: Blurs while preserving edges
            # Parameters:
            #   d=9: Diameter of pixel neighborhood
            #   sigmaColor=75: Filter sigma in color space
            #   sigmaSpace=75: Filter sigma in coordinate space
            filtered = cv2.bilateralFilter(img, 9, 75, 75)
            
        else:
            # Unknown filter type, return original
            print(f"Warning: Unknown filter type '{filter_type}'. Returning original.")
            filtered = img
        
        return filtered


# ==============================================================================
# END OF FILE
# ==============================================================================
# 
# SUMMARY:
# This file provides the ImagePreprocessor class for preparing images
# before feeding them to the deepfake detection model.
# 
# MAIN FUNCTIONS:
# 1. preprocess() - Complete preprocessing pipeline (most used!)
# 2. load_image() - Load image from file
# 3. extract_faces() - Detect and crop faces
# 4. remove_noise() - Apply denoising filters
# 5. denormalize() - Convert tensor back to viewable image
# 
# TYPICAL USAGE:
#     preprocessor = ImagePreprocessor()
#     tensor = preprocessor.preprocess('image.jpg')
#     output = model(tensor)
# ==============================================================================
