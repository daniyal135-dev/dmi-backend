"""
==============================================================================
IMAGE DETECTION INFERENCE SERVICE
==============================================================================

WHAT THIS FILE DOES:
Image detection using ViT (phase4_replay_best.pth) only.
Loads ViT, preprocesses images, runs prediction, and generates attention heatmaps.

WHAT IT PROVIDES:
- Complete inference pipeline (image → prediction + heatmap)
- Batch processing (analyze multiple images at once)
- Metadata extraction (EXIF data from images)
- Human-readable explanations
- Ready-to-use service for Django API

TYPICAL WORKFLOW:
1. User uploads image through web interface
2. Django API calls ImageDetectionService.predict_with_heatmap()
3. This file loads image, preprocesses it, runs model, generates heatmap
4. Returns: verdict, confidence, probabilities, heatmap, explanation
5. Django saves results to database
6. Frontend displays results to user

FILE STRUCTURE:
1. ImageDetectionService class - Main inference service
2. Methods for prediction, batch processing, and metadata extraction
==============================================================================
"""

# ============== IMPORTS ==============
import torch
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torchvision import transforms
from transformers import ViTForImageClassification


def _vit_attention_heatmap(model, input_tensor, device):
    """ViT: get attention map from last layer (class token to patches), shape (14,14) -> (224,224)."""
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=input_tensor, output_attentions=True)
    if not out.attentions:
        return np.zeros((224, 224), dtype=np.float32)
    att = out.attentions[-1]   # (batch, heads, 197, 197)
    att = att.mean(dim=1)     # (1, 197, 197)
    att = att[0, 0, 1:].cpu().numpy().reshape(14, 14)
    att = (att - att.min()) / (att.max() - att.min() + 1e-8)
    cam = cv2.resize(att.astype(np.float32), (224, 224))
    return cam


def _save_heatmap_overlay(original_image, cam, save_path, alpha=0.5):
    """Overlay heatmap on image and save. cam: (H,W) 0-1 float."""
    if isinstance(original_image, Image.Image):
        img = np.array(original_image.convert("RGB"))
    else:
        img = np.array(original_image) if hasattr(original_image, 'convert') else original_image
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    cam_u8 = np.uint8(255 * np.clip(cam_resized, 0, 1))
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


class ImageDetectionService:
    """
    Image detection service: ViT (phase4_replay_best) only.
    Predicts Real/Fake, returns confidence and heatmap.
    """

    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"✓ Using device: {self.device}")

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"ViT model not found at {model_path}. Put phase4_replay_best.pth in ml_models/weights/.")

        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=2,
            id2label={0: "Real", 1: "Fake"},
            label2id={"Real": 0, "Fake": 1},
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.class_names = ["Real", "Fake"]
        print(f"✓ Loaded ViT (phase4_replay_best) from: {model_path}")
        print("="*50)
        print("✓ ImageDetectionService ready!")
        print("="*50)
    
    def predict(self, image_path):
        """
        PREDICT IF IMAGE IS REAL OR FAKE
        
        Basic prediction without heatmap generation.
        Faster than predict_with_heatmap(), use when you don't need explanations.
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            dict: Prediction results containing:
                - verdict (str): 'real' or 'fake'
                - confidence (float): Confidence score (0-1)
                - probabilities (dict): {'real': 0.05, 'fake': 0.95}
                - predicted_class (int): Class index (0 or 1)
        
        EXAMPLE OUTPUT:
        {
            'verdict': 'fake',
            'confidence': 0.95,
            'probabilities': {'real': 0.05, 'fake': 0.95},
            'predicted_class': 1
        }
        """
        try:
            img = Image.open(image_path).convert("RGB")
            input_tensor = self.vit_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(pixel_values=input_tensor).logits
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            pred_class_idx = predicted_class.item()
            pred_label = self.class_names[pred_class_idx]
            probs = probabilities[0].cpu().numpy()
            result = {
                'verdict': pred_label.lower(),
                'confidence': float(confidence.item()),
                'probabilities': {'real': float(probs[0]), 'fake': float(probs[1])},
                'predicted_class': int(pred_class_idx),
            }
            return result
            
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")
    
    def predict_with_heatmap(self, image_path, output_dir=None):
        """
        PREDICT WITH GRAD-CAM HEATMAP GENERATION
        
        Complete prediction + explainability.
        This is what you'll use in production!
        
        Args:
            image_path (str): Path to image file
            output_dir (str): Directory to save heatmap
                - None: Don't save heatmap (faster)
                - Path: Save heatmap to this directory
        
        Returns:
            dict: Complete results containing:
                - verdict: 'real' or 'fake'
                - confidence: Confidence score
                - probabilities: Dict of class probabilities
                - predicted_class: Class index
                - heatmap_path: Path to saved heatmap (if output_dir provided)
                - explanation: Human-readable explanation
        
        USAGE:
            result = service.predict_with_heatmap(
                'uploads/image.jpg',
                'media/heatmaps/'
            )
        """
        try:
            result = self.predict(image_path)
            heatmap_path = None
            if output_dir:
                img = Image.open(image_path).convert("RGB")
                input_tensor = self.vit_transform(img).unsqueeze(0).to(self.device)
                cam = _vit_attention_heatmap(self.model, input_tensor, self.device)
                os.makedirs(output_dir, exist_ok=True)
                image_name = Path(image_path).stem
                heatmap_path = os.path.join(output_dir, f'{image_name}_heatmap.jpg')
                _save_heatmap_overlay(img, cam, heatmap_path)
            result['heatmap_path'] = heatmap_path
            result['explanation'] = self.generate_explanation(result)
            return result
            
        except Exception as e:
            raise ValueError(f"Error during prediction with heatmap: {str(e)}")
    
    def predict_batch(self, image_paths, generate_heatmaps=False, output_dir=None):
        """
        PREDICT MULTIPLE IMAGES
        
        Processes multiple images efficiently.
        Useful for batch analysis or testing on datasets.
        
        Args:
            image_paths (list): List of image file paths
            generate_heatmaps (bool): Whether to generate heatmaps
            output_dir (str): Directory for heatmaps (if generate_heatmaps=True)
        
        Returns:
            list: List of prediction result dictionaries
        
        USAGE:
            image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
            results = service.predict_batch(
                image_paths,
                generate_heatmaps=True,
                output_dir='heatmaps/'
            )
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                # Progress indicator
                print(f"  [{i+1}/{len(image_paths)}] Processing: {image_path}")
                
                # Predict with or without heatmap
                if generate_heatmaps:
                    result = self.predict_with_heatmap(image_path, output_dir)
                else:
                    result = self.predict(image_path)
                
                # Add image path to result
                result['image_path'] = image_path
                results.append(result)
                
            except Exception as e:
                # If one image fails, continue with others
                print(f"  ✗ Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        print(f"✓ Batch processing complete: {len(results)} results")
        return results
    
    def generate_explanation(self, result):
        """
        GENERATE HUMAN-READABLE EXPLANATION
        
        Converts technical prediction into plain English explanation.
        Makes results understandable for non-technical users.
        
        Args:
            result (dict): Prediction result dictionary
        
        Returns:
            str: Plain English explanation
        
        EXPLANATION STYLE:
        - High confidence (>90%): "highly likely"
        - Medium confidence (70-90%): "likely"
        - Low confidence (<70%): "may be"
        """
        verdict = result['verdict']
        confidence = result['confidence']
        
        # ============== GENERATE EXPLANATION BASED ON CONFIDENCE ==============
        if verdict == 'fake':
            # Image predicted as FAKE
            if confidence > 0.9:
                explanation = (
                    f"The image is highly likely to be FAKE (confidence: {confidence:.2%}). "
                    f"The model detected strong indicators of manipulation or AI generation. "
                    f"The highlighted regions in the heatmap show suspicious artifacts."
                )
            elif confidence > 0.7:
                explanation = (
                    f"The image is likely to be FAKE (confidence: {confidence:.2%}). "
                    f"The model found notable signs of manipulation or AI generation. "
                    f"Review the heatmap to see areas that influenced this decision."
                )
            else:
                explanation = (
                    f"The image may be FAKE (confidence: {confidence:.2%}). "
                    f"Some manipulation indicators were detected, but with moderate confidence. "
                    f"Manual review is recommended."
                )
        else:
            # Image predicted as REAL
            if confidence > 0.9:
                explanation = (
                    f"The image appears to be REAL (confidence: {confidence:.2%}). "
                    f"No significant manipulation indicators were detected. "
                    f"The image shows natural characteristics consistent with authentic media."
                )
            elif confidence > 0.7:
                explanation = (
                    f"The image is likely REAL (confidence: {confidence:.2%}). "
                    f"Minimal manipulation indicators were found. "
                    f"The image appears mostly authentic."
                )
            else:
                explanation = (
                    f"The image may be REAL (confidence: {confidence:.2%}). "
                    f"The classification is uncertain. This could be due to image quality, "
                    f"compression, or unusual characteristics. Further analysis recommended."
                )
        
        explanation += " The heatmap highlights regions that most influenced the model's decision."
        return explanation
    
    def analyze_image_with_metadata(self, image_path, output_dir=None):
        """
        COMPLETE ANALYSIS INCLUDING PREDICTION AND METADATA
        
        Combines deepfake detection with metadata extraction.
        Provides comprehensive analysis of the image.
        
        Args:
            image_path (str): Path to image
            output_dir (str): Directory for heatmap output
        
        Returns:
            dict: Complete analysis including prediction, heatmap, and metadata
        
        METADATA INCLUDES:
        - Image dimensions (width, height)
        - File format (JPEG, PNG, etc.)
        - Color mode (RGB, RGBA, etc.)
        - EXIF data (camera info, GPS, timestamp, etc.)
        """
        # ============== STEP 1: GET PREDICTION WITH HEATMAP ==============
        result = self.predict_with_heatmap(image_path, output_dir)
        
        # ============== STEP 2: EXTRACT METADATA ==============
        try:
            from PIL.ExifTags import TAGS
            image = Image.open(image_path)
            
            # Basic metadata
            metadata = {
                'size': image.size,          # (width, height)
                'format': image.format,      # 'JPEG', 'PNG', etc.
                'mode': image.mode,          # 'RGB', 'RGBA', etc.
            }
            
            # Extract EXIF data (if available)
            # EXIF = Exchangeable Image File Format
            # Contains camera settings, GPS, timestamp, etc.
            exif_data = {}
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                for tag_id, value in exif.items():
                    # Convert tag ID to human-readable name
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)  # Convert to string for JSON compatibility
            
            metadata['exif'] = exif_data
            result['metadata'] = metadata
            
            print("✓ Metadata extracted successfully")
            
        except Exception as e:
            # Metadata extraction failed (not critical)
            print(f"⚠ Could not extract metadata: {str(e)}")
            result['metadata'] = {}
        
        return result
    
    def set_model_path(self, model_path):
        """Load a different ViT checkpoint."""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.to(self.device)
            print(f"✓ Loaded ViT from {model_path}")
        else:
            print(f"✗ Model not found at {model_path}")


# ==============================================================================
# END OF FILE
# ==============================================================================
# 
# SUMMARY:
# This file provides the ImageDetectionService class that brings together
# the model, preprocessor, and Grad-CAM for complete image analysis.
# 
# MAIN FUNCTIONS:
# 1. predict() - Basic prediction (Real/Fake)
# 2. predict_with_heatmap() - Prediction + explainability (RECOMMENDED!)
# 3. predict_batch() - Process multiple images
# 4. analyze_image_with_metadata() - Complete analysis with EXIF
# 5. generate_explanation() - Human-readable explanations
# 
# TYPICAL USAGE IN DJANGO:
#     # In views.py
#     from ml_models.image_detection.inference import ImageDetectionService
#     
#     service = ImageDetectionService(model_path='weights/trained_model.pth')
#     result = service.predict_with_heatmap(
#         image_path=request.FILES['image'].path,
#         output_dir='media/heatmaps/'
#     )
#     
#     # Save to database
#     AnalysisResult.objects.create(
#         verdict=result['verdict'],
#         confidence=result['confidence'],
#         heatmap_path=result['heatmap_path'],
#         explanation=result['explanation']
#     )
# 
# THIS IS THE FILE DJANGO WILL USE FOR IMAGE DETECTION!
# ==============================================================================
