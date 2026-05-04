"""
==============================================================================
TEXT DETECTION INFERENCE SERVICE
==============================================================================

WHAT THIS FILE DOES:
This is the "glue" that connects everything together!
It uses the model and preprocessor to make predictions on text.

WHY THIS FILE EXISTS:
- model.py defines the architecture (blueprint)
- text_preprocessor.py prepares text
- THIS FILE brings them all together for actual predictions!

WHAT IT PROVIDES:
- Complete inference pipeline (text → prediction + explanation)
- Batch processing (analyze multiple texts at once)
- Human-readable explanations
- Ready-to-use service for Django API

TYPICAL WORKFLOW:
1. User uploads text through web interface
2. Django API calls TextDetectionService.predict()
3. This file loads text, preprocesses it, runs model
4. Returns: verdict, confidence, probabilities, explanation
5. Django saves results to database
6. Frontend displays results to user

FILE STRUCTURE:
1. TextDetectionService class - Main inference service
2. Methods for prediction, batch processing, and explanation generation
==============================================================================
"""

# ============== IMPORTS ==============
import torch                           # PyTorch: Deep learning framework
import os                              # File operations
import numpy as np                     # NumPy: Array operations

# Import our modules
from .model import AIGeneratedTextDetector, DEFAULT_TEXT_ENCODER
from ..preprocessing.text_preprocessor import TextPreprocessor  # Text preprocessing


class TextDetectionService:
    """
    ==============================================================================
    TEXT DETECTION SERVICE
    ==============================================================================
    
    Complete service for AI-generated text detection.
    This is what Django will use to analyze text!
    
    FEATURES:
    - Load trained model weights
    - Predict Human/AI-generated for text
    - Batch processing for multiple texts
    - Generate human-readable explanations
    - Extract attention weights for explainability
    
    USAGE IN DJANGO:
        # In views.py
        service = TextDetectionService(model_path='weights/deberta-v3-bestmodel.pth')
        result = service.predict("This is AI-generated text.")
        # result = {
        #     'verdict': 'ai-generated',
        #     'confidence': 0.95,
        #     'probabilities': {'human': 0.05, 'ai-generated': 0.95},
        #     'explanation': 'The text shows patterns typical of AI generation...'
        # }
    
    INITIALIZATION:
        # With trained model
        service = TextDetectionService(model_path='weights/deberta-v3-bestmodel.pth')
        
        # Without trained model (for testing, uses pretrained encoder weights only)
        service = TextDetectionService()
    """
    
    def __init__(self, model_path=None, device=None):
        """
        INITIALIZE DETECTION SERVICE
        
        Sets up the model and preprocessor for inference.
        
        Args:
            model_path (str): Path to trained model weights (.pth file)
                - None: Use pretrained encoder only (NOT RECOMMENDED for production!)
                - Path to .pth: Load trained weights (RECOMMENDED)
            
            device (str): Device to run inference on
                - None: Auto-detect (use GPU if available)
                - 'cpu': Force CPU
                - 'cuda': Force GPU
        
        WHAT HAPPENS HERE:
        1. Detect device (CPU or GPU)
        2. Initialize preprocessor (tokenizer)
        3. Load model (trained or base)
        4. Set model to evaluation mode
        """
        # ============== DEVICE SETUP ==============
        if device is None:
            # Auto-detect device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"✓ TextDetectionService initialized on: {self.device}")

        # ============== LOAD MODEL + TOKENIZER (tokenizer must match checkpoint) ==============
        if model_path and os.path.exists(model_path):
            print(f"✓ Loading trained model from: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            encoder_name = checkpoint.get("model_name", DEFAULT_TEXT_ENCODER)
            self.preprocessor = TextPreprocessor(model_name=encoder_name, max_length=512)
            self.model = AIGeneratedTextDetector(
                num_classes=checkpoint.get("num_classes", 2),
                model_name=encoder_name,
                dropout=checkpoint.get("dropout", 0.3),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            del checkpoint
            self.model.to(self.device)
            self.model.eval()
        else:
            self.preprocessor = TextPreprocessor(
                model_name=DEFAULT_TEXT_ENCODER, max_length=512
            )
            if model_path:
                print(f"⚠ Model not found at {model_path}. Using pretrained encoder only.")
            else:
                print("⚠ No model path provided. Using pretrained encoder only.")

            self.model = AIGeneratedTextDetector(
                num_classes=2, model_name=DEFAULT_TEXT_ENCODER
            )
            self.model.to(self.device)
            self.model.eval()
        
        # Class names
        self.class_names = ['human', 'ai-generated']
    
    def predict(self, text):
        """
        PREDICT IF TEXT IS AI-GENERATED
        
        Main prediction function. Use this for single text analysis.
        
        Args:
            text (str): Input text to analyze
                Example: "This is a human-written article about deepfakes."
        
        Returns:
            dict: Prediction results containing:
                - verdict: 'human' or 'ai-generated'
                - confidence: Confidence score (0.0 to 1.0)
                - probabilities: Dict of class probabilities
                - predicted_class: Class index (0=human, 1=ai-generated)
        
        EXAMPLE OUTPUT:
        {
            'verdict': 'ai-generated',
            'confidence': 0.95,
            'probabilities': {'human': 0.05, 'ai-generated': 0.95},
            'predicted_class': 1
        }
        """
        try:
            # ============== STEP 1: PREPROCESS TEXT ==============
            # Tokenize text (convert to token IDs)
            encoded = self.preprocessor.tokenize(text)
            
            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # ============== STEP 2: FORWARD PASS (PREDICTION) ==============
            # Disable gradient computation (faster inference, saves memory)
            with torch.no_grad():
                # Forward pass through model
                logits = self.model(input_ids, attention_mask)
                # logits shape: (1, 2) → [[Human_score, AI_score]]
                # Example: [[-1.5, 2.1]] → AI-generated is more likely (class 1)
                
                # Convert logits to probabilities using softmax
                probabilities = torch.softmax(logits, dim=1)
                # Example: [[-1.5, 2.1]] → [[0.05, 0.95]]
                
                # Get predicted class and confidence
                confidence, predicted_class = torch.max(probabilities, 1)
                # confidence: highest probability
                # predicted_class: index of highest probability (0 or 1)
            
            # ============== STEP 3: EXTRACT RESULTS ==============
            # Convert from tensors to Python numbers
            pred_class_idx = predicted_class.item()         # 0 or 1
            pred_label = self.class_names[pred_class_idx]   # "human" or "ai-generated"
            conf_score = confidence.item()                  # e.g., 0.95
            
            # Get probabilities for both classes
            probs = probabilities[0].cpu().numpy()  # Convert to numpy array
            
            # ============== STEP 4: CREATE RESULT DICTIONARY ==============
            # IMPORTANT: probs[0] = probability of class 0 (Human)
            #            probs[1] = probability of class 1 (AI-generated)
            result = {
                'verdict': pred_label,  # 'human' or 'ai-generated'
                'confidence': float(conf_score),  # 0.95
                'probabilities': {
                    'human': float(probs[0]),  # Probability of class 0 (Human)
                    'ai-generated': float(probs[1])   # Probability of class 1 (AI-generated)
                },
                'predicted_class': int(pred_class_idx)  # 0 or 1
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")
    
    def predict_with_explanation(self, text):
        """
        PREDICT WITH EXPLANATION
        
        Complete prediction + human-readable explanation.
        This is what you'll use in production!
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            dict: Complete results containing:
                - verdict: 'human' or 'ai-generated'
                - confidence: Confidence score
                - probabilities: Dict of class probabilities
                - predicted_class: Class index
                - explanation: Human-readable explanation
        
        USAGE:
            result = service.predict_with_explanation("This is AI-generated text.")
        """
        try:
            # ============== STEP 1: GET BASIC PREDICTION ==============
            result = self.predict(text)
            
            # ============== STEP 2: GENERATE EXPLANATION ==============
            result['explanation'] = self.generate_explanation(result, text)
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error during prediction with explanation: {str(e)}")
    
    def predict_batch(self, texts):
        """
        PREDICT MULTIPLE TEXTS
        
        Processes multiple texts efficiently.
        Useful for batch analysis or testing on datasets.
        
        Args:
            texts (list): List of text strings
                Example: ["Text 1", "Text 2", "Text 3"]
        
        Returns:
            list: List of result dictionaries (one per text)
        
        USAGE:
            texts = ["This is real text.", "This is AI-generated text."]
            results = service.predict_batch(texts)
        """
        try:
            results = []
            
            # Process each text
            for text in texts:
                result = self.predict(text)
                results.append(result)
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error during batch prediction: {str(e)}")
    
    def generate_explanation(self, result, text=None):
        """
        GENERATE HUMAN-READABLE EXPLANATION
        
        Creates a plain-English explanation of the prediction.
        
        Args:
            result (dict): Prediction result from predict()
            text (str, optional): Original text (for detailed explanations)
        
        Returns:
            str: Human-readable explanation
        
        EXAMPLE:
        "The text is highly likely to be AI-GENERATED (95% confidence).
        The model detected patterns typical of AI-generated content, such as
        repetitive phrasing and uniform sentence structure."
        """
        verdict = result['verdict']
        confidence = result['confidence']
        prob_human = result['probabilities']['human']
        prob_ai = result['probabilities']['ai-generated']
        
        # Determine confidence level
        if confidence >= 0.9:
            conf_level = "highly likely"
        elif confidence >= 0.7:
            conf_level = "likely"
        elif confidence >= 0.5:
            conf_level = "possibly"
        else:
            conf_level = "uncertain"
        
        # Generate explanation based on verdict
        if verdict == 'ai-generated':
            explanation = (
                f"The text is {conf_level} to be AI-GENERATED ({confidence*100:.1f}% confidence). "
                f"The model detected patterns typical of AI-generated content, such as "
                f"repetitive phrasing, uniform sentence structure, or lack of natural variation. "
                f"Human-written probability: {prob_human*100:.1f}%, "
                f"AI-generated probability: {prob_ai*100:.1f}%."
            )
        else:  # human
            explanation = (
                f"The text is {conf_level} to be HUMAN-WRITTEN ({confidence*100:.1f}% confidence). "
                f"The model detected natural language patterns, varied sentence structure, "
                f"and authentic writing style typical of human authors. "
                f"Human-written probability: {prob_human*100:.1f}%, "
                f"AI-generated probability: {prob_ai*100:.1f}%."
            )
        
        return explanation
    
    def get_attention_weights(self, text):
        """
        GET ATTENTION WEIGHTS FOR EXPLAINABILITY
        
        Returns attention weights to see which words the model focuses on.
        Useful for generating detailed explanations.
        
        Args:
            text (str): Input text
        
        Returns:
            list: Attention weights from all layers
                Shape: (num_layers, num_heads, seq_len, seq_len)
        """
        try:
            # Preprocess text
            encoded = self.preprocessor.tokenize(text)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get attention weights
            with torch.no_grad():
                attentions = self.model.get_attention_weights(input_ids, attention_mask)
            
            return attentions
            
        except Exception as e:
            raise ValueError(f"Error getting attention weights: {str(e)}")


# ==============================================================================
# END OF FILE
# ==============================================================================
# 
# SUMMARY:
# This file provides the TextDetectionService class for detecting
# AI-generated text using RoBERTa-based model.
# 
# MAIN FUNCTIONS:
# 1. predict() - Basic prediction (verdict, confidence, probabilities)
# 2. predict_with_explanation() - Prediction + human-readable explanation
# 3. predict_batch() - Process multiple texts at once
# 4. generate_explanation() - Create explanation text
# 5. get_attention_weights() - Get attention for explainability
# 
# TYPICAL USAGE:
#     service = TextDetectionService(model_path='weights/deberta-v3-bestmodel.pth')
#     result = service.predict_with_explanation("This is AI-generated text.")
# ==============================================================================

