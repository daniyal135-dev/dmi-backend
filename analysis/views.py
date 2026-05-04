from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from pathlib import Path

from .models import AnalysisResult
from .serializers import AnalysisResultSerializer

# Import our trained model services
from ml_models.image_detection.inference import ImageDetectionService
from ml_models.text_detection.inference import TextDetectionService

class AnalysisResultListCreateView(generics.ListCreateAPIView):
    serializer_class = AnalysisResultSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return AnalysisResult.objects.filter(user=self.request.user)

class AnalysisResultDetailView(generics.RetrieveAPIView):
    serializer_class = AnalysisResultSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return AnalysisResult.objects.filter(user=self.request.user)

# ==============================================================================
# GLOBAL MODEL SERVICE INSTANCE
# ==============================================================================
# Initialize the model service once (lazy loading - only when first needed)
# This avoids loading the model on every request (which is slow!)
_model_service = None
_current_model_name = None

def get_model_service():
    """Get or create the image detection service (ViT phase4_replay_best). Loads once."""
    global _model_service, _current_model_name
    model_name = getattr(settings, 'MODEL_NAME', 'phase4_replay_best.pth')
    model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'weights', model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ViT model not found at {model_path}. "
            f"Copy phase4_replay_best.pth to ml_models/weights/."
        )
    if _model_service is None or _current_model_name != model_path:
        _model_service = ImageDetectionService(model_path=model_path)
        _current_model_name = model_path
        print(f"✓ Image detection (ViT) loaded from {model_path}")
    return _model_service


_text_detection_service = None
_text_detection_service_key = None


def get_text_detection_service():
    """
    Load TextDetectionService once per process.
    Avoids reloading ~700MB weights + tokenizer on every /api/analysis/text/ request.
    """
    global _text_detection_service, _text_detection_service_key
    text_model_name = getattr(settings, "TEXT_MODEL_NAME", "deberta-v3-bestmodel.pth")
    model_path = os.path.join(
        settings.BASE_DIR,
        "ml_models",
        "weights",
        text_model_name,
    )
    path_key = model_path if os.path.exists(model_path) else f"__none__:{text_model_name}"
    td = getattr(settings, "TEXT_INFERENCE_DEVICE", "") or ""
    device_kw = td if td in ("cpu", "cuda") else None
    key = f"{path_key}|{device_kw or 'auto'}"

    if _text_detection_service is None or _text_detection_service_key != key:
        try:
            _text_detection_service = TextDetectionService(
                model_path=model_path if os.path.exists(model_path) else None,
                device=device_kw,
            )
        except RuntimeError as e:
            err = str(e).lower()
            if device_kw != "cpu" and (
                "out of memory" in err or "cuda" in err or "cublas" in err
            ):
                print(f"WARN: Text model load on GPU failed ({e!r}); retrying on CPU.")
                device_kw = "cpu"
                key = f"{path_key}|cpu"
                _text_detection_service = TextDetectionService(
                    model_path=model_path if os.path.exists(model_path) else None,
                    device="cpu",
                )
            else:
                raise
        _text_detection_service_key = key
        print(f"✓ Text detection (DeBERTa) loaded: {key}")
    return _text_detection_service


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_image(request):
    """
    ==============================================================================
    ANALYZE IMAGE ENDPOINT
    ==============================================================================
    
    Accepts an image file upload, analyzes it using the ViT model (phase4_replay_best),
    generates an attention heatmap, and saves results to database.
    
    REQUEST:
        POST /api/analysis/image/
        Content-Type: multipart/form-data
        Body: { 'image': <file> }
    
    RESPONSE:
        {
            'id': 123,
            'verdict': 'fake',
            'confidence': 0.95,
            'heatmap_path': '/media/heatmaps/image_heatmap.jpg',
            'explanation': 'The image is highly likely to be FAKE...',
            'metadata': {...},
            'file_path': '/media/uploads/image.jpg'
        }
    """
    try:
        # ============== STEP 1: VALIDATE FILE UPLOAD ==============
        if 'image' not in request.FILES:
            return Response(
                {'error': 'No image file provided. Please upload an image.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES['image']
        
        # Validate file type
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return Response(
                {
                    'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}',
                    'received': file_ext
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if uploaded_file.size > max_size:
            return Response(
                {'error': f'File too large. Maximum size: 10MB'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # ============== STEP 2: SAVE UPLOADED FILE ==============
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        file_path = default_storage.save(
            os.path.join('uploads', uploaded_file.name),
            ContentFile(uploaded_file.read())
        )
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        # ============== STEP 3: LOAD MODEL SERVICE ==============
        service = get_model_service()
        
        # ============== STEP 4: ANALYZE IMAGE ==============
        # Create heatmap output directory
        heatmap_dir = os.path.join(settings.MEDIA_ROOT, 'heatmaps')
        os.makedirs(heatmap_dir, exist_ok=True)
        
        # Run prediction with heatmap
        result = service.predict_with_heatmap(
            image_path=full_file_path,
            output_dir=heatmap_dir
        )
        
        # ============== STEP 5: DETERMINE VERDICT ==============
        # Convert to 'uncertain' if confidence is low
        verdict = result['verdict']
        confidence = result['confidence']
        
        if confidence < 0.6:  # Low confidence threshold
            verdict = 'uncertain'
        
        # ============== STEP 6: CONVERT HEATMAP PATH TO RELATIVE ==============
        # Convert full path to relative path for media URL
        heatmap_full_path = result.get('heatmap_path', '')
        heatmap_relative_path = ''
        if heatmap_full_path:
            # Convert absolute path to relative path
            # e.g., /path/to/media/heatmaps/image.jpg → heatmaps/image.jpg
            if heatmap_full_path.startswith(settings.MEDIA_ROOT):
                heatmap_relative_path = os.path.relpath(heatmap_full_path, settings.MEDIA_ROOT)
                # Normalize path separators for URLs
                heatmap_relative_path = heatmap_relative_path.replace('\\', '/')
        
        # ============== STEP 7: SAVE TO DATABASE ==============
        analysis_result = AnalysisResult.objects.create(
            user=request.user,
            file_type='image',
            file_path=file_path,  # Relative path for database
            verdict=verdict,
            confidence=float(confidence),
            heatmap_path=heatmap_relative_path,  # Relative path to heatmap
            metadata={
                'probabilities': result.get('probabilities', {}),
                'predicted_class': result.get('predicted_class', -1),
                'file_size': uploaded_file.size,
                'file_name': uploaded_file.name
            },
            explanation=result.get('explanation', '')
        )
        
        # ============== STEP 8: RETURN RESPONSE ==============
        serializer = AnalysisResultSerializer(analysis_result)
        
        return Response(
            {
                'message': 'Image analyzed successfully',
                'result': serializer.data
            },
            status=status.HTTP_200_OK
        )
        
    except FileNotFoundError as e:
        # Model not found
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    except Exception as e:
        # Any other error
        return Response(
            {
                'error': 'An error occurred during image analysis',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_video(request):
    """
    ==============================================================================
    ANALYZE VIDEO ENDPOINT
    ==============================================================================
    
    Accepts a video file upload, extracts frames, analyzes each frame using
    the trained image model, and aggregates results.
    
    REQUEST:
        POST /api/analysis/video/
        Content-Type: multipart/form-data
        Body: { 'video': <file> }
    
    RESPONSE:
        {
            'id': 123,
            'verdict': 'fake',
            'confidence': 0.85,
            'fake_percentage': 70.5,
            'frames_analyzed': 30,
            'suspicious_frames': [...],
            'explanation': 'The video is likely FAKE...'
        }
    
    HOW IT WORKS:
    1. Save uploaded video
    2. Extract frames (1 per second, max 30 frames)
    3. Analyze each frame with image detection model
    4. Aggregate results (>50% fake frames = FAKE video)
    5. Return comprehensive results
    """
    try:
        # ============== STEP 1: VALIDATE FILE UPLOAD ==============
        if 'video' not in request.FILES:
            return Response(
                {'error': 'No video file provided. Please upload a video.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES['video']
        
        # Validate file type
        allowed_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return Response(
                {
                    'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}',
                    'received': file_ext
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB in bytes
        if uploaded_file.size > max_size:
            return Response(
                {'error': f'File too large. Maximum size: 100MB'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # ============== STEP 2: SAVE UPLOADED VIDEO ==============
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', 'videos')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        file_path = default_storage.save(
            os.path.join('uploads', 'videos', uploaded_file.name),
            ContentFile(uploaded_file.read())
        )
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        # ============== STEP 3: LOAD VIDEO SERVICE ==============
        from ml_models.video_detection.inference import VideoDetectionService
        
        # Get model path
        model_name = getattr(settings, 'MODEL_NAME', 'best_model.pth')
        model_path = os.path.join(
            settings.BASE_DIR,
            'ml_models',
            'weights',
            model_name
        )
        
        # Initialize video service
        video_service = VideoDetectionService(model_path=model_path)
        
        # ============== STEP 4: ANALYZE VIDEO ==============
        # Create output directory for heatmaps
        output_dir = os.path.join(settings.MEDIA_ROOT, 'video_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Run video analysis
        result = video_service.analyze_video(
            video_path=full_file_path,
            output_dir=output_dir
        )
        
        # ============== STEP 5: DETERMINE VERDICT ==============
        verdict = result['verdict'].lower()
        confidence = result['confidence']
        
        if confidence < 0.6:
            verdict = 'uncertain'
        
        # ============== STEP 6: SAVE TO DATABASE ==============
        analysis_result = AnalysisResult.objects.create(
            user=request.user,
            file_type='video',
            file_path=file_path,
            verdict=verdict,
            confidence=float(confidence),
            heatmap_path='',  # Videos don't have single heatmap
            metadata={
                'fake_percentage': result.get('fake_percentage', 0),
                'frames_analyzed': result.get('frames_analyzed', 0),
                'fake_frames_count': result.get('fake_frames_count', 0),
                'video_info': result.get('video_info', {}),
                'suspicious_frames': [
                    {
                        'frame_number': f['frame_number'],
                        'confidence': f['confidence'],
                        'heatmap_path': f.get('heatmap_path', '')
                    }
                    for f in result.get('suspicious_frames', [])[:5]
                ],
                'file_size': uploaded_file.size,
                'file_name': uploaded_file.name
            },
            explanation=f"Video analysis complete. {result.get('frames_analyzed', 0)} frames analyzed. "
                       f"{result.get('fake_frames_count', 0)} frames detected as fake "
                       f"({result.get('fake_percentage', 0):.1f}%). "
                       f"The video is {'likely manipulated' if verdict == 'fake' else 'likely authentic'}."
        )
        
        # ============== STEP 7: RETURN RESPONSE ==============
        serializer = AnalysisResultSerializer(analysis_result)
        
        return Response(
            {
                'message': 'Video analyzed successfully',
                'result': serializer.data,
                'details': {
                    'frames_analyzed': result.get('frames_analyzed', 0),
                    'fake_frames_count': result.get('fake_frames_count', 0),
                    'fake_percentage': result.get('fake_percentage', 0),
                    'suspicious_frames': result.get('suspicious_frames', [])[:5]
                }
            },
            status=status.HTTP_200_OK
        )
        
    except FileNotFoundError as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(
            {
                'error': 'An error occurred during video analysis',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_text(request):
    """
    ==============================================================================
    ANALYZE TEXT ENDPOINT
    ==============================================================================
    
    Accepts text content, analyzes it using the trained RoBERTa model,
    and saves results to database.
    
    REQUEST:
        POST /api/analysis/text/
        Content-Type: application/json
        Body: { 'text': 'The text content to analyze...' }
    
    RESPONSE:
        {
            'id': 123,
            'verdict': 'ai-generated',
            'confidence': 0.95,
            'explanation': 'The text is highly likely to be AI-GENERATED...',
            'text_content': 'The text content...'
        }
    
    HOW IT WORKS:
    1. Validate text input
    2. Load text detection service (RoBERTa model)
    3. Analyze text for AI-generated patterns
    4. Save results to database
    5. Return comprehensive results
    """
    try:
        # ============== STEP 1: VALIDATE TEXT INPUT ==============
        text_content = request.data.get('text', '').strip()
        
        if not text_content:
            return Response(
                {'error': 'No text provided. Please provide text content to analyze.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate text length (min 10 characters, max 10,000 characters)
        min_length = 10
        max_length = 10000
        
        if len(text_content) < min_length:
            return Response(
                {'error': f'Text too short. Minimum length: {min_length} characters.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if len(text_content) > max_length:
            return Response(
                {'error': f'Text too long. Maximum length: {max_length} characters.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # ============== STEP 2: TEXT DETECTION SERVICE (singleton, first request loads model) ==============
        text_service = get_text_detection_service()
        
        # ============== STEP 3: ANALYZE TEXT ==============
        # Run prediction with explanation
        result = text_service.predict_with_explanation(text_content)
        
        # ============== STEP 4: DETERMINE VERDICT ==============
        # ML labels: human | ai-generated | uncertain (low confidence → uncertain)
        verdict_ml = result["verdict"]
        confidence = result["confidence"]

        if confidence < 0.6:
            verdict_ml = "uncertain"

        # AnalysisResult.verdict is varchar(10) + choices real/fake/uncertain (same as image/video).
        # ai-generated is 13 chars → DB error if stored raw. Map for UI + DB consistency.
        verdict_store = {
            "human": "real",
            "ai-generated": "fake",
            "uncertain": "uncertain",
            "real": "real",
            "fake": "fake",
        }.get(verdict_ml, "uncertain")

        # ============== STEP 5: SAVE TO DATABASE ==============
        # For text, we store the text content in metadata
        # Note: We don't save the full text in file_path (it's for files only)
        analysis_result = AnalysisResult.objects.create(
            user=request.user,
            file_type="text",
            file_path="",  # No file path for text
            verdict=verdict_store,
            confidence=float(confidence),
            heatmap_path="",  # No heatmap for text
            metadata={
                "text_content": text_content[:500],
                "text_length": len(text_content),
                "text_verdict_ml": verdict_ml,
                "probabilities": result.get("probabilities", {}),
                "predicted_class": result.get("predicted_class", -1),
            },
            explanation=result.get("explanation", ""),
        )
        
        # ============== STEP 6: RETURN RESPONSE ==============
        serializer = AnalysisResultSerializer(analysis_result)
        
        return Response(
            {
                'message': 'Text analyzed successfully',
                'result': serializer.data,
                "details": {
                    "text_length": len(text_content),
                    "probabilities": result.get("probabilities", {}),
                    "verdict_ml": verdict_ml,
                    "verdict": verdict_store,
                    "confidence": confidence,
                }
            },
            status=status.HTTP_200_OK
        )
        
    except FileNotFoundError as e:
        # Model not found (but service will use base RoBERTa)
        return Response(
            {
                'warning': 'Trained text model not found. Using base RoBERTa (not fine-tuned).',
                'error': str(e)
            },
            status=status.HTTP_200_OK  # Still return 200, but with warning
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(
            {
                'error': 'An error occurred during text analysis',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )