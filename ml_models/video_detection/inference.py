"""
Video Detection Service – uses ViT (phase4_replay_best) via ImageDetectionService.
Extracts frames with FFmpeg, runs ViT on each frame, aggregates → verdict.
"""

import os
import shutil
import tempfile
from typing import Dict, List
from pathlib import Path

from ml_models.preprocessing.video_preprocessor import VideoPreprocessor
from ml_models.image_detection.inference import ImageDetectionService


class VideoDetectionService:
    """
    Detects deepfake videos by analyzing individual frames with ViT.
    
    Flow:
      Video → FFmpeg (1 fps, max 30 frames) → ViT per frame → Aggregate → verdict
    
    Aggregation:
      >50% frames fake → video fake; confidence = avg of frame confidences.
    """

    def __init__(self, model_path: str, fps: int = 1, max_frames: int = 30):
        print("=" * 60)
        print("INITIALIZING VIDEO DETECTION SERVICE (ViT)")
        print("=" * 60)
        print(f"Loading ViT model from: {model_path}")
        self.image_service = ImageDetectionService(model_path=model_path)
        print(f"Video preprocessor: {fps} fps, max {max_frames} frames")
        self.video_preprocessor = VideoPreprocessor(fps=fps, max_frames=max_frames)
        self.fps = fps
        self.max_frames = max_frames
        print("Video Detection Service ready!")
        print("=" * 60)

    def analyze_video(self, video_path: str, output_dir: str) -> Dict:
        """
        Analyze video: extract frames → ViT predict each → aggregate.
        Returns dict with verdict ('fake'/'real'), confidence (0-1), etc.
        """
        print(f"\nANALYZING VIDEO: {video_path}")

        # Step 1: video info
        try:
            video_info = self.video_preprocessor.get_video_info(video_path)
            print(f"  {video_info['width']}x{video_info['height']}, "
                  f"{video_info['duration']:.1f}s, {video_info['fps']} fps")
        except Exception as e:
            print(f"  Could not get video info: {e}")
            video_info = {'width': 0, 'height': 0, 'duration': 0, 'fps': 0}

        # Step 2: extract frames
        frames_dir = tempfile.mkdtemp(prefix='video_frames_')
        try:
            frame_paths = self.video_preprocessor.extract_frames(video_path, frames_dir)
            print(f"  Extracted {len(frame_paths)} frames")
            if not frame_paths:
                raise ValueError("No frames could be extracted from video")
        except Exception as e:
            shutil.rmtree(frames_dir, ignore_errors=True)
            raise ValueError(f"Frame extraction failed: {e}")

        # Step 3: analyze each frame with ViT
        heatmaps_dir = os.path.join(output_dir, 'frame_heatmaps')
        os.makedirs(heatmaps_dir, exist_ok=True)

        frame_results: List[Dict] = []
        fake_count = 0
        total_fake_prob = 0.0

        for i, frame_path in enumerate(frame_paths):
            try:
                res = self.image_service.predict_with_heatmap(frame_path, heatmaps_dir)
                verdict_lower = res['verdict'].lower()
                conf = float(res['confidence'])
                probs = res['probabilities']

                frame_results.append({
                    'frame_number': i + 1,
                    'frame_path': frame_path,
                    'verdict': verdict_lower,
                    'confidence': conf,
                    'probabilities': probs,
                    'heatmap_path': res.get('heatmap_path'),
                })

                if verdict_lower == 'fake':
                    fake_count += 1
                    total_fake_prob += probs.get('fake', conf)

                print(f"  Frame {i+1}/{len(frame_paths)}: {verdict_lower} ({conf:.2%})")
            except Exception as e:
                print(f"  Frame {i+1}/{len(frame_paths)}: ERROR ({e})")
                frame_results.append({
                    'frame_number': i + 1,
                    'frame_path': frame_path,
                    'verdict': 'error',
                    'confidence': 0,
                    'error': str(e),
                })

        # Step 4: aggregate
        valid_frames = [r for r in frame_results if r['verdict'] != 'error']
        total_valid = len(valid_frames)
        if total_valid == 0:
            shutil.rmtree(frames_dir, ignore_errors=True)
            raise ValueError("No frames could be analyzed successfully")

        fake_percentage = (fake_count / total_valid) * 100.0
        FAKE_THRESHOLD = 50

        if fake_percentage >= FAKE_THRESHOLD:
            final_verdict = 'fake'
            avg_fake_prob = total_fake_prob / fake_count if fake_count else 0.0
            final_confidence = avg_fake_prob
        else:
            final_verdict = 'real'
            real_confs = [r['confidence'] for r in valid_frames if r['verdict'] == 'real']
            final_confidence = sum(real_confs) / len(real_confs) if real_confs else (1.0 - fake_percentage / 100.0)

        print(f"  Fake frames: {fake_count}/{total_valid} ({fake_percentage:.1f}%)")
        print(f"  Verdict: {final_verdict} (confidence {final_confidence:.2%})")

        suspicious_frames = sorted(
            [r for r in valid_frames if r['verdict'] == 'fake'],
            key=lambda x: x['probabilities'].get('fake', 0),
            reverse=True,
        )[:5]

        # Step 5: cleanup temp frames
        shutil.rmtree(frames_dir, ignore_errors=True)

        return {
            'verdict': final_verdict,
            'confidence': round(final_confidence, 4),
            'fake_percentage': round(fake_percentage, 2),
            'frames_analyzed': total_valid,
            'fake_frames_count': fake_count,
            'frame_results': frame_results,
            'suspicious_frames': suspicious_frames,
            'video_info': video_info,
            'analysis_settings': {
                'fps': self.fps,
                'max_frames': self.max_frames,
                'fake_threshold': FAKE_THRESHOLD,
            },
        }

    def get_frame_timeline(self, frame_results: List[Dict]) -> List[Dict]:
        """Timeline of per-frame predictions for frontend visualization."""
        return [
            {
                'frame': r['frame_number'],
                'is_fake': r['verdict'] == 'fake',
                'fake_probability': r['probabilities'].get('fake', 0),
                'real_probability': r['probabilities'].get('real', 0),
            }
            for r in frame_results
            if r['verdict'] != 'error'
        ]


_video_service_instance = None

def get_video_service(model_path: str = None) -> VideoDetectionService:
    """Singleton: load ViT model once, reuse for all video analyses."""
    global _video_service_instance
    if _video_service_instance is None:
        if model_path is None:
            from django.conf import settings
            model_name = getattr(settings, 'MODEL_NAME', 'phase4_replay_best.pth')
            model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'weights', model_name)
        _video_service_instance = VideoDetectionService(model_path=model_path)
    return _video_service_instance

