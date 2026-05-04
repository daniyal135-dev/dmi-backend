"""
==============================================================================
VIDEO PREPROCESSING MODULE
==============================================================================

WHAT THIS FILE DOES:
Prepares videos BEFORE analyzing them for deepfake detection.
Videos are too large to process directly, so we extract frames (images).

WHY VIDEO PREPROCESSING IS NEEDED:
- Videos are sequences of frames (images)
- Can't feed entire video to model (too much data!)
- Extract representative frames → analyze each frame → aggregate results
- Need to handle different formats, resolutions, frame rates

WHAT PREPROCESSING INCLUDES:
- Video metadata extraction (fps, duration, resolution)
- Frame extraction using FFmpeg (fast) or OpenCV (fallback)
- Keyframe detection (frames with significant changes)
- Uniform frame sampling
- Thumbnail generation

VIDEO DETECTION WORKFLOW:
1. Upload video → Extract frames (1 per second)
2. Preprocess each frame (using ImagePreprocessor)
3. Run deepfake detection on each frame
4. Aggregate results → Final verdict

FILE STRUCTURE:
1. VideoPreprocessor class - Main preprocessing pipeline
2. FFmpeg methods (fast, recommended)
3. OpenCV methods (fallback if FFmpeg fails)
==============================================================================
"""

# ============== IMPORTS ==============
import cv2            # OpenCV: Computer vision library (video processing)
import os             # File operations
import subprocess     # Run shell commands (for FFmpeg)
import numpy as np    # NumPy: Array operations
from PIL import Image # PIL: Image handling
import ffmpeg         # FFmpeg Python wrapper (video manipulation)
from pathlib import Path  # Path handling


class VideoPreprocessor:
    """
    ==============================================================================
    VIDEO PREPROCESSOR
    ==============================================================================
    
    Handles all video preprocessing tasks for deepfake video detection.
    
    TYPICAL WORKFLOW:
    1. Get video metadata (duration, fps, resolution)
    2. Extract frames at specified rate (e.g., 1 frame per second)
    3. Save frames as images
    4. Process each frame with ImagePreprocessor
    5. Run model on each frame
    6. Aggregate results
    
    WHY EXTRACT FRAMES?
    - Videos are just sequences of images (frames)
    - 30 fps video = 30 frames per second (too many!)
    - We sample: 1 fps = 1 frame per second (manageable)
    - 10-second video → 10 frames → 10 predictions → Final result
    
    USAGE EXAMPLE:
        preprocessor = VideoPreprocessor(fps=1, max_frames=30)
        frames = preprocessor.extract_frames('video.mp4', 'output_dir/')
        # Now analyze each frame with image model!
    """
    
    def __init__(self, fps=1, max_frames=30):
        """
        INITIALIZE VIDEO PREPROCESSOR
        
        Args:
            fps (int): Frames per second to extract
                - Default: 1 (one frame per second)
                - Higher = More frames = Better accuracy but slower
                - Lower = Fewer frames = Faster but might miss manipulation
            
            max_frames (int): Maximum number of frames to extract
                - Default: 30 (analyze up to 30 seconds)
                - Prevents extremely long processing for long videos
                - Balance between coverage and processing time
        
        FRAME EXTRACTION STRATEGY:
        - 1 fps for 10-second video → 10 frames
        - 1 fps for 60-second video, max_frames=30 → 30 frames (not 60!)
        """
        self.fps = fps
        self.max_frames = max_frames
        print(f"✓ VideoPreprocessor initialized: {fps} fps, max {max_frames} frames")
    
    def get_video_info(self, video_path):
        """
        GET VIDEO METADATA
        
        Extracts technical information about the video.
        Useful for validation and debugging.
        
        Args:
            video_path (str): Path to video file
        
        Returns:
            dict: Video information containing:
                - width: Video width in pixels
                - height: Video height in pixels
                - fps: Frames per second (e.g., 30.0)
                - duration: Video length in seconds
                - frames: Total number of frames
                - codec: Video codec (e.g., 'h264')
        
        USAGE:
            info = preprocessor.get_video_info('video.mp4')
            print(f"Video: {info['width']}x{info['height']}, {info['duration']}s")
        """
        try:
            # Use ffmpeg.probe to read video metadata
            probe = ffmpeg.probe(video_path)
            
            # Find video stream (videos can have multiple streams: video, audio, subtitles)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # Extract relevant information
            return {
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                
                # FPS: frames per second (r_frame_rate is a fraction like "30/1")
                # eval() converts "30/1" → 30.0
                'fps': eval(video_info['r_frame_rate']),
                
                # Duration in seconds (may not always be present)
                'duration': float(video_info.get('duration', 0)),
                
                # Total number of frames
                'frames': int(video_info.get('nb_frames', 0)),
                
                # Video codec (h264, h265, vp9, etc.)
                'codec': video_info['codec_name']
            }
            
        except Exception as e:
            raise ValueError(f"Error reading video info from {video_path}: {str(e)}")
    
    def extract_frames_ffmpeg(self, video_path, output_dir, fps=None):
        """
        EXTRACT FRAMES USING FFMPEG (RECOMMENDED METHOD)
        
        FFmpeg is the industry standard for video processing.
        It's MUCH faster than OpenCV and handles more formats.
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save extracted frames
            fps (int): Frames per second (None = use self.fps)
        
        Returns:
            list: Paths to extracted frame images
        
        HOW IT WORKS:
        1. FFmpeg reads video
        2. Extracts frames at specified fps (e.g., 1 per second)
        3. Saves as JPEG images (frame_0001.jpg, frame_0002.jpg, ...)
        4. Returns list of frame paths
        
        FFMPEG COMMAND EQUIVALENT:
        ffmpeg -i video.mp4 -vf "fps=1" -frames:v 30 output/frame_%04d.jpg
        
        WHY FFMPEG?
        - 10x faster than OpenCV
        - Better format support
        - More reliable with corrupted videos
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use instance fps if not provided
        if fps is None:
            fps = self.fps
        
        # Output pattern: frame_0001.jpg, frame_0002.jpg, etc.
        # %04d = 4-digit number with zero padding
        output_pattern = os.path.join(output_dir, 'frame_%04d.jpg')
        
        try:
            # ============== FFMPEG FRAME EXTRACTION ==============
            # Build and execute FFmpeg command
            (
                ffmpeg
                .input(video_path)                           # Input video
                .filter('fps', fps=fps)                      # Extract at specified fps
                .output(
                    output_pattern,                           # Output pattern
                    format='image2',                          # Image sequence format
                    vframes=self.max_frames                   # Maximum frames to extract
                )
                .overwrite_output()                           # Overwrite existing files
                .run(capture_stdout=True, capture_stderr=True)  # Run command
            )
            
            # ============== COLLECT FRAME PATHS ==============
            # List all JPG files in output directory
            frame_paths = sorted([
                os.path.join(output_dir, f) 
                for f in os.listdir(output_dir) 
                if f.endswith('.jpg')
            ])
            
            # Limit to max_frames (in case more were extracted)
            frame_paths = frame_paths[:self.max_frames]
            
            print(f"✓ Extracted {len(frame_paths)} frames using FFmpeg")
            return frame_paths
            
        except ffmpeg.Error as e:
            # FFmpeg failed - raise error with stderr message
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise ValueError(f"FFmpeg error: {error_msg}")
    
    def extract_frames_opencv(self, video_path, output_dir):
        """
        EXTRACT FRAMES USING OPENCV (FALLBACK METHOD)
        
        OpenCV is slower than FFmpeg but doesn't require FFmpeg installation.
        Use this as a fallback if FFmpeg isn't available.
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save frames
        
        Returns:
            list: Paths to extracted frame images
        
        HOW IT WORKS:
        1. Open video with cv2.VideoCapture
        2. Read frames one by one
        3. Save every Nth frame (based on fps setting)
        4. Stop at max_frames
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # ============== OPEN VIDEO ==============
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video FPS
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame interval
        # Example: video is 30 fps, we want 1 fps → save every 30th frame
        frame_interval = int(video_fps / self.fps) if self.fps < video_fps else 1
        
        # ============== EXTRACT FRAMES ==============
        frame_paths = []
        frame_count = 0      # Total frames read
        saved_count = 0      # Frames saved
        
        while cap.isOpened() and saved_count < self.max_frames:
            # Read next frame
            ret, frame = cap.read()
            
            # If frame reading failed, we've reached the end
            if not ret:
                break
            
            # Save frame at specified interval
            if frame_count % frame_interval == 0:
                # Create filename with zero-padded number
                frame_filename = f'frame_{saved_count:04d}.jpg'
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Save frame as JPEG
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
            
            frame_count += 1
        
        # Release video capture
        cap.release()
        
        print(f"✓ Extracted {len(frame_paths)} frames using OpenCV")
        return frame_paths
    
    def extract_frames(self, video_path, output_dir, method='ffmpeg'):
        """
        EXTRACT FRAMES (SMART METHOD SELECTION)
        
        Tries FFmpeg first (faster), falls back to OpenCV if FFmpeg fails.
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save frames
            method (str): Preferred method ('ffmpeg' or 'opencv')
        
        Returns:
            list: Paths to extracted frame images
        
        USAGE (RECOMMENDED):
            frames = preprocessor.extract_frames('video.mp4', 'frames/')
            # Automatically uses best available method!
        """
        if method == 'ffmpeg':
            try:
                # Try FFmpeg first (faster, better)
                return self.extract_frames_ffmpeg(video_path, output_dir)
            except Exception as e:
                # FFmpeg failed, fall back to OpenCV
                print(f"⚠ FFmpeg failed: {str(e)}")
                print("  Falling back to OpenCV...")
                return self.extract_frames_opencv(video_path, output_dir)
        else:
            # User explicitly requested OpenCV
            return self.extract_frames_opencv(video_path, output_dir)
    
    def extract_keyframes(self, video_path, output_dir, threshold=0.3):
        """
        EXTRACT ONLY KEYFRAMES (SMART SAMPLING)
        
        Instead of uniform sampling (every second), extract frames with significant changes.
        This is smarter for videos with varying content.
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save keyframes
            threshold (float): Difference threshold for keyframe detection
                - Higher = More selective (fewer keyframes)
                - Lower = More permissive (more keyframes)
                - Range: 0.0 to 1.0
        
        Returns:
            list: Paths to extracted keyframes
        
        WHAT IS A KEYFRAME?
        A frame that's significantly different from the previous frame.
        Example: Scene changes, new objects appearing, etc.
        
        WHY KEYFRAMES?
        - Reduces redundancy (no need for nearly identical frames)
        - Focuses on important moments
        - Better for videos with static scenes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        prev_frame = None       # Previous frame (for comparison)
        keyframe_paths = []
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened() and saved_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for comparison (faster, sufficient)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Decide if this is a keyframe
            if prev_frame is None:
                # First frame is always a keyframe
                is_keyframe = True
            else:
                # Calculate difference from previous frame
                diff = cv2.absdiff(prev_frame, gray)
                
                # Average difference across all pixels (normalized to 0-1)
                diff_score = np.mean(diff) / 255.0
                
                # Is difference above threshold?
                is_keyframe = diff_score > threshold
            
            # Save if keyframe
            if is_keyframe:
                frame_filename = f'keyframe_{saved_count:04d}.jpg'
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                keyframe_paths.append(frame_path)
                saved_count += 1
                
                # Update previous frame
                prev_frame = gray
            
            frame_count += 1
        
        cap.release()
        print(f"✓ Extracted {len(keyframe_paths)} keyframes (threshold={threshold})")
        return keyframe_paths
    
    def sample_uniform_frames(self, video_path, n_frames=10):
        """
        SAMPLE N FRAMES UNIFORMLY FROM VIDEO
        
        Extracts exactly N frames evenly distributed across the video.
        Good for getting a representative sample of the entire video.
        
        Args:
            video_path (str): Path to video file
            n_frames (int): Number of frames to sample
        
        Returns:
            list: List of frame numpy arrays (not saved to disk)
        
        EXAMPLE:
        Video has 300 frames, n_frames=10
        → Sample frames at indices: [0, 33, 66, 99, 132, 165, 198, 231, 264, 297]
        
        WHEN TO USE:
        - When you want exactly N frames (not time-based)
        - When you want to cover the entire video evenly
        - For quick analysis without saving files
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get total number of frames in video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        # linspace creates evenly spaced numbers
        # Example: n_frames=10, total=300 → [0, 33, 66, ..., 297]
        if total_frames < n_frames:
            # Video has fewer frames than requested
            indices = range(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        # Extract frames at calculated indices
        frames = []
        for idx in indices:
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            # Read frame
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        print(f"✓ Sampled {len(frames)} frames uniformly")
        return frames
    
    def create_video_thumbnail(self, video_path, output_path, timestamp=1.0):
        """
        CREATE THUMBNAIL FROM VIDEO
        
        Extracts a single frame at specified timestamp and saves as thumbnail.
        Useful for video previews in the UI.
        
        Args:
            video_path (str): Path to video file
            output_path (str): Path to save thumbnail
            timestamp (float): Time in seconds to capture thumbnail
                - Default: 1.0 (1 second into video)
        
        Returns:
            str: Path to saved thumbnail
        
        USAGE:
            thumbnail = preprocessor.create_video_thumbnail(
                'video.mp4',
                'thumbnails/video_thumb.jpg',
                timestamp=2.0  # 2 seconds into video
            )
        """
        try:
            # Use FFmpeg to extract single frame at timestamp
            (
                ffmpeg
                .input(video_path, ss=timestamp)      # ss = seek to timestamp
                .filter('scale', 320, -1)             # Scale to 320px width (maintain aspect ratio)
                .output(output_path, vframes=1)       # Output 1 frame
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            print(f"✓ Created thumbnail at {timestamp}s")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise ValueError(f"Error creating thumbnail: {error_msg}")


# ==============================================================================
# END OF FILE
# ==============================================================================
# 
# SUMMARY:
# This file provides the VideoPreprocessor class for extracting frames
# from videos before deepfake detection.
# 
# MAIN FUNCTIONS:
# 1. extract_frames() - Extract frames at specified fps (most used!)
# 2. get_video_info() - Get video metadata
# 3. extract_keyframes() - Smart keyframe extraction
# 4. sample_uniform_frames() - Sample N frames evenly
# 5. create_video_thumbnail() - Generate video preview
# 
# TYPICAL USAGE:
#     preprocessor = VideoPreprocessor(fps=1, max_frames=30)
#     frames = preprocessor.extract_frames('video.mp4', 'frames/')
#     # Now process each frame with ImagePreprocessor and model!
# ==============================================================================
