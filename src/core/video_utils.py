"""
Video processing utilities
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def get_video_info(video_path: str) -> Dict:
    """Get video metadata without loading frames"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    
    cap.release()
    return info

def extract_frames(
    video_path: str, 
    max_frames: int = 32, 
    target_resolution: Tuple[int, int] = (448, 448),
    sampling_strategy: str = "uniform"  # "uniform", "motion", "keyframe"
) -> Tuple[List[Image.Image], float, List[int]]:
    """
    Extract frames from video with various sampling strategies
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        target_resolution: (width, height) to resize to
        sampling_strategy: How to sample frames
    
    Returns:
        frames: List of PIL Images
        fps: Original video FPS
        indices: Frame indices that were extracted
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames")
    
    # Calculate which frames to extract
    if sampling_strategy == "uniform":
        # Evenly spaced frames
        indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    elif sampling_strategy == "motion":
        # TODO: Implement motion-based sampling (extract more frames where motion is high)
        indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {idx}")
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target resolution
        frame_resized = cv2.resize(frame_rgb, target_resolution)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_resized)
        frames.append(pil_image)
    
    cap.release()
    
    logger.info(f"Extracted {len(frames)} frames from {video_path} (strategy: {sampling_strategy})")
    return frames, fps, indices.tolist()

def save_debug_video(
    frames: List[np.ndarray], 
    trajectories: Dict, 
    output_path: str,
    fps: int = 5
):
    """
    Save visualization video with trajectories overlaid
    Useful for debugging/validation
    """
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i, frame in enumerate(frames):
        # Draw trajectories
        for obj_id, traj in trajectories.items():
            if i < len(traj):
                x, y = int(traj[i]['x']), int(traj[i]['y'])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(obj_id), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Debug video saved to {output_path}")