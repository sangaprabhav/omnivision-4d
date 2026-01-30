import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import tempfile
import os
from pathlib import Path

from src.models.sam2_model import SAM2Model  # Create similar to CosmosModel
from src.models.depth_model import DepthModel
from src.models.cosmos_model import CosmosModel
from src.models.fusion import fuse_4d
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class OmnivisionAnnotator:
    """Production 4D annotation pipeline"""
    
    def __init__(self):
        self.sam2 = SAM2Model()
        self.depth = DepthModel()
        self.cosmos = CosmosModel()
        
    def extract_frames(self, video_path: str) -> Tuple[List[Image.Image], float, List[int]]:
        """Production-grade frame extraction"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate indices
        indices = np.linspace(0, total-1, settings.MAX_FRAMES, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize((448, 448))
                frames.append(pil)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames, fps, indices.tolist()
    
    def process(self, video_path: str, prompt: str = None, job_id: str = None) -> Dict:
        """End-to-end 4D annotation"""
        job_id = job_id or os.urandom(4).hex()
        logger.info(f"[{job_id}] Starting annotation")
        
        try:
            # Validate video
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            size_mb = os.path.getsize(video_path) / (1024*1024)
            if size_mb > settings.MAX_UPLOAD_SIZE / (1024*1024):
                raise ValueError(f"Video too large: {size_mb:.1f}MB")
            
            # Step 1: Frames
            frames, fps, indices = self.extract_frames(video_path)
            
            # Step 2: SAM-2 (geometry)
            logger.info(f"[{job_id}] Running SAM-2...")
            sam_masks, obj_ids = self.sam2.track(video_path, indices)
            self.sam2.unload()  # Free VRAM
            
            # Step 3: Depth (3D)
            logger.info(f"[{job_id}] Running Depth...")
            depth_maps = self.depth.estimate(frames)
            self.depth.unload()
            
            # Step 4: Cosmos (semantics)
            logger.info(f"[{job_id}] Running Cosmos...")
            if not prompt:
                prompt = "Track robot movements and output trajectories as JSON"
            cosmos_out = self.cosmos.annotate(frames, prompt)
            # Keep Cosmos loaded? No, unload to save VRAM for next request
            self.cosmos.unload()
            
            # Step 5: Fusion
            logger.info(f"[{job_id}] Fusing modalities...")
            result = fuse_4d(sam_masks, depth_maps, cosmos_out, indices, fps)
            result["job_id"] = job_id
            result["status"] = "success"
            
            logger.info(f"[{job_id}] Annotation complete")
            return result
            
        except Exception as e:
            logger.error(f"[{job_id}] Annotation failed: {e}")
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }

# Global singleton (one instance per worker)
annotator = OmnivisionAnnotator()