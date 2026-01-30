import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SAM2Model:
    """Singleton wrapper for SAM-2 video segmentation"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, checkpoint_dir: str = "/app/models"):
        if self._initialized:
            return
            
        self.checkpoint_dir = checkpoint_dir
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
        
        logger.info(f"SAM2Model singleton created on {self.device}")
    
    def load(self):
        """Lazy load SAM-2"""
        if self.predictor is not None:
            return
            
        try:
            logger.info("Loading SAM-2 Video Predictor...")
            from sam2.build_sam import build_sam2_video_predictor
            
            checkpoint = os.path.join(self.checkpoint_dir, "sam2_hiera_large.pt")
            if not os.path.exists(checkpoint):
                logger.info("Downloading SAM-2 checkpoint...")
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
                import urllib.request
                urllib.request.urlretrieve(url, checkpoint)
            
            self.predictor = build_sam2_video_predictor(
                "sam2_hiera_l.yaml",
                checkpoint,
                device=self.device
            )
            logger.info("✅ SAM-2 loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM-2: {e}")
            raise
    
    def unload(self):
        """Free GPU memory"""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
            torch.cuda.empty_cache()
            logger.info("🧹 SAM-2 unloaded")
    
    def track(self, video_path: str, frame_indices: List[int]) -> Tuple[Dict, List[int]]:
        """
        Track objects through video
        
        Returns:
            video_segments: {frame_idx: {obj_id: mask}}
            object_ids: list of tracked object IDs
        """
        self.load()
        
        try:
            inference_state = self.predictor.init_state(video_path)
            
            # Auto-detect objects using center point
            height, width = 448, 448  # Resized dimensions
            
            self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=np.array([[width//2, height//2]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32)
            )
            
            # Propagate through video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                masks = (out_mask_logits > 0.0).cpu().numpy()
                
                # Ensure masks are 2D (squeeze extra dimensions)
                processed_masks = {}
                for i, obj_id in enumerate(out_obj_ids):
                    mask = masks[i]
                    if mask.ndim > 2:
                        mask = np.squeeze(mask)
                    processed_masks[obj_id] = mask
                
                video_segments[out_frame_idx] = processed_masks
            
            self.predictor.reset_state(inference_state)
            
            # Get all unique object IDs
            all_obj_ids = list(set().union(*[set(v.keys()) for v in video_segments.values()]))
            
            logger.info(f"SAM-2 tracked {len(all_obj_ids)} objects across {len(video_segments)} frames")
            return video_segments, all_obj_ids
            
        except Exception as e:
            logger.error(f"SAM-2 tracking failed: {e}")
            # Fallback: return empty masks for all frames
            return {}, []