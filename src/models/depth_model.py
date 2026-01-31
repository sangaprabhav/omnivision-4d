
import torch
import numpy as np
from typing import List
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class DepthModel:
    """Singleton wrapper for ZoeDepth (metric depth estimation)"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_name: str = "Intel/zoedepth-nyu-kitti"):
        if self._initialized:
            return
            
        self.model_name = model_name
        self.pipe = None
        self.device = 0 if torch.cuda.is_available() else -1
        self._initialized = True
        
        logger.info("DepthModel singleton created")
    
    def load(self):
        """Lazy load ZoeDepth"""
        if self.pipe is not None:
            return
            
        try:
            logger.info(f"Loading ZoeDepth ({self.model_name})...")
            from transformers import pipeline
            
            self.pipe = pipeline(
                task="depth-estimation",
                model=self.model_name,
                device=self.device
            )
            logger.info("✅ Depth model loaded (metric in meters)")
            
        except Exception as e:
            logger.error(f"Failed to load Depth model: {e}")
            raise
    
    def unload(self):
        """Free GPU memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            logger.info("🧹 Depth model unloaded")
    
    def estimate(self, frames: List[Image.Image]) -> List[np.ndarray]:
        """
        Estimate metric depth for each frame
        
        Returns:
            List of depth maps (numpy arrays) in meters
        """
        self.load()
        
        try:
            depth_maps = []
            
            for i, frame in enumerate(frames):
                result = self.pipe(frame)
                depth = np.array(result['predicted_depth'])
                
                # ZoeDepth outputs metric depth in meters
                # Validate range (should be 0.1m to 100m for indoor/outdoor)
                if np.any(depth < 0) or np.any(depth > 1000):
                    logger.warning(f"Frame {i}: Suspicious depth values detected, clipping")
                    depth = np.clip(depth, 0.1, 100.0)
                
                depth_maps.append(depth)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(frames)} depth maps")
            
            logger.info(f"✅ Generated {len(depth_maps)} metric depth maps")
            return depth_maps
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            # Return empty list to trigger fallback
            return []
    
    def get_depth_at_point(self, depth_map: np.ndarray, x: float, y: float) -> float:
        """
        Bilinear interpolation of depth at sub-pixel coordinates
        
        Args:
            depth_map: H x W depth array
            x, y: Float coordinates (can be sub-pixel)
        
        Returns:
            Interpolated depth value in meters
        """
        h, w = depth_map.shape
        
        # Clamp to valid range
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Get integer coordinates
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        # Calculate weights
        dx, dy = x - x0, y - y0
        
        # Bilinear interpolation
        depth = (1 - dx) * (1 - dy) * depth_map[y0, x0] + \
                dx * (1 - dy) * depth_map[y0, x1] + \
                (1 - dx) * dy * depth_map[y1, x0] + \
                dx * dy * depth_map[y1, x1]
        
        return float(depth)