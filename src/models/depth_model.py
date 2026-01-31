
import torch
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Depth estimation result with uncertainty metrics"""
    depth_maps: List[np.ndarray]  # List of depth maps in meters
    uncertainty_maps: List[np.ndarray]  # Uncertainty/variance for each pixel
    confidence_scores: List[float]  # Overall confidence per frame (0-1)

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
    
    def estimate(
        self,
        frames: List[Image.Image],
        compute_uncertainty: bool = True
    ) -> DepthResult:
        """
        Estimate metric depth for each frame with uncertainty quantification

        Args:
            frames: List of PIL images
            compute_uncertainty: If True, estimate per-pixel uncertainty

        Returns:
            DepthResult with depth_maps, uncertainty_maps, confidence_scores
        """
        self.load()

        try:
            depth_maps = []
            uncertainty_maps = []
            confidence_scores = []

            for i, frame in enumerate(frames):
                result = self.pipe(frame)
                depth = np.array(result['predicted_depth'])

                # Validate range (should be 0.1m to 100m for indoor/outdoor)
                valid_mask = (depth >= 0.1) & (depth <= 100.0)
                if not np.all(valid_mask):
                    logger.warning(f"Frame {i}: Suspicious depth values detected, clipping")
                    depth = np.clip(depth, 0.1, 100.0)

                # Compute uncertainty (variance estimation)
                if compute_uncertainty:
                    uncertainty = self._estimate_uncertainty(depth, valid_mask)
                else:
                    uncertainty = np.zeros_like(depth)

                # Compute overall confidence for this frame
                confidence = self._compute_frame_confidence(depth, uncertainty, valid_mask)

                depth_maps.append(depth)
                uncertainty_maps.append(uncertainty)
                confidence_scores.append(confidence)

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Processed {i+1}/{len(frames)} depth maps "
                        f"(avg confidence: {np.mean(confidence_scores):.3f})"
                    )

            logger.info(
                f"✅ Generated {len(depth_maps)} metric depth maps "
                f"(mean confidence: {np.mean(confidence_scores):.3f})"
            )

            return DepthResult(
                depth_maps=depth_maps,
                uncertainty_maps=uncertainty_maps,
                confidence_scores=confidence_scores
            )

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            # Return empty result
            return DepthResult(
                depth_maps=[],
                uncertainty_maps=[],
                confidence_scores=[]
            )

    def _estimate_uncertainty(self, depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Estimate per-pixel depth uncertainty

        Uses local variance and edge detection to estimate confidence.
        Higher variance at edges = lower confidence.
        """
        # Compute local variance using a sliding window
        from scipy.ndimage import uniform_filter

        # Local mean
        local_mean = uniform_filter(depth, size=5)

        # Local variance
        local_var = uniform_filter(depth**2, size=5) - local_mean**2

        # Edge magnitude (high gradient = low confidence)
        grad_y, grad_x = np.gradient(depth)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Combine variance and edge information
        # Normalize to [0, 1] range
        uncertainty = local_var + 0.5 * edge_magnitude

        # Normalize
        if uncertainty.max() > 0:
            uncertainty = uncertainty / uncertainty.max()

        # Mark invalid regions with high uncertainty
        uncertainty[~valid_mask] = 1.0

        return uncertainty

    def _compute_frame_confidence(
        self,
        depth: np.ndarray,
        uncertainty: np.ndarray,
        valid_mask: np.ndarray
    ) -> float:
        """
        Compute overall confidence score for a depth map

        Returns value in [0, 1] where 1.0 = very confident
        """
        # Average uncertainty (lower is better)
        avg_uncertainty = uncertainty[valid_mask].mean() if valid_mask.sum() > 0 else 1.0

        # Fraction of valid pixels
        valid_ratio = valid_mask.sum() / valid_mask.size

        # Depth range consistency (indoor scenes typically 0.5-10m)
        depth_std = depth[valid_mask].std() if valid_mask.sum() > 0 else 100.0
        depth_consistency = 1.0 - min(1.0, depth_std / 50.0)  # Normalize

        # Combined confidence
        confidence = (
            0.5 * (1.0 - avg_uncertainty) +
            0.3 * valid_ratio +
            0.2 * depth_consistency
        )

        return float(np.clip(confidence, 0.0, 1.0))
    
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