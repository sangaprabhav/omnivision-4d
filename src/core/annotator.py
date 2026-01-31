import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import tempfile
import os
from pathlib import Path

from src.models.sam2_model import SAM2Model, SAM2Result
from src.models.depth_model import DepthModel, DepthResult
from src.models.cosmos_model import CosmosModel, CosmosResult
from src.models.fusion import fuse_4d
from src.core.adaptive_sampling import sample_frames_adaptive
from src.core.mask_quality import filter_low_quality_masks
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class OmnivisionAnnotator:
    """Production 4D annotation pipeline"""
    
    def __init__(self):
        self.sam2 = SAM2Model()
        self.depth = DepthModel()
        self.cosmos = CosmosModel()
        
    def extract_frames(
        self,
        video_path: str,
        use_adaptive_sampling: bool = True,
        max_frames: int = None
    ) -> Tuple[List[Image.Image], float, List[int], Dict]:
        """
        Production-grade frame extraction with adaptive sampling

        Args:
            video_path: Path to video file
            use_adaptive_sampling: If True, use motion-based adaptive sampling
            max_frames: Maximum frames to extract (defaults to settings.MAX_FRAMES)

        Returns:
            frames: List of PIL images
            fps: Video frame rate
            indices: Frame indices selected
            sampling_metadata: Metadata about sampling strategy
        """
        if max_frames is None:
            max_frames = settings.MAX_FRAMES

        # Use adaptive sampling for intelligent frame selection
        if use_adaptive_sampling:
            indices, sampling_metadata = sample_frames_adaptive(
                video_path,
                max_frames=max_frames,
                enable_adaptive=True
            )
        else:
            # Fallback to uniform sampling
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()
            sampling_metadata = {
                "total_frames": total,
                "sampled_frames": len(indices),
                "fps": fps_val,
                "sampling_strategy": "uniform"
            }

        # Extract frames at selected indices
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize((448, 448))
                frames.append(pil)

        cap.release()

        logger.info(
            f"Extracted {len(frames)} frames from {video_path} "
            f"(strategy: {sampling_metadata['sampling_strategy']})"
        )

        return frames, fps, indices, sampling_metadata
    
    def process(
        self,
        video_path: str,
        prompt: str = None,
        job_id: str = None,
        use_adaptive_sampling: bool = True,
        enable_quality_filtering: bool = True,
        grid_size: int = 3,
        min_quality_score: float = 0.4
    ) -> Dict:
        """
        End-to-end 4D annotation with Phase 1 enhancements

        Args:
            video_path: Path to video file
            prompt: Custom prompt for Cosmos (optional)
            job_id: Job identifier (auto-generated if None)
            use_adaptive_sampling: Enable motion-based adaptive frame sampling
            enable_quality_filtering: Enable mask quality filtering
            grid_size: Grid size for multi-object auto-detection (e.g., 3 = 3x3 = 9 objects)
            min_quality_score: Minimum quality score for mask filtering (0-1)

        Returns:
            Annotated 4D result with confidence scores
        """
        job_id = job_id or os.urandom(4).hex()
        logger.info(f"[{job_id}] Starting annotation with Phase 1 enhancements")

        try:
            # Validate video
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")

            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            if size_mb > settings.MAX_UPLOAD_SIZE / (1024 * 1024):
                raise ValueError(f"Video too large: {size_mb:.1f}MB")

            # Step 1: Adaptive Frame Sampling
            logger.info(f"[{job_id}] Extracting frames with adaptive sampling...")
            frames, fps, indices, sampling_metadata = self.extract_frames(
                video_path,
                use_adaptive_sampling=use_adaptive_sampling
            )

            # Step 2: SAM-2 Multi-Object Tracking with Confidence
            logger.info(f"[{job_id}] Running SAM-2 with multi-object detection (grid: {grid_size}x{grid_size})...")
            sam_result: SAM2Result = self.sam2.track(
                video_path,
                indices,
                auto_detect=True,
                grid_size=grid_size
            )
            self.sam2.unload()  # Free VRAM

            logger.info(
                f"[{job_id}] SAM-2 detected {len(sam_result.object_ids)} objects "
                f"(avg confidence: {self._avg_dict_values(sam_result.confidence_scores):.3f})"
            )

            # Step 3: Mask Quality Filtering
            if enable_quality_filtering:
                logger.info(f"[{job_id}] Filtering low-quality masks...")
                sam_result.video_segments = filter_low_quality_masks(
                    sam_result.video_segments,
                    min_quality_score=min_quality_score,
                    min_temporal_iou=0.5
                )

                # Update object IDs after filtering
                sam_result.object_ids = list(
                    set().union(*[set(v.keys()) for v in sam_result.video_segments.values()])
                )

                logger.info(f"[{job_id}] After filtering: {len(sam_result.object_ids)} objects retained")

            # Step 4: Depth Estimation with Uncertainty
            logger.info(f"[{job_id}] Running depth estimation with uncertainty quantification...")
            depth_result: DepthResult = self.depth.estimate(frames, compute_uncertainty=True)
            self.depth.unload()

            logger.info(
                f"[{job_id}] Depth estimation complete "
                f"(avg confidence: {np.mean(depth_result.confidence_scores):.3f})"
            )

            # Step 5: Cosmos Semantic Annotation with Confidence
            logger.info(f"[{job_id}] Running Cosmos semantic annotation...")
            if not prompt:
                prompt = "Track objects in the video and output their trajectories as JSON with object labels and behaviors"

            cosmos_result: CosmosResult = self.cosmos.annotate(
                frames,
                prompt,
                return_logits=False  # Set to True for logit-based confidence (slower)
            )
            self.cosmos.unload()

            logger.info(
                f"[{job_id}] Cosmos annotation complete "
                f"(confidence: {cosmos_result.confidence_score:.3f}, parsed: {cosmos_result.parse_success})"
            )

            # Step 6: 4D Fusion with Confidence Propagation
            logger.info(f"[{job_id}] Fusing modalities into 4D trajectories...")
            result = fuse_4d(
                sam_result.video_segments,
                depth_result.depth_maps,
                cosmos_result.annotation,
                indices,
                fps,
                self.depth
            )

            # Add Phase 1 metadata
            result["job_id"] = job_id
            result["status"] = "success"

            result["annotation_metadata"]["phase1_enhancements"] = {
                "adaptive_sampling": use_adaptive_sampling,
                "quality_filtering": enable_quality_filtering,
                "multi_object_detection": True,
                "confidence_propagation": True
            }

            result["annotation_metadata"]["sampling_info"] = sampling_metadata

            result["annotation_metadata"]["model_confidence"] = {
                "sam2_avg_confidence": self._avg_dict_values(sam_result.confidence_scores),
                "depth_avg_confidence": float(np.mean(depth_result.confidence_scores))
                if depth_result.confidence_scores else 0.0,
                "cosmos_confidence": cosmos_result.confidence_score
            }

            # Add quality metrics to each object
            for i, obj in enumerate(result.get("objects", [])):
                obj_id = sam_result.object_ids[i] if i < len(sam_result.object_ids) else i + 1

                # Collect confidence scores for this object
                obj_confidences = []
                for frame_idx in sam_result.confidence_scores:
                    if obj_id in sam_result.confidence_scores[frame_idx]:
                        obj_confidences.append(sam_result.confidence_scores[frame_idx][obj_id])

                obj["quality_metrics"] = {
                    "sam2_avg_confidence": float(np.mean(obj_confidences))
                    if obj_confidences else 0.0,
                    "tracking_frames": len(obj_confidences)
                }

            logger.info(f"[{job_id}] Annotation complete with {len(result.get('objects', []))} objects")
            return result

        except Exception as e:
            logger.error(f"[{job_id}] Annotation failed: {e}", exc_info=True)
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }

    def _avg_dict_values(self, nested_dict: Dict[int, Dict[int, float]]) -> float:
        """Compute average of all values in nested dictionary"""
        all_values = []
        for inner_dict in nested_dict.values():
            all_values.extend(inner_dict.values())
        return float(np.mean(all_values)) if all_values else 0.0

# Global singleton (one instance per worker)
annotator = OmnivisionAnnotator()