import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)


@dataclass
class SAM2Result:
    """SAM-2 tracking result with confidence scores"""
    video_segments: Dict[int, Dict[int, np.ndarray]]  # {frame_idx: {obj_id: mask}}
    object_ids: List[int]
    confidence_scores: Dict[int, Dict[int, float]]  # {frame_idx: {obj_id: confidence}}
    iou_predictions: Dict[int, Dict[int, float]]  # {frame_idx: {obj_id: iou_pred}}

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
    
    def track(
        self,
        video_path: str,
        frame_indices: List[int],
        init_points: Optional[List[Tuple[float, float]]] = None,
        auto_detect: bool = True,
        grid_size: int = 3,
        min_mask_area: float = 0.001
    ) -> SAM2Result:
        """
        Track objects through video with multi-object support

        Args:
            video_path: Path to video
            frame_indices: List of frame indices to process
            init_points: Optional list of (x, y) initialization points. If None, auto-detect
            auto_detect: If True, automatically detect objects using grid-based sampling
            grid_size: Grid size for auto-detection (e.g., 3x3 = 9 points)
            min_mask_area: Minimum mask area as fraction of frame (filter tiny objects)

        Returns:
            SAM2Result with video_segments, object_ids, confidence_scores, iou_predictions
        """
        self.load()

        try:
            inference_state = self.predictor.init_state(video_path)

            # Auto-detect objects using grid-based initialization
            height, width = 448, 448  # Resized dimensions

            if init_points is None and auto_detect:
                # Generate grid of initialization points
                init_points = self._generate_grid_points(width, height, grid_size)
                logger.info(f"Auto-detecting objects using {len(init_points)} grid points")
            elif init_points is None:
                # Fallback: single center point
                init_points = [(width // 2, height // 2)]

            # Initialize tracking for each point
            obj_id = 1
            for point in init_points:
                self.predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=np.array([[point[0], point[1]]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32)
                )
                obj_id += 1

            # Propagate through video and extract confidence scores
            video_segments = {}
            confidence_scores = {}
            iou_predictions = {}

            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                masks = (out_mask_logits > 0.0).cpu().numpy()

                # Extract mask logits for confidence estimation
                mask_logits_np = out_mask_logits.cpu().numpy()

                # Process each object
                processed_masks = {}
                frame_confidences = {}
                frame_ious = {}

                for i, obj_id in enumerate(out_obj_ids):
                    mask = masks[i]
                    logits = mask_logits_np[i]

                    if mask.ndim > 2:
                        mask = np.squeeze(mask)
                        logits = np.squeeze(logits)

                    # Filter by minimum area
                    mask_area = mask.sum() / mask.size
                    if mask_area < min_mask_area:
                        continue

                    # Compute confidence from logits
                    # Higher absolute logit values = more confident predictions
                    confidence = self._compute_confidence(logits, mask)

                    # Estimate IoU prediction (use logit magnitude as proxy)
                    iou_pred = self._estimate_iou_prediction(logits, mask)

                    processed_masks[obj_id] = mask
                    frame_confidences[obj_id] = confidence
                    frame_ious[obj_id] = iou_pred

                if processed_masks:  # Only add frame if it has valid masks
                    video_segments[out_frame_idx] = processed_masks
                    confidence_scores[out_frame_idx] = frame_confidences
                    iou_predictions[out_frame_idx] = frame_ious

            self.predictor.reset_state(inference_state)

            # Apply Non-Maximum Suppression to remove duplicate detections
            video_segments, confidence_scores, iou_predictions = self._apply_nms(
                video_segments,
                confidence_scores,
                iou_predictions,
                iou_threshold=0.7
            )

            # Get all unique object IDs
            all_obj_ids = list(set().union(*[set(v.keys()) for v in video_segments.values()]))

            logger.info(
                f"SAM-2 tracked {len(all_obj_ids)} objects across {len(video_segments)} frames "
                f"(avg confidence: {self._avg_confidence(confidence_scores):.3f})"
            )

            return SAM2Result(
                video_segments=video_segments,
                object_ids=all_obj_ids,
                confidence_scores=confidence_scores,
                iou_predictions=iou_predictions
            )

        except Exception as e:
            logger.error(f"SAM-2 tracking failed: {e}")
            # Fallback: return empty result
            return SAM2Result(
                video_segments={},
                object_ids=[],
                confidence_scores={},
                iou_predictions={}
            )

    def _generate_grid_points(self, width: int, height: int, grid_size: int) -> List[Tuple[float, float]]:
        """
        Generate grid of initialization points for multi-object detection

        Returns list of (x, y) coordinates
        """
        points = []

        # Create grid with margins
        margin_x = width * 0.1
        margin_y = height * 0.1

        x_coords = np.linspace(margin_x, width - margin_x, grid_size)
        y_coords = np.linspace(margin_y, height - margin_y, grid_size)

        for x in x_coords:
            for y in y_coords:
                points.append((float(x), float(y)))

        return points

    def _compute_confidence(self, logits: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute confidence score from mask logits

        Higher absolute logit values indicate more confident predictions
        """
        if mask.sum() == 0:
            return 0.0

        # Average absolute logit value within the mask
        mask_logits = logits[mask]

        if len(mask_logits) == 0:
            return 0.0

        # Use sigmoid of mean absolute logit as confidence
        mean_abs_logit = np.abs(mask_logits).mean()
        confidence = 1.0 / (1.0 + np.exp(-mean_abs_logit))

        return float(confidence)

    def _estimate_iou_prediction(self, logits: np.ndarray, mask: np.ndarray) -> float:
        """
        Estimate IoU prediction from logits

        SAM models output IoU predictions which correlate with mask quality
        """
        if mask.sum() == 0:
            return 0.0

        # Use mean logit within mask region as IoU proxy
        mask_logits = logits[mask]

        if len(mask_logits) == 0:
            return 0.0

        # Normalize to [0, 1] range
        mean_logit = mask_logits.mean()
        iou_pred = 1.0 / (1.0 + np.exp(-mean_logit))

        return float(np.clip(iou_pred, 0.0, 1.0))

    def _apply_nms(
        self,
        video_segments: Dict[int, Dict[int, np.ndarray]],
        confidence_scores: Dict[int, Dict[int, float]],
        iou_predictions: Dict[int, Dict[int, float]],
        iou_threshold: float = 0.7
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate/overlapping detections

        For each frame, keeps only the highest-confidence mask among overlapping masks
        """
        filtered_segments = {}
        filtered_confidences = {}
        filtered_ious = {}

        for frame_idx in video_segments.keys():
            masks = video_segments[frame_idx]
            confidences = confidence_scores[frame_idx]
            ious = iou_predictions[frame_idx]

            # Sort objects by confidence (descending)
            sorted_obj_ids = sorted(
                masks.keys(),
                key=lambda obj_id: confidences[obj_id],
                reverse=True
            )

            keep_obj_ids = []

            for obj_id in sorted_obj_ids:
                current_mask = masks[obj_id]

                # Check overlap with already-kept masks
                should_keep = True
                for kept_id in keep_obj_ids:
                    kept_mask = masks[kept_id]

                    # Compute IoU
                    intersection = np.logical_and(current_mask, kept_mask).sum()
                    union = np.logical_or(current_mask, kept_mask).sum()

                    if union > 0:
                        iou = intersection / union

                        # If high overlap, discard current mask
                        if iou > iou_threshold:
                            should_keep = False
                            break

                if should_keep:
                    keep_obj_ids.append(obj_id)

            # Filter to kept objects
            if keep_obj_ids:
                filtered_segments[frame_idx] = {obj_id: masks[obj_id] for obj_id in keep_obj_ids}
                filtered_confidences[frame_idx] = {obj_id: confidences[obj_id] for obj_id in keep_obj_ids}
                filtered_ious[frame_idx] = {obj_id: ious[obj_id] for obj_id in keep_obj_ids}

        # Log NMS results
        total_before = sum(len(objs) for objs in video_segments.values())
        total_after = sum(len(objs) for objs in filtered_segments.values())
        logger.info(f"NMS: {total_after}/{total_before} masks retained")

        return filtered_segments, filtered_confidences, filtered_ious

    def _avg_confidence(self, confidence_scores: Dict[int, Dict[int, float]]) -> float:
        """Compute average confidence across all frames and objects"""
        all_confidences = []
        for frame_scores in confidence_scores.values():
            all_confidences.extend(frame_scores.values())

        return np.mean(all_confidences) if all_confidences else 0.0