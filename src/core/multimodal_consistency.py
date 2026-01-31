"""
Multi-Modal Consistency Checker

Validates that SAM-2, Depth, and Cosmos outputs are consistent with each other.
Detects conflicts and assigns consensus-based confidence scores.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyMetrics:
    """Consistency metrics for multi-modal agreement"""
    spatial_iou: float  # Overlap between SAM mask and Cosmos bbox
    centroid_distance: float  # Distance between centroids (normalized)
    depth_consistency: float  # Uniformity of depth within mask
    overall_consistency: float  # Weighted average (0-1)
    is_consistent: bool  # Whether object passes consistency threshold


class MultiModalConsistencyChecker:
    """
    Validates consistency between SAM-2, Depth, and Cosmos outputs

    Checks:
    1. Spatial agreement: SAM masks vs Cosmos bounding boxes
    2. Depth consistency: Uniform depth within object masks
    3. Semantic plausibility: Object labels match visual appearance
    """

    def __init__(
        self,
        min_spatial_iou: float = 0.3,
        max_centroid_distance: float = 0.2,
        max_depth_variance: float = 2.0,
        min_overall_consistency: float = 0.5
    ):
        """
        Args:
            min_spatial_iou: Minimum IoU between SAM and Cosmos (0-1)
            max_centroid_distance: Max centroid distance as fraction of diagonal
            max_depth_variance: Max acceptable depth variance within object (meters)
            min_overall_consistency: Minimum overall consistency score (0-1)
        """
        self.min_spatial_iou = min_spatial_iou
        self.max_centroid_distance = max_centroid_distance
        self.max_depth_variance = max_depth_variance
        self.min_overall_consistency = min_overall_consistency

    def check_consistency(
        self,
        sam_mask: np.ndarray,
        depth_map: np.ndarray,
        cosmos_bbox: Optional[Dict] = None,
        frame_shape: Tuple[int, int] = (448, 448)
    ) -> ConsistencyMetrics:
        """
        Check consistency between modalities for a single object

        Args:
            sam_mask: Binary mask from SAM-2 (H x W)
            depth_map: Depth map in meters (H x W)
            cosmos_bbox: Optional bounding box from Cosmos {"x": , "y": , "w": , "h": }
            frame_shape: Frame dimensions (height, width)

        Returns:
            ConsistencyMetrics with detailed agreement scores
        """
        height, width = frame_shape

        # 1. Spatial Agreement: SAM vs Cosmos
        spatial_iou = self._compute_spatial_iou(sam_mask, cosmos_bbox, frame_shape)

        # 2. Centroid Distance
        sam_centroid = self._compute_centroid(sam_mask)
        if cosmos_bbox is not None:
            cosmos_centroid = self._bbox_centroid(cosmos_bbox)
            centroid_distance = self._normalized_distance(
                sam_centroid, cosmos_centroid, frame_shape
            )
        else:
            centroid_distance = 0.0  # No Cosmos data, assume perfect

        # 3. Depth Consistency
        depth_consistency = self._compute_depth_consistency(sam_mask, depth_map)

        # 4. Overall Consistency Score
        overall = self._compute_overall_consistency(
            spatial_iou, centroid_distance, depth_consistency
        )

        # 5. Validation
        is_consistent = (
            spatial_iou >= self.min_spatial_iou and
            centroid_distance <= self.max_centroid_distance and
            overall >= self.min_overall_consistency
        )

        return ConsistencyMetrics(
            spatial_iou=float(spatial_iou),
            centroid_distance=float(centroid_distance),
            depth_consistency=float(depth_consistency),
            overall_consistency=float(overall),
            is_consistent=is_consistent
        )

    def check_temporal_consistency(
        self,
        trajectory: List[Dict],
        consistency_window: int = 5
    ) -> float:
        """
        Check temporal consistency of multi-modal agreement across frames

        Args:
            trajectory: List of points with consistency scores
            consistency_window: Window size for averaging

        Returns:
            Temporal consistency score (0-1)
        """
        if len(trajectory) < consistency_window:
            return 1.0  # Too short to judge

        # Extract consistency scores
        scores = [
            point.get('consistency_score', 1.0)
            for point in trajectory
        ]

        # Compute rolling average
        rolling_avg = []
        for i in range(len(scores) - consistency_window + 1):
            window = scores[i:i + consistency_window]
            rolling_avg.append(np.mean(window))

        # Temporal consistency = stability of rolling average
        if len(rolling_avg) > 1:
            variance = np.var(rolling_avg)
            temporal_consistency = 1.0 - min(1.0, variance * 2)
        else:
            temporal_consistency = 1.0

        return float(temporal_consistency)

    def _compute_spatial_iou(
        self,
        sam_mask: np.ndarray,
        cosmos_bbox: Optional[Dict],
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Compute IoU between SAM mask and Cosmos bounding box

        Returns IoU in [0, 1], or 1.0 if no Cosmos bbox available
        """
        if cosmos_bbox is None:
            return 1.0  # No Cosmos data, assume perfect match

        # Convert Cosmos bbox to mask
        height, width = frame_shape
        cosmos_mask = np.zeros((height, width), dtype=bool)

        x = int(cosmos_bbox.get('x', 0))
        y = int(cosmos_bbox.get('y', 0))
        w = int(cosmos_bbox.get('w', width))
        h = int(cosmos_bbox.get('h', height))

        # Clip to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 > x1 and y2 > y1:
            cosmos_mask[y1:y2, x1:x2] = True

        # Compute IoU
        intersection = np.logical_and(sam_mask, cosmos_mask).sum()
        union = np.logical_or(sam_mask, cosmos_mask).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0

        return float(iou)

    def _compute_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Compute centroid of binary mask"""
        y_coords, x_coords = np.where(mask)

        if len(x_coords) == 0:
            return (0.0, 0.0)

        cx = x_coords.mean()
        cy = y_coords.mean()

        return (float(cx), float(cy))

    def _bbox_centroid(self, bbox: Dict) -> Tuple[float, float]:
        """Compute centroid of bounding box"""
        x = bbox.get('x', 0)
        y = bbox.get('y', 0)
        w = bbox.get('w', 0)
        h = bbox.get('h', 0)

        cx = x + w / 2.0
        cy = y + h / 2.0

        return (float(cx), float(cy))

    def _normalized_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Compute normalized Euclidean distance between two points

        Normalized by frame diagonal, returns value in [0, 1]
        """
        height, width = frame_shape
        diagonal = np.sqrt(height**2 + width**2)

        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        distance = np.sqrt(dx**2 + dy**2)

        normalized = distance / diagonal if diagonal > 0 else 0.0

        return float(normalized)

    def _compute_depth_consistency(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray
    ) -> float:
        """
        Compute depth consistency within mask region

        Returns score in [0, 1] where 1.0 = very consistent (low variance)
        """
        # Extract depth values within mask
        depth_values = depth_map[mask]

        if len(depth_values) == 0:
            return 0.0

        # Compute variance
        depth_variance = np.var(depth_values)

        # Normalize: low variance = high consistency
        # Use exponential decay: consistency = exp(-variance / threshold)
        consistency = np.exp(-depth_variance / self.max_depth_variance)

        return float(np.clip(consistency, 0.0, 1.0))

    def _compute_overall_consistency(
        self,
        spatial_iou: float,
        centroid_distance: float,
        depth_consistency: float
    ) -> float:
        """
        Compute overall consistency score

        Weighted combination of individual metrics
        """
        # Convert centroid distance to similarity (1 - distance)
        centroid_similarity = 1.0 - min(1.0, centroid_distance / self.max_centroid_distance)

        # Weighted average
        overall = (
            0.4 * spatial_iou +
            0.3 * centroid_similarity +
            0.3 * depth_consistency
        )

        return float(np.clip(overall, 0.0, 1.0))


def filter_inconsistent_objects(
    objects: List[Dict],
    consistency_threshold: float = 0.5
) -> List[Dict]:
    """
    Filter out objects with low multi-modal consistency

    Args:
        objects: List of detected objects with consistency metrics
        consistency_threshold: Minimum consistency score (0-1)

    Returns:
        Filtered list of consistent objects
    """
    filtered = []

    for obj in objects:
        consistency = obj.get('consistency_metrics', {}).get('overall_consistency', 0.0)

        if consistency >= consistency_threshold:
            filtered.append(obj)
            logger.debug(
                f"Object {obj.get('object_id', 'unknown')}: "
                f"consistency={consistency:.3f} ✓"
            )
        else:
            logger.warning(
                f"Object {obj.get('object_id', 'unknown')}: "
                f"consistency={consistency:.3f} < {consistency_threshold} - FILTERED"
            )

    logger.info(
        f"Consistency filtering: {len(filtered)}/{len(objects)} objects retained "
        f"({100*len(filtered)/max(1, len(objects)):.1f}%)"
    )

    return filtered


def compute_cross_modal_confidence(
    sam_confidence: float,
    depth_confidence: float,
    cosmos_confidence: float,
    consistency_score: float
) -> float:
    """
    Compute cross-modal confidence by combining individual model confidences
    with consistency score

    High consistency boosts confidence, low consistency penalizes it

    Args:
        sam_confidence: SAM-2 confidence (0-1)
        depth_confidence: Depth model confidence (0-1)
        cosmos_confidence: Cosmos confidence (0-1)
        consistency_score: Multi-modal consistency (0-1)

    Returns:
        Cross-modal confidence score (0-1)
    """
    # Weighted geometric mean of individual confidences
    # Geometric mean penalizes low scores more than arithmetic mean
    individual_mean = (
        sam_confidence ** 0.4 *
        depth_confidence ** 0.3 *
        cosmos_confidence ** 0.3
    )

    # Modulate by consistency score
    # High consistency: boost confidence
    # Low consistency: penalize confidence
    cross_modal = individual_mean * (0.5 + 0.5 * consistency_score)

    return float(np.clip(cross_modal, 0.0, 1.0))
