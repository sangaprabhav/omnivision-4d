"""
Mask Quality Assessment Module

Evaluates the quality of segmentation masks to filter out low-quality annotations.
"""

import numpy as np
import cv2
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaskQualityMetrics:
    """Quality metrics for a segmentation mask"""
    coverage: float  # Percentage of frame covered (0-1)
    compactness: float  # Ratio of area to perimeter (higher = more compact)
    edge_sharpness: float  # Average gradient magnitude at edges (0-1)
    fragmentation: float  # Number of connected components (1 = perfect)
    aspect_ratio_score: float  # Penalty for extreme aspect ratios (0-1)
    overall_score: float  # Weighted average (0-1)
    is_valid: bool  # Whether mask meets minimum quality threshold


class MaskQualityAssessor:
    """Assesses segmentation mask quality"""

    def __init__(
        self,
        min_coverage: float = 0.001,  # At least 0.1% of frame
        max_coverage: float = 0.95,   # At most 95% of frame
        min_compactness: float = 0.1,
        min_edge_sharpness: float = 0.2,
        max_fragments: int = 5,
        min_overall_score: float = 0.4
    ):
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.min_compactness = min_compactness
        self.min_edge_sharpness = min_edge_sharpness
        self.max_fragments = max_fragments
        self.min_overall_score = min_overall_score

    def assess(self, mask: np.ndarray) -> MaskQualityMetrics:
        """
        Compute quality metrics for a segmentation mask

        Args:
            mask: Binary mask (H x W), values 0 or 1 (or boolean)

        Returns:
            MaskQualityMetrics with detailed quality assessment
        """
        # Ensure binary mask
        if mask.dtype != bool:
            mask = mask > 0.5

        # 1. Coverage: percentage of frame covered
        total_pixels = mask.size
        object_pixels = mask.sum()
        coverage = object_pixels / total_pixels if total_pixels > 0 else 0.0

        # 2. Compactness: ratio of area to perimeter^2 (normalized to 0-1)
        # Circle has compactness of 1/(4*pi) ≈ 0.08, square ≈ 0.0625
        # We normalize so square = 0.5, circle = 1.0
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0 and object_pixels > 0:
            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)

            if perimeter > 0:
                # Isoperimetric quotient: 4π * area / perimeter²
                # Circle = 1.0, square ≈ 0.785
                compactness = 4 * np.pi * object_pixels / (perimeter ** 2)
                compactness = min(1.0, compactness)  # Clip to [0, 1]
            else:
                compactness = 0.0
        else:
            compactness = 0.0

        # 3. Edge sharpness: average gradient magnitude at mask boundaries
        edge_sharpness = self._compute_edge_sharpness(mask)

        # 4. Fragmentation: number of connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        num_fragments = num_labels - 1  # Subtract background

        # Score: penalize multiple fragments (1.0 for single component)
        fragmentation_score = 1.0 / max(1, num_fragments) if num_fragments <= self.max_fragments else 0.0

        # 5. Aspect ratio: penalize extremely elongated shapes
        aspect_ratio_score = self._compute_aspect_ratio_score(mask)

        # 6. Overall score: weighted average
        overall_score = (
            0.15 * self._normalize_coverage(coverage) +
            0.30 * compactness +
            0.20 * edge_sharpness +
            0.20 * fragmentation_score +
            0.15 * aspect_ratio_score
        )

        # 7. Validation: check if mask meets minimum thresholds
        is_valid = (
            self.min_coverage <= coverage <= self.max_coverage and
            compactness >= self.min_compactness and
            edge_sharpness >= self.min_edge_sharpness and
            num_fragments <= self.max_fragments and
            overall_score >= self.min_overall_score
        )

        return MaskQualityMetrics(
            coverage=float(coverage),
            compactness=float(compactness),
            edge_sharpness=float(edge_sharpness),
            fragmentation=float(fragmentation_score),
            aspect_ratio_score=float(aspect_ratio_score),
            overall_score=float(overall_score),
            is_valid=is_valid
        )

    def _compute_edge_sharpness(self, mask: np.ndarray) -> float:
        """
        Compute sharpness of mask edges using gradient magnitude

        Sharp edges indicate confident segmentation.
        Blurry edges suggest uncertainty.
        """
        # Convert to float for gradient computation
        mask_float = mask.astype(np.float32)

        # Sobel gradients
        grad_x = cv2.Sobel(mask_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_float, cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Focus on edge pixels (where gradient is non-zero)
        edge_pixels = gradient_mag > 0.1

        if edge_pixels.sum() > 0:
            # Average gradient magnitude at edges
            avg_sharpness = gradient_mag[edge_pixels].mean()
            # Normalize to [0, 1] (typical range is 0-1.41 for binary masks)
            return min(1.0, avg_sharpness / 1.41)
        else:
            return 0.0

    def _compute_aspect_ratio_score(self, mask: np.ndarray) -> float:
        """
        Penalize extreme aspect ratios (very elongated objects)

        Returns score in [0, 1] where 1.0 is ideal (aspect ratio near 1.0)
        """
        # Find bounding box
        coords = np.column_stack(np.where(mask))

        if len(coords) < 2:
            return 0.0

        # PCA to find principal axes
        coords_centered = coords - coords.mean(axis=0)

        if coords_centered.shape[0] < 2:
            return 1.0  # Too small to judge

        # Covariance matrix
        cov = np.cov(coords_centered.T)

        # Eigenvalues = variance along principal axes
        eigenvalues = np.linalg.eigvalsh(cov)

        if eigenvalues[1] > 1e-6:  # Avoid division by zero
            aspect_ratio = np.sqrt(eigenvalues[1] / eigenvalues[0])

            # Penalize ratios far from 1.0
            # Use exponential decay: score = exp(-(log(ratio))^2 / 2)
            # aspect_ratio=1 → score=1, aspect_ratio=3 → score≈0.3, aspect_ratio=10 → score≈0.04
            score = np.exp(-0.5 * (np.log(aspect_ratio) ** 2))
            return float(score)
        else:
            return 1.0

    def _normalize_coverage(self, coverage: float) -> float:
        """
        Normalize coverage to [0, 1] with optimal range

        Too small or too large coverage is penalized
        Optimal: 0.1% - 50% of frame
        """
        if coverage < self.min_coverage or coverage > self.max_coverage:
            return 0.0

        # Optimal range: 1% - 30%
        optimal_min = 0.01
        optimal_max = 0.30

        if optimal_min <= coverage <= optimal_max:
            return 1.0
        elif coverage < optimal_min:
            # Linear interpolation from min_coverage to optimal_min
            return (coverage - self.min_coverage) / (optimal_min - self.min_coverage)
        else:  # coverage > optimal_max
            # Linear interpolation from optimal_max to max_coverage
            return 1.0 - (coverage - optimal_max) / (self.max_coverage - optimal_max)

    def assess_temporal_consistency(
        self,
        mask_current: np.ndarray,
        mask_previous: Optional[np.ndarray] = None
    ) -> float:
        """
        Assess temporal consistency between consecutive frames

        Returns IoU (Intersection over Union) score in [0, 1]
        High IoU = stable tracking, Low IoU = possible tracking failure
        """
        if mask_previous is None:
            return 1.0  # First frame, assume valid

        # Ensure binary
        if mask_current.dtype != bool:
            mask_current = mask_current > 0.5
        if mask_previous.dtype != bool:
            mask_previous = mask_previous > 0.5

        # Compute IoU
        intersection = np.logical_and(mask_current, mask_previous).sum()
        union = np.logical_or(mask_current, mask_previous).sum()

        if union > 0:
            iou = intersection / union
            return float(iou)
        else:
            # Both masks empty
            return 1.0 if intersection == 0 else 0.0


def filter_low_quality_masks(
    video_segments: Dict[int, Dict[int, np.ndarray]],
    min_quality_score: float = 0.4,
    min_temporal_iou: float = 0.5
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Filter out low-quality masks from video segmentation results

    Args:
        video_segments: {frame_idx: {obj_id: mask}}
        min_quality_score: Minimum overall quality score
        min_temporal_iou: Minimum IoU with previous frame

    Returns:
        Filtered video_segments with low-quality masks removed
    """
    assessor = MaskQualityAssessor(min_overall_score=min_quality_score)
    filtered_segments = {}

    # Track previous masks for temporal consistency
    previous_masks = {}

    for frame_idx in sorted(video_segments.keys()):
        filtered_segments[frame_idx] = {}

        for obj_id, mask in video_segments[frame_idx].items():
            # Assess quality
            quality = assessor.assess(mask)

            # Check temporal consistency if available
            temporal_iou = 1.0
            if obj_id in previous_masks:
                temporal_iou = assessor.assess_temporal_consistency(
                    mask, previous_masks[obj_id]
                )

            # Filter based on quality and temporal consistency
            if quality.is_valid and temporal_iou >= min_temporal_iou:
                filtered_segments[frame_idx][obj_id] = mask
                previous_masks[obj_id] = mask

                logger.debug(
                    f"Frame {frame_idx}, Object {obj_id}: "
                    f"quality={quality.overall_score:.3f}, "
                    f"temporal_iou={temporal_iou:.3f} ✓"
                )
            else:
                logger.warning(
                    f"Frame {frame_idx}, Object {obj_id}: "
                    f"quality={quality.overall_score:.3f} (valid={quality.is_valid}), "
                    f"temporal_iou={temporal_iou:.3f} - FILTERED"
                )

    # Summary
    total_original = sum(len(objs) for objs in video_segments.values())
    total_filtered = sum(len(objs) for objs in filtered_segments.values())
    logger.info(
        f"Mask quality filtering: {total_filtered}/{total_original} masks retained "
        f"({100*total_filtered/max(1, total_original):.1f}%)"
    )

    return filtered_segments
