"""
Robust Semantic-Geometric Matching

Matches SAM-2 object masks with Cosmos semantic labels using:
1. Spatial consistency (IoU, centroid distance)
2. Temporal consistency (label stability across frames)
3. Hierarchical labeling (coarse and fine-grained)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticMatch:
    """Semantic matching result"""
    sam_object_id: int
    semantic_label: str
    confidence: float  # Matching confidence (0-1)
    match_method: str  # "spatial", "temporal", "hierarchical"
    spatial_iou: float
    temporal_consistency: float


class SemanticGeometricMatcher:
    """
    Matches geometric detections (SAM masks) with semantic labels (Cosmos)

    Strategy:
    1. Spatial matching: IoU between SAM masks and Cosmos bboxes
    2. Temporal matching: Track label consistency across frames
    3. Hierarchical matching: Support coarse ("robot") and fine ("robot_gripper") labels
    """

    def __init__(
        self,
        min_iou_threshold: float = 0.3,
        min_temporal_consensus: int = 3,
        enable_hierarchical: bool = True
    ):
        """
        Args:
            min_iou_threshold: Minimum IoU for spatial match (0-1)
            min_temporal_consensus: Minimum frames for temporal consensus
            enable_hierarchical: Enable hierarchical label matching
        """
        self.min_iou_threshold = min_iou_threshold
        self.min_temporal_consensus = min_temporal_consensus
        self.enable_hierarchical = enable_hierarchical

        # Hierarchical label relationships
        self.label_hierarchy = {
            "robot": ["robot", "robot_arm", "robot_gripper", "robot_base"],
            "person": ["person", "human", "pedestrian", "worker"],
            "vehicle": ["vehicle", "car", "truck", "bus", "van"],
            "ball": ["ball", "sphere", "sports_ball"],
            "object": ["object", "thing", "item"]
        }

    def match_objects_to_labels(
        self,
        sam_masks: Dict[int, np.ndarray],  # {obj_id: mask}
        cosmos_objects: List[Dict],  # [{"label": str, "bbox": {...}}]
        frame_shape: Tuple[int, int] = (448, 448)
    ) -> List[SemanticMatch]:
        """
        Match SAM objects to Cosmos semantic labels

        Args:
            sam_masks: Dictionary of object masks {obj_id: mask}
            cosmos_objects: List of Cosmos detections with labels and bboxes
            frame_shape: Frame dimensions (height, width)

        Returns:
            List of semantic matches
        """
        matches = []

        # For each SAM object, find best matching Cosmos label
        for obj_id, mask in sam_masks.items():
            best_match = None
            best_score = 0.0

            for cosmos_obj in cosmos_objects:
                label = cosmos_obj.get('label', cosmos_obj.get('object_id', 'unknown'))
                bbox = cosmos_obj.get('bbox', cosmos_obj.get('bounding_box'))

                if bbox is None:
                    continue

                # Compute spatial IoU
                iou = self._compute_iou(mask, bbox, frame_shape)

                # Compute matching confidence
                confidence = self._compute_match_confidence(iou, label)

                if confidence > best_score and iou >= self.min_iou_threshold:
                    best_score = confidence
                    best_match = SemanticMatch(
                        sam_object_id=obj_id,
                        semantic_label=label,
                        confidence=confidence,
                        match_method="spatial",
                        spatial_iou=iou,
                        temporal_consistency=0.0  # Will be computed later
                    )

            # If no match found, assign generic label
            if best_match is None:
                best_match = SemanticMatch(
                    sam_object_id=obj_id,
                    semantic_label=f"object_{obj_id}",
                    confidence=0.5,  # Low confidence for fallback
                    match_method="fallback",
                    spatial_iou=0.0,
                    temporal_consistency=0.0
                )

            matches.append(best_match)

        logger.info(
            f"Semantic matching: {len(matches)} objects matched "
            f"(avg IoU: {np.mean([m.spatial_iou for m in matches]):.3f})"
        )

        return matches

    def match_with_temporal_consensus(
        self,
        frame_matches: Dict[int, List[SemanticMatch]]  # {frame_idx: [matches]}
    ) -> Dict[int, str]:
        """
        Improve matching using temporal consensus

        For each object, find most frequent label across frames

        Args:
            frame_matches: Dictionary mapping frame index to list of matches

        Returns:
            Dictionary mapping object ID to consensus label
        """
        # Collect labels for each object across frames
        object_labels = {}  # {obj_id: [labels]}

        for frame_idx, matches in frame_matches.items():
            for match in matches:
                obj_id = match.sam_object_id

                if obj_id not in object_labels:
                    object_labels[obj_id] = []

                object_labels[obj_id].append(match.semantic_label)

        # Find consensus label for each object
        consensus_labels = {}

        for obj_id, labels in object_labels.items():
            if len(labels) >= self.min_temporal_consensus:
                # Find most common label
                unique_labels, counts = np.unique(labels, return_counts=True)
                consensus_idx = np.argmax(counts)
                consensus_label = unique_labels[consensus_idx]
                consensus_ratio = counts[consensus_idx] / len(labels)

                consensus_labels[obj_id] = consensus_label

                logger.debug(
                    f"Object {obj_id}: consensus='{consensus_label}' "
                    f"({consensus_ratio:.1%} of {len(labels)} frames)"
                )
            else:
                # Not enough frames for consensus
                consensus_labels[obj_id] = labels[0] if labels else f"object_{obj_id}"

        return consensus_labels

    def refine_with_hierarchical_labels(
        self,
        matches: List[SemanticMatch],
        frame_context: Optional[Dict] = None
    ) -> List[SemanticMatch]:
        """
        Refine matches using hierarchical label relationships

        Args:
            matches: Initial semantic matches
            frame_context: Optional context about scene (e.g., {"scene_type": "warehouse"})

        Returns:
            Refined matches with hierarchical labels
        """
        if not self.enable_hierarchical:
            return matches

        refined = []

        for match in matches:
            original_label = match.semantic_label

            # Check if label is in hierarchy
            parent_label = self._find_parent_label(original_label)

            if parent_label != original_label:
                # Create hierarchical label
                hierarchical_label = {
                    "coarse": parent_label,
                    "fine": original_label
                }

                # Update match
                refined_match = SemanticMatch(
                    sam_object_id=match.sam_object_id,
                    semantic_label=original_label,  # Keep fine-grained
                    confidence=match.confidence * 1.1,  # Boost for hierarchy match
                    match_method="hierarchical",
                    spatial_iou=match.spatial_iou,
                    temporal_consistency=match.temporal_consistency
                )

                refined.append(refined_match)
            else:
                refined.append(match)

        return refined

    def _compute_iou(
        self,
        mask: np.ndarray,
        bbox: Dict,
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Compute IoU between mask and bounding box

        Args:
            mask: Binary mask (H x W)
            bbox: Bounding box {"x": , "y": , "w": , "h": }
            frame_shape: Frame dimensions

        Returns:
            IoU score in [0, 1]
        """
        height, width = frame_shape

        # Create bbox mask
        bbox_mask = np.zeros((height, width), dtype=bool)

        x = int(bbox.get('x', 0))
        y = int(bbox.get('y', 0))
        w = int(bbox.get('w', width))
        h = int(bbox.get('h', height))

        # Clip to bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 > x1 and y2 > y1:
            bbox_mask[y1:y2, x1:x2] = True

        # Compute IoU
        intersection = np.logical_and(mask, bbox_mask).sum()
        union = np.logical_or(mask, bbox_mask).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0

        return float(iou)

    def _compute_match_confidence(self, iou: float, label: str) -> float:
        """
        Compute matching confidence based on IoU and label quality

        High IoU + valid label = high confidence
        """
        # Base confidence from IoU
        iou_confidence = iou

        # Boost for non-generic labels
        label_boost = 1.0
        if label and not label.startswith("object_"):
            label_boost = 1.2

        confidence = iou_confidence * label_boost

        return float(np.clip(confidence, 0.0, 1.0))

    def _find_parent_label(self, label: str) -> str:
        """
        Find parent (coarse) label in hierarchy

        Args:
            label: Fine-grained label (e.g., "robot_gripper")

        Returns:
            Parent label (e.g., "robot") or original if no parent
        """
        label_lower = label.lower()

        for parent, children in self.label_hierarchy.items():
            if label_lower in [c.lower() for c in children]:
                return parent

        return label  # No parent found

    def create_label_mapping(
        self,
        matches: List[SemanticMatch]
    ) -> Dict[int, str]:
        """
        Create simple mapping from object ID to semantic label

        Args:
            matches: List of semantic matches

        Returns:
            Dictionary {obj_id: label}
        """
        mapping = {}

        for match in matches:
            mapping[match.sam_object_id] = match.semantic_label

        return mapping


def extract_cosmos_objects(cosmos_json: Dict) -> List[Dict]:
    """
    Extract object list from Cosmos JSON output

    Handles various Cosmos output formats

    Args:
        cosmos_json: Cosmos annotation JSON

    Returns:
        List of objects with labels and bboxes
    """
    objects = []

    # Try different JSON structures
    if "objects" in cosmos_json:
        objects = cosmos_json["objects"]
    elif "trajectories" in cosmos_json:
        objects = cosmos_json["trajectories"]
    elif "detections" in cosmos_json:
        objects = cosmos_json["detections"]
    else:
        # Try to interpret entire JSON as object list
        if isinstance(cosmos_json, list):
            objects = cosmos_json

    # Normalize object format
    normalized = []
    for obj in objects:
        if isinstance(obj, dict):
            normalized.append({
                "label": obj.get("label") or obj.get("object_id") or obj.get("class") or "unknown",
                "bbox": obj.get("bbox") or obj.get("bounding_box") or obj.get("box"),
                "confidence": obj.get("confidence", 1.0)
            })

    return normalized
