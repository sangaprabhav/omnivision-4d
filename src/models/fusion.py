"""
4D Fusion Logic
Combines SAM-2 (geometry), Depth (metric), Cosmos (semantics) into unified 4D trajectories

Phase 2 Enhancements:
- Multi-modal consistency checking
- Physics validation with class-specific constraints
- Robust semantic-geometric matching
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import savgol_filter
import logging

from src.core.multimodal_consistency import (
    MultiModalConsistencyChecker,
    compute_cross_modal_confidence
)
from src.core.physics_validator import PhysicsValidator, classify_motion_type
from src.core.semantic_matcher import (
    SemanticGeometricMatcher,
    extract_cosmos_objects
)

logger = logging.getLogger(__name__)

def smooth_trajectory(trajectory: List[Dict], window: int = 5) -> List[Dict]:
    """
    Apply Savitzky-Golay filter to remove jitter while preserving trends
    
    Args:
        trajectory: List of {x, y, z, t} points
        window: Smoothing window size (must be odd)
    
    Returns:
        Smoothed trajectory with velocity vectors added
    """
    if len(trajectory) < window or len(trajectory) < 3:
        return trajectory
    
    if window % 2 == 0:
        window = window + 1  # Make odd
    
    try:
        xs = np.array([p['x'] for p in trajectory])
        ys = np.array([p['y'] for p in trajectory])
        zs = np.array([p['z'] for p in trajectory])
        
        # Savitzky-Golay: polynomial smoothing (preserves peaks better than moving average)
        xs_smooth = savgol_filter(xs, window, polyorder=min(2, window-1))
        ys_smooth = savgol_filter(ys, window, polyorder=min(2, window-1))
        zs_smooth = savgol_filter(zs, window, polyorder=min(2, window-1))
        
        # Update trajectory and add velocities
        smoothed = []
        for i, p in enumerate(trajectory):
            point = p.copy()
            point['x'] = float(xs_smooth[i])
            point['y'] = float(ys_smooth[i])
            point['z'] = float(zs_smooth[i])
            
            # Calculate velocity (derivative)
            if i > 0:
                dt = p['timestamp'] - trajectory[i-1]['timestamp']
                if dt > 0:
                    point['vx'] = float((xs_smooth[i] - xs_smooth[i-1]) / dt)
                    point['vy'] = float((ys_smooth[i] - ys_smooth[i-1]) / dt)
                    point['vz'] = float((zs_smooth[i] - zs_smooth[i-1]) / dt)
                else:
                    point['vx'] = point['vy'] = point['vz'] = 0.0
            else:
                point['vx'] = point['vy'] = point['vz'] = 0.0
            
            point['smoothed'] = True
            smoothed.append(point)
        
        return smoothed
        
    except Exception as e:
        logger.warning(f"Smoothing failed: {e}, returning original")
        return trajectory

def validate_physics(trajectory: List[Dict], max_speed: float = 5000.0) -> List[Dict]:
    """
    Remove physically impossible jumps (teleportation detection)
    
    Args:
        trajectory: List of 4D points
        max_speed: Maximum allowable speed (pixels/second or m/s)
    
    Returns:
        Validated trajectory with interpolation for outliers
    """
    if len(trajectory) < 2:
        return trajectory
    
    validated = [trajectory[0]]
    
    for i in range(1, len(trajectory)):
        prev = validated[-1]
        curr = trajectory[i]
        
        dt = curr['timestamp'] - prev['timestamp']
        if dt <= 0:
            dt = 0.033  # Assume 30fps if timestamp missing
        
        # Calculate speed
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        dz = curr['z'] - prev['z']
        speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt
        
        if speed > max_speed:
            # Interpolate between prev and next valid point
            logger.warning(f"Frame {i}: Speed {speed:.1f} exceeds max {max_speed}, interpolating")
            
            interpolated = curr.copy()
            interpolated['x'] = (prev['x'] + curr['x']) / 2
            interpolated['y'] = (prev['y'] + curr['y']) / 2
            interpolated['z'] = (prev['z'] + curr['z']) / 2
            interpolated['interpolated'] = True
            validated.append(interpolated)
        else:
            validated.append(curr)
    
    return validated

def subpixel_centroid(mask: np.ndarray, depth: np.ndarray, 
                     depth_model) -> Tuple[float, float, float]:
    """
    Calculate sub-pixel accurate centroid with depth interpolation
    
    Args:
        mask: 2D binary mask (H x W)
        depth: 2D depth map (H x W)
        depth_model: DepthModel instance for interpolation
    
    Returns:
        (x, y, z) sub-pixel coordinates and metric depth
    """
    # Get coordinates where mask > 0
    y_indices, x_indices = np.where(mask > 0)
    
    if len(x_indices) == 0:
        return None
    
    # Weight by mask intensity (for soft masks)
    weights = mask[y_indices, x_indices].astype(np.float32)
    
    # Weighted centroid (sub-pixel)
    x_center = np.average(x_indices, weights=weights)
    y_center = np.average(y_indices, weights=weights)
    
    # Get metric depth at sub-pixel location
    z = depth_model.get_depth_at_point(depth, x_center, y_center)
    
    return float(x_center), float(y_center), float(z)

def fuse_4d(
    sam_masks: Dict,
    depth_maps: List[np.ndarray],
    cosmos_json: Dict,
    frame_indices: List[int],
    fps: float,
    depth_model,
    sam_confidences: Optional[Dict] = None,
    depth_confidences: Optional[List[float]] = None,
    cosmos_confidence: Optional[float] = None,
    enable_phase2: bool = True
) -> Dict:
    """
    Production-grade 4D fusion combining all three modalities

    Pipeline:
    1. Extract sub-pixel centroids from SAM masks
    2. Get metric depth at centroids
    3. Match with Cosmos semantic labels (Phase 2: robust matching)
    4. Apply temporal smoothing
    5. Validate physics constraints (Phase 2: class-specific)
    6. Check multi-modal consistency (Phase 2)

    Args:
        sam_masks: SAM-2 video segments {frame_idx: {obj_id: mask}}
        depth_maps: List of depth maps
        cosmos_json: Cosmos annotation JSON
        frame_indices: List of frame indices
        fps: Video frame rate
        depth_model: DepthModel instance
        sam_confidences: Optional SAM confidence scores
        depth_confidences: Optional depth confidence scores
        cosmos_confidence: Optional Cosmos confidence
        enable_phase2: Enable Phase 2 validation features

    Returns:
        Unified 4D annotation structure with Phase 2 enhancements
    """
    logger.info(f"Starting 4D fusion (Phase 2: {enable_phase2})...")
    
    # Get all unique object IDs from SAM
    all_obj_ids = list(set().union(*[set(v.keys()) for v in sam_masks.values()]))
    logger.info(f"Fusing {len(all_obj_ids)} objects")

    # Initialize Phase 2 components
    if enable_phase2:
        consistency_checker = MultiModalConsistencyChecker()
        physics_validator = PhysicsValidator()
        semantic_matcher = SemanticGeometricMatcher()

        # Extract Cosmos objects for semantic matching
        cosmos_objects = extract_cosmos_objects(cosmos_json)
        logger.info(f"Extracted {len(cosmos_objects)} Cosmos objects for matching")

    fused_objects = []
    
    for obj_id in all_obj_ids:
        trajectory_raw = []
        consistency_scores = []  # Phase 2: track consistency per frame

        # Build raw trajectory from SAM + Depth
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx not in sam_masks or obj_id not in sam_masks[frame_idx]:
                continue

            mask = sam_masks[frame_idx][obj_id]
            depth = depth_maps[i] if i < len(depth_maps) else None

            if depth is None:
                continue

            # Ensure mask is 2D
            if isinstance(mask, np.ndarray) and mask.ndim > 2:
                mask = np.squeeze(mask)

            if mask.ndim != 2:
                continue

            # Get sub-pixel centroid with metric depth
            centroid = subpixel_centroid(mask, depth, depth_model)
            if centroid is None:
                continue

            x, y, z = centroid

            # Phase 2: Check multi-modal consistency
            if enable_phase2:
                # Find matching Cosmos bbox for this frame
                cosmos_bbox = None
                for cosmos_obj in cosmos_objects:
                    if cosmos_obj.get('frame_idx') == frame_idx or cosmos_obj.get('bbox'):
                        cosmos_bbox = cosmos_obj.get('bbox')
                        break

                consistency = consistency_checker.check_consistency(
                    mask,
                    depth,
                    cosmos_bbox,
                    frame_shape=(448, 448)
                )
                consistency_scores.append(consistency.overall_consistency)
            else:
                consistency_scores.append(1.0)

            trajectory_raw.append({
                "frame": int(frame_idx),
                "timestamp": round(frame_idx / fps, 3) if fps > 0 else round(i / 30.0, 3),
                "x": x,
                "y": y,
                "z": z,
                "mask_coverage": float(np.sum(mask > 0) / mask.size),
                "consistency_score": consistency_scores[-1] if consistency_scores else 1.0
            })

        if not trajectory_raw:
            continue

        # Apply temporal smoothing
        trajectory_smoothed = smooth_trajectory(trajectory_raw, window=5)

        # Phase 2: Classify motion type first (for physics validation)
        motion_type = "unknown"
        if enable_phase2:
            motion_type = classify_motion_type(trajectory_smoothed)

        # Validate physics with class-specific constraints
        if enable_phase2:
            # We'll determine object class after semantic matching
            # For now, use motion type as proxy
            trajectory_validated, physics_report = physics_validator.validate_trajectory(
                trajectory_smoothed,
                object_class="unknown",  # Will be refined later
                units="pixels"
            )
        else:
            trajectory_validated = validate_physics(trajectory_smoothed, max_speed=1000.0)
            physics_report = None
        
        # Calculate spatial statistics
        xs = [p['x'] for p in trajectory_validated]
        ys = [p['y'] for p in trajectory_validated]
        zs = [p['z'] for p in trajectory_validated]

        # Phase 2: Robust semantic matching
        semantic_label = f"object_{obj_id}"
        semantic_confidence = 0.5
        behavior = "stationary"

        if enable_phase2 and cosmos_objects:
            # Use robust semantic matcher
            frame_masks = {
                frame_idx: sam_masks[frame_idx].get(obj_id)
                for frame_idx in frame_indices
                if frame_idx in sam_masks and obj_id in sam_masks[frame_idx]
            }

            # Match using first valid frame
            if frame_masks:
                first_frame_idx = list(frame_masks.keys())[0]
                first_mask = frame_masks[first_frame_idx]

                matches = semantic_matcher.match_objects_to_labels(
                    {obj_id: first_mask},
                    cosmos_objects,
                    frame_shape=(448, 448)
                )

                if matches:
                    match = matches[0]
                    semantic_label = match.semantic_label
                    semantic_confidence = match.confidence
        else:
            # Fallback: Try to extract from Cosmos output
            if isinstance(cosmos_json, dict):
                cosmos_traj = cosmos_json.get('trajectories', []) or cosmos_json.get('objects', [])
                for c_obj in cosmos_traj:
                    if isinstance(c_obj, dict):
                        obj_desc = c_obj.get('object_id', '') or c_obj.get('label', '')
                        if obj_desc and obj_desc != f"object_{obj_id}":
                            semantic_label = obj_desc
                        if 'action' in c_obj or 'behavior' in c_obj:
                            behavior = c_obj.get('action') or c_obj.get('behavior', 'stationary')
                        break

        # Determine if moving or stationary
        total_distance = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2 + (zs[-1] - zs[0])**2)
        if total_distance > 10:  # Moved more than 10 pixels/meters
            behavior = "translating"

        # Phase 2: Use motion classification
        if enable_phase2:
            behavior = motion_type
        
        # Phase 2: Compute cross-modal confidence
        if enable_phase2 and sam_confidences and depth_confidences and cosmos_confidence:
            # Get average SAM confidence for this object
            sam_conf = 0.5
            if sam_confidences:
                obj_confs = []
                for frame_idx in frame_indices:
                    if frame_idx in sam_confidences and obj_id in sam_confidences[frame_idx]:
                        obj_confs.append(sam_confidences[frame_idx][obj_id])
                sam_conf = np.mean(obj_confs) if obj_confs else 0.5

            depth_conf = np.mean(depth_confidences) if depth_confidences else 0.5
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0

            cross_modal_conf = compute_cross_modal_confidence(
                sam_conf,
                depth_conf,
                cosmos_confidence or 0.5,
                avg_consistency
            )
        else:
            cross_modal_conf = 0.7  # Default

        obj_dict = {
            "object_id": f"omnivision_{obj_id}",
            "semantic_label": semantic_label,
            "behavior_classification": behavior,
            "tracking_confidence": "high" if len(trajectory_validated) > 10 else "medium",
            "trajectory_4d": {
                "coordinate_frame": "camera",
                "units": {"x": "pixels", "y": "pixels", "z": "meters", "t": "seconds"},
                "num_frames": len(trajectory_validated),
                "duration_seconds": trajectory_validated[-1]['timestamp'] - trajectory_validated[0]['timestamp'] if len(trajectory_validated) > 1 else 0,
                "points": trajectory_validated
            },
            "spatial_extent": {
                "x_range": [float(min(xs)), float(max(xs))],
                "y_range": [float(min(ys)), float(max(ys))],
                "z_range": [float(min(zs)), float(max(zs))],
            },
            "motion_statistics": {
                "total_distance_3d": float(total_distance),
                "average_speed": float(np.mean([np.sqrt(p.get('vx', 0)**2 + p.get('vy', 0)**2 + p.get('vz', 0)**2) for p in trajectory_validated])),
                "motion_type": motion_type if enable_phase2 else "unknown"
            }
        }

        # Add Phase 2 validation metrics
        if enable_phase2:
            obj_dict["phase2_validation"] = {
                "cross_modal_confidence": float(cross_modal_conf),
                "semantic_confidence": float(semantic_confidence),
                "avg_consistency_score": float(np.mean(consistency_scores)) if consistency_scores else 1.0,
                "physics_validation": physics_report if physics_report else {"status": "not_validated"},
                "temporal_consistency": consistency_checker.check_temporal_consistency(trajectory_validated) if consistency_scores else 1.0
            }

        fused_objects.append(obj_dict)
    
    result = {
        "annotation_metadata": {
            "fusion_engine": "omnivision_4d_v1",
            "models": ["sam2_hiera_large", "zoedepth_metric", "cosmos_reason2_8b"],
            "fusion_timestamp": str(np.datetime64('now')),
            "video_fps": fps,
            "frames_processed": len(frame_indices),
            "total_objects_detected": len(fused_objects)
        },
        "scene_analysis": {
            "scene_type": "dynamic" if len(fused_objects) > 0 and fused_objects[0]['behavior_classification'] != 'stationary' else "static",
            "dominant_motion": "translational" if any(obj['behavior_classification'] == 'translating' for obj in fused_objects) else "none",
            "complexity_score": len(fused_objects) * len(frame_indices) / 100
        },
        "objects": fused_objects,
        "raw_semantics": cosmos_json  # Keep for debugging/reference
    }
    
    logger.info(f"Fusion complete: {len(fused_objects)} objects with 4D trajectories")
    return result