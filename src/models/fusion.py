"""
4D Fusion Logic
Combines SAM-2 (geometry), Depth (metric), Cosmos (semantics) into unified 4D trajectories
"""
import numpy as np
from typing import List, Dict, Tuple
from scipy.signal import savgol_filter
import logging

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

def fuse_4d(sam_masks: Dict, depth_maps: List[np.ndarray], 
            cosmos_json: Dict, frame_indices: List[int], 
            fps: float, depth_model) -> Dict:
    """
    Production-grade 4D fusion combining all three modalities
    
    Pipeline:
    1. Extract sub-pixel centroids from SAM masks
    2. Get metric depth at centroids
    3. Match with Cosmos semantic labels
    4. Apply temporal smoothing
    5. Validate physics constraints
    
    Returns:
        Unified 4D annotation structure
    """
    logger.info("Starting 4D fusion...")
    
    # Get all unique object IDs from SAM
    all_obj_ids = list(set().union(*[set(v.keys()) for v in sam_masks.values()]))
    logger.info(f"Fusing {len(all_obj_ids)} objects")
    
    fused_objects = []
    
    for obj_id in all_obj_ids:
        trajectory_raw = []
        
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
            
            trajectory_raw.append({
                "frame": int(frame_idx),
                "timestamp": round(frame_idx / fps, 3) if fps > 0 else round(i / 30.0, 3),
                "x": x,
                "y": y,
                "z": z,
                "mask_coverage": float(np.sum(mask > 0) / mask.size)
            })
        
        if not trajectory_raw:
            continue
        
        # Apply temporal smoothing
        trajectory_smoothed = smooth_trajectory(trajectory_raw, window=5)
        
        # Validate physics (remove teleportation)
        trajectory_validated = validate_physics(trajectory_smoothed, max_speed=1000.0)
        
        # Calculate spatial statistics
        xs = [p['x'] for p in trajectory_validated]
        ys = [p['y'] for p in trajectory_validated]
        zs = [p['z'] for p in trajectory_validated]
        
        # Match with Cosmos semantic understanding
        semantic_label = f"object_{obj_id}"
        behavior = "stationary"
        
        # Try to extract from Cosmos output
        if isinstance(cosmos_json, dict):
            # Check trajectories
            cosmos_traj = cosmos_json.get('trajectories', []) or cosmos_json.get('objects', [])
            for c_obj in cosmos_traj:
                if isinstance(c_obj, dict):
                    # Simple matching: if Cosmos mentions this object type
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
        
        fused_objects.append({
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
                "average_speed": float(np.mean([np.sqrt(p.get('vx', 0)**2 + p.get('vy', 0)**2 + p.get('vz', 0)**2) for p in trajectory_validated]))
            }
        })
    
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