"""
Post-processing utilities for 4D trajectories
"""
import numpy as np
from typing import List, Dict, Tuple

def normalize_trajectory(trajectory: List[Dict], method: str = "minmax") -> List[Dict]:
    """
    Normalize trajectory to 0-1 range or standardize
    Useful for ML training input
    """
    if not trajectory:
        return trajectory
    
    xs = np.array([p['x'] for p in trajectory])
    ys = np.array([p['y'] for p in trajectory])
    zs = np.array([p['z'] for p in trajectory])
    
    if method == "minmax":
        # Normalize to 0-1
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        z_min, z_max = zs.min(), zs.max()
        
        normalized = []
        for p in trajectory:
            norm_p = p.copy()
            norm_p['x'] = (p['x'] - x_min) / (x_max - x_min) if x_max != x_min else 0.5
            norm_p['y'] = (p['y'] - y_min) / (y_max - y_min) if y_max != y_min else 0.5
            norm_p['z'] = (p['z'] - z_min) / (z_max - z_min) if z_max != z_min else 0.5
            normalized.append(norm_p)
        
        return normalized
    
    elif method == "standardize":
        # Zero mean, unit variance
        x_mean, x_std = xs.mean(), xs.std()
        y_mean, y_std = ys.mean(), ys.std()
        z_mean, z_std = zs.mean(), zs.std()
        
        standardized = []
        for p in trajectory:
            std_p = p.copy()
            std_p['x'] = (p['x'] - x_mean) / x_std if x_std > 0 else 0
            std_p['y'] = (p['y'] - y_mean) / y_std if y_std > 0 else 0
            std_p['z'] = (p['z'] - z_mean) / z_std if z_std > 0 else 0
            standardized.append(std_p)
        
        return standardized
    
    return trajectory

def validate_coordinate_system(trajectory: List[Dict]) -> Tuple[bool, str]:
    """
    Validate that trajectory makes sense (no NaN, Inf, etc.)
    Returns (is_valid, error_message)
    """
    if not trajectory:
        return False, "Empty trajectory"
    
    for i, point in enumerate(trajectory):
        # Check required fields
        required = ['x', 'y', 'z', 'timestamp']
        for field in required:
            if field not in point:
                return False, f"Point {i} missing field: {field}"
        
        # Check for NaN or Inf
        for field in ['x', 'y', 'z']:
            val = point[field]
            if np.isnan(val) or np.isinf(val):
                return False, f"Point {i} has invalid {field}: {val}"
        
        # Check timestamp is monotonic
        if i > 0 and point['timestamp'] < trajectory[i-1]['timestamp']:
            return False, f"Point {i} has timestamp earlier than previous"
    
    return True, "Valid"

def interpolate_missing_frames(
    trajectory: List[Dict], 
    target_frame_indices: List[int]
) -> List[Dict]:
    """
    Interpolate trajectory to specific frame indices
    Useful if detection skipped some frames
    """
    if not trajectory:
        return []
    
    # Get existing frame numbers
    existing_frames = {p['frame']: p for p in trajectory if 'frame' in p}
    
    interpolated = []
    for frame_idx in target_frame_indices:
        if frame_idx in existing_frames:
            interpolated.append(existing_frames[frame_idx])
        else:
            # Find nearest frames for interpolation
            prev_frame = max([f for f in existing_frames.keys() if f < frame_idx], default=None)
            next_frame = min([f for f in existing_frames.keys() if f > frame_idx], default=None)
            
            if prev_frame is None or next_frame is None:
                continue  # Can't interpolate
            
            prev_point = existing_frames[prev_frame]
            next_point = existing_frames[next_frame]
            
            # Linear interpolation
            t = (frame_idx - prev_frame) / (next_frame - prev_frame)
            
            interp_point = {
                'frame': frame_idx,
                'timestamp': prev_point['timestamp'] + t * (next_point['timestamp'] - prev_point['timestamp']),
                'x': prev_point['x'] + t * (next_point['x'] - prev_point['x']),
                'y': prev_point['y'] + t * (next_point['y'] - prev_point['y']),
                'z': prev_point['z'] + t * (next_point['z'] - prev_point['z']),
                'interpolated': True
            }
            interpolated.append(interp_point)
    
    return interpolated

def calculate_acceleration(trajectory: List[Dict]) -> List[Dict]:
    """
    Calculate acceleration (second derivative) for each point
    Useful for detecting manipulation events (high acceleration = contact)
    """
    if len(trajectory) < 3:
        return trajectory
    
    for i in range(1, len(trajectory) - 1):
        dt = trajectory[i+1]['timestamp'] - trajectory[i-1]['timestamp']
        if dt == 0:
            continue
        
        prev_vx = trajectory[i].get('vx', 0)
        next_vx = trajectory[i+1].get('vx', 0)
        
        ax = (next_vx - prev_vx) / dt
        
        trajectory[i]['ax'] = float(ax)
    
    return trajectory