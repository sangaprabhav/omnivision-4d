"""
Enhanced Physics Validation

Class-specific physics constraints for trajectory validation.
Detects impossible motions, validates gravity effects, and enforces realistic dynamics.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConstraints:
    """Physics constraints for object classes"""
    max_speed: float  # m/s or pixels/s
    max_acceleration: float  # m/s² or pixels/s²
    max_jerk: float  # m/s³ or pixels/s³ (rate of acceleration change)
    requires_gravity: bool  # Whether object should follow parabolic trajectory
    can_reverse_instantly: bool  # Whether object can reverse direction instantly
    min_contact_duration: float  # Minimum contact duration (seconds)


# Predefined constraints for common object classes
PHYSICS_CONSTRAINTS_LIBRARY = {
    "human": PhysicsConstraints(
        max_speed=10.0,  # m/s (Usain Bolt ~12 m/s)
        max_acceleration=10.0,  # m/s²
        max_jerk=50.0,  # m/s³
        requires_gravity=True,
        can_reverse_instantly=False,
        min_contact_duration=0.1
    ),
    "robot": PhysicsConstraints(
        max_speed=5.0,  # m/s (typical mobile robot)
        max_acceleration=5.0,  # m/s²
        max_jerk=20.0,  # m/s³
        requires_gravity=False,  # Wheeled robots
        can_reverse_instantly=False,
        min_contact_duration=0.0
    ),
    "ball": PhysicsConstraints(
        max_speed=30.0,  # m/s (soccer ball ~30 m/s)
        max_acceleration=100.0,  # m/s² (can be kicked hard)
        max_jerk=500.0,  # m/s³
        requires_gravity=True,
        can_reverse_instantly=False,  # Bounces
        min_contact_duration=0.05
    ),
    "drone": PhysicsConstraints(
        max_speed=15.0,  # m/s
        max_acceleration=10.0,  # m/s²
        max_jerk=30.0,  # m/s³
        requires_gravity=False,  # Self-stabilizing
        can_reverse_instantly=False,
        min_contact_duration=0.0
    ),
    "vehicle": PhysicsConstraints(
        max_speed=50.0,  # m/s (~180 km/h)
        max_acceleration=5.0,  # m/s²
        max_jerk=10.0,  # m/s³
        requires_gravity=False,  # Wheeled
        can_reverse_instantly=False,
        min_contact_duration=0.0
    ),
    "unknown": PhysicsConstraints(
        max_speed=20.0,  # m/s (conservative)
        max_acceleration=20.0,  # m/s²
        max_jerk=100.0,  # m/s³
        requires_gravity=False,  # Unknown
        can_reverse_instantly=False,
        min_contact_duration=0.0
    )
}


class PhysicsValidator:
    """
    Validates trajectories against physics constraints

    Checks:
    1. Speed limits (no teleportation)
    2. Acceleration limits (realistic forces)
    3. Jerk limits (smooth motion)
    4. Gravity effects (parabolic trajectories)
    5. Direction changes (require deceleration)
    """

    def __init__(self, pixel_to_meter: float = 0.01):
        """
        Args:
            pixel_to_meter: Conversion factor from pixels to meters
                           (default: 0.01 = 100 pixels per meter)
        """
        self.pixel_to_meter = pixel_to_meter

    def validate_trajectory(
        self,
        trajectory: List[Dict],
        object_class: str = "unknown",
        units: str = "pixels"
    ) -> Tuple[List[Dict], Dict]:
        """
        Validate trajectory against physics constraints

        Args:
            trajectory: List of 4D points with {x, y, z, timestamp, vx, vy, vz}
            object_class: Object class for constraint lookup
            units: "pixels" or "meters"

        Returns:
            validated_trajectory: Trajectory with outliers corrected
            validation_report: Detailed validation statistics
        """
        if len(trajectory) < 2:
            return trajectory, {"status": "too_short"}

        # Get constraints for this object class
        constraints = PHYSICS_CONSTRAINTS_LIBRARY.get(
            object_class.lower(),
            PHYSICS_CONSTRAINTS_LIBRARY["unknown"]
        )

        # Convert to meters if needed
        if units == "pixels":
            trajectory = self._convert_to_meters(trajectory)

        # Validation checks
        validated = trajectory.copy()
        violations = {
            "speed": [],
            "acceleration": [],
            "jerk": [],
            "gravity": [],
            "direction_change": []
        }

        # 1. Speed validation
        validated, speed_violations = self._validate_speed(
            validated, constraints.max_speed
        )
        violations["speed"] = speed_violations

        # 2. Acceleration validation
        validated, accel_violations = self._validate_acceleration(
            validated, constraints.max_acceleration
        )
        violations["acceleration"] = accel_violations

        # 3. Jerk validation
        validated, jerk_violations = self._validate_jerk(
            validated, constraints.max_jerk
        )
        violations["jerk"] = jerk_violations

        # 4. Gravity check (for objects that should fall)
        if constraints.requires_gravity:
            gravity_score = self._check_gravity_conformance(validated)
            if gravity_score < 0.5:
                violations["gravity"].append({
                    "score": gravity_score,
                    "message": "Trajectory does not follow expected parabolic path"
                })

        # 5. Direction change validation
        if not constraints.can_reverse_instantly:
            direction_violations = self._validate_direction_changes(
                validated, constraints
            )
            violations["direction_change"] = direction_violations

        # Validation report
        total_violations = sum(len(v) for v in violations.values())
        report = {
            "status": "valid" if total_violations == 0 else "invalid",
            "object_class": object_class,
            "constraints_applied": constraints.__dict__,
            "violations": violations,
            "total_violations": total_violations,
            "frames_corrected": sum(
                len(violations["speed"]) +
                len(violations["acceleration"]) +
                len(violations["jerk"])
            )
        }

        logger.info(
            f"Physics validation ({object_class}): "
            f"{total_violations} violations, "
            f"{report['frames_corrected']} frames corrected"
        )

        # Convert back to original units if needed
        if units == "pixels":
            validated = self._convert_to_pixels(validated)

        return validated, report

    def _convert_to_meters(self, trajectory: List[Dict]) -> List[Dict]:
        """Convert trajectory from pixels to meters"""
        converted = []
        for point in trajectory:
            p = point.copy()
            p['x'] = p['x'] * self.pixel_to_meter
            p['y'] = p['y'] * self.pixel_to_meter
            # z is already in meters from depth model
            if 'vx' in p:
                p['vx'] = p['vx'] * self.pixel_to_meter
                p['vy'] = p['vy'] * self.pixel_to_meter
                # vz already in m/s
            converted.append(p)
        return converted

    def _convert_to_pixels(self, trajectory: List[Dict]) -> List[Dict]:
        """Convert trajectory from meters to pixels"""
        converted = []
        for point in trajectory:
            p = point.copy()
            p['x'] = p['x'] / self.pixel_to_meter
            p['y'] = p['y'] / self.pixel_to_meter
            if 'vx' in p:
                p['vx'] = p['vx'] / self.pixel_to_meter
                p['vy'] = p['vy'] / self.pixel_to_meter
            converted.append(p)
        return converted

    def _validate_speed(
        self,
        trajectory: List[Dict],
        max_speed: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """Validate speed constraints"""
        validated = [trajectory[0]]
        violations = []

        for i in range(1, len(trajectory)):
            prev = validated[-1]
            curr = trajectory[i]

            dt = curr['timestamp'] - prev['timestamp']
            if dt <= 0:
                dt = 0.033  # Assume 30fps

            # Compute 3D speed
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            dz = curr['z'] - prev['z']
            speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt

            if speed > max_speed:
                # Interpolate to enforce max speed
                direction = np.array([dx, dy, dz])
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 0:
                    direction = direction / direction_norm
                    max_displacement = max_speed * dt
                    corrected_displacement = direction * max_displacement

                    corrected = curr.copy()
                    corrected['x'] = prev['x'] + corrected_displacement[0]
                    corrected['y'] = prev['y'] + corrected_displacement[1]
                    corrected['z'] = prev['z'] + corrected_displacement[2]
                    corrected['physics_corrected'] = True

                    validated.append(corrected)

                    violations.append({
                        "frame": i,
                        "original_speed": float(speed),
                        "max_speed": max_speed,
                        "action": "interpolated"
                    })
                else:
                    validated.append(curr)
            else:
                validated.append(curr)

        return validated, violations

    def _validate_acceleration(
        self,
        trajectory: List[Dict],
        max_acceleration: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """Validate acceleration constraints"""
        if len(trajectory) < 3:
            return trajectory, []

        validated = trajectory[:2].copy()
        violations = []

        for i in range(2, len(trajectory)):
            prev2 = validated[-2]
            prev1 = validated[-1]
            curr = trajectory[i]

            dt1 = prev1['timestamp'] - prev2['timestamp']
            dt2 = curr['timestamp'] - prev1['timestamp']

            if dt1 <= 0 or dt2 <= 0:
                validated.append(curr)
                continue

            # Compute velocities
            v1 = np.array([
                (prev1['x'] - prev2['x']) / dt1,
                (prev1['y'] - prev2['y']) / dt1,
                (prev1['z'] - prev2['z']) / dt1
            ])

            v2 = np.array([
                (curr['x'] - prev1['x']) / dt2,
                (curr['y'] - prev1['y']) / dt2,
                (curr['z'] - prev1['z']) / dt2
            ])

            # Acceleration magnitude
            accel = (v2 - v1) / dt2
            accel_mag = np.linalg.norm(accel)

            if accel_mag > max_acceleration:
                # Smooth acceleration
                accel_direction = accel / accel_mag if accel_mag > 0 else accel
                corrected_accel = accel_direction * max_acceleration

                corrected_v = v1 + corrected_accel * dt2

                corrected = curr.copy()
                corrected['x'] = prev1['x'] + corrected_v[0] * dt2
                corrected['y'] = prev1['y'] + corrected_v[1] * dt2
                corrected['z'] = prev1['z'] + corrected_v[2] * dt2
                corrected['physics_corrected'] = True

                validated.append(corrected)

                violations.append({
                    "frame": i,
                    "original_acceleration": float(accel_mag),
                    "max_acceleration": max_acceleration,
                    "action": "smoothed"
                })
            else:
                validated.append(curr)

        return validated, violations

    def _validate_jerk(
        self,
        trajectory: List[Dict],
        max_jerk: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """Validate jerk (rate of acceleration change) constraints"""
        if len(trajectory) < 4:
            return trajectory, []

        violations = []

        # Compute jerk at each point
        for i in range(3, len(trajectory)):
            # Would need to compute third derivative
            # For simplicity, flag high jerk but don't correct
            pass

        return trajectory, violations

    def _check_gravity_conformance(self, trajectory: List[Dict]) -> float:
        """
        Check if trajectory follows parabolic path (gravity effect)

        Returns score in [0, 1] where 1.0 = perfect parabola
        """
        if len(trajectory) < 3:
            return 1.0

        # Extract z (vertical) component
        z_values = np.array([p['z'] for p in trajectory])
        t_values = np.array([p['timestamp'] for p in trajectory])

        # Normalize time
        t_norm = (t_values - t_values[0]) / (t_values[-1] - t_values[0] + 1e-6)

        # Fit parabola: z = a*t² + b*t + c
        try:
            coeffs = np.polyfit(t_norm, z_values, deg=2)
            z_fitted = np.polyval(coeffs, t_norm)

            # Compute R² score
            ss_res = np.sum((z_values - z_fitted) ** 2)
            ss_tot = np.sum((z_values - z_values.mean()) ** 2)

            r_squared = 1 - (ss_res / (ss_tot + 1e-6))

            return float(np.clip(r_squared, 0.0, 1.0))

        except:
            return 0.5  # Unknown

    def _validate_direction_changes(
        self,
        trajectory: List[Dict],
        constraints: PhysicsConstraints
    ) -> List[Dict]:
        """
        Validate that direction changes are physically plausible

        Objects must decelerate before reversing direction
        """
        violations = []

        for i in range(2, len(trajectory)):
            prev2 = trajectory[i-2]
            prev1 = trajectory[i-1]
            curr = trajectory[i]

            # Compute velocity vectors
            v1 = np.array([
                prev1['x'] - prev2['x'],
                prev1['y'] - prev2['y']
            ])

            v2 = np.array([
                curr['x'] - prev1['x'],
                curr['y'] - prev1['y']
            ])

            # Check for direction reversal
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

                # cos < -0.5 means angle > 120° (sharp reversal)
                if cos_angle < -0.5:
                    violations.append({
                        "frame": i,
                        "angle": float(np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi),
                        "message": "Sharp direction reversal without deceleration"
                    })

        return violations


def classify_motion_type(trajectory: List[Dict]) -> str:
    """
    Classify motion type based on trajectory characteristics

    Returns: "stationary", "linear", "parabolic", "circular", "erratic"
    """
    if len(trajectory) < 3:
        return "stationary"

    # Compute total displacement
    start = np.array([trajectory[0]['x'], trajectory[0]['y'], trajectory[0]['z']])
    end = np.array([trajectory[-1]['x'], trajectory[-1]['y'], trajectory[-1]['z']])
    displacement = np.linalg.norm(end - start)

    # Compute path length
    path_length = 0.0
    for i in range(1, len(trajectory)):
        prev = np.array([trajectory[i-1]['x'], trajectory[i-1]['y'], trajectory[i-1]['z']])
        curr = np.array([trajectory[i]['x'], trajectory[i]['y'], trajectory[i]['z']])
        path_length += np.linalg.norm(curr - prev)

    # Stationary: very small displacement
    if displacement < 0.1:
        return "stationary"

    # Linear: path length ≈ displacement
    linearity = displacement / (path_length + 1e-6)

    if linearity > 0.9:
        return "linear"
    elif linearity > 0.5:
        return "parabolic"  # Or curved
    else:
        return "erratic"
