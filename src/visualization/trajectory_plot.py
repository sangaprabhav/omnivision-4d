"""
3D Trajectory Visualization

Creates interactive 3D plots of object trajectories with matplotlib.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TrajectoryPlotter:
    """
    Create 3D visualizations of object trajectories
    """

    def __init__(self, figsize: tuple = (12, 8), dpi: int = 100):
        """
        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for output images
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_trajectories_3d(
        self,
        annotation_result: Dict,
        output_path: str,
        show_velocity: bool = False,
        color_by: str = 'object'  # 'object', 'speed', 'confidence'
    ) -> str:
        """
        Create 3D plot of all object trajectories

        Args:
            annotation_result: Annotation result from annotator
            output_path: Path to save plot image
            show_velocity: If True, draw velocity vectors
            color_by: How to color trajectories

        Returns:
            Path to saved plot
        """
        logger.info(f"Creating 3D trajectory plot: {output_path}")

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        objects = annotation_result.get('objects', [])

        if not objects:
            logger.warning("No objects to plot")
            return output_path

        # Plot each object's trajectory
        for obj_idx, obj in enumerate(objects):
            trajectory = obj.get('trajectory_4d', {}).get('points', [])

            if not trajectory:
                continue

            # Extract coordinates
            xs = [p['x'] for p in trajectory]
            ys = [p['y'] for p in trajectory]
            zs = [p['z'] for p in trajectory]

            # Determine color
            if color_by == 'object':
                color = f'C{obj_idx % 10}'
            elif color_by == 'speed':
                speeds = [
                    np.sqrt(p.get('vx', 0)**2 + p.get('vy', 0)**2 + p.get('vz', 0)**2)
                    for p in trajectory
                ]
                color = speeds
            elif color_by == 'confidence':
                confidences = [p.get('consistency_score', 1.0) for p in trajectory]
                color = confidences
            else:
                color = 'blue'

            # Plot trajectory line
            if isinstance(color, str):
                ax.plot(
                    xs, ys, zs,
                    label=obj.get('semantic_label', f"Object {obj_idx}"),
                    linewidth=2,
                    color=color
                )
            else:
                # Color map for speed/confidence
                scatter = ax.scatter(
                    xs, ys, zs,
                    c=color,
                    cmap='viridis',
                    label=obj.get('semantic_label', f"Object {obj_idx}"),
                    s=20
                )

            # Mark start and end
            ax.scatter(xs[0], ys[0], zs[0], c='green', marker='o', s=100, alpha=0.7)
            ax.scatter(xs[-1], ys[-1], zs[-1], c='red', marker='x', s=100, alpha=0.7)

            # Draw velocity vectors (optional)
            if show_velocity:
                self._draw_velocity_vectors(ax, trajectory)

        # Labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Object Trajectories')

        # Legend
        ax.legend(loc='upper right')

        # Grid
        ax.grid(True, alpha=0.3)

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"✓ Trajectory plot saved: {output_path}")
        return output_path

    def plot_trajectory_analysis(
        self,
        annotation_result: Dict,
        output_path: str
    ) -> str:
        """
        Create multi-panel analysis plot with various trajectory metrics

        Panels:
        - Position vs time (X, Y, Z)
        - Velocity vs time
        - Speed vs time
        - Acceleration vs time
        """
        logger.info(f"Creating trajectory analysis plot: {output_path}")

        objects = annotation_result.get('objects', [])

        if not objects:
            logger.warning("No objects to plot")
            return output_path

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)

        for obj_idx, obj in enumerate(objects):
            trajectory = obj.get('trajectory_4d', {}).get('points', [])

            if not trajectory:
                continue

            label = obj.get('semantic_label', f"Object {obj_idx}")
            color = f'C{obj_idx % 10}'

            # Extract data
            times = [p['timestamp'] for p in trajectory]
            xs = [p['x'] for p in trajectory]
            ys = [p['y'] for p in trajectory]
            zs = [p['z'] for p in trajectory]
            vxs = [p.get('vx', 0) for p in trajectory]
            vys = [p.get('vy', 0) for p in trajectory]
            vzs = [p.get('vz', 0) for p in trajectory]

            # Compute speeds
            speeds = [np.sqrt(vx**2 + vy**2 + vz**2) for vx, vy, vz in zip(vxs, vys, vzs)]

            # Panel 1: Position vs time
            axes[0, 0].plot(times, xs, label=f'{label} (X)', color=color, linestyle='-')
            axes[0, 0].plot(times, ys, label=f'{label} (Y)', color=color, linestyle='--')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Position (pixels)')
            axes[0, 0].set_title('Position over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Panel 2: Depth vs time
            axes[0, 1].plot(times, zs, label=label, color=color)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Depth (meters)')
            axes[0, 1].set_title('Depth over Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Panel 3: Velocity components
            axes[1, 0].plot(times, vxs, label=f'{label} (Vx)', color=color, linestyle='-')
            axes[1, 0].plot(times, vys, label=f'{label} (Vy)', color=color, linestyle='--')
            axes[1, 0].plot(times, vzs, label=f'{label} (Vz)', color=color, linestyle=':')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Velocity')
            axes[1, 0].set_title('Velocity Components')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Panel 4: Speed (magnitude)
            axes[1, 1].plot(times, speeds, label=label, color=color)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Speed')
            axes[1, 1].set_title('Speed over Time')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"✓ Analysis plot saved: {output_path}")
        return output_path

    def plot_motion_heatmap(
        self,
        annotation_result: Dict,
        output_path: str,
        frame_shape: tuple = (448, 448)
    ) -> str:
        """
        Create 2D heatmap showing areas of motion activity

        Args:
            annotation_result: Annotation result
            output_path: Output path for heatmap
            frame_shape: Video frame dimensions (height, width)

        Returns:
            Path to saved heatmap
        """
        logger.info(f"Creating motion heatmap: {output_path}")

        height, width = frame_shape
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Accumulate motion activity
        for obj in annotation_result.get('objects', []):
            trajectory = obj.get('trajectory_4d', {}).get('points', [])

            for point in trajectory:
                x = int(np.clip(point['x'], 0, width - 1))
                y = int(np.clip(point['y'], 0, height - 1))

                # Add Gaussian blob around point
                self._add_gaussian_blob(heatmap, x, y, sigma=10, intensity=1.0)

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear', origin='upper')
        ax.set_title('Motion Activity Heatmap')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activity')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"✓ Heatmap saved: {output_path}")
        return output_path

    def _draw_velocity_vectors(self, ax, trajectory: List[Dict], stride: int = 5):
        """Draw velocity vectors as arrows"""
        for i in range(0, len(trajectory), stride):
            point = trajectory[i]

            x, y, z = point['x'], point['y'], point['z']
            vx = point.get('vx', 0)
            vy = point.get('vy', 0)
            vz = point.get('vz', 0)

            # Scale velocity for visibility
            scale = 10.0
            ax.quiver(
                x, y, z,
                vx * scale, vy * scale, vz * scale,
                color='gray',
                alpha=0.5,
                arrow_length_ratio=0.3
            )

    def _add_gaussian_blob(
        self,
        heatmap: np.ndarray,
        x: int,
        y: int,
        sigma: float = 10.0,
        intensity: float = 1.0
    ):
        """Add Gaussian blob to heatmap at given location"""
        height, width = heatmap.shape

        # Create meshgrid around point
        radius = int(3 * sigma)
        x_min = max(0, x - radius)
        x_max = min(width, x + radius)
        y_min = max(0, y - radius)
        y_max = min(height, y + radius)

        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]

        # Compute Gaussian
        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))

        # Add to heatmap
        heatmap[y_min:y_max, x_min:x_max] += gaussian * intensity


def plot_trajectories(
    annotation_result: Dict,
    output_path: str,
    plot_type: str = '3d'
) -> str:
    """
    Convenience function for trajectory plotting

    Args:
        annotation_result: Annotation result
        output_path: Output path
        plot_type: '3d', 'analysis', or 'heatmap'

    Returns:
        Path to saved plot
    """
    plotter = TrajectoryPlotter()

    if plot_type == '3d':
        return plotter.plot_trajectories_3d(annotation_result, output_path)
    elif plot_type == 'analysis':
        return plotter.plot_trajectory_analysis(annotation_result, output_path)
    elif plot_type == 'heatmap':
        return plotter.plot_motion_heatmap(annotation_result, output_path)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
