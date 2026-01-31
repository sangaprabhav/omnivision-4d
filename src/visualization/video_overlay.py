"""
Video Annotation Overlay

Draws annotations (boxes, labels, trajectories, confidence) on video frames.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import colorsys
import logging

logger = logging.getLogger(__name__)


class VideoOverlay:
    """
    Overlay annotations on video frames with customizable styling
    """

    def __init__(
        self,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_trajectories: bool = True,
        show_depth: bool = False,
        trajectory_length: int = 30,
        font_scale: float = 0.6,
        line_thickness: int = 2
    ):
        """
        Args:
            show_boxes: Draw bounding boxes around objects
            show_labels: Show semantic labels
            show_confidence: Show confidence scores
            show_trajectories: Draw trajectory paths
            show_depth: Show depth values
            trajectory_length: Number of past frames to show in trajectory
            font_scale: Font size for text
            line_thickness: Thickness of lines and boxes
        """
        self.show_boxes = show_boxes
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_trajectories = show_trajectories
        self.show_depth = show_depth
        self.trajectory_length = trajectory_length
        self.font_scale = font_scale
        self.line_thickness = line_thickness

        # Color palette (distinct colors for different objects)
        self.colors = self._generate_color_palette(50)

    def create_annotated_video(
        self,
        video_path: str,
        annotation_result: Dict,
        output_path: str,
        original_resolution: bool = True
    ) -> str:
        """
        Create annotated video with all overlays

        Args:
            video_path: Path to original video
            annotation_result: Annotation result from annotator.process()
            output_path: Path for output video
            original_resolution: If True, use original video resolution

        Returns:
            Path to created annotated video
        """
        logger.info(f"Creating annotated video: {output_path}")

        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} @ {fps:.1f} fps ({total_frames} frames)")

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Extract trajectory history for smooth path drawing
        trajectory_history = self._build_trajectory_history(annotation_result)

        # Process each frame
        frame_idx = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Overlay annotations
            annotated_frame = self._annotate_frame(
                frame,
                frame_idx,
                annotation_result,
                trajectory_history
            )

            # Write frame
            out.write(annotated_frame)

            frame_idx += 1
            processed_frames += 1

            if processed_frames % 100 == 0:
                logger.info(f"Processed {processed_frames}/{total_frames} frames")

        # Cleanup
        cap.release()
        out.release()

        logger.info(f"✓ Annotated video saved: {output_path}")
        return output_path

    def _annotate_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        annotation_result: Dict,
        trajectory_history: Dict
    ) -> np.ndarray:
        """
        Annotate a single frame with all overlays

        Args:
            frame: Original frame (BGR)
            frame_idx: Frame index
            annotation_result: Full annotation result
            trajectory_history: Pre-computed trajectory paths

        Returns:
            Annotated frame (BGR)
        """
        annotated = frame.copy()

        # Get objects present in this frame
        objects = annotation_result.get('objects', [])

        for obj_idx, obj in enumerate(objects):
            obj_id = obj.get('object_id', f'obj_{obj_idx}')
            color = self.colors[obj_idx % len(self.colors)]

            # Get trajectory points
            trajectory = obj.get('trajectory_4d', {}).get('points', [])

            # Find current point (closest frame index)
            current_point = self._find_point_at_frame(trajectory, frame_idx)

            if current_point is None:
                continue  # Object not visible in this frame

            # 1. Draw trajectory path
            if self.show_trajectories:
                self._draw_trajectory(
                    annotated,
                    trajectory_history.get(obj_id, []),
                    frame_idx,
                    color
                )

            # 2. Draw bounding box
            if self.show_boxes:
                self._draw_bounding_box(
                    annotated,
                    current_point,
                    color,
                    obj.get('semantic_label', 'object')
                )

            # 3. Draw label and info
            if self.show_labels or self.show_confidence:
                self._draw_label_info(
                    annotated,
                    current_point,
                    obj,
                    color
                )

        # Add frame info overlay
        self._draw_frame_info(annotated, frame_idx, annotation_result)

        return annotated

    def _draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Dict],
        current_frame: int,
        color: Tuple[int, int, int]
    ):
        """Draw trajectory path with fade effect"""
        if len(trajectory) < 2:
            return

        # Filter to recent frames only
        recent_points = [
            p for p in trajectory
            if p['frame'] <= current_frame and
            current_frame - p['frame'] <= self.trajectory_length
        ]

        if len(recent_points) < 2:
            return

        # Draw lines between consecutive points
        for i in range(1, len(recent_points)):
            pt1 = recent_points[i - 1]
            pt2 = recent_points[i]

            # Compute fade factor (older points are more transparent)
            age = current_frame - pt2['frame']
            alpha = 1.0 - (age / self.trajectory_length)
            alpha = max(0.3, alpha)  # Minimum visibility

            # Blend color with alpha
            faded_color = tuple(int(c * alpha) for c in color)

            # Draw line
            cv2.line(
                frame,
                (int(pt1['x']), int(pt1['y'])),
                (int(pt2['x']), int(pt2['y'])),
                faded_color,
                self.line_thickness
            )

    def _draw_bounding_box(
        self,
        frame: np.ndarray,
        point: Dict,
        color: Tuple[int, int, int],
        label: str
    ):
        """Draw bounding box around object"""
        x, y = int(point['x']), int(point['y'])

        # Estimate box size (simple heuristic: 40x40 pixels)
        box_size = 40
        x1 = max(0, x - box_size // 2)
        y1 = max(0, y - box_size // 2)
        x2 = min(frame.shape[1], x + box_size // 2)
        y2 = min(frame.shape[0], y + box_size // 2)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

        # Draw center point
        cv2.circle(frame, (x, y), 5, color, -1)

    def _draw_label_info(
        self,
        frame: np.ndarray,
        point: Dict,
        obj: Dict,
        color: Tuple[int, int, int]
    ):
        """Draw label and confidence information"""
        x, y = int(point['x']), int(point['y'])

        # Build text
        lines = []

        if self.show_labels:
            label = obj.get('semantic_label', 'object')
            lines.append(label)

        if self.show_confidence:
            if 'phase2_validation' in obj:
                conf = obj['phase2_validation'].get('cross_modal_confidence', 0.5)
                lines.append(f"Conf: {conf:.2f}")

        if self.show_depth:
            z = point.get('z', 0.0)
            lines.append(f"Depth: {z:.2f}m")

        # Draw text background
        text_y = y - 50
        for line in lines:
            text_size = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.line_thickness
            )[0]

            # Background rectangle
            cv2.rectangle(
                frame,
                (x - 5, text_y - text_size[1] - 5),
                (x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1
            )

            # Text
            cv2.putText(
                frame,
                line,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                self.line_thickness
            )

            text_y += text_size[1] + 10

    def _draw_frame_info(
        self,
        frame: np.ndarray,
        frame_idx: int,
        annotation_result: Dict
    ):
        """Draw frame index and metadata overlay"""
        # Top-left info
        info_lines = [
            f"Frame: {frame_idx}",
            f"Objects: {len(annotation_result.get('objects', []))}",
        ]

        # Add model confidence if available
        if 'annotation_metadata' in annotation_result:
            metadata = annotation_result['annotation_metadata']
            if 'model_confidence' in metadata:
                sam_conf = metadata['model_confidence'].get('sam2_avg_confidence', 0)
                info_lines.append(f"SAM: {sam_conf:.2f}")

        # Draw info
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 25

    def _build_trajectory_history(self, annotation_result: Dict) -> Dict:
        """
        Pre-compute trajectory history for efficient lookup

        Returns:
            Dict mapping object_id to list of trajectory points
        """
        history = {}

        for obj in annotation_result.get('objects', []):
            obj_id = obj.get('object_id')
            trajectory = obj.get('trajectory_4d', {}).get('points', [])

            history[obj_id] = trajectory

        return history

    def _find_point_at_frame(
        self,
        trajectory: List[Dict],
        frame_idx: int
    ) -> Optional[Dict]:
        """Find trajectory point closest to given frame index"""
        if not trajectory:
            return None

        # Find closest point
        closest = min(
            trajectory,
            key=lambda p: abs(p.get('frame', 0) - frame_idx)
        )

        # Only return if within reasonable distance (e.g., 5 frames)
        if abs(closest.get('frame', 0) - frame_idx) > 5:
            return None

        return closest

    def _generate_color_palette(self, n_colors: int) -> List[Tuple[int, int, int]]:
        """
        Generate distinct colors for objects

        Uses HSV color space for maximum distinction
        """
        colors = []

        for i in range(n_colors):
            hue = i / n_colors
            saturation = 0.9
            value = 0.9

            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)

            # Convert to BGR (OpenCV format) and scale to 0-255
            bgr = (
                int(rgb[2] * 255),
                int(rgb[1] * 255),
                int(rgb[0] * 255)
            )

            colors.append(bgr)

        return colors


def create_annotated_video(
    video_path: str,
    annotation_result: Dict,
    output_path: str,
    **overlay_options
) -> str:
    """
    Convenience function to create annotated video

    Args:
        video_path: Path to original video
        annotation_result: Annotation result from annotator
        output_path: Output video path
        **overlay_options: Options for VideoOverlay

    Returns:
        Path to created video
    """
    overlay = VideoOverlay(**overlay_options)
    return overlay.create_annotated_video(
        video_path,
        annotation_result,
        output_path
    )
