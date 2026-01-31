"""
Adaptive Frame Sampling Module

Implements motion-aware frame sampling to prioritize high-action segments
and optimize annotation quality.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotionSegment:
    """Represents a segment of video with specific motion characteristics"""
    start_frame: int
    end_frame: int
    motion_intensity: float  # Average optical flow magnitude
    segment_type: str  # "static", "low_motion", "medium_motion", "high_motion"


class AdaptiveFrameSampler:
    """
    Motion-aware adaptive frame sampling

    Samples more densely in high-motion segments and sparsely in static regions.
    """

    def __init__(
        self,
        static_threshold: float = 1.0,      # Flow magnitude < 1.0 = static
        low_motion_threshold: float = 5.0,  # Flow magnitude < 5.0 = low motion
        high_motion_threshold: float = 15.0, # Flow magnitude > 15.0 = high motion
        scene_change_threshold: float = 0.3  # Histogram difference > 0.3 = scene change
    ):
        self.static_threshold = static_threshold
        self.low_motion_threshold = low_motion_threshold
        self.high_motion_threshold = high_motion_threshold
        self.scene_change_threshold = scene_change_threshold

    def sample_frames(
        self,
        video_path: str,
        max_frames: int = 64,
        min_frames: int = 16,
        uniform_fallback: bool = True
    ) -> Tuple[List[int], Dict[str, any]]:
        """
        Adaptively sample frames based on motion and scene changes

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to sample
            min_frames: Minimum number of frames to sample
            uniform_fallback: If True, fall back to uniform sampling on error

        Returns:
            frame_indices: List of frame indices to sample
            metadata: Dictionary with sampling statistics
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"Analyzing video: {total_frames} frames @ {fps:.1f} fps")

            # Step 1: Compute optical flow for motion analysis
            motion_scores = self._compute_motion_scores(cap, total_frames)

            # Step 2: Detect scene changes
            scene_changes = self._detect_scene_changes(cap, total_frames)

            # Step 3: Segment video by motion intensity
            segments = self._segment_by_motion(motion_scores)

            # Step 4: Allocate frames to each segment based on motion
            frame_indices = self._allocate_frames(
                segments,
                scene_changes,
                total_frames,
                max_frames,
                min_frames
            )

            cap.release()

            # Metadata
            metadata = {
                "total_frames": total_frames,
                "sampled_frames": len(frame_indices),
                "fps": fps,
                "motion_segments": len(segments),
                "scene_changes": len(scene_changes),
                "avg_motion_intensity": float(np.mean(motion_scores)),
                "sampling_strategy": "adaptive_motion_based"
            }

            logger.info(
                f"Adaptive sampling: {len(frame_indices)} frames selected "
                f"({len(segments)} motion segments, {len(scene_changes)} scene changes)"
            )

            return sorted(frame_indices), metadata

        except Exception as e:
            logger.warning(f"Adaptive sampling failed: {e}")

            if uniform_fallback:
                logger.info("Falling back to uniform sampling")
                return self._uniform_sample(video_path, max_frames)
            else:
                raise

    def _compute_motion_scores(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        sample_rate: int = 5  # Analyze every 5th frame for speed
    ) -> np.ndarray:
        """
        Compute motion scores using optical flow

        Returns array of motion magnitudes for each frame
        """
        motion_scores = np.zeros(total_frames)
        prev_gray = None

        for frame_idx in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Convert to grayscale and resize for speed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))

            if prev_gray is not None:
                # Compute dense optical flow (Farneback)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )

                # Compute flow magnitude
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                avg_magnitude = magnitude.mean()

                # Store motion score
                motion_scores[frame_idx] = avg_magnitude

            prev_gray = gray.copy()

        # Interpolate missing values (frames not analyzed)
        if sample_rate > 1:
            analyzed_indices = np.arange(0, total_frames, sample_rate)
            analyzed_scores = motion_scores[analyzed_indices]

            # Linear interpolation
            motion_scores = np.interp(
                np.arange(total_frames),
                analyzed_indices,
                analyzed_scores
            )

        return motion_scores

    def _detect_scene_changes(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        sample_rate: int = 10  # Check every 10th frame
    ) -> List[int]:
        """
        Detect scene changes using histogram comparison

        Returns list of frame indices where scene changes occur
        """
        scene_changes = [0]  # Always include first frame
        prev_hist = None

        for frame_idx in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Compute color histogram
            frame_small = cv2.resize(frame, (160, 120))
            hist = cv2.calcHist(
                [frame_small],
                [0, 1, 2],
                None,
                [8, 8, 8],
                [0, 256, 0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                # Compare histograms using Chi-Square distance
                diff = cv2.compareHist(
                    prev_hist,
                    hist,
                    cv2.HISTCMP_CHISQR
                )

                # Threshold for scene change
                if diff > self.scene_change_threshold:
                    scene_changes.append(frame_idx)
                    logger.debug(f"Scene change detected at frame {frame_idx} (diff={diff:.3f})")

            prev_hist = hist.copy()

        # Always include last frame
        scene_changes.append(total_frames - 1)

        return sorted(set(scene_changes))

    def _segment_by_motion(self, motion_scores: np.ndarray) -> List[MotionSegment]:
        """
        Segment video into regions by motion intensity

        Returns list of motion segments
        """
        segments = []
        current_start = 0
        current_type = self._classify_motion(motion_scores[0])

        for i in range(1, len(motion_scores)):
            motion_type = self._classify_motion(motion_scores[i])

            # Check if segment type changed
            if motion_type != current_type or i == len(motion_scores) - 1:
                # Compute average motion for segment
                avg_motion = motion_scores[current_start:i].mean()

                segments.append(MotionSegment(
                    start_frame=current_start,
                    end_frame=i,
                    motion_intensity=float(avg_motion),
                    segment_type=current_type
                ))

                current_start = i
                current_type = motion_type

        return segments

    def _classify_motion(self, motion_magnitude: float) -> str:
        """Classify motion intensity into categories"""
        if motion_magnitude < self.static_threshold:
            return "static"
        elif motion_magnitude < self.low_motion_threshold:
            return "low_motion"
        elif motion_magnitude < self.high_motion_threshold:
            return "medium_motion"
        else:
            return "high_motion"

    def _allocate_frames(
        self,
        segments: List[MotionSegment],
        scene_changes: List[int],
        total_frames: int,
        max_frames: int,
        min_frames: int
    ) -> List[int]:
        """
        Allocate frame budget across segments based on motion intensity

        Sampling density:
        - high_motion: every 2 frames
        - medium_motion: every 5 frames
        - low_motion: every 10 frames
        - static: every 20 frames
        """
        frame_indices = set()

        # Always include scene changes
        frame_indices.update(scene_changes)

        # Define sampling rates by motion type
        sampling_rates = {
            "high_motion": 2,
            "medium_motion": 5,
            "low_motion": 10,
            "static": 20
        }

        # Sample each segment
        for segment in segments:
            rate = sampling_rates[segment.segment_type]

            # Sample frames in this segment
            segment_frames = list(range(
                segment.start_frame,
                segment.end_frame,
                rate
            ))

            frame_indices.update(segment_frames)

            # Always include segment boundaries
            frame_indices.add(segment.start_frame)
            frame_indices.add(segment.end_frame - 1)

        # Convert to sorted list
        frame_indices = sorted(frame_indices)

        # Enforce max_frames constraint
        if len(frame_indices) > max_frames:
            # Downsample by selecting evenly spaced frames
            logger.warning(
                f"Adaptive sampling produced {len(frame_indices)} frames, "
                f"downsampling to {max_frames}"
            )
            step = len(frame_indices) / max_frames
            frame_indices = [
                frame_indices[int(i * step)]
                for i in range(max_frames)
            ]

        # Enforce min_frames constraint
        if len(frame_indices) < min_frames:
            logger.warning(
                f"Adaptive sampling produced only {len(frame_indices)} frames, "
                f"upsampling to {min_frames}"
            )
            # Add uniformly sampled frames to reach min_frames
            uniform_indices = np.linspace(0, total_frames - 1, min_frames, dtype=int)
            frame_indices = sorted(set(frame_indices) | set(uniform_indices))

        return frame_indices

    def _uniform_sample(self, video_path: str, max_frames: int) -> Tuple[List[int], Dict]:
        """Fallback to uniform sampling"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

        metadata = {
            "total_frames": total_frames,
            "sampled_frames": len(indices),
            "fps": fps,
            "sampling_strategy": "uniform_fallback"
        }

        return indices, metadata


def sample_frames_adaptive(
    video_path: str,
    max_frames: int = 64,
    enable_adaptive: bool = True
) -> Tuple[List[int], Dict[str, any]]:
    """
    Convenience function for adaptive frame sampling

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to sample
        enable_adaptive: If False, use uniform sampling

    Returns:
        frame_indices: List of selected frame indices
        metadata: Sampling statistics
    """
    if enable_adaptive:
        sampler = AdaptiveFrameSampler()
        return sampler.sample_frames(video_path, max_frames)
    else:
        # Uniform sampling
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

        metadata = {
            "total_frames": total_frames,
            "sampled_frames": len(indices),
            "fps": fps,
            "sampling_strategy": "uniform"
        }

        return indices, metadata
