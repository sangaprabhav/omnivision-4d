"""
Video Preprocessing

Handles diverse video formats, codecs, resolutions, and preprocessing tasks.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import subprocess
import os
import tempfile
import logging

logger = logging.getLogger(__name__)


# Supported codecs (in order of preference)
SUPPORTED_CODECS = [
    'h264', 'avc1',  # H.264
    'h265', 'hevc',  # H.265
    'vp8', 'vp9',    # VP8/VP9
    'av1',           # AV1
    'mp4v', 'xvid'   # MPEG-4
]

# Supported containers
SUPPORTED_FORMATS = [
    '.mp4', '.avi', '.mov', '.mkv', '.webm',
    '.flv', '.wmv', '.m4v', '.mpg', '.mpeg'
]


@dataclass
class VideoMetadata:
    """Video file metadata"""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # seconds
    codec: str
    format: str  # Container format
    rotation: int  # Rotation angle (0, 90, 180, 270)
    file_size_mb: float
    is_valid: bool
    warnings: list


class VideoPreprocessor:
    """
    Preprocess videos for annotation pipeline

    Handles:
    - Format validation and conversion
    - Codec compatibility
    - Resolution adaptation
    - Rotation correction
    - Frame rate normalization
    - Corruption detection
    """

    def __init__(
        self,
        max_resolution: Tuple[int, int] = (1920, 1080),
        preserve_aspect: bool = True,
        target_fps: Optional[float] = None,
        enable_transcoding: bool = True
    ):
        """
        Args:
            max_resolution: Maximum allowed resolution (width, height)
            preserve_aspect: Preserve aspect ratio when resizing
            target_fps: Target frame rate (None = keep original)
            enable_transcoding: Enable codec transcoding if needed
        """
        self.max_resolution = max_resolution
        self.preserve_aspect = preserve_aspect
        self.target_fps = target_fps
        self.enable_transcoding = enable_transcoding

    def preprocess(self, video_path: str) -> 'ProcessedVideo':
        """
        Preprocess video for annotation

        Args:
            video_path: Path to input video

        Returns:
            ProcessedVideo with preprocessed path and metadata
        """
        logger.info(f"Preprocessing video: {video_path}")

        # 1. Validate video
        metadata = self.extract_metadata(video_path)

        if not metadata.is_valid:
            raise ValueError(f"Invalid video: {metadata.warnings}")

        # 2. Handle rotation
        if metadata.rotation != 0:
            logger.info(f"Rotating video by {metadata.rotation}°")
            video_path = self._rotate_video(video_path, metadata.rotation)
            metadata = self.extract_metadata(video_path)  # Re-extract after rotation

        # 3. Check codec compatibility
        if self.enable_transcoding and not self._is_codec_supported(metadata.codec):
            logger.info(f"Transcoding from {metadata.codec} to h264")
            video_path = self._transcode_video(video_path, target_codec='h264')
            metadata = self.extract_metadata(video_path)

        # 4. Compute target resolution
        target_res = self._compute_target_resolution(
            (metadata.width, metadata.height)
        )

        # 5. Resize if needed
        needs_resize = (
            metadata.width > target_res[0] or
            metadata.height > target_res[1]
        )

        if needs_resize:
            logger.info(f"Resizing from {metadata.width}x{metadata.height} to {target_res[0]}x{target_res[1]}")
            video_path = self._resize_video(video_path, target_res)
            metadata = self.extract_metadata(video_path)

        # 6. Normalize frame rate if requested
        if self.target_fps and abs(metadata.fps - self.target_fps) > 1.0:
            logger.info(f"Normalizing FPS from {metadata.fps:.1f} to {self.target_fps}")
            video_path = self._change_fps(video_path, self.target_fps)
            metadata = self.extract_metadata(video_path)

        logger.info(f"✓ Preprocessing complete: {metadata.width}x{metadata.height} @ {metadata.fps:.1f} fps")

        return ProcessedVideo(
            path=video_path,
            metadata=metadata,
            target_resolution=target_res,
            preprocessing_applied=True
        )

    def extract_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract comprehensive video metadata

        Args:
            video_path: Path to video file

        Returns:
            VideoMetadata with all video properties
        """
        if not os.path.exists(video_path):
            return VideoMetadata(
                path=video_path,
                width=0, height=0, fps=0, total_frames=0,
                duration=0, codec='', format='',
                rotation=0, file_size_mb=0,
                is_valid=False,
                warnings=['File does not exist']
            )

        warnings = []

        try:
            # Open video
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return VideoMetadata(
                    path=video_path,
                    width=0, height=0, fps=0, total_frames=0,
                    duration=0, codec='', format='',
                    rotation=0, file_size_mb=0,
                    is_valid=False,
                    warnings=['Cannot open video file']
                )

            # Extract basic properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Handle invalid FPS
            if fps <= 0 or fps > 240:
                warnings.append(f"Suspicious FPS: {fps}, assuming 30")
                fps = 30.0

            duration = total_frames / fps if fps > 0 else 0

            # Get codec (FourCC code)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = self._fourcc_to_string(fourcc)

            cap.release()

            # Get file format and size
            file_format = os.path.splitext(video_path)[1].lower()
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

            # Get rotation from metadata (if available)
            rotation = self._get_rotation(video_path)

            # Validation checks
            is_valid = True

            if width <= 0 or height <= 0:
                warnings.append(f"Invalid resolution: {width}x{height}")
                is_valid = False

            if total_frames <= 0:
                warnings.append(f"No frames detected")
                is_valid = False

            if file_format not in SUPPORTED_FORMATS:
                warnings.append(f"Unsupported format: {file_format}")

            return VideoMetadata(
                path=video_path,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                codec=codec,
                format=file_format,
                rotation=rotation,
                file_size_mb=file_size_mb,
                is_valid=is_valid,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return VideoMetadata(
                path=video_path,
                width=0, height=0, fps=0, total_frames=0,
                duration=0, codec='', format='',
                rotation=0, file_size_mb=0,
                is_valid=False,
                warnings=[f'Metadata extraction failed: {str(e)}']
            )

    def _compute_target_resolution(
        self,
        original_resolution: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Compute target resolution respecting max limits and aspect ratio

        Args:
            original_resolution: (width, height)

        Returns:
            target_resolution: (width, height)
        """
        orig_width, orig_height = original_resolution
        max_width, max_height = self.max_resolution

        # If already within limits, keep original
        if orig_width <= max_width and orig_height <= max_height:
            return (orig_width, orig_height)

        if self.preserve_aspect:
            # Compute scaling factor
            scale_w = max_width / orig_width
            scale_h = max_height / orig_height
            scale = min(scale_w, scale_h)

            target_width = int(orig_width * scale)
            target_height = int(orig_height * scale)

            # Ensure even dimensions (required by some codecs)
            target_width = target_width - (target_width % 2)
            target_height = target_height - (target_height % 2)

            return (target_width, target_height)
        else:
            # Simply clamp to max
            return (
                min(orig_width, max_width),
                min(orig_height, max_height)
            )

    def _is_codec_supported(self, codec: str) -> bool:
        """Check if codec is supported"""
        return codec.lower() in SUPPORTED_CODECS

    def _fourcc_to_string(self, fourcc: int) -> str:
        """Convert FourCC code to string"""
        try:
            return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        except:
            return "unknown"

    def _get_rotation(self, video_path: str) -> int:
        """
        Get video rotation from metadata

        Uses ffprobe if available, otherwise returns 0
        """
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream_tags=rotate',
                    '-of', 'default=nw=1:nk=1',
                    video_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())
        except:
            pass

        return 0

    def _rotate_video(self, video_path: str, rotation: int) -> str:
        """
        Rotate video by specified angle

        Uses ffmpeg if available
        """
        if rotation not in [90, 180, 270]:
            logger.warning(f"Unsupported rotation: {rotation}, skipping")
            return video_path

        try:
            # Create temporary output file
            output_path = tempfile.mktemp(suffix='.mp4')

            # FFmpeg rotation filter
            if rotation == 90:
                transpose = 'transpose=1'
            elif rotation == 180:
                transpose = 'transpose=1,transpose=1'
            elif rotation == 270:
                transpose = 'transpose=2'

            # Run ffmpeg
            subprocess.run(
                [
                    'ffmpeg', '-i', video_path,
                    '-vf', transpose,
                    '-c:a', 'copy',
                    '-y', output_path
                ],
                capture_output=True,
                check=True,
                timeout=300
            )

            return output_path

        except Exception as e:
            logger.error(f"Rotation failed: {e}")
            return video_path

    def _transcode_video(self, video_path: str, target_codec: str = 'h264') -> str:
        """
        Transcode video to target codec

        Uses ffmpeg
        """
        try:
            output_path = tempfile.mktemp(suffix='.mp4')

            subprocess.run(
                [
                    'ffmpeg', '-i', video_path,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-y', output_path
                ],
                capture_output=True,
                check=True,
                timeout=600
            )

            return output_path

        except Exception as e:
            logger.error(f"Transcoding failed: {e}")
            return video_path

    def _resize_video(self, video_path: str, target_resolution: Tuple[int, int]) -> str:
        """
        Resize video to target resolution

        Uses ffmpeg
        """
        try:
            output_path = tempfile.mktemp(suffix='.mp4')
            width, height = target_resolution

            subprocess.run(
                [
                    'ffmpeg', '-i', video_path,
                    '-vf', f'scale={width}:{height}',
                    '-c:a', 'copy',
                    '-y', output_path
                ],
                capture_output=True,
                check=True,
                timeout=600
            )

            return output_path

        except Exception as e:
            logger.error(f"Resize failed: {e}")
            return video_path

    def _change_fps(self, video_path: str, target_fps: float) -> str:
        """
        Change video frame rate

        Uses ffmpeg
        """
        try:
            output_path = tempfile.mktemp(suffix='.mp4')

            subprocess.run(
                [
                    'ffmpeg', '-i', video_path,
                    '-r', str(target_fps),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-c:a', 'copy',
                    '-y', output_path
                ],
                capture_output=True,
                check=True,
                timeout=600
            )

            return output_path

        except Exception as e:
            logger.error(f"FPS change failed: {e}")
            return video_path


@dataclass
class ProcessedVideo:
    """Result of video preprocessing"""
    path: str
    metadata: VideoMetadata
    target_resolution: Tuple[int, int]
    preprocessing_applied: bool


def preprocess_video(
    video_path: str,
    max_resolution: Tuple[int, int] = (1920, 1080),
    **kwargs
) -> ProcessedVideo:
    """
    Convenience function for video preprocessing

    Args:
        video_path: Path to video file
        max_resolution: Maximum resolution
        **kwargs: Additional VideoPreprocessor options

    Returns:
        ProcessedVideo
    """
    preprocessor = VideoPreprocessor(max_resolution=max_resolution, **kwargs)
    return preprocessor.preprocess(video_path)
