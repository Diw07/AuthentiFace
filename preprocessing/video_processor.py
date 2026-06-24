"""
Video processing module.
Extracts frames from videos using OpenCV (backed by FFmpeg).
"""

import cv2
import numpy as np
from typing import List, Tuple

from utils.config import (
    MAX_FRAMES_PER_VIDEO,
    FRAME_SAMPLING_STRATEGY,
    FRAMES_PER_SECOND,
)


class VideoProcessor:
    """
    Extracts frames from video files using OpenCV.

    Usage:
        processor = VideoProcessor()
        frames = processor.extract_frames("path/to/video.mp4")
        # frames is a list of (frame_array, frame_index) tuples
    """

    def __init__(
        self,
        max_frames=MAX_FRAMES_PER_VIDEO,
        strategy=FRAME_SAMPLING_STRATEGY,
        fps=FRAMES_PER_SECOND,
    ):
        """
        Args:
            max_frames: Maximum number of frames to extract.
            strategy: "uniform" (evenly spaced) or "fps" (N frames per second).
            fps: Frames per second to extract (only if strategy="fps").
        """
        self.max_frames = max_frames
        self.strategy = strategy
        self.fps = fps

    def extract_frames(self, video_path: str) -> List[Tuple[np.ndarray, int]]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of tuples: (frame, frame_index)
            - frame: numpy array (BGR), as read by OpenCV
            - frame_index: original frame number in the video
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps if video_fps > 0 else 0

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")

        # Calculate which frames to extract
        frame_indices = self._get_frame_indices(total_frames, video_fps)

        # Extract frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append((frame, idx))

        cap.release()
        return frames

    def get_video_info(self, video_path: str) -> dict:
        """
        Get metadata about a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict with keys: total_frames, fps, duration, width, height
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        info = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration"] = (
            info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
        )

        cap.release()
        return info

    def _get_frame_indices(self, total_frames: int, video_fps: float) -> List[int]:
        """
        Calculate which frame indices to extract based on the sampling strategy.

        Args:
            total_frames: Total number of frames in the video.
            video_fps: Video frame rate.

        Returns:
            List of frame indices to extract.
        """
        if self.strategy == "uniform":
            # Evenly spaced frames across the entire video
            n_frames = min(self.max_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

        elif self.strategy == "fps":
            # Extract at a specific FPS rate
            if video_fps <= 0:
                # Fallback to uniform
                return self._get_frame_indices_uniform(total_frames)

            frame_interval = max(1, int(video_fps / self.fps))
            indices = list(range(0, total_frames, frame_interval))

            # Cap at max_frames
            if len(indices) > self.max_frames:
                step = len(indices) / self.max_frames
                indices = [indices[int(i * step)] for i in range(self.max_frames)]
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

        return indices

    def _get_frame_indices_uniform(self, total_frames: int) -> List[int]:
        """Fallback: uniform sampling."""
        n_frames = min(self.max_frames, total_frames)
        return np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()
