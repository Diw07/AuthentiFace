"""Tests for the video processing module."""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.video_processor import VideoProcessor


class TestVideoProcessor:
    """Tests for OpenCV-based video frame extraction."""

    @pytest.fixture
    def processor(self):
        return VideoProcessor(max_frames=10, strategy="uniform")

    @pytest.fixture
    def temp_video(self, tmp_path):
        """Create a minimal test video (30 frames, 640x480)."""
        video_path = str(tmp_path / "test_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

        for i in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()
        return video_path

    def test_processor_initializes(self, processor):
        """VideoProcessor should initialize with default settings."""
        assert processor.max_frames == 10
        assert processor.strategy == "uniform"

    def test_extract_frames_returns_list(self, processor, temp_video):
        """extract_frames should return a list of (frame, index) tuples."""
        frames = processor.extract_frames(temp_video)
        assert isinstance(frames, list)
        assert len(frames) > 0
        assert len(frames) <= 10  # max_frames

    def test_frames_are_numpy_arrays(self, processor, temp_video):
        """Each frame should be a numpy array."""
        frames = processor.extract_frames(temp_video)
        for frame, idx in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3  # H x W x C
            assert isinstance(idx, int)

    def test_uniform_sampling_count(self, temp_video):
        """Uniform sampling with max_frames=5 should return 5 frames."""
        processor = VideoProcessor(max_frames=5, strategy="uniform")
        frames = processor.extract_frames(temp_video)
        assert len(frames) == 5

    def test_video_info(self, processor, temp_video):
        """get_video_info should return correct metadata."""
        info = processor.get_video_info(temp_video)
        assert info['total_frames'] == 30
        assert info['width'] == 640
        assert info['height'] == 480
        assert info['fps'] > 0

    def test_invalid_path_raises_error(self, processor):
        """Should raise ValueError for non-existent video."""
        with pytest.raises(ValueError):
            processor.extract_frames("nonexistent_video.mp4")
