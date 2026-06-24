"""Tests for the face extraction module."""

import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import INPUT_SIZE


class TestFaceExtractor:
    """Tests for RetinaFace-based face extraction."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Try to import FaceExtractor — skip if insightface not installed."""
        try:
            from preprocessing.face_extractor import FaceExtractor
            self.FaceExtractor = FaceExtractor
        except ImportError:
            pytest.skip("insightface not installed")

    def test_extractor_initializes(self):
        """FaceExtractor should initialize without errors."""
        extractor = self.FaceExtractor(ctx_id=-1)
        assert extractor is not None

    def test_extract_faces_returns_list(self):
        """extract_faces should return a list."""
        extractor = self.FaceExtractor(ctx_id=-1)
        # Create a dummy image (no face expected)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.extract_faces(dummy_image)
        assert isinstance(result, list)

    def test_face_crop_correct_size(self):
        """Extracted face crops should be INPUT_SIZE x INPUT_SIZE."""
        extractor = self.FaceExtractor(ctx_id=-1)
        # Use a real face image from sample dir if available
        sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample")
        if not os.path.exists(sample_dir) or len(os.listdir(sample_dir)) == 0:
            pytest.skip("No sample images available")

        import cv2
        for fname in os.listdir(sample_dir):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(sample_dir, fname))
                faces = extractor.extract_faces(img)
                for face_crop, bbox, conf in faces:
                    assert face_crop.shape == (INPUT_SIZE, INPUT_SIZE, 3), \
                        f"Expected ({INPUT_SIZE}, {INPUT_SIZE}, 3), got {face_crop.shape}"
                break

    def test_empty_image_returns_nothing(self):
        """An empty/blank image should return no faces."""
        extractor = self.FaceExtractor(ctx_id=-1)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        result = extractor.extract_faces(blank)
        assert len(result) == 0

    def test_none_image_returns_nothing(self):
        """Passing None should return empty list."""
        extractor = self.FaceExtractor(ctx_id=-1)
        result = extractor.extract_faces(None)
        assert result == []
