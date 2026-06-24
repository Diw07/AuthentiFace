"""Tests for the ConvNeXt-Tiny deepfake detector model."""

import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import INPUT_SIZE


class TestDeepfakeDetector:
    """Tests for the ConvNeXt-Tiny model architecture."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from models.convnext_detector import DeepfakeDetector
            self.DeepfakeDetector = DeepfakeDetector
        except ImportError:
            pytest.skip("Required dependencies not installed")

    def test_model_initializes(self):
        """Model should initialize without errors."""
        model = self.DeepfakeDetector(pretrained_backbone=False)
        assert model is not None

    def test_forward_pass_shape(self):
        """Forward pass should return (B, 1) logit tensor."""
        model = self.DeepfakeDetector(pretrained_backbone=False)
        model.eval()
        dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"

    def test_batch_forward_pass(self):
        """Forward pass with batch size > 1."""
        model = self.DeepfakeDetector(pretrained_backbone=False)
        model.eval()
        dummy_input = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"

    def test_sigmoid_output_range(self):
        """Sigmoid of model output should be in [0, 1]."""
        model = self.DeepfakeDetector(pretrained_backbone=False)
        model.eval()
        dummy_input = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            logits = model(dummy_input)
            probs = torch.sigmoid(logits)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_model_parameter_count(self):
        """Model should have roughly ~28.6M parameters."""
        model = self.DeepfakeDetector(pretrained_backbone=False)
        param_count = sum(p.numel() for p in model.parameters())
        # Allow some flexibility (28-30M expected)
        assert 27_000_000 < param_count < 31_000_000, \
            f"Unexpected param count: {param_count:,}"
