"""
Grad-CAM visualization for the DeepFake Detector.
Highlights regions that influenced the model's prediction.
"""

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMVisualizer:
    """
    Generates Grad-CAM heatmaps for the ConvNeXt-Tiny deepfake detector.

    Target layer: backbone.features[7] (last ConvNeXt stage)
    — this is where the highest-level features are, best for understanding
      what spatial regions drive the real/fake decision.
    """

    def __init__(self, model, device):
        """
        Args:
            model: DeepfakeDetector instance (nn.Module).
            device: torch device.
        """
        self.model = model
        self.device = device

        # Target the last ConvNeXt feature stage
        target_layer = self.model.backbone.features[7]
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def generate(self, input_tensor, original_face_rgb):
        """
        Generate Grad-CAM overlay on the original face image.

        Args:
            input_tensor: Preprocessed tensor (1, 3, 224, 224), already on device.
            original_face_rgb: Original face crop (numpy, RGB, 224x224).

        Returns:
            Numpy array (RGB, 224x224) with heatmap overlay, or None on failure.
        """
        try:
            # Generate CAM
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]  # Take first image in batch

            # Normalize original face to [0, 1] for overlay
            face_normalized = original_face_rgb.astype(np.float32) / 255.0

            # Resize face if needed
            if face_normalized.shape[:2] != grayscale_cam.shape[:2]:
                face_normalized = cv2.resize(
                    face_normalized,
                    (grayscale_cam.shape[1], grayscale_cam.shape[0]),
                )

            # Create overlay
            overlay = show_cam_on_image(
                face_normalized, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
            )

            # Analyze face regions from the raw grayscale cam
            region_scores = self.analyze_regions(grayscale_cam)

            return overlay, region_scores

        except Exception as e:
            print(f"[!] Grad-CAM generation failed: {e}")
            return None, None

    def analyze_regions(self, grayscale_cam):
        """
        Calculates the average activation (suspicion score) for 6 face regions.
        Expects a 224x224 heatmap array. Returns a dict of percentages (0-100%).
        """
        if grayscale_cam.shape != (224, 224):
            grayscale_cam = cv2.resize(grayscale_cam, (224, 224))

        regions = {
            "Forehead":  grayscale_cam[0:56, :],
            "Left Eye":  grayscale_cam[56:106, 0:112],
            "Right Eye": grayscale_cam[56:106, 112:224],
            "Nose":      grayscale_cam[106:146, :],
            "Mouth":     grayscale_cam[146:186, :],
            "Jawline":   grayscale_cam[186:224, :]
        }

        # Calculate mean activation per region (scaled to 100%)
        scores = {}
        for name, crop in regions.items():
            score = float(np.mean(crop)) * 100
            scores[name] = round(score, 1)

        return scores
