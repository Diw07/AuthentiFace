"""
DeepFake Detector model using ConvNeXt-Tiny.
Handles model definition, loading weights, and image/video inference.
"""

import os
import base64
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from utils.config import (
    DEVICE,
    INPUT_SIZE,
    CLASSIFIER_HIDDEN,
    DROPOUT_RATE,
    MODEL_WEIGHTS_FILE,
    CONFIDENCE_THRESHOLD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_FRAMES_PER_VIDEO,
    VIDEO_AGGREGATION,
)
from preprocessing.face_extractor import FaceExtractor
from preprocessing.video_processor import VideoProcessor
from utils.gradcam import GradCAMVisualizer


class DeepfakeDetector(nn.Module):
    """
    ConvNeXt-Tiny based binary classifier for deepfake detection.

    Architecture:
        ConvNeXt-Tiny backbone (pretrained ImageNet-1K)
        → Custom classifier: Linear(768, 256) → ReLU → Dropout → Linear(256, 1)
    """

    def __init__(self, pretrained_backbone=True):
        super().__init__()

        # Load ConvNeXt-Tiny backbone
        if pretrained_backbone:
            self.backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
        else:
            self.backbone = models.convnext_tiny(weights=None)

        # Replace the classifier head
        # ConvNeXt-Tiny classifier: Sequential(LayerNorm, Flatten, Linear(768, 1000))
        # We replace the last Linear layer with our custom head
        self.backbone.classifier[2] = nn.Sequential(
            nn.Linear(768, CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, 3, 224, 224)

        Returns:
            Raw logits of shape (B, 1). Apply sigmoid for probability.
        """
        return self.backbone(x)


class DeepfakePredictor:
    """
    High-level inference class that ties together:
    - Face extraction (RetinaFace)
    - Classification — HuggingFace model (primary, better generalization)
    - Grad-CAM visualization (ConvNeXt-Tiny, kept for explainability)
    - Video analysis with frame aggregation
    """

    HF_MODEL_ID = "dima806/deepfake_vs_real_image_detection"

    def __init__(self, weights_path=None, device=None):
        """
        Args:
            weights_path: Path to trained .pth file. If None, uses config default.
            device: torch device. If None, uses config default (auto GPU/CPU).
        """
        self.device = device or DEVICE
        self.weights_path = weights_path or MODEL_WEIGHTS_FILE

        # ── Primary classifier: HuggingFace pre-trained model ──
        from transformers import pipeline
        print("[*] Loading HuggingFace deepfake detection model...")
        self.hf_classifier = pipeline(
            "image-classification",
            model=self.HF_MODEL_ID,
            device=0 if str(self.device) == "cuda" else -1,
        )
        print(f"[✓] HuggingFace model loaded: {self.HF_MODEL_ID}")

        # ── Secondary model: ConvNeXt for Grad-CAM only ──
        self.model = DeepfakeDetector(pretrained_backbone=False)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

        # Initialize face extractor (CPU for inference)
        self.face_extractor = FaceExtractor(ctx_id=-1)

        # Initialize video processor
        self.video_processor = VideoProcessor()

        # Initialize Grad-CAM (uses ConvNeXt model)
        self.gradcam = GradCAMVisualizer(self.model, self.device)

        # Image transform for ConvNeXt (same normalization as training)
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_weights(self):
        """Load trained model weights from .pth file."""
        if os.path.exists(self.weights_path):
            state_dict = torch.load(self.weights_path, map_location=self.device)
            # Handle weights from V3 Kaggle script which used 'self.b' instead of 'self.backbone'
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('b.'):
                    # Map `b.` to `backbone.`
                    new_state_dict[k.replace('b.', 'backbone.', 1)] = v
                else:
                    new_state_dict[k] = v
                    
            # Try to load, allowing strict=False in case of minor feature discrepancies 
            # (e.g., if the Kaggle script added an extra wrapper layer)
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing:
                print(f"[!] Warning: Missing keys when loading weights: {missing[:5]}...")
            if unexpected:
                print(f"[!] Warning: Unexpected keys in weights file: {unexpected[:5]}...")
                
            print(f"[✓] Loaded weights from: {self.weights_path}")
        else:
            print(f"[!] No weights found at: {self.weights_path}")
            print("[!] Model will use random weights. Train or download weights first.")

    def predict_image(self, image, return_gradcam=True):
        """
        Predict whether an image is real or fake.

        Args:
            image: numpy array (BGR from cv2) or PIL Image or file path string.
            return_gradcam: If True, generate Grad-CAM heatmap.

        Returns:
            dict with keys:
                - label: "REAL" or "FAKE"
                - confidence: float (0-1, probability of being fake)
                - face_bbox: [x1, y1, x2, y2] or None
                - face_detected: bool
                - gradcam_image: base64 encoded PNG string (if return_gradcam=True)
        """
        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return self._error_result("Cannot read image file")

        # Detect face
        face_result = self.face_extractor.extract_single_face(image)

        if face_result is not None:
            face_crop, bbox, det_confidence = face_result  # 224x224 RGB for Grad-CAM
            face_detected = True
            # Extract HIGH-RES face crop for HF model (don't resize to 224)
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            margin = 0.3
            mx = int((x2 - x1) * margin)
            my = int((y2 - y1) * margin)
            hx1, hy1 = max(0, x1 - mx), max(0, y1 - my)
            hx2, hy2 = min(w, x2 + mx), min(h, y2 + my)
            hires_crop = cv2.cvtColor(image[hy1:hy2, hx1:hx2], cv2.COLOR_BGR2RGB)
            hires_pil = Image.fromarray(hires_crop)
        else:
            # Fallback: no face detected — use the full original image
            print("[!] No face detected by RetinaFace. Using full-image fallback.")
            h, w = image.shape[:2]
            # High-res: full image for HF model (no resizing!)
            hires_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Low-res: center-crop to 224x224 for ConvNeXt Grad-CAM
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            cropped = image[top:top + min_dim, left:left + min_dim]
            face_crop = cv2.cvtColor(
                cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            bbox = [left, top, left + min_dim, top + min_dim]
            det_confidence = 0.0
            face_detected = False

        # ── Primary classification via HuggingFace model (full resolution!) ──
        hf_results = self.hf_classifier(hires_pil)
        # Parse HF output: [{"label": "Fake", "score": 0.95}, {"label": "Real", "score": 0.05}]
        fake_score = 0.0
        for item in hf_results:
            if item["label"].lower() == "fake":
                fake_score = item["score"]
                break

        probability = fake_score
        label = "FAKE" if probability >= CONFIDENCE_THRESHOLD else "REAL"

        # ── Grad-CAM via ConvNeXt (for explainability only, uses 224px crop) ──
        face_pil = Image.fromarray(face_crop)
        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

        result = {
            "label": label,
            "confidence": round(probability, 4),
            "face_bbox": bbox,
            "face_detected": True,  # Always True now (we have a fallback)
            "detection_confidence": round(det_confidence, 4),
        }

        if not face_detected:
            result["fallback_used"] = True

        # Generate Grad-CAM
        if return_gradcam:
            gradcam_img, region_scores = self.gradcam.generate(input_tensor, face_crop)
            result["gradcam_image"] = self._image_to_base64(gradcam_img)
            result["region_scores"] = region_scores

        return result

    def predict_video(self, video_path, return_gradcam=True):
        """
        Predict whether a video is real or fake by analyzing multiple frames.

        Args:
            video_path: Path to video file.
            return_gradcam: If True, generate Grad-CAM for each frame.

        Returns:
            dict with keys:
                - verdict: "REAL" or "FAKE"
                - confidence: float (aggregated)
                - total_frames_analyzed: int
                - frames_with_faces: int
                - per_frame_results: list of per-frame dicts
                - video_info: dict of video metadata
        """
        # Get video info
        video_info = self.video_processor.get_video_info(video_path)

        # Extract frames
        frames = self.video_processor.extract_frames(video_path)

        if not frames:
            return {
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "error": "Could not extract frames from video",
                "video_info": video_info,
            }

        # Analyze each frame
        per_frame_results = []
        fake_scores = []
        all_region_scores = {}

        for frame, frame_idx in frames:
            result = self.predict_image(frame, return_gradcam=return_gradcam)
            result["frame_index"] = frame_idx

            # Include the original analyzed frame as base64 for frontend display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (320, 240))
            result["frame_image"] = self._image_to_base64(frame_resized)

            if result["face_detected"]:
                fake_scores.append(result["confidence"])

            # Aggregate region scores
            if "region_scores" in result and result["region_scores"]:
                for region, score in result["region_scores"].items():
                    if region not in all_region_scores:
                        all_region_scores[region] = []
                    all_region_scores[region].append(score)

            per_frame_results.append(result)

        # Aggregate results — smart threshold detection
        if not fake_scores:
            verdict = "UNKNOWN"
            avg_confidence = 0.0
        else:
            fake_count = sum(1 for s in fake_scores if s >= CONFIDENCE_THRESHOLD)
            fake_ratio = fake_count / len(fake_scores)
            max_fake = max(fake_scores)
            avg_confidence = sum(fake_scores) / len(fake_scores)

            # Verdict: FAKE if >20% of frames are fake OR any frame is >85% fake
            if fake_ratio >= 0.2 or max_fake >= 0.85:
                verdict = "FAKE"
                # Use the stronger signal as confidence
                avg_confidence = max(avg_confidence, fake_ratio, max_fake)
            else:
                verdict = "REAL"

        # Average region scores across frames
        avg_region_scores = {}
        for region, scores in all_region_scores.items():
            avg_region_scores[region] = round(sum(scores) / len(scores), 1)

        return {
            "verdict": verdict,
            "confidence": round(avg_confidence, 4),
            "total_frames_analyzed": len(frames),
            "frames_with_faces": len(fake_scores),
            "per_frame_results": per_frame_results,
            "video_info": video_info,
            "region_scores": avg_region_scores if avg_region_scores else None,
        }

    @staticmethod
    def _error_result(message):
        """Return an error result dict."""
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "face_bbox": None,
            "face_detected": False,
            "error": message,
        }

    @staticmethod
    def _image_to_base64(image_array):
        """Convert numpy image array (RGB) to base64 PNG string."""
        if image_array is None:
            return None
        pil_image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
