"""
Configuration for the DeepFake Detector project.
All hyperparameters, paths, and device settings in one place.
"""

import os
import torch

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PRETRAINED_DIR = os.path.join(BASE_DIR, "models", "pretrained")
UPLOAD_DIR = os.path.join(BASE_DIR, "app", "uploads")
TEMP_DIR = os.path.join(BASE_DIR, "app", "temp")

# Create directories if they don't exist
for d in [SAMPLE_DIR, RAW_DIR, PROCESSED_DIR, PRETRAINED_DIR, UPLOAD_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Device ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model ───────────────────────────────────────────────
MODEL_NAME = "convnext_tiny"
NUM_CLASSES = 1                     # Binary classification (sigmoid output)
INPUT_SIZE = 224                    # ConvNeXt-Tiny expects 224x224
CLASSIFIER_HIDDEN = 256             # Hidden layer in custom classifier head
DROPOUT_RATE = 0.3
MODEL_WEIGHTS_FILE = os.path.join(PRETRAINED_DIR, "deepfake_convnext_tiny.pth")

# ─── Face Detection (RetinaFace) ─────────────────────────
FACE_DET_SIZE = (640, 640)          # RetinaFace detection resolution
FACE_CONFIDENCE_THRESHOLD = 0.3     # Minimum confidence to accept a face detection (lowered for better recall)
FACE_MARGIN = 30                    # Percentage margin around detected face (matches V3 training: 0.3)
FACE_OUTPUT_SIZE = INPUT_SIZE       # Crop faces to model input size

# ─── Video Processing ────────────────────────────────────
MAX_FRAMES_PER_VIDEO = 30           # Max frames to sample from a video
FRAME_SAMPLING_STRATEGY = "uniform" # "uniform" or "fps"
FRAMES_PER_SECOND = 1               # If strategy is "fps", extract 1 frame/sec

# ─── Training (for Kaggle notebook reference) ────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 30
FREEZE_EPOCHS = 3                   # Freeze backbone for first N epochs
EARLY_STOP_PATIENCE = 5
SCHEDULER = "cosine"                # CosineAnnealingLR

# ─── Image Normalization (ImageNet stats) ─────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Inference ────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5          # Above = FAKE, below = REAL
VIDEO_AGGREGATION = "weighted"      # "majority" or "weighted"

# ─── API ──────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 100            # Max file upload size
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
