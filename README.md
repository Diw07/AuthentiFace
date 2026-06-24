# 🔍 DeepFake.ai — Synthetic Media Forensics Engine

A production-grade deepfake detection platform that analyzes **images and videos** for AI-generated manipulation using a dual-model architecture. Powered by a **HuggingFace ViT classifier** for robust detection and **ConvNeXt-Tiny Grad-CAM** for explainable forensics.

---

## ✨ Key Features

### 🧠 Dual-Model AI Architecture
| Component | Model | Purpose |
|---|---|---|
| **Classification** | HuggingFace ViT (`dima806/deepfake_vs_real_image_detection`) | FAKE/REAL prediction with 99%+ accuracy |
| **Explainability** | ConvNeXt-Tiny (custom-trained) | Grad-CAM heatmap & facial region analysis |
| **Face Detection** | RetinaFace (InsightFace `buffalo_l`) | Face extraction with intelligent fallback |

### 📸 Image Analysis
- Upload any image — faces are auto-detected and cropped
- Full-resolution input to the ViT classifier for maximum accuracy
- **Grad-CAM heatmap** overlaid on the detected face
- **6 facial region scores**: Forehead, Left Eye, Right Eye, Nose, Mouth, Jawline

### 🎬 Video Analysis
- Frame-by-frame temporal sampling with uniform extraction
- **Interactive frame timeline** — click any frame to inspect its Grad-CAM and per-frame confidence
- Smart verdict aggregation: FAKE if >20% of frames flagged OR any frame exceeds 85% suspicion
- Aggregated region scores across all analyzed frames

### 🛡️ Robust Face Detection
- Primary: **RetinaFace** (InsightFace) with configurable confidence threshold
- Fallback: Automatic **center-crop** for full-frame AI faces (e.g., thispersondoesnotexist.com)
- Full-resolution images passed to classifier — no quality loss from pre-downscaling

### 🎨 Modern Dashboard UI
- Glassmorphism design with dark mode support
- Real-time analysis workflow visualization
- Drag-and-drop file upload
- Scan history with local storage persistence

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **AI / ML** | PyTorch, HuggingFace Transformers, ConvNeXt-Tiny, Grad-CAM |
| **Face Detection** | InsightFace (RetinaFace), ONNX Runtime |
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JavaScript |
| **Training** | Kaggle (GPU), Albumentations, scikit-learn |

---

## 🚀 Local Setup & Installation

### Prerequisites
- **Python 3.11** (recommended for `insightface` compatibility on Windows)
- **Microsoft C++ Build Tools** (required for InsightFace on Windows)
- NVIDIA GPU with CUDA (optional — CPU works but is slower for video)

### 1. Clone the Repository
```bash
git clone https://github.com/pranay9981/deepfake_predictor.git
cd DeepFake-Predictor
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Provide Model Weights
The ConvNeXt-Tiny weights are used for Grad-CAM heatmap generation:
1. Train on Kaggle using `kaggle/deepfake_training_v3.py`, **OR** use the pre-trained weights.
2. Place the `.pth` file in `models/pretrained/` as `deepfake_convnext_tiny.pth`.

> **Note:** The primary classifier (HuggingFace ViT) downloads automatically on first run (~350MB, cached locally).

### 5. Start the Server
```bash
cd app
python server.py
```
Open **http://localhost:8000** in your browser.

---

## 📁 Project Structure

```text
DeepFake-Predictor/
│
├── app/                              # Web Application
│   ├── server.py                     # FastAPI server, API routes (/predict, /predict-video)
│   ├── static/
│   │   ├── css/style.css             # Custom CSS animations & effects
│   │   └── js/main.js                # Frontend logic (upload, results, history)
│   ├── templates/
│   │   └── index.html                # Dashboard UI (Tailwind + glassmorphism)
│   ├── uploads/                      # Temporary upload storage
│   └── temp/                         # Temporary processing files
│
├── models/
│   ├── __init__.py
│   ├── convnext_detector.py          # Dual-model predictor (HF + ConvNeXt + Grad-CAM)
│   └── pretrained/                   # Model weights (.pth files — gitignored)
│
├── preprocessing/
│   ├── __init__.py
│   ├── face_extractor.py             # RetinaFace wrapper with margin cropping
│   └── video_processor.py            # OpenCV frame extraction & video metadata
│
├── utils/
│   ├── __init__.py
│   ├── config.py                     # All hyperparameters, paths, thresholds
│   ├── gradcam.py                    # Grad-CAM heatmap + 6-region face analysis
│   └── metrics.py                    # AUC, accuracy, and evaluation helpers
│
├── kaggle/
│   ├── deepfake_training.py          # V1 training script (FaceForensics++ only)
│   ├── deepfake_training_v2.py       # V2 training script (multi-dataset)
│   └── deepfake_training_v3.py       # V3 training script (8 datasets, 0.9977 AUC)
│
├── tests/
│   ├── test_detector.py              # Model inference tests
│   ├── test_face_extractor.py        # Face detection tests
│   └── test_video_processor.py       # Video processing tests
│
├── data/                             # Dataset storage (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔬 How It Works

### Detection Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Upload      │────▶│  Face        │────▶│  HuggingFace     │────▶│  Result +      │
│  Image/Video │     │  Detection   │     │  ViT Classifier  │     │  Grad-CAM      │
│              │     │  (RetinaFace)│     │  (Full-Res)      │     │  Heatmap       │
└──────────────┘     └──────┬───────┘     └──────────────────┘     └────────────────┘
                            │
                     No face found?
                            │
                     ┌──────▼───────┐
                     │  Center-Crop │
                     │  Fallback    │
                     └──────────────┘
```

### Training Data (V3)
The ConvNeXt-Tiny model was fine-tuned across **8 diverse datasets** covering both legacy and modern AI generators:

| Dataset | Type | Era |
|---|---|---|
| FaceForensics++ (C23) | Face Swap | 2019 |
| CelebDF v2 | Face Swap | 2020 |
| 140k StyleGAN Faces | GAN | 2020 |
| DFDC Challenge | Various | 2020 |
| GRAVEX-200K | AI vs Real | 2023 |
| 130k SDXL & FLUX Faces | Diffusion | 2024 |

**Training Performance:** 0.9977 AUC on validation set (20 epochs, 16,000 balanced face crops).

---

## 📡 API Reference

### `POST /api/predict`
Analyze a single image.

**Request:** `multipart/form-data` with `file` field.

**Response:**
```json
{
  "label": "FAKE",
  "confidence": 0.9821,
  "face_bbox": [120, 45, 380, 410],
  "face_detected": true,
  "detection_confidence": 0.92,
  "gradcam_image": "base64...",
  "region_scores": {
    "Forehead": 42.3,
    "Left Eye": 28.1,
    "Right Eye": 25.7,
    "Nose": 15.2,
    "Mouth": 12.8,
    "Jawline": 38.5
  }
}
```

### `POST /api/predict-video`
Analyze a video frame-by-frame.

**Request:** `multipart/form-data` with `file` field.

**Response:**
```json
{
  "verdict": "FAKE",
  "confidence": 0.95,
  "total_frames_analyzed": 8,
  "frames_with_faces": 8,
  "per_frame_results": [...],
  "region_scores": {...},
  "video_info": {
    "fps": 30,
    "total_frames": 240,
    "duration": 8.0,
    "resolution": [1920, 1080]
  }
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

If you find a new AI generator that defeats the current model, open an issue so we can incorporate it into the training pipeline.

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.