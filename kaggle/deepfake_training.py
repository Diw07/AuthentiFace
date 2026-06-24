# ============================================================
# 🛡️ DeepFake Detector — Kaggle Training Notebook
# ============================================================
# Dataset: FaceForensics++ (C23) — https://www.kaggle.com/datasets/xdxd003/ff-c23
# Model:   ConvNeXt-Tiny (pretrained ImageNet-1K)
# Task:    Binary Classification (Real vs Fake)
#
# HOW TO USE:
#   1. Create a new Kaggle Notebook
#   2. Add dataset "xdxd003/ff-c23"
#   3. Enable GPU P100 | Turn Internet ON
#   4. Copy each "CELL" section below into a separate notebook cell
#   5. Run all cells in order
# ============================================================


# ═══════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies & Verify GPU
# ═══════════════════════════════════════════════════════════════

# !pip install -q insightface onnxruntime-gpu albumentations grad-cam

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import cv2
import numpy as np
import os
import random
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ═══════════════════════════════════════════════════════════════
# CELL 2: Configuration
# ═══════════════════════════════════════════════════════════════

class CFG:
    # Dataset paths (Kaggle input)
    DATA_ROOT = "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23"
    REAL_DIR = os.path.join(DATA_ROOT, "original")
    FAKE_DIRS = [
        os.path.join(DATA_ROOT, "Deepfakes"),
        os.path.join(DATA_ROOT, "Face2Face"),
        os.path.join(DATA_ROOT, "FaceSwap"),
        os.path.join(DATA_ROOT, "FaceShifter"),
        os.path.join(DATA_ROOT, "NeuralTextures"),
        os.path.join(DATA_ROOT, "DeepFakeDetection"),
    ]

    # Face extraction
    FACE_DET_SIZE = (640, 640)
    FACE_CONF_THRESHOLD = 0.5
    FACE_MARGIN = 0.3
    FACE_OUTPUT_SIZE = 224

    # Video sampling
    FRAMES_PER_VIDEO = 10   # Extract 10 frames per video

    # Model
    INPUT_SIZE = 224
    CLASSIFIER_HIDDEN = 256
    DROPOUT_RATE = 0.3

    # Training
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULER = "cosine"    # cosine annealing
    WARMUP_EPOCHS = 2

    # Balance
    MAX_FAKE_PER_METHOD = 200   # Limit fake videos per method to balance
    MAX_REAL = 800              # Limit real videos
    VAL_SPLIT = 0.2

    # Output
    MODEL_SAVE_PATH = "/kaggle/working/deepfake_convnext_tiny.pth"
    BEST_MODEL_PATH = "/kaggle/working/deepfake_convnext_tiny_best.pth"

print("✅ Config loaded")
print(f"   Real dir: {CFG.REAL_DIR}")
print(f"   Fake dirs: {len(CFG.FAKE_DIRS)} methods")


# ═══════════════════════════════════════════════════════════════
# CELL 3: Initialize RetinaFace
# ═══════════════════════════════════════════════════════════════

from insightface.app import FaceAnalysis

face_app = FaceAnalysis(
    allowed_modules=["detection"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=CFG.FACE_DET_SIZE)
print("✅ RetinaFace initialized on GPU")


def extract_face(image_bgr, margin=CFG.FACE_MARGIN, output_size=CFG.FACE_OUTPUT_SIZE):
    """
    Detect and crop the largest face from a BGR image.
    Returns RGB face crop (224x224) or None if no face found.
    """
    faces = face_app.get(image_bgr)
    faces = [f for f in faces if f.det_score >= CFG.FACE_CONF_THRESHOLD]

    if not faces:
        return None

    # Pick largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = face.bbox.astype(int)
    h, w = image_bgr.shape[:2]

    # Add margin
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (output_size, output_size))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop


# ═══════════════════════════════════════════════════════════════
# CELL 4: Extract Faces from Videos
# ═══════════════════════════════════════════════════════════════

def extract_frames_from_video(video_path, num_frames=CFG.FRAMES_PER_VIDEO):
    """Extract uniformly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def process_videos(video_dir, label, max_videos=None, desc="Processing"):
    """
    Extract faces from all videos in a directory.
    Returns list of (face_crop_rgb, label) tuples.
    """
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if max_videos:
        video_files = video_files[:max_videos]

    samples = []
    skipped = 0

    for vf in tqdm(video_files, desc=desc, leave=False):
        video_path = os.path.join(video_dir, vf)
        frames = extract_frames_from_video(video_path)

        for frame in frames:
            face = extract_face(frame)
            if face is not None:
                samples.append((face, label))
            else:
                skipped += 1

    print(f"   ✓ {desc}: {len(samples)} faces extracted, {skipped} frames skipped (no face)")
    return samples


# ---------- Process Real Videos ----------
print("📂 Processing REAL videos...")
real_samples = process_videos(
    CFG.REAL_DIR, label=0,
    max_videos=CFG.MAX_REAL,
    desc="Real"
)

# ---------- Process Fake Videos ----------
fake_samples = []
for fake_dir in CFG.FAKE_DIRS:
    method_name = os.path.basename(fake_dir)
    print(f"📂 Processing FAKE ({method_name})...")
    method_samples = process_videos(
        fake_dir, label=1,
        max_videos=CFG.MAX_FAKE_PER_METHOD,
        desc=method_name
    )
    fake_samples.extend(method_samples)

# ---------- Summary ----------
all_samples = real_samples + fake_samples
print(f"\n{'='*50}")
print(f"📊 Dataset Summary:")
print(f"   Real faces: {len(real_samples)}")
print(f"   Fake faces: {len(fake_samples)}")
print(f"   Total:      {len(all_samples)}")
print(f"   Ratio:      1:{len(fake_samples)/max(len(real_samples),1):.1f} (real:fake)")
print(f"{'='*50}")


# ═══════════════════════════════════════════════════════════════
# CELL 5: Create Dataset & DataLoaders
# ═══════════════════════════════════════════════════════════════

# Split data
all_faces = [s[0] for s in all_samples]
all_labels = [s[1] for s in all_samples]

X_train, X_val, y_train, y_val = train_test_split(
    all_faces, all_labels,
    test_size=CFG.VAL_SPLIT,
    random_state=SEED,
    stratify=all_labels
)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")
print(f"Train Real: {y_train.count(0)} | Train Fake: {y_train.count(1)}")
print(f"Val   Real: {y_val.count(0)} | Val   Fake: {y_val.count(1)}")

# Augmentation pipelines
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(10, 20), hole_width_range=(10, 20), p=0.2),
    A.ImageCompression(quality_range=(50, 95), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class FaceDataset(Dataset):
    def __init__(self, faces, labels, transform=None):
        self.faces = faces       # list of numpy arrays (224, 224, 3) RGB
        self.labels = labels     # list of ints (0 or 1)
        self.transform = transform

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        face = self.faces[idx]   # numpy RGB (224, 224, 3)
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=face)
            face = augmented["image"]
        else:
            face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0

        return face, torch.tensor(label, dtype=torch.float32)


train_dataset = FaceDataset(X_train, y_train, transform=train_transform)
val_dataset = FaceDataset(X_val, y_val, transform=val_transform)

# Handle class imbalance with weighted sampler
train_label_counts = np.bincount(y_train)
class_weights = 1.0 / train_label_counts
sample_weights = [class_weights[l] for l in y_train]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(
    train_dataset, batch_size=CFG.BATCH_SIZE,
    sampler=sampler, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=CFG.BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=True
)

# Quick sanity check
batch_imgs, batch_labels = next(iter(train_loader))
print(f"\n✅ DataLoader ready!")
print(f"   Batch shape: {batch_imgs.shape}")
print(f"   Labels: {batch_labels[:8].tolist()}")


# ═══════════════════════════════════════════════════════════════
# CELL 6: Define ConvNeXt-Tiny Model
# ═══════════════════════════════════════════════════════════════

class DeepfakeDetector(nn.Module):
    """ConvNeXt-Tiny with custom binary classification head."""

    def __init__(self):
        super().__init__()
        self.backbone = models.convnext_tiny(weights="IMAGENET1K_V1")

        # Replace classifier: Linear(768, 1000) → our custom head
        self.backbone.classifier[2] = nn.Sequential(
            nn.Linear(768, CFG.CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(CFG.DROPOUT_RATE),
            nn.Linear(CFG.CLASSIFIER_HIDDEN, 1),
        )

    def forward(self, x):
        return self.backbone(x)


model = DeepfakeDetector().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model loaded on {device}")
print(f"   Total params:     {total_params:,}")
print(f"   Trainable params: {trainable_params:,}")


# ═══════════════════════════════════════════════════════════════
# CELL 7: Training Setup
# ═══════════════════════════════════════════════════════════════

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=CFG.LEARNING_RATE,
    weight_decay=CFG.WEIGHT_DECAY
)

# Cosine annealing with warmup
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG.EPOCHS - CFG.WARMUP_EPOCHS, eta_min=1e-6
)

# Warmup scheduler (linear warmup for first N epochs)
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=CFG.WARMUP_EPOCHS
)

# Combined scheduler
combined_scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, scheduler],
    milestones=[CFG.WARMUP_EPOCHS]
)

print("✅ Optimizer: AdamW | Loss: BCEWithLogitsLoss")
print(f"   LR: {CFG.LEARNING_RATE} | WD: {CFG.WEIGHT_DECAY}")
print(f"   Schedule: {CFG.WARMUP_EPOCHS} warmup + cosine annealing")


# ═══════════════════════════════════════════════════════════════
# CELL 8: Training Loop
# ═══════════════════════════════════════════════════════════════

history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": [], "lr": []}
best_val_auc = 0.0

for epoch in range(CFG.EPOCHS):
    # ─── Train ────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_steps = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = train_loss / train_steps

    # ─── Validate ─────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_steps = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS} [Val]", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            val_loss += loss.item()
            val_steps += 1
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / val_steps
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    val_acc = accuracy_score(all_targets, (all_preds >= 0.5).astype(int))
    val_auc = roc_auc_score(all_targets, all_preds)

    # ─── Step scheduler ──────────────────────────────
    current_lr = optimizer.param_groups[0]["lr"]
    combined_scheduler.step()

    # ─── Log ──────────────────────────────────────────
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)
    history["val_auc"].append(val_auc)
    history["lr"].append(current_lr)

    print(f"\n📊 Epoch {epoch+1}/{CFG.EPOCHS}")
    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val Loss:   {avg_val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
    print(f"   LR: {current_lr:.2e}")

    # ─── Save best model ─────────────────────────────
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), CFG.BEST_MODEL_PATH)
        print(f"   🏆 New best AUC! Saved to {CFG.BEST_MODEL_PATH}")

print(f"\n{'='*50}")
print(f"🎯 Training complete! Best Val AUC: {best_val_auc:.4f}")
print(f"{'='*50}")


# ═══════════════════════════════════════════════════════════════
# CELL 9: Training Curves
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history["train_loss"], label="Train", color="#6366f1", linewidth=2)
axes[0].plot(history["val_loss"], label="Val", color="#ec4899", linewidth=2)
axes[0].set_title("Loss", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy & AUC
axes[1].plot(history["val_acc"], label="Accuracy", color="#22c55e", linewidth=2)
axes[1].plot(history["val_auc"], label="AUC-ROC", color="#f59e0b", linewidth=2)
axes[1].set_title("Validation Metrics", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0.5, 1.0)
axes[1].legend()
axes[1].grid(alpha=0.3)

# Learning Rate
axes[2].plot(history["lr"], color="#6366f1", linewidth=2)
axes[2].set_title("Learning Rate", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Epoch")
axes[2].set_yscale("log")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/kaggle/working/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Training curves saved")


# ═══════════════════════════════════════════════════════════════
# CELL 10: Final Evaluation on Best Model
# ═══════════════════════════════════════════════════════════════

# Load best model
model.load_state_dict(torch.load(CFG.BEST_MODEL_PATH, map_location=device))
model.eval()
print("✅ Loaded best model")

# Run validation
all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Final Evaluation"):
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu().numpy())
        all_targets.extend(labels.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# Metrics
print(f"\n{'='*50}")
print("📊 FINAL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"Accuracy:  {accuracy_score(all_targets, (all_preds >= 0.5).astype(int)):.4f}")
print(f"AUC-ROC:   {roc_auc_score(all_targets, all_preds):.4f}")
print(f"F1 Score:  {f1_score(all_targets, (all_preds >= 0.5).astype(int)):.4f}")
print()
print(classification_report(
    all_targets, (all_preds >= 0.5).astype(int),
    target_names=["REAL", "FAKE"]
))

# Confusion Matrix
cm = confusion_matrix(all_targets, (all_preds >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["REAL", "FAKE"])
ax.set_yticklabels(["REAL", "FAKE"])
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=18, fontweight="bold",
                color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════
# CELL 11: Save Final Model & Download Instructions
# ═══════════════════════════════════════════════════════════════

# Save final model (same as best)
torch.save(model.state_dict(), CFG.MODEL_SAVE_PATH)

print(f"\n{'='*50}")
print("🎉 MODEL SAVED SUCCESSFULLY!")
print(f"{'='*50}")
print(f"\n📦 File: {CFG.MODEL_SAVE_PATH}")
print(f"📦 Best: {CFG.BEST_MODEL_PATH}")
print(f"\n📥 DOWNLOAD INSTRUCTIONS:")
print(f"   1. Click 'Save Version' (top right) → Save & Run All")
print(f"   2. Go to the 'Output' tab on the right sidebar")
print(f"   3. Download 'deepfake_convnext_tiny.pth'")
print(f"   4. Place in: models/pretrained/deepfake_convnext_tiny.pth")
print(f"\n🚀 Then run locally:")
print(f"   cd app")
print(f"   python server.py")
print(f"   # Open http://localhost:8000")
