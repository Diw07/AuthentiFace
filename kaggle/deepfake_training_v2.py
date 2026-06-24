# ============================================================
# DeepFake Detector v2 — Kaggle Training Notebook
# ============================================================
# Datasets:
#   1. FaceForensics++ (C23) — video deepfakes (xdxd003/ff-c23)
#   2. 140k Real and Fake Faces — StyleGAN  (xhlulu/140k-real-and-fake-faces)
#   3. Deepfake and Real Images            (manjilkarki/deepfake-and-real-images)
#
# HOW TO USE:
#   1. Create a new Kaggle Notebook
#   2. Add ALL 3 datasets above
#   3. Enable GPU P100 | Turn Internet ON
#   4. Copy each "CELL" section into separate cells
#   5. Run all cells OR use "Save & Run All (Commit)"
# ============================================================


# ═══════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies & Verify GPU
# ═══════════════════════════════════════════════════════════════

!pip install -q insightface onnxruntime-gpu albumentations grad-cam

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
from glob import glob
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
print(f"Device: {device}")
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
    # ─── Dataset 1: FaceForensics++ (videos) ───
    FF_ROOT = "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23"
    FF_REAL_DIR = os.path.join(FF_ROOT, "original")
    FF_FAKE_DIRS = [
        os.path.join(FF_ROOT, "Deepfakes"),
        os.path.join(FF_ROOT, "Face2Face"),
        os.path.join(FF_ROOT, "FaceSwap"),
        os.path.join(FF_ROOT, "FaceShifter"),
        os.path.join(FF_ROOT, "NeuralTextures"),
        os.path.join(FF_ROOT, "DeepFakeDetection"),
    ]

    # ─── Dataset 2: 140k Real and Fake Faces (images) ───
    STYLEGAN_ROOT = "/kaggle/input/140k-real-and-fake-faces"

    # ─── Dataset 3: Deepfake and Real Images ───
    DFREAL_ROOT = "/kaggle/input/deepfake-and-real-images"

    # Face extraction
    FACE_DET_SIZE = (640, 640)
    FACE_CONF_THRESHOLD = 0.5
    FACE_MARGIN = 0.3
    FACE_OUTPUT_SIZE = 224

    # Video sampling
    FRAMES_PER_VIDEO = 10

    # Model
    INPUT_SIZE = 224
    CLASSIFIER_HIDDEN = 256
    DROPOUT_RATE = 0.3

    # Training
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2

    # Balance
    FF_MAX_FAKE_PER_METHOD = 150  # FF++ videos per method
    FF_MAX_REAL = 600             # FF++ real videos
    IMG_MAX_PER_CLASS = 8000      # Max images per class from image datasets
    VAL_SPLIT = 0.2

    # Output
    MODEL_SAVE_PATH = "/kaggle/working/deepfake_convnext_tiny_v2.pth"
    BEST_MODEL_PATH = "/kaggle/working/deepfake_convnext_tiny_v2_best.pth"


# ─── Auto-detect dataset paths ───
print("Checking dataset paths...")
for name, path in [("FF++ Root", CFG.FF_ROOT), ("StyleGAN", CFG.STYLEGAN_ROOT), ("DF Real", CFG.DFREAL_ROOT)]:
    if os.path.exists(path):
        print(f"  [OK] {name}: {path}")
    else:
        print(f"  [!!] {name}: NOT FOUND at {path}")
        print(f"       Available: {os.listdir('/kaggle/input/')}")


# ═══════════════════════════════════════════════════════════════
# CELL 3: Initialize RetinaFace
# ═══════════════════════════════════════════════════════════════

from insightface.app import FaceAnalysis

face_app = FaceAnalysis(
    allowed_modules=["detection"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=CFG.FACE_DET_SIZE)
print("RetinaFace initialized on GPU")


def extract_face(image_bgr, margin=CFG.FACE_MARGIN, output_size=CFG.FACE_OUTPUT_SIZE):
    """Detect and crop the largest face from a BGR image."""
    faces = face_app.get(image_bgr)
    faces = [f for f in faces if f.det_score >= CFG.FACE_CONF_THRESHOLD]
    if not faces:
        return None

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = face.bbox.astype(int)
    h, w = image_bgr.shape[:2]

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (output_size, output_size))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop


# ═══════════════════════════════════════════════════════════════
# CELL 4: Extract Faces — FaceForensics++ Videos
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


def process_ff_videos(video_dir, label, max_videos=None, desc="Processing"):
    """Extract faces from FF++ videos."""
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if max_videos:
        random.shuffle(video_files)
        video_files = video_files[:max_videos]

    samples = []
    for vf in tqdm(video_files, desc=desc, leave=False):
        video_path = os.path.join(video_dir, vf)
        frames = extract_frames_from_video(video_path)
        for frame in frames:
            face = extract_face(frame)
            if face is not None:
                samples.append((face, label))
    print(f"   {desc}: {len(samples)} faces")
    return samples


# Process FF++ real videos
print("=== FaceForensics++ (Videos) ===")
ff_real = process_ff_videos(CFG.FF_REAL_DIR, label=0, max_videos=CFG.FF_MAX_REAL, desc="FF++ Real")

# Process FF++ fake videos
ff_fake = []
for fake_dir in CFG.FF_FAKE_DIRS:
    if os.path.exists(fake_dir):
        name = os.path.basename(fake_dir)
        samples = process_ff_videos(fake_dir, label=1, max_videos=CFG.FF_MAX_FAKE_PER_METHOD, desc=f"FF++ {name}")
        ff_fake.extend(samples)

print(f"FF++ Total: {len(ff_real)} real + {len(ff_fake)} fake\n")


# ═══════════════════════════════════════════════════════════════
# CELL 5: Load Image Datasets (StyleGAN + Deepfake/Real Images)
# ═══════════════════════════════════════════════════════════════

def load_image_dataset(base_path, max_per_class=CFG.IMG_MAX_PER_CLASS):
    """
    Load images from a dataset with real/fake subfolders.
    Auto-detects folder structure.
    """
    samples = []

    # Try common folder structures
    possible_structures = [
        # Structure: base/real/*.jpg, base/fake/*.jpg
        {"real": ["real", "Real", "REAL", "real_faces", "training_real"],
         "fake": ["fake", "Fake", "FAKE", "fake_faces", "training_fake"]},
        # Structure: base/train/real/, base/train/fake/
        {"real": ["train/real", "train/Real", "training/real"],
         "fake": ["train/fake", "train/Fake", "training/fake"]},
    ]

    real_dir = None
    fake_dir = None

    # First, list available folders
    if os.path.exists(base_path):
        print(f"   Contents of {base_path}:")
        for item in sorted(os.listdir(base_path)):
            full = os.path.join(base_path, item)
            if os.path.isdir(full):
                count = len(os.listdir(full))
                print(f"     {item}/ ({count} items)")

    for structure in possible_structures:
        for r in structure["real"]:
            candidate = os.path.join(base_path, r)
            if os.path.exists(candidate):
                real_dir = candidate
                break
        for f in structure["fake"]:
            candidate = os.path.join(base_path, f)
            if os.path.exists(candidate):
                fake_dir = candidate
                break
        if real_dir and fake_dir:
            break

    if not real_dir or not fake_dir:
        print(f"   [WARN] Could not find real/fake folders in {base_path}")
        return []

    # Load real images
    real_files = glob(os.path.join(real_dir, "**", "*.*"), recursive=True)
    real_files = [f for f in real_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.shuffle(real_files)
    real_files = real_files[:max_per_class]

    for img_path in tqdm(real_files, desc="Real images", leave=False):
        img = cv2.imread(img_path)
        if img is None:
            continue
        face = extract_face(img)
        if face is not None:
            samples.append((face, 0))

    print(f"   Real: {sum(1 for s in samples if s[1]==0)} faces from {len(real_files)} images")

    # Load fake images
    before = len(samples)
    fake_files = glob(os.path.join(fake_dir, "**", "*.*"), recursive=True)
    fake_files = [f for f in fake_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.shuffle(fake_files)
    fake_files = fake_files[:max_per_class]

    for img_path in tqdm(fake_files, desc="Fake images", leave=False):
        img = cv2.imread(img_path)
        if img is None:
            continue
        face = extract_face(img)
        if face is not None:
            samples.append((face, 1))

    fake_loaded = len(samples) - before
    print(f"   Fake: {fake_loaded} faces from {len(fake_files)} images")
    return samples


# Load image datasets
img_samples = []

print("=== 140k Real and Fake Faces (StyleGAN) ===")
if os.path.exists(CFG.STYLEGAN_ROOT):
    img_samples.extend(load_image_dataset(CFG.STYLEGAN_ROOT))

print("\n=== Deepfake and Real Images ===")
if os.path.exists(CFG.DFREAL_ROOT):
    img_samples.extend(load_image_dataset(CFG.DFREAL_ROOT))

print(f"\nImage datasets total: {len(img_samples)} faces")


# ═══════════════════════════════════════════════════════════════
# CELL 6: Combine All Data & Create DataLoaders
# ═══════════════════════════════════════════════════════════════

# Combine FF++ + image datasets
all_samples = ff_real + ff_fake + img_samples
random.shuffle(all_samples)

all_faces = [s[0] for s in all_samples]
all_labels = [s[1] for s in all_samples]

real_count = sum(1 for l in all_labels if l == 0)
fake_count = sum(1 for l in all_labels if l == 1)

print("=" * 50)
print("COMBINED DATASET:")
print(f"  Real: {real_count}")
print(f"  Fake: {fake_count}")
print(f"  Total: {len(all_labels)}")
print(f"  Sources: FF++ videos + StyleGAN images + Deepfake/Real images")
print("=" * 50)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    all_faces, all_labels,
    test_size=CFG.VAL_SPLIT,
    random_state=SEED,
    stratify=all_labels
)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")
print(f"Train Real: {y_train.count(0)} | Train Fake: {y_train.count(1)}")
print(f"Val   Real: {y_val.count(0)} | Val   Fake: {y_val.count(1)}")

# Augmentations
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
        self.faces = faces
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        face = self.faces[idx]
        label = self.labels[idx]
        if self.transform:
            face = self.transform(image=face)["image"]
        else:
            face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
        return face, torch.tensor(label, dtype=torch.float32)


train_dataset = FaceDataset(X_train, y_train, transform=train_transform)
val_dataset = FaceDataset(X_val, y_val, transform=val_transform)

# Weighted sampler for class balance
train_label_counts = np.bincount(y_train)
class_weights = 1.0 / train_label_counts
sample_weights = [class_weights[l] for l in y_train]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"\nDataLoader ready! Batch shape: {next(iter(train_loader))[0].shape}")


# ═══════════════════════════════════════════════════════════════
# CELL 7: Define Model + Training Setup
# ═══════════════════════════════════════════════════════════════

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
        self.backbone.classifier[2] = nn.Sequential(
            nn.Linear(768, CFG.CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(CFG.DROPOUT_RATE),
            nn.Linear(CFG.CLASSIFIER_HIDDEN, 1),
        )
    def forward(self, x):
        return self.backbone(x)

model = DeepfakeDetector().to(device)
print(f"Model on {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)

warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=CFG.WARMUP_EPOCHS)
cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS - CFG.WARMUP_EPOCHS, eta_min=1e-6)
scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[CFG.WARMUP_EPOCHS])


# ═══════════════════════════════════════════════════════════════
# CELL 8: Training Loop
# ═══════════════════════════════════════════════════════════════

history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": [], "lr": []}
best_val_auc = 0.0

for epoch in range(CFG.EPOCHS):
    # --- Train ---
    model.train()
    train_loss, train_steps = 0.0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # --- Validate ---
    model.eval()
    val_loss, val_steps = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS} [Val]", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            val_loss += loss.item()
            val_steps += 1
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_train = train_loss / train_steps
    avg_val = val_loss / val_steps
    preds_arr = np.array(all_preds)
    targs_arr = np.array(all_targets)
    val_acc = accuracy_score(targs_arr, (preds_arr >= 0.5).astype(int))
    val_auc = roc_auc_score(targs_arr, preds_arr)

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["val_acc"].append(val_acc)
    history["val_auc"].append(val_auc)
    history["lr"].append(current_lr)

    print(f"\nEpoch {epoch+1}/{CFG.EPOCHS}")
    print(f"  Train Loss: {avg_train:.4f}")
    print(f"  Val Loss: {avg_val:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | LR: {current_lr:.2e}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), CFG.BEST_MODEL_PATH)
        print(f"  NEW BEST! Saved to {CFG.BEST_MODEL_PATH}")

# Also save final model
torch.save(model.state_dict(), CFG.MODEL_SAVE_PATH)
print(f"\n{'='*50}")
print(f"Training complete! Best Val AUC: {best_val_auc:.4f}")
print(f"{'='*50}")


# ═══════════════════════════════════════════════════════════════
# CELL 9: Training Curves
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history["train_loss"], label="Train", color="#6366f1", linewidth=2)
axes[0].plot(history["val_loss"], label="Val", color="#ec4899", linewidth=2)
axes[0].set_title("Loss", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history["val_acc"], label="Accuracy", color="#22c55e", linewidth=2)
axes[1].plot(history["val_auc"], label="AUC-ROC", color="#f59e0b", linewidth=2)
axes[1].set_title("Validation Metrics", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0.5, 1.0); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(history["lr"], color="#6366f1", linewidth=2)
axes[2].set_title("Learning Rate", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Epoch"); axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/kaggle/working/training_curves_v2.png", dpi=150, bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════
# CELL 10: Final Evaluation
# ═══════════════════════════════════════════════════════════════

model.load_state_dict(torch.load(CFG.BEST_MODEL_PATH, map_location=device))
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Final Evaluation"):
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu().numpy())
        all_targets.extend(labels.numpy())

preds_arr = np.array(all_preds)
targs_arr = np.array(all_targets)

print(f"\n{'='*50}")
print("FINAL EVALUATION (v2 — Multi-Dataset)")
print(f"{'='*50}")
print(f"Accuracy:  {accuracy_score(targs_arr, (preds_arr >= 0.5).astype(int)):.4f}")
print(f"AUC-ROC:   {roc_auc_score(targs_arr, preds_arr):.4f}")
print(f"F1 Score:  {f1_score(targs_arr, (preds_arr >= 0.5).astype(int)):.4f}")
print()
print(classification_report(targs_arr, (preds_arr >= 0.5).astype(int), target_names=["REAL", "FAKE"]))

# Confusion matrix
cm = confusion_matrix(targs_arr, (preds_arr >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["REAL", "FAKE"]); ax.set_yticklabels(["REAL", "FAKE"])
ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix (v2)", fontsize=14, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=18, fontweight="bold",
                color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im); plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrix_v2.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nModel files saved:")
print(f"  {CFG.MODEL_SAVE_PATH}")
print(f"  {CFG.BEST_MODEL_PATH}")
print(f"\nDownload from Output tab and place in: models/pretrained/deepfake_convnext_tiny.pth")
