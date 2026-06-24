# ============================================================
# DeepFake Detector V3 — Multi-Dataset Kaggle Training
# ============================================================
# Trains on 8 diverse datasets for high robustness:
#   1. FaceForensics++ (Vids)
#   2. CelebDF_V2 (Imgs)
#   3. Celeb-DF Preprocessed (Imgs) 
#   4. 130k Real vs Fake Face (Imgs - FLUX, SDXL)
#   5. Celeb DF v2 (Vids)
#   6. GRAVEX-200K (Imgs)
#   7. 140k Real and Fake Faces (Imgs - StyleGAN)
#   8. DFDC faces (Imgs)
#
# HOW TO USE: Add all 8 datasets to a Kaggle notebook,
# copy this script, and select "Save & Run All".
# ============================================================

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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

from insightface.app import FaceAnalysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class CFG:
    # ─── Video Datasets ───
    FF_ROOT = "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23"
    CELEB_DF_V2_VIDS_ROOT = "/kaggle/input/celeb-df-v2"
    
    # ─── Image Datasets ───
    CELEB_DF_V2_IMGS_ROOT = "/kaggle/input/celebdf-v2-image-dataset"
    CELEB_DF_PRE_ROOT = "/kaggle/input/celeb-df-preprocessed"
    REAL_VS_FAKE_130K_ROOT = "/kaggle/input/130k-real-vs-fake-face"
    GRAVEX_ROOT = "/kaggle/input/gravex-200k"
    STYLEGAN_140K_ROOT = "/kaggle/input/140k-real-and-fake-faces"
    DFDC_FACES_ROOT = "/kaggle/input/dfdc-faces-of-the-train-sample"

    # Face Extraction
    FACE_DET_SIZE = (640, 640)
    FACE_CONF_THRESHOLD = 0.5
    FACE_MARGIN = 0.3
    FACE_OUTPUT_SIZE = 224
    FRAMES_PER_VIDEO = 8

    # Training
    BATCH_SIZE = 64  # Increased for larger dataset
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2

    # Sampling limits (to keep training manageable and balanced)
    MAX_VIDEOS_PER_CLASS = 1000
    MAX_IMAGES_PER_DATASET = 8000
    VAL_SPLIT = 0.15

    MODEL_SAVE_PATH = "/kaggle/working/deepfake_convnext_tiny_v3_best.pth"

# ==========================================
# 1. Face Extraction Setup
# ==========================================
face_app = FaceAnalysis(allowed_modules=["detection"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=CFG.FACE_DET_SIZE)

def extract_face(image_bgr, margin=CFG.FACE_MARGIN, output_size=CFG.FACE_OUTPUT_SIZE):
    faces = face_app.get(image_bgr)
    faces = [f for f in faces if f.det_score >= CFG.FACE_CONF_THRESHOLD]
    if not faces: return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    bbox = face.bbox.astype(int)
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2-x1)*margin), int((y2-y1)*margin)
    crop = image_bgr[max(0, y1-my):min(h, y2+my), max(0, x1-mx):min(w, x2+mx)]
    if crop.size == 0: return None
    return cv2.cvtColor(cv2.resize(crop, (output_size, output_size)), cv2.COLOR_BGR2RGB)

# ==========================================
# 2. Data Loading Helpers
# ==========================================
def extract_frames(video_path, num_frames=CFG.FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: return []
    frames = []
    for idx in np.linspace(0, total-1, num=min(num_frames, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None: frames.append(frame)
    cap.release()
    return frames

def load_video_folder(folder_path, label, max_vids):
    if not os.path.exists(folder_path): return []
    vids = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp4')]
    random.shuffle(vids)
    samples = []
    for v in tqdm(vids[:max_vids], desc=f"Video: {os.path.basename(folder_path)}", leave=False):
        for frame in extract_frames(v):
            face = extract_face(frame)
            if face is not None: samples.append((face, label))
    return samples

def load_image_from_dirs(base_path, max_imgs):
    if not os.path.exists(base_path): return []
    real_files, fake_files = [], []
    
    # Simple heuristic to find real/fake
    for root, _, files in os.walk(base_path):
        imgs = [os.path.join(root, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if 'real' in root.lower() or 'original' in root.lower():
            real_files.extend(imgs)
        elif 'fake' in root.lower() or 'synth' in root.lower() or 'ai' in root.lower() or 'flux' in root.lower() or 'sdxl' in root.lower():
            fake_files.extend(imgs)
            
    random.shuffle(real_files); random.shuffle(fake_files)
    real_files, fake_files = real_files[:max_imgs], fake_files[:max_imgs]
    
    samples = []
    for paths, label in [(real_files, 0), (fake_files, 1)]:
        for p in tqdm(paths, desc=f"Images: {os.path.basename(base_path)} [{label}]", leave=False):
            img = cv2.imread(p)
            if img is not None:
                face = extract_face(img)
                if face is not None: samples.append((face, label))
    return samples

# ==========================================
# 3. Load All Datasets
# ==========================================
print("Loading datasets...")
all_samples = []

# FF++ Videos
if os.path.exists(CFG.FF_ROOT):
    all_samples.extend(load_video_folder(os.path.join(CFG.FF_ROOT, "original"), 0, int(CFG.MAX_VIDEOS_PER_CLASS*1.5)))
    for m in ["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]:
        all_samples.extend(load_video_folder(os.path.join(CFG.FF_ROOT, m), 1, CFG.MAX_VIDEOS_PER_CLASS//5))

# Celeb DF v2 Videos
if os.path.exists(CFG.CELEB_DF_V2_VIDS_ROOT):
    all_samples.extend(load_video_folder(os.path.join(CFG.CELEB_DF_V2_VIDS_ROOT, "Celeb-real"), 0, CFG.MAX_VIDEOS_PER_CLASS//2))
    all_samples.extend(load_video_folder(os.path.join(CFG.CELEB_DF_V2_VIDS_ROOT, "YouTube-real"), 0, CFG.MAX_VIDEOS_PER_CLASS//2))
    all_samples.extend(load_video_folder(os.path.join(CFG.CELEB_DF_V2_VIDS_ROOT, "Celeb-synthesis"), 1, CFG.MAX_VIDEOS_PER_CLASS))

# Image Datasets
for root in [CFG.CELEB_DF_V2_IMGS_ROOT, CFG.CELEB_DF_PRE_ROOT, CFG.REAL_VS_FAKE_130K_ROOT, 
             CFG.GRAVEX_ROOT, CFG.STYLEGAN_140K_ROOT, CFG.DFDC_FACES_ROOT]:
    all_samples.extend(load_image_from_dirs(root, CFG.MAX_IMAGES_PER_DATASET))

random.shuffle(all_samples)
faces, labels = [s[0] for s in all_samples], [s[1] for s in all_samples]
print(f"Total Combined Faces: {len(faces)} (Real: {labels.count(0)}, Fake: {labels.count(1)})")

# ==========================================
# 4. DataLoader
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(faces, labels, test_size=CFG.VAL_SPLIT, stratify=labels)

transform = A.Compose([
    A.HorizontalFlip(p=0.5), A.Rotate(limit=15, p=0.3), A.ColorJitter(p=0.4),
    A.GaussianBlur(p=0.2), A.ImageCompression(quality_range=(50, 95), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(),
])
val_transform = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

class FaceDS(Dataset):
    def __init__(self, f, l, t): self.f, self.l, self.t = f, l, t
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        return self.t(image=self.f[i])["image"], torch.tensor(self.l[i], dtype=torch.float32)

train_loader = DataLoader(FaceDS(X_train, y_train, transform), batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(FaceDS(X_val, y_val, val_transform), batch_size=CFG.BATCH_SIZE, shuffle=False)

# ==========================================
# 5. Training Loop
# ==========================================
class ConvNeXtTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = models.convnext_tiny(weights="IMAGENET1K_V1")
        self.b.classifier[2] = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(self, x): return self.b(x)

model = ConvNeXtTiny().to(device)
crit = nn.BCEWithLogitsLoss()
opt = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.EPOCHS)

best_auc = 0
history = {"train_loss": [], "val_auc": []}

for ep in range(CFG.EPOCHS):
    model.train(); tl = 0
    pbar = tqdm(train_loader, desc=f"Ep {ep+1}/{CFG.EPOCHS}")
    for img, lbl in pbar:
        img, lbl = img.to(device), lbl.to(device)
        opt.zero_grad()
        loss = crit(model(img).squeeze(1), lbl)
        loss.backward(); opt.step(); tl += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    model.eval(); preds, targs = [], []
    with torch.no_grad():
        for img, lbl in val_loader:
            preds.extend(torch.sigmoid(model(img.to(device)).squeeze(1)).cpu().numpy())
            targs.extend(lbl.numpy())
            
    auc = roc_auc_score(targs, preds)
    history["train_loss"].append(tl/len(train_loader)); history["val_auc"].append(auc)
    print(f"Epoch {ep+1} | Train Loss: {history['train_loss'][-1]:.4f} | Val AUC: {auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), CFG.MODEL_SAVE_PATH)
    sched.step()

print(f"Done! Best AUC: {best_auc:.4f}. Model saved to {CFG.MODEL_SAVE_PATH}")
