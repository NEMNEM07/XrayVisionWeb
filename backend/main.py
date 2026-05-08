# backend/main.py
import io
import cv2
import torch
import numpy as np
import pydicom
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torch.nn import functional as F
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
import torch.nn as nn
import base64
import asyncio
from contextlib import asynccontextmanager
from transformers import ViTModel


# =====================
# 모델 정의
# =====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        mask = (target.sum(dim=(1, 2, 3)) > 0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred = pred[mask]
        target = target[mask]
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)


class XrayViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.cls_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0, :]
        cls_prob = self.cls_head(cls_token)
        patch_tokens = out.last_hidden_state[:, 1:, :]
        B = patch_tokens.shape[0]
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, 768, 14, 14)
        seg_map = self.seg_head(patch_tokens)
        seg_map = F.interpolate(seg_map, size=(224, 224), mode='bilinear', align_corners=False)
        return cls_prob, seg_map


# =====================
# 전역 모델
# =====================
DEVICE = torch.device('cpu')
model = None


def load_model_sync():
    global model
    print("HuggingFace에서 모델 다운로드 중...")
    path = hf_hub_download(repo_id="NEMNEM0702/xrayvision", filename="best_model.pt")
    m = XrayViT().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    model = m
    print("모델 로드 완료!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_model_sync)
    yield


# =====================
# FastAPI 앱
# =====================
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================
# 전처리
# =====================
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image(file_bytes, filename):
    ext = filename.split('.')[-1].lower()
    if ext == 'dcm':
        dcm = pydicom.dcmread(io.BytesIO(file_bytes))
        img = dcm.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img_pil = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    else:
        img_pil = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    return img_pil.resize((224, 224), Image.BILINEAR)


def overlay_mask(img_np, mask_np, color, alpha=0.45, threshold=0.3):
    binary  = (mask_np >= threshold).astype(np.uint8)
    colored = np.zeros_like(img_np)
    colored[binary == 1] = color
    return cv2.addWeighted(img_np.copy(), 1.0, colored, alpha, 0)


def img_to_base64(img_np):
    _, buf = cv2.imencode('.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8')


# =====================
# API
# =====================
@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            {"error": "모델 로딩 중입니다. 잠시 후 다시 시도해주세요."},
            status_code=503
        )

    file_bytes = await file.read()
    img_pil    = load_image(file_bytes, file.filename)
    img_np     = np.array(img_pil)
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_prob, seg_map = model(img_tensor)

    prob      = torch.sigmoid(cls_prob).item()
    pred_mask = seg_map.squeeze().cpu().numpy()
    status    = 'ABNORMAL' if prob >= 0.5 else 'NORMAL'
    color     = (255, 50, 50) if prob >= 0.5 else (50, 200, 50)
    overlaid  = overlay_mask(img_np, pred_mask, color=color)

    mask_colored = (pred_mask * 255).astype(np.uint8)
    mask_colored = cv2.applyColorMap(mask_colored, cv2.COLORMAP_HOT)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)

    return JSONResponse({
        "status":      status,
        "probability": round(prob * 100, 1),
        "original":    img_to_base64(img_np),
        "heatmap":     img_to_base64(mask_colored),
        "overlay":     img_to_base64(overlaid),
    })