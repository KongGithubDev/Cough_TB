import os, io, base64, gc, subprocess, tempfile
import numpy as np
import librosa
import torch
import torch.nn as nn
import timm
from scipy.ndimage import zoom
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import uvicorn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use OMP_NUM_THREADS env var (e.g., set to 4 on VPS with 10GB RAM)
# Defaults to 1 for Render Free Tier (512MB RAM)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
SR = 16000
CROP = 0.5
N_MELS = 224
FMAX = 8000


class TemporalShift(nn.Module):
    def __init__(self, channels, shift_div=8):
        super().__init__()
        self.fold = channels // shift_div
    def forward(self, x):
        B, C, T, F = x.size()
        t = x.permute(0, 2, 1, 3).contiguous()
        out = torch.zeros_like(t)
        out[:, :-1, :self.fold, :] = t[:, 1:, :self.fold, :]
        out[:, 1:, self.fold:2*self.fold, :] = t[:, :-1, self.fold:2*self.fold, :]
        out[:, :, 2*self.fold:, :] = t[:, :, 2*self.fold:, :]
        return out.permute(0, 2, 1, 3)


class Res2TSMBlock(nn.Module):
    def __init__(self, channels, scale=4, shift_div=8):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        self.temporal_shift = TemporalShift(channels, shift_div)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=(3, 1),
                      padding=(1, 0), groups=self.width, bias=False)
            for _ in range(scale-1)
        ])
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.temporal_shift(x)
        splits = torch.split(x, self.width, dim=1)
        y = splits[0]
        outs = [y]
        for i in range(1, self.scale):
            sp = splits[i] + y
            sp = self.convs[i-1](sp)
            y = sp
            outs.append(sp)
        out = torch.cat(outs, dim=1)
        return self.act(self.bn(out))


class MobileNetV4_Res2TSM(nn.Module):
    def __init__(self, model_key, scale=4, shift_div=8, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_key, pretrained=False, features_only=True)
        C = self.backbone.feature_info.channels()[-1]
        self.res2tsm = Res2TSMBlock(C, scale=scale, shift_div=shift_div)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(C, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = self.res2tsm(feat)
        out = self.global_pool(feat).view(feat.size(0), -1)
        return self.fc(out).squeeze(1)


model = None


def load_model():
    global model
    if model is not None:
        return model
    model = MobileNetV4_Res2TSM('mobilenetv4_conv_blur_medium').to(DEVICE)
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    if not os.path.exists(model_path):
        os.system("git clone https://github.com/yop-dev/tb-cough-detection.git")
        src = "tb-cough-detection/final_best_mobilenetv4_conv_blur_medium_res2tsm_tb_classifier.pth"
        import shutil
        shutil.move(src, model_path)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    sd = state.get('state_dict') or state.get('model_state_dict') or state
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model


def load_and_segment(path_or_bytes):
    if isinstance(path_or_bytes, bytes):
        try:
            y, _ = librosa.load(io.BytesIO(path_or_bytes), sr=SR)
        except Exception:
            y, _ = _ffmpeg_decode(path_or_bytes)
    else:
        y, _ = librosa.load(path_or_bytes, sr=SR)
    if len(y) == 0:
        return None
    y, _ = librosa.effects.trim(y, top_db=20)
    target_len = int(SR * CROP)
    if len(y) >= target_len:
        energy = np.convolve(y**2, np.ones(target_len), mode='valid')
        start = np.argmax(energy)
        seg = y[start:start+target_len]
    else:
        seg = np.zeros(target_len, dtype=y.dtype)
        seg[:len(y)] = y
    return seg


def _ffmpeg_decode(data: bytes, sr=SR):
    tmp_in = tmp_out = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(data)
            tmp_in = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_out = f.name
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-ar", str(sr), "-ac", "1",
             "-f", "wav", tmp_out],
            capture_output=True, check=True
        )
        y, _ = librosa.load(tmp_out, sr=sr)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise HTTPException(400, "Audio format not supported. Use .wav or install ffmpeg.")
    finally:
        for p in (tmp_in, tmp_out):
            if p and os.path.exists(p):
                os.unlink(p)
    return y, sr


def make_mel_rgb(y_seg):
    mel = librosa.feature.melspectrogram(
        y=y_seg, sr=SR, n_mels=N_MELS, fmax=FMAX,
        hop_length=512, win_length=2048
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    target_shape = (224, 224)
    if mel_db.shape != target_shape:
        zf = (target_shape[0]/mel_db.shape[0], target_shape[1]/mel_db.shape[1])
        resized = zoom(mel_db, zf, order=1)
    else:
        resized = mel_db
    if np.ptp(resized) > 0:
        normed = (resized - resized.min()) / np.ptp(resized)
    else:
        normed = np.zeros_like(resized)
    rgb = np.stack([normed] * 3, axis=0)
    return mel_db, rgb


def mel_to_image(mel_db):
    """Generate spectrogram image using PIL only (no matplotlib — saves ~150MB RAM)."""
    norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(255 - norm, mode='L').resize((672, 336), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def predict(audio_bytes):
    y_seg = load_and_segment(audio_bytes)
    if y_seg is None:
        return {"error": "Cannot process audio"}
    mel_db, mel_rgb = make_mel_rgb(y_seg)
    with torch.no_grad():
        input_tensor = torch.from_numpy(mel_rgb).float().unsqueeze(0).to(DEVICE)
        prob = model(input_tensor).cpu().numpy()[0]
    # Clean up large tensors immediately
    del input_tensor, mel_rgb
    gc.collect()
    is_tb = bool(prob > 0.5)
    result = {
        "tb_probability": float(prob),
        "is_tb": is_tb,
        "confidence_tb": float(prob) if is_tb else float(1 - prob),
        "label": "TB Detected" if is_tb else "No TB Detected",
        "spectrogram": mel_to_image(mel_db)
    }
    del mel_db
    gc.collect()
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # Cleanup on shutdown
    global model
    model = None
    gc.collect()


app = FastAPI(title="CoughTB", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty file")
    result = predict(audio_bytes)
    if "error" in result:
        raise HTTPException(422, result["error"])
    return JSONResponse(result)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
