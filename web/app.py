import os, io, base64, gc, subprocess, tempfile, time
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
from PIL import Image, ImageDraw
import uvicorn
from dotenv import load_dotenv

# Load .env file (try web/.env first, then project root)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

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


def _make_magma_lut(size=256):
    """Create a magma-like colormap LUT (256x3 uint8) — medical-grade, no matplotlib."""
    xp = np.linspace(0, 1, 9)
    x = np.linspace(0, 1, size)
    r = np.interp(x, xp, [0.001, 0.042, 0.162, 0.362, 0.565, 0.754, 0.911, 0.983, 1.000])
    g = np.interp(x, xp, [0.000, 0.010, 0.021, 0.070, 0.162, 0.312, 0.491, 0.709, 0.910])
    b = np.interp(x, xp, [0.014, 0.079, 0.298, 0.456, 0.498, 0.473, 0.386, 0.286, 0.180])
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)

_MAGMA_LUT = _make_magma_lut()

def _img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def mel_to_image(mel_db):
    """Spectrogram with magma colormap."""
    norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 255).astype(np.uint8)
    colored = _MAGMA_LUT[norm[::-1, :]]
    img = Image.fromarray(colored, mode='RGB').resize((672, 336), Image.LANCZOS)
    return _img_to_b64(img)

def waveform_to_image(y, width=672, height=140):
    """Draw audio waveform as filled cyan line."""
    img = Image.new('RGB', (width, height), '#f8fafc')
    draw = ImageDraw.Draw(img)
    # Normalize and downsample
    y_norm = y / (np.max(np.abs(y)) + 1e-8)
    step = max(1, len(y_norm) // width)
    y_down = y_norm[::step][:width]
    x_vals = np.arange(len(y_down))
    # Map to pixel coords
    cx = width / 2
    cy = height / 2
    scale_v = (height - 20) / 2
    points = []
    for i, v in enumerate(y_down):
        px = int(3 + i * (width - 6) / max(len(y_down) - 1, 1))
        py = int(cy - v * scale_v)
        points.append((px, py))
    if len(points) > 1:
        # Draw fill
        fill_pts = [(points[0][0], cy)] + points + [(points[-1][0], cy)]
        draw.polygon(fill_pts, fill='#ccfbf1')
        # Draw line
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill='#0d9488', width=2)
    # Center line
    draw.line([(3, cy), (width - 3, cy)], fill='#e2e8f0', width=1)
    return _img_to_b64(img)

def freq_spectrum_to_image(mel_db, width=672, height=140):
    """Average frequency spectrum as horizontal bar chart."""
    img = Image.new('RGB', (width, height), '#f8fafc')
    draw = ImageDraw.Draw(img)
    # Average across time
    profile = mel_db.mean(axis=1)  # (n_mels,)
    if np.ptp(profile) > 0:
        profile = (profile - profile.min()) / np.ptp(profile)
    # Draw bars
    n_bars = len(profile)
    bar_w = max(1, (width - 20) // n_bars)
    gap = max(1, bar_w // 4)
    bar_w = max(1, bar_w - gap)
    for i, v in enumerate(profile):
        bar_h = int(v * (height - 24))
        x = 10 + i * (bar_w + gap)
        y_bottom = height - 12
        y_top = y_bottom - bar_h
        # Color from magma LUT
        idx = min(int(v * 255), 255)
        color = tuple(_MAGMA_LUT[idx].tolist())
        draw.rectangle([x, y_top, x + bar_w, y_bottom], fill=color)
    # Baseline
    draw.line([(8, height - 12), (width - 8, height - 12)], fill='#e2e8f0', width=1)
    return _img_to_b64(img)

def mfcc_to_image(y, width=672, height=140):
    """MFCC heatmap with magma colormap."""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=2048, hop_length=512)
    target_w = width // 2  # reduce width to avoid tiny pixels
    if mfcc.shape[1] != target_w:
        zf = target_w / mfcc.shape[1]
        mfcc = zoom(mfcc, (1, zf), order=1)
    if np.ptp(mfcc) > 0:
        norm = ((mfcc - mfcc.min()) / np.ptp(mfcc) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(mfcc, dtype=np.uint8)
    colored = _MAGMA_LUT[255 - norm]  # invert: high energy = warm
    img = Image.fromarray(colored, mode='RGB').resize((width, height), Image.LANCZOS)
    return _img_to_b64(img)


def predict(audio_bytes):
    t0 = time.perf_counter()
    y_seg = load_and_segment(audio_bytes)
    if y_seg is None:
        return {"error": "Cannot process audio"}
    audio_duration = len(y_seg) / SR
    mel_db, mel_rgb = make_mel_rgb(y_seg)
    t1 = time.perf_counter()
    with torch.no_grad():
        input_tensor = torch.from_numpy(mel_rgb).float().unsqueeze(0).to(DEVICE)
        prob = model(input_tensor).cpu().numpy()[0]
    t2 = time.perf_counter()
    # Clean up large tensors immediately
    del input_tensor, mel_rgb
    gc.collect()
    is_tb = bool(prob > 0.5)
    result = {
        "tb_probability": float(prob),
        "is_tb": is_tb,
        "confidence_tb": float(prob) if is_tb else float(1 - prob),
        "label": "TB Detected" if is_tb else "No TB Detected",
        "threshold": 0.5,
        "audio_duration_sec": round(audio_duration, 2),
        "sample_rate": SR,
        "device": str(DEVICE),
        "model": "MobileNetV4_Res2TSM",
        "processing_time_ms": round((t2 - t0) * 1000, 1),
        "spectrogram": mel_to_image(mel_db),
        "waveform": waveform_to_image(y_seg),
        "freq_spectrum": freq_spectrum_to_image(mel_db),
        "mfcc": mfcc_to_image(y_seg)
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
