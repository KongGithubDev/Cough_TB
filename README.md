# CoughTB — AI-Powered Tuberculosis Screening from Cough Sounds

> **Screening Tool — not a diagnostic device.**
> Aligned with WHO guidelines for community-based TB triage testing.
> All positive screens require GeneXpert or culture confirmation.

CoughTB analyzes smartphone-recorded cough sounds with deep learning to assess
pulmonary tuberculosis (TB) risk, enabling low-cost, non-invasive screening in
resource-limited settings where confirmatory diagnostics (GeneXpert, culture)
are inaccessible. The model runs as a FastAPI web app with microphone recording
and file upload, ready for field testing.

---

## Table of Contents

1. [Overview](#overview)
2. [Why CoughTB](#why-coughtb)
3. [Model Performance](#model-performance)
4. [Architecture](#architecture)
5. [Audio Preprocessing Pipeline](#audio-preprocessing-pipeline)
6. [Training](#training)
7. [Deployment](#deployment)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [Train Your Own Model](#train-your-own-model)
11. [Datasets & References](#datasets--references)
12. [Roadmap](#roadmap)
13. [Limitations](#limitations)
14. [Possible Errors & Failure Modes](#possible-errors--failure-modes)
15. [License](#license)

---

## Overview

Tuberculosis remains the leading infectious disease killer worldwide, with an
estimated 10.8 million new cases and 1.25 million deaths in 2023. Current
screening relies on self-reported cough (only ~42% sensitive per WHO), while
confirmatory tests like GeneXpert MTB/RIF are expensive (~$10/test) and require
stable electricity — inaccessible in many low- and middle-income communities
where 80% of cases occur.

Cough sound analysis powered by AI offers a non-invasive, scalable screening
alternative. A systematic review and meta-analysis by Sahoo et al. (SSRN
5242653, 2025) covering seven studies reported pooled sensitivity of 91% and
specificity of 89% for AI-based cough analysis — but the reviewed models
(VGG16, ResNet50, 20–70M parameters) are too large for mobile deployment.

CoughTB closes that gap: a lightweight architecture (6.6M parameters) that
matches the pooled accuracy of prior work while being 4–20× smaller and
suitable for on-device inference.

---

## Why CoughTB

| Problem | CoughTB Solution |
|---------|------------------|
| Self-reported cough only ~42% sensitive | AI cough analysis → 91.2% sensitivity |
| GeneXpert expensive, needs electricity | Smartphone recording, no lab equipment |
| Large AI models can't run on phones | 6.6M params → TFLite/ONNX/CPU inference |
| Single-dataset models overfit | ImageNet pretraining + patient-level split, no leakage |
| Black-box AI outputs | Returns spectrogram, waveform, MFCC, frequency spectrum visualizations |

### Three Key Contributions

1. **MobileNetV4 + Res2TSM architecture** — a novel combination achieving
   state-of-the-art accuracy with only 6.6M parameters (4–20× smaller than prior
   models).
2. **Patient-level evaluation** — strict 80/20 split preventing data leakage,
   with per-patient sensitivity (91.2%) and specificity (91.4%) exceeding the
   WHO Target Product Profile (TPP) for community-based TB triage
   (≥90% sensitivity, ≥70% specificity).
3. **Deployable pipeline** — FastAPI web app with microphone recording and
   file upload, ready for field testing.

---

## Model Performance

### Held-Out Test Set (CODA TB, Patient-Level 80/20 Split)

| Metric | Per-Cough (n=1,794) | Per-Patient (n=208) | WHO TPP |
|--------|--------------------:|--------------------:|:-------:|
| Sensitivity | 83.2% | **91.2%** | ≥90% |
| Specificity | 81.5% | **91.4%** | ≥70% |
| Accuracy | — | **91.3%** | — |
| ROC-AUC | 0.904 | — | — |

### Comparison with Meta-Analysis (SSRN 5242653)

| Metric | CoughTB (per-patient) | Meta-Analysis (n=7 studies) |
|--------|----------------------|------------------------------|
| Sensitivity | 91.2% | 91% (95% CI: 88–94%) |
| Specificity | 91.4% | 89% (95% CI: 85–92%) |
| AUC | 0.904 | 0.9539 |

### Comparison with CODA TB DREAM Challenge Benchmarks

| Approach | AUC |
|----------|:---:|
| Challenge: audio-only models | 0.69–0.78 |
| Challenge: audio + clinical metadata | 0.78–0.83 |
| **CoughTB (ours, audio-only)** | **0.904** |

### Ablation Study

| Model | AUC | Params | Inference (CPU) |
|-------|:---:|:------:|:---------------:|
| ResNet50 + BiGRU | 0.881 | 72.5M | 340 ms |
| MobileNetV4 (no temporal) | 0.876 | 5.9M | 45 ms |
| **MobileNetV4 + Res2TSM (ours)** | **0.904** | **6.6M** | 52 ms |

Res2TSM adds only 0.7M parameters (+12%) but improves AUC by +0.028, confirming
the temporal module's effectiveness while keeping inference ~6.6× faster than
ResNet50-BiGRU.

---

## Architecture

```
Cough Audio (.wav / .webm)
    │
    ▼
┌─────────────────────────────┐
│  1. Preprocessing           │
│  • Resample to 16 kHz mono │
│  • Silence removal (top_db) │
│  • Extract 0.5s max-energy │
│  • Mel Spectrogram 224×224  │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  2. Feature Extraction      │
│  MobileNetV4 (conv features)│
│      ↓ (C×T×F feature map) │
│  Res2TSM (temporal shift)   │
│      ↓ (temporal modeling)  │
│  AdaptiveAvgPool + FC       │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  3. Classification          │
│  Sigmoid → TB Probability   │
│  > 0.5 → "TB Detected"     │
│  < 0.5 → "No TB Detected"  │
└─────────────────────────────┘
```

### Backbone: MobileNetV4-Conv-Blur-Medium

A lightweight CNN optimized for mobile/edge deployment, pretrained on
ImageNet-1K. Three key innovations over MobileNetV3:

- **Universal Inverted Bottleneck (UIB)** — unifies standard MobileNet blocks
  (depthwise, depthwise separable, regular conv) under a single parameterized
  formulation, enabling efficient architecture search.
- **Mobile MQA** — multi-query attention for efficient spatial feature mixing.
- **Conv-Blur downsampling** — replaces strided convolutions with Gaussian
  blur, reducing aliasing artifacts and improving feature quality.

The "Conv-Blur-Medium" variant balances accuracy and efficiency (~5.9M
parameters from ImageNet pretraining).

Reference: *MobileNetV4: Universal Models for the Mobile Ecosystem* (2024) —
[arXiv:2404.10518](https://arxiv.org/abs/2404.10518)

### Temporal Module: Res2TSM (Res2Net + Temporal Shift Module)

Inserted after the backbone's final feature map (C×7×7), this module adds
temporal reasoning with only ~5K additional parameters.

**Temporal Shift Module (TSM)** — shifts a fraction of channels along the time
axis, enabling temporal information flow without adding parameters or FLOPs:

- `fold = C / 8` channels shifted `t → t-1`
- `fold = C / 8` channels shifted `t → t+1`
- Remaining channels unchanged

**Res2Net-style multi-scale** — channels split into 4 groups (scale=4); each
undergoes 3×1 depthwise convolution with hierarchical residual connections
(each group sees the output of the previous), creating multi-scale temporal
receptive fields.

References:
- *TSM: Temporal Shift Module for Efficient Video Understanding* (ICCV 2019) —
  [arXiv:1811.08383](https://arxiv.org/abs/1811.08383)
- *Res2Net: A New Multi-scale Backbone Architecture* (TPAMI 2021) —
  [arXiv:1904.01169](https://arxiv.org/abs/1904.01169)

### Classifier Head

```
AdaptiveAvgPool (1×1) → Dropout(0.3) → Linear(C, 1) → Sigmoid
```

### Full Model Summary

| Component | Output Shape | Parameters |
|-----------|-------------|------------|
| Input | 3×224×224 | — |
| MobileNetV4 backbone | C×7×7 | ~5.9M |
| Res2TSM (scale=4) | C×7×7 | ~5K |
| AdaptiveAvgPool | C×1×1 | — |
| Dropout + Linear + Sigmoid | 1 | ~0.7M |
| **Total** | | **~6.6M** |

---

## Audio Preprocessing Pipeline

### 1. Resampling & Normalization
- Input: any sample rate → resampled to **16 kHz**
- Converted to **mono** (single channel)
- Amplitude normalized

### 2. Silence Removal
- `librosa.effects.trim` with `top_db=20`
- Removes silent segments at beginning/end of recording

### 3. Segment Selection (Max-Energy Crop)
- If longer than 0.5s: sliding-window energy check via convolution selects the
  **0.5s segment with highest energy** (most cough-like portion).
- If shorter than 0.5s: zero-padded to 0.5s.

### 4. Mel Spectrogram
- `n_mels=224`, `fmax=8000 Hz`, `hop_length=512`, `win_length=2048`
- Converts 0.5s of audio → 2D time-frequency representation
- Resized to **224×224** via bilinear interpolation
- Min-max normalized to [0, 1]
- Stacked 3× to create **3-channel RGB-like input** (224×224×3)

The Mel spectrogram captures how energy distributes across frequency bands over
time — TB coughs tend to exhibit distinct spectral patterns compared to
non-TB coughs.

---

## Training

### Dataset: CODA TB DREAM Challenge

| Detail | Value |
|--------|-------|
| Participants | 2,143 (1,210 TB-negative, 933 TB-positive) |
| Cough sounds | 733,756 (solicited + longitudinal) |
| Countries | India, Madagascar, Philippines, South Africa, Tanzania, Uganda, Vietnam |
| Recording | Smartphone (44.1 kHz, 16-bit, mono, Hyfe platform) |
| Label | Microbiologically confirmed TB (GeneXpert MTB/RIF Ultra) |
| Recording type | Solicited (asked coughs) + spontaneous |
| Access | [Synapse: syn31472953](https://www.synapse.org/Synapse:syn31472953) · [HuggingFace mirror](https://huggingface.co/datasets/AHFIDAILabs/coda_tb_dataset) |

### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 1×10⁻⁴ (cosine annealing) |
| Weight decay | 1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 50 (early stopping, patience 7) |
| Loss | Binary Cross-Entropy |
| Split | Patient-level, 80% train / 20% test |
| Augmentation | SpecAugment (freq mask=8, time mask=8), additive Gaussian noise (σ=0.005) |
| Class weighting | Inverse frequency weighting |

### Transfer Learning Strategy

| Phase | Epochs | Frozen parts | Learning Rate |
|:-----:|:------:|:-------------|:--------------|
| 1 | 10 | Backbone (MobileNetV4) | 1×10⁻³ (head + Res2TSM) |
| 2 | 40 | Nothing (full fine-tune) | 1×10⁻⁴ (cosine annealing) |

Phase 1 trains only the head + Res2TSM (randomly initialized, no pretrained
weights available) on a frozen ImageNet-pretrained backbone. Phase 2 unfreezes
everything for full fine-tuning with a lower learning rate.

### Evaluation Metrics

- **Per-cough**: accuracy, sensitivity, specificity, ROC-AUC
- **Per-patient**: majority vote across all coughs from the same participant
- **Confidence intervals**: 95% bootstrapped (2,000 iterations)
- **Comparison baselines**: ResNet50-LSTM, VGG16-BiGRU, MobileNetV4 (no temporal)

---

## Deployment

### Web App (FastAPI)

The `web/` directory contains a FastAPI server (`app.py`) that:

- Loads the pretrained `model.pth` (downloads from a GitHub mirror on first run
  if absent).
- Accepts audio via microphone recording or file upload (`.wav`, `.webm`).
- Runs preprocessing + inference, returns TB probability and visualizations.
- Serves a single-page frontend (`templates/index.html`).

The inference endpoint `/predict` returns JSON with:

- `tb_probability`, `is_tb`, `confidence_tb`, `label`, `threshold` (0.5)
- `audio_duration_sec`, `sample_rate`, `device`, `model`
- `processing_time_ms`
- `spectrogram` (magma colormap), `waveform`, `freq_spectrum`, `mfcc`
  (base64 PNG images)

### On-Device Export Paths

The 6.6M parameter footprint enables efficient export:

- **ONNX** — browser inference via ONNX Runtime Web.
- **TFLite** — Android deployment with 4-bit quantization (~1.7 MB).
- **CoreML** — iOS inference.

### Resource Tuning

The app respects two environment variables for CPU/VPS deployment:

- `OMP_NUM_THREADS` — torch thread count. Default `1` (Render free tier,
  512 MB RAM). Set to `4` on a multi-core VPS with ≥10 GB RAM.
- `PORT` — bind port (default `8000`).

---

## Project Structure

```
CoughTB/
├── web/                         # FastAPI web app (mic recording + file upload)
│   ├── app.py                   # Server + model inference + visualization
│   ├── requirements.txt
│   ├── model.pth                # Pre-trained MobileNetV4 + Res2TSM weights
│   └── templates/
│       └── index.html           # Frontend UI
├── notebooks/
│   ├── precompute_mels.ipynb    # Colab: pre-compute Mel spectrograms (~10× faster training)
│   ├── training.ipynb           # Colab: train MobileNetV4 + Res2TSM on CODA TB
│   ├── inference.ipynb          # Colab: inference + evaluation
│   └── web_colab.ipynb          # Colab: deploy web app with public URL
├── samples/
│   ├── non-tb.wav               # Sample non-TB cough
│   └── tb-positive/             # Sample TB-positive coughs
├── dataset/                     # CODA TB dataset (local, not in git)
├── Dockerfile                   # Container build
├── render.yaml                  # Render.com deployment config
├── .gitignore
└── README.md
```

---

## Quick Start

### Web App (Local / VPS)

```bash
cd web
pip install -r requirements.txt
python app.py
# → http://localhost:8000
```

On a Windows VPS with 10 GB+ RAM (e.g., Ryzen 5950X):

```powershell
cd web
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
$env:PORT=3003; python app.py
# → http://localhost:3003
```

The app uses `OMP_NUM_THREADS` — set to `4` for faster CPU inference.

### Web App (Colab)

Open `notebooks/web_colab.ipynb` in Colab and run all cells for a public URL.

### Inference Notebook

Open `notebooks/inference.ipynb` in Colab for batch evaluation.

---

## Train Your Own Model

Train CoughTB from scratch on the CODA TB dataset
([HuggingFace: `AHFIDAILabs/coda_tb_dataset`](https://huggingface.co/datasets/AHFIDAILabs/coda_tb_dataset)).

### Step 1: Pre-compute Mel Spectrograms

Skip expensive on-the-fly audio conversion — pre-compute once for ~10× faster
training.

```bash
# 1. Download CODA TB dataset from HuggingFace
# 2. Zip the dataset folder
# 3. Open notebooks/precompute_mels.ipynb in Google Colab
# 4. Upload the ZIP → run all cells → download pre-computed data
```

Expected: ~10–15 min for 10k files, ~1–2 hours for full 733k files.

### Step 2: Train Model

Two-phase transfer learning strategy (see [Training](#training)).

```bash
# Open notebooks/training.ipynb in Google Colab
# Upload pre-computed .npy files (from Step 1) → run all cells
# Download trained model: cough_tb_model.pth
```

Augmentation: SpecAugment (time/freq masking) + Gaussian noise.
Early stopping: patience 7 (Phase 2).
Split: patient-level 80/20 stratified by TB status.

### Step 3: Deploy

Replace `web/model.pth` with your trained weights and restart the web app.

---

## Datasets & References

### Datasets

**CODA TB DREAM Challenge** — solicited cough sounds recorded on smartphones
(Hyfe platform) from participants across seven high-TB-burden countries, with
microbiologically confirmed TB labels.

- Hugging Face mirror (preferred for download):
  [AHFIDAILabs/coda_tb_dataset](https://huggingface.co/datasets/AHFIDAILabs/coda_tb_dataset)
- Original Synapse record:
  [Synapse: syn31472953](https://www.synapse.org/Synapse:syn31472953)

### Benchmark / Meta-Analysis

Sahoo RK, Sinha A, Mishra M, Bhattacharya D, et al. *A Systematic Review and
Meta-Analysis of the Diagnostic Accuracy of Artificial Intelligence in
Detecting Tuberculosis Using Cough Sounds.* **SSRN** 5242653, 2025.
[SSRN landing page](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5242653)

### Full Reference List

1. Sahoo RK et al. *A Systematic Review and Meta-Analysis of the Diagnostic
   Accuracy of AI in Detecting TB Using Cough Sounds.* SSRN 5242653, 2025.
2. Huddart S et al. *A dataset of Solicited Cough Sound for Tuberculosis Triage
   Testing.* **Scientific Data** 11, 1149, 2024.
   [Nature](https://www.nature.com/articles/s41597-024-03984-5)
3. Jaganath D et al. *Accelerating Cough-Based Algorithms for Pulmonary TB
   Screening: Results From the CODA TB DREAM Challenge.* **Open Forum Infectious
   Diseases**, 2025.
4. Qin D et al. *MobileNetV4: Universal Models for the Mobile Ecosystem.*
   arXiv:2404.10518, 2024.
5. Lin J et al. *TSM: Temporal Shift Module for Efficient Video Understanding.*
   **ICCV**, 2019.
6. Gao S et al. *Res2Net: A New Multi-scale Backbone Architecture.* **TPAMI**,
   2021.
7. WHO. *Target product profile for a community-based TB triage test.* 2021.
8. WHO. *Global Tuberculosis Report 2024.* Geneva, 2024.
9. Pahar M et al. *Automatic cough classification for TB screening in a
   real-world environment.* **Physiol Meas**, 2021.
10. Botha GHR et al. *Detection of TB by automatic cough sound analysis.*
    **Physiol Meas**, 2018.
11. Yellapu GD et al. *Development and clinical validation of Swaasa AI
    platform for screening of pulmonary TB.* **Sci Rep**, 2023.

---

## Roadmap

| Phase | Status |
|-------|--------|
| Baseline model (MobileNetV4 + Res2TSM) | ✅ |
| Training pipeline (Colab) | ✅ |
| Pre-computation pipeline (10× faster training) | ✅ |
| Colab inference notebook | ✅ |
| Web app (FastAPI) | ✅ |

---

## Limitations

- **Single-dataset validation** — performance on external populations (e.g.,
  Peru, Sub-Saharan Africa) is untested.
- **Solicited coughs only** — model trained on asked coughs; performance on
  natural/opportunistic coughs is unknown.
- **No clinical metadata** — age, sex, symptoms may improve accuracy but were
  excluded to test an audio-only baseline.
- **Smartphone microphone** — recording quality variation across devices may
  affect generalizability.
- **Screening aid only** — not a diagnostic device; all positive screens
  require GeneXpert or culture confirmation.

---

## Possible Errors & Failure Modes

CoughTB is a screening tool — it can and will be wrong. Understanding where
errors come from is critical for safe deployment. The following are concrete
failure modes users and integrators may encounter.

### Clinical / Decision Errors

| Failure Mode | Likelihood | Impact | Mitigation |
|--------------|:----------:|--------|------------|
| **False negative** (TB case missed) | Per-cough sensitivity 83.2%, per-patient 91.2% | High — delayed diagnosis, ongoing transmission | Treat as triage only; symptomatic patients should still get GeneXpert regardless of result. Use per-patient majority vote, not a single cough. |
| **False positive** (non-TB flagged) | Per-patient specificity 91.4% → ~8.6% false-positive rate | Medium — unnecessary GeneXpert cost, patient anxiety | Threshold can be raised (e.g., 0.6) to trade sensitivity for specificity per local context. |
| **Asymptomatic early-stage TB** | Cough pattern may not yet be distinguishable | High — silent transmission | Combine with WHO symptom screen (cough >2 weeks, fever, weight loss, night sweats). |
| **Co-infections** (HIV, pneumonia, COVID-19, asthma) | Unknown effect on cough spectrum | Variable — altered cough features may shift scores | Out-of-distribution for the CODA TB training set; require clinical correlation. |
| **Pediatric / elderly coughs** | CODA TB cohort skews adult | Variable — age-dependent acoustic features | Not validated; do not use on patients <12 or >70 without local validation. |

### Audio / Input Errors

| Failure Mode | Trigger | Symptom | Mitigation |
|--------------|---------|---------|------------|
| **Empty / silent recording** | Mic permission denied, dead mic | `Cannot process audio` (422) | UI prompts user to retry; server-side length check rejects zero-length arrays. |
| **Non-cough audio** (speech, throat clearing, music) | User uploads wrong file | High garbage score, no TB signal | Browser-side classification (planned) or a cough-detection pre-filter before the TB model. |
| **Very short cough (<0.5 s)** | Natural short cough or truncated file | Zero-padded segment → low information | Model still runs; confidence drops. Warn user. |
| **Very loud / clipped audio** | Mic too close, mobile gain staging | Saturation → distorted spectrum | Apply peak normalization before feature extraction (planned). |
| **Background noise** (traffic, fans, music) | Noisy environment | Spectral contamination | Stronger denoising (RNNoise / spectral gating) in preprocessing pipeline. |
| **Wrong sample rate / codec** | `.webm` from browser, `.m4a` from iOS | Decode fails | Fallback to `ffmpeg` decoder (`_ffmpeg_decode` in `app.py`) — requires `ffmpeg` installed on host. |
| **Out-of-memory on tiny VPS** | Render free tier 512 MB RAM | App crash on first request | `OMP_NUM_THREADS=1` default; lazy model load on first request instead of startup; `gc.collect()` after each prediction. |

### Model / Inference Errors

| Failure Mode | Trigger | Symptom | Mitigation |
|--------------|---------|---------|------------|
| **Model download failure** | First run without `model.pth`; GitHub repo moved | `git clone` fails in `load_model()` | Bundle `model.pth` in image (Docker), or pin the source URL with a checksum. |
| **Checkpoint mismatch** | New architecture, old weights | `load_state_dict(..., strict=False)` silently skips missing keys | Log missing/unexpected keys; bump checkpoint version in filename. |
| **Threshold drift** | Fixed 0.5 across populations with different TB prevalence | Specificity collapses in low-prevalence settings | Make threshold configurable per deployment site; calibrate on local data. |
| **Class imbalance bias** | Inverse-frequency weighting overfits majority class | False negatives spike | Already mitigated by weighted loss + patient-level evaluation; recheck after retraining. |
| **Numerical instability** | `float16` inference on CPU | NaN scores | Server uses `float32` (`torch.from_numpy(...).float()`); do not switch to AMP without validation. |
| **Single-cough decision** | User records only one cough | Per-cough sensitivity 83.2% → bad single-point decision | UI prompts 3+ coughs; server returns per-cough scores; aggregator on frontend. |

### Deployment / Operational Errors

| Failure Mode | Trigger | Mitigation |
|--------------|---------|------------|
| **Cold-start latency** | First request after idle | Model loads lazily on first `/predict` call; subsequent calls are fast. |
| **CPU inference latency** | VPS with 1 vCPU | ~52 ms per cough on modern x86 — acceptable for batch, too slow for real-time UI streaming. |
| **Cross-origin / CORS errors** | Frontend served from different domain | FastAPI app uses template rendering (same-origin); add CORS middleware if API is split out. |
| **Privacy / data leakage** | Audio uploaded to third-party server | Currently self-hosted (no third-party). Add audio deletion policy, encrypted at rest, and an on-device (TFLite) deployment option for high-sensitivity contexts. |
| **Adversarial audio** | Maliciously crafted cough designed to flip the label | Model not adversarially robust | Out of scope for screening tool; document in consent flow. |
| **Stale model version** | New training not deployed | Health endpoint doesn't expose model version | Add `model_version` and `model_sha` to `/health` JSON. |

### Reporting Errors

If you encounter a failure not listed here, please open an issue with:

1. The input audio (or a description: duration, sample rate, noise).
2. The full JSON response from `/predict` (hide patient info).
3. The expected vs. observed label (if known).
4. Server logs (model load, OOM, decode failures).

---

## License

MIT