# CoughTB — AI-Powered Tuberculosis Screening from Cough Sounds

> **Screening Tool** — not a diagnostic device.
> Aligned with WHO guidelines for community-based TB triage testing.

## Overview

CoughTB analyzes cough sounds using deep learning to assess pulmonary TB risk, enabling low-cost screening in resource-limited settings.

**Reference**: Sahoo et al., *A Systematic Review and Meta-Analysis of the Diagnostic Accuracy of AI in Detecting TB Using Cough Sounds* (SSRN 5242653) — pooled sensitivity 91%, specificity 89%.

## Project Structure

```
CoughTB/
├── web/                         # FastAPI web app (mic recording + file upload)
│   ├── app.py                   # Server + model inference
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
├── .gitignore
├── README.md
├── TECHNICAL.md                 # Full technical pipeline documentation
└── PAPER_DRAFT.md               # TICTA 2026 submission draft
```

## Model Performance

| Metric | CoughTB (CODA TB test set) | Meta-Analysis (SSRN 5242653) |
|--------|---------------------------|-------------------------------|
| Sensitivity (per-patient) | 91.2% | 91% |
| Specificity (per-patient) | 91.4% | 89% |
| ROC-AUC | 0.904 | 0.9539 |
| Parameters | 6.6M | VGG16/ResNet50 (20–70M) |

## Quick Start

### Web App (Local / VPS)

```bash
cd web
pip install -r requirements.txt
python app.py
# → http://localhost:8000
```

> On a Windows VPS with 10GB+ RAM (e.g., Ryzen 5950X):
> ```powershell
> cd web
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> $env:PORT=3003; python app.py
> # → http://localhost:3003
> ```
> The app uses `OMP_NUM_THREADS` env var — set to `4` for faster CPU inference.

### Web App (Colab)

Open `notebooks/web_colab.ipynb` in Colab and run all cells for a public URL.

### Inference Notebook

Open `notebooks/inference.ipynb` in Colab for batch evaluation.

## Train Your Own Model

Train CoughTB from scratch on the CODA TB dataset (HuggingFace: `AHFIDAILabs/coda_tb_dataset`).

### Step 1: Pre-compute Mel Spectrograms

Skip expensive on-the-fly audio conversion — pre-compute once for ~10× faster training.

```bash
# 1. Download CODA TB dataset from HuggingFace
# 2. Zip the dataset folder
# 3. Open notebooks/precompute_mels.ipynb in Google Colab
# 4. Upload the ZIP → run all cells → download pre-computed data
```

Expected: ~10-15 min for 10k files, ~1-2 hours for full 733k files.

### Step 2: Train Model

Two-phase transfer learning strategy:

| Phase | Epochs | Frozen parts | Learning Rate |
|:-----:|:------:|:-------------|:--------------|
| 1 | 10 | Backbone (MobileNetV4) | 1e-3 (head + Res2TSM) |
| 2 | 40 | Nothing (full fine-tune) | 1e-4 (cosine annealing) |

```bash
# Open notebooks/training.ipynb in Google Colab
# Upload pre-computed .npy files (from Step 1) → run all cells
# Download trained model: cough_tb_model.pth
```

Augmentation: SpecAugment (time/freq masking) + Gaussian noise.  
Early stopping: patience 7 (Phase 2).  
Split: Patient-level 80/20 stratified by TB status.

### Step 3: Deploy

Replace `web/model.pth` with your trained weights and restart the web app.

## Architecture

```
Cough Audio → Segment (0.5s) → Mel Spectrogram (224×224)
    → MobileNetV4 Backbone → Res2TSM (temporal shift)
    → AdaptiveAvgPool → FC → Sigmoid → TB Probability
```

**MobileNetV4** extracts spatial features (ImageNet-pretrained).  
**Res2TSM** (Res2Net + Temporal Shift Module) adds temporal reasoning with only ~5K additional parameters.

## Roadmap

| Phase | Status |
|-------|--------|
| Baseline model (MobileNetV4 + Res2TSM) | ✅ |
| Training pipeline (Colab) | ✅ |
| Pre-computation pipeline (10× faster training) | ✅ |
| Colab inference notebook | ✅ |
| Web app (FastAPI) | ✅ |
| TICTA 2026 submission | ⏳ 20 July 2026 |
| Mobile app (Flutter) | ⬜ |
| External validation (TBscreen, etc.) | ⬜ |
| On-device inference (TFLite/ONNX) | ⬜ |

## References

1. Sahoo RK et al. *A Systematic Review and Meta-Analysis of the Diagnostic Accuracy of AI in Detecting TB Using Cough Sounds*. SSRN 5242653, 2025.
2. Huddart S et al. *A dataset of Solicited Cough Sound for Tuberculosis Triage Testing*. Scientific Data 11, 1149, 2024.
3. Jaganath D et al. *Accelerating Cough-Based Algorithms for Pulmonary TB Screening: Results From the CODA TB DREAM Challenge*. Open Forum Infectious Diseases, 2025.
4. WHO *Target product profile for a community-based TB triage test*, 2021.

## License

MIT
