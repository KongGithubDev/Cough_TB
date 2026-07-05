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
│   ├── inference.ipynb          # Colab: inference + evaluation
│   └── web_colab.ipynb          # Colab: deploy web app with public URL
├── samples/
│   ├── non-tb.wav               # Sample non-TB cough
│   └── tb-positive/             # Sample TB-positive coughs
├── dataset/                     # CODA TB dataset (local, not in git)
├── .gitignore
└── README.md
```

## Model Performance

| Metric | CoughTB (CODA TB test set) | Meta-Analysis (SSRN 5242653) |
|--------|---------------------------|-------------------------------|
| Sensitivity (per-patient) | 91.2% | 91% |
| Specificity (per-patient) | 91.4% | 89% |
| ROC-AUC | 0.904 | 0.9539 |

## Quick Start

### Web App (Local)

```bash
cd web
pip install -r requirements.txt
python app.py
# → http://localhost:8000
```

### Web App (Colab)

Open `notebooks/web_colab.ipynb` in Colab and run all cells for a public URL.

### Inference Notebook

Open `notebooks/inference.ipynb` in Colab for batch evaluation.

## Architecture

```
Cough Audio → Segment (0.5s) → Mel Spectrogram (224×224)
    → MobileNetV4 Backbone → Res2TSM (temporal shift)
    → AdaptiveAvgPool → FC → Sigmoid → TB Probability
```

## Roadmap

| Phase | Status |
|-------|--------|
| Baseline model (MobileNetV4 + Res2TSM) | ✅ |
| Colab inference notebook | ✅ |
| Web app (FastAPI) | ✅ |
| TICTA 2026 submission | ⏳ 20 July 2026 |
| Mobile app (Flutter) | ⬜ |
| External validation | ⬜ |
| On-device inference (TFLite/ONNX) | ⬜ |

## References

1. Sahoo RK et al. *A Systematic Review and Meta-Analysis of the Diagnostic Accuracy of AI in Detecting TB Using Cough Sounds*. SSRN 5242653, 2025.
2. Huddart S et al. *A dataset of Solicited Cough Sound for Tuberculosis Triage Testing*. Scientific Data 11, 1149, 2024.
3. Jaganath D et al. *Accelerating Cough-Based Algorithms for Pulmonary TB Screening: Results From the CODA TB DREAM Challenge*. Open Forum Infectious Diseases, 2025.
4. WHO *Target product profile for a community-based TB triage test*, 2021.

## License

MIT
