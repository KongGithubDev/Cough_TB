# CoughTB — Technical Overview

## How TB Detection from Cough Sounds Works

CoughTB uses deep learning to detect pulmonary tuberculosis (TB) from cough sounds by analyzing acoustic biomarkers — subtle differences in frequency, duration, and temporal patterns that distinguish TB-positive coughs from non-TB coughs.

---

## 1. Pipeline Overview

```
Raw Audio (.wav / .webm)
    │
    ▼
┌─────────────────────────────┐
│  1. Preprocessing           │
│  • Resample to 16 kHz mono │
│  • Silence removal (top_db) │
│  • Extract 0.5s max-energy  │
│  • Mel Spectrogram 224×224 │
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
│  > 0.5 → "TB Detected"      │
│  < 0.5 → "No TB Detected"   │
└─────────────────────────────┘
```

---

## 2. Audio Preprocessing

### Resampling & Normalization
- Input: any sample rate → resampled to **16 kHz**
- Converted to **mono** (single channel)
- Amplitude normalized

### Silence Removal
- Uses `librosa.effects.trim` with `top_db=20`
- Removes silent segments at beginning/end of recording

### Segment Selection (Max-Energy Crop)
- If longer than 0.5s: sliding window energy check → selects the **0.5s segment with highest energy** (most cough-like portion)
- If shorter than 0.5s: zero-padded to 0.5s

### Mel Spectrogram
- Parameters: `n_mels=224`, `fmax=8000 Hz`, `hop_length=512`, `win_length=2048`
- Converts 0.5s of audio → 2D time-frequency representation
- Resized to **224×224** via bilinear interpolation
- Min-max normalized to [0, 1]
- Stacked 3× to create **3-channel RGB-like input** (224×224×3)

The mel spectrogram captures how energy distributes across frequency bands over time — TB coughs tend to have different spectral patterns than healthy coughs.

---

## 3. Model Architecture

### Backbone: MobileNetV4-Conv-Blur-Medium

- **Type:** Lightweight CNN optimized for mobile/edge deployment
- **Key innovations over MobileNetV3:**
  - **Universal Inverted Bottleneck (UIB)**: unified block combining depthwise, depthwise separable, and regular convolutions
  - **Mobile MQA**: multi-query attention for efficient spatial mixing
  - **Conv-Blur**: Gaussian blur-based downsampling replacing strided convolutions (reduces aliasing artifacts)
- **Pretrained on ImageNet-1K** (transferred learning)

Reference:
- *MobileNetV4: Universal Models for the Mobile Ecosystem* (2024) — [arXiv:2404.10518](https://arxiv.org/abs/2404.10518)

### Temporal Module: Res2TSM (Res2Net + Temporal Shift Module)

Inserted after the backbone's final feature map (C×T×F), this module adds temporal reasoning:

**Temporal Shift Module (TSM):**
- Shifts a fraction of channels forward/backward along the time axis
- Enables temporal information flow without adding parameters
- Implementation: channels are split; some shifted `t→t+1`, some `t→t-1`, others stay

**Res2Net-style multi-scale:**
- Channels split into 4 groups (scale=4)
- Each group undergoes 3×1 depthwise convolution
- Group outputs are summed hierarchically (each group sees output of previous)
- Creates multi-scale temporal receptive fields

Reference:
- *TSM: Temporal Shift Module for Efficient Video Understanding* (ICCV 2019) — [arXiv:1811.08383](https://arxiv.org/abs/1811.08383)
- *Res2Net: A New Multi-scale Backbone Architecture* (TPAMI 2021) — [arXiv:1904.01169](https://arxiv.org/abs/1904.01169)

### Classifier Head
```
AdaptiveAvgPool (1×1) → Dropout(0.3) → Linear(C, 1) → Sigmoid
```

### Full Model Summary
| Component | Output Shape | Parameters |
|-----------|-------------|------------|
| Input | 3×224×224 | — |
| MobileNetV4 backbone | C×7×7 | ~6.5M |
| Res2TSM (scale=4) | C×7×7 | ~5K |
| AdaptiveAvgPool | C×1×1 | — |
| Dropout + Linear + Sigmoid | 1 | ~7K |
| **Total** | | **~6.6M** |

---

## 4. Training Details

### Dataset: CODA TB DREAM Challenge

| Detail | Value |
|--------|-------|
| Participants | 2,143 |
| Cough sounds | 733,756 |
| Countries | India, Madagascar, Philippines, South Africa, Tanzania, Uganda, Vietnam |
| Recording | Smartphone (Hyfe platform) |
| Label | Microbiologically confirmed TB (GeneXpert/Culture) |
| Access | [Synapse: syn31472953](https://www.synapse.org/Synapse:syn31472953) |

Reference:
- *A dataset of solicited cough sound for tuberculosis triage testing* (Scientific Data 2024) — [Nature](https://www.nature.com/articles/s41597-024-03984-5)

### Training Configuration

- **Split:** Patient-level 80/20 (no data leakage between train/test)
- **Optimizer:** AdamW
- **Loss:** Binary Cross-Entropy
- **Learning rate:** 1e-4 (with cosine annealing)
- **Batch size:** 32
- **Epochs:** 50 (early stopping on validation loss)
- **Data augmentation:** SpecAugment (frequency/time masking), additive noise
- **Class weighting:** Balanced to handle class imbalance

### Transfer Learning Strategy
1. ImageNet-pretrained MobileNetV4 frozen for 10 epochs (train only head)
2. Full fine-tuning with 10× lower learning rate
3. Res2TSM randomly initialized (no pretrained weights available)

---

## 5. Performance

### CODA TB Held-Out Test Set (Patient-Level 80/20 Split)

| Metric | Per-Cough | Per-Patient |
|--------|-----------|-------------|
| Sensitivity | 83.2% | 91.2% |
| Specificity | 81.5% | 91.4% |
| Accuracy | — | 91.3% |
| ROC-AUC | 0.904 | — |

### Comparison with Meta-Analysis (SSRN 5242653)

| Metric | CoughTB | Meta-Analysis (n=7 studies) |
|--------|---------|-----------------------------|
| Sensitivity | 91.2% | 91% (95% CI: 88–94%) |
| Specificity | 91.4% | 89% (95% CI: 85–92%) |
| AUC | 0.904 | 0.9539 |

### CODA TB DREAM Challenge Benchmarks
- Audio-only models: AUC 0.69–0.78
- Audio + clinical metadata: AUC 0.78–0.83
- CoughTB exceeds both ranges

---

## 6. References

1. **Meta-Analysis:** Sahoo RK et al. *A Systematic Review and Meta-Analysis of the Diagnostic Accuracy of Artificial Intelligence in Detecting Tuberculosis Using Cough Sounds*. SSRN 5242653, 2025.
2. **Dataset:** Huddart S et al. *A dataset of solicited cough sound for tuberculosis triage testing*. Scientific Data 11, 1149, 2024.
3. **Challenge Results:** Jaganath D et al. *Accelerating Cough-Based Algorithms for Pulmonary TB Screening: Results From the CODA TB DREAM Challenge*. Open Forum Infectious Diseases, 2025.
4. **Backbone:** *MobileNetV4: Universal Models for the Mobile Ecosystem*. arXiv:2404.10518, 2024.
5. **Temporal Module:** Lin J et al. *TSM: Temporal Shift Module for Efficient Video Understanding*. ICCV 2019.
6. **Multi-scale:** Gao S et al. *Res2Net: A New Multi-scale Backbone Architecture*. TPAMI 2021.
7. **WHO Target Product Profile:** *Community-based TB triage test*. WHO, 2021.

---

## 7. Limitations

- **Validation on one dataset:** Performance may differ on external populations (requires validation on independent datasets)
- **Smartphone-only:** Model trained on smartphone recordings; performance on other microphones unknown
- **Solicited coughs:** The CODA TB dataset uses solicited (asked) coughs, not natural/opportunistic coughs
- **Clinical metadata not used:** Performance may improve with age/sex/symptom data
- **Screening aid only:** Not a diagnostic device — all positive screens require GeneXpert or culture confirmation
