# Market Pattern CNN 📈

> Predicting stock market movement from candlestick chart images using Convolutional Neural Networks

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

Instead of feeding raw price numbers into a model, this project converts historical OHLC (Open, High, Low, Close) stock data into **candlestick chart images** and trains a CNN to recognize visual price patterns — the same way a human trader reads a chart, but automated and at scale.

```
Past OHLC data  →  Candlestick chart images  →  CNN learns patterns  →  Predicts Up / Down / Sideways
```

This approach is validated by peer-reviewed research (Kim & Kim 2019; Lin et al. 2026; Brim & Flann 2022) which showed that CNNs trained on candlestick images can extract meaningful predictive signals from visual chart patterns.

---

## Results

| Model | Test Accuracy | vs Naive Baseline (59%) |
|---|---|---|
| ResNet18 (Transfer Learning) | 65% | 59% |
| VGG16 (Transfer Learning) | 62% | 59% |
| Random baseline | 33.3% | — |
| Naive baseline (always SIDEWAYS) | 59.0% | — |

> Fill in your actual accuracy numbers after training.

---

## Pipeline

```
Stage 1  →  Data Collection       yfinance pulls 9 years of OHLC data (SPY, AAPL, GLD, RELIANCE.NS)
Stage 2  →  Image Generation      8,873 candlestick PNGs at 128×128px via 30-day sliding window
Stage 3  →  Dataset Pipeline      70/15/15 split, weighted sampler to fix class imbalance
Stage 4a →  VGG16 Training        2-phase: head-only → selective unfreeze (blocks 4-5)
Stage 4b →  ResNet18 Training     2-phase: head-only → full fine-tune with CosineAnnealingLR
Stage 5  →  Evaluation            Accuracy, confusion matrix, F1, confidence histograms
Stage 6  →  Grad-CAM              Visualize which chart regions the CNN focuses on
Stage 7  →  Calibration           Temperature scaling to fix model overconfidence
Stage 8  →  Simulation            Confidence-threshold BUY/SELL decision system
Stage 9  →  Per-ticker Analysis   Breakdown accuracy by asset (SPY, AAPL, GLD, RELIANCE.NS)
```

---

## Dataset

| Ticker | Asset Type | Market | Rows | Usable Windows |
|---|---|---|---|---|
| SPY | S&P 500 ETF | US Equity | 2,264 | 2,229 |
| AAPL | Apple Inc. | US Tech Stock | 2,264 | 2,229 |
| GLD | Gold ETF | Commodity | 2,264 | 2,229 |
| RELIANCE.NS | Reliance Industries | Indian Market (NSE) | 2,221 | 2,186 |
| **Total** | | | | **8,873 images** |

- **Date range:** 2015–2024 (9 years)
- **Window:** 30 trading days per image
- **Lookahead:** 5 days to determine label
- **Threshold:** ±2% price change = Up or Down, else Sideways
- **Label split:** UP 24% / DOWN 17% / SIDEWAYS 59%

Each image is a 128×128px candlestick chart — no axes, no labels, pure visual price pattern. Color is preserved (green = bullish candle, red = bearish) as Huang & Chu (2024) showed color significantly enhances CNN feature extraction over black-and-white.

---

## Models

### VGG16 (Transfer Learning)
- Pretrained on ImageNet
- `AdaptiveAvgPool2d(4,4)` to handle 128×128 input
- Custom 3-layer classifier head: 8192 → 512 → 128 → 3
- **Phase 1:** Train classifier head only (frozen backbone)
- **Phase 2:** Unfreeze last 2 conv blocks (features[17:]) + fine-tune at 5e-5 with CosineAnnealingLR

### ResNet18 (Transfer Learning)
- Pretrained on ImageNet
- Deeper FC head: 512 → 256 → 64 → 3 with Dropout(0.4) and Dropout(0.2)
- **Phase 1:** Train FC head only
- **Phase 2:** Unfreeze all layers, fine-tune at 1e-4 with CosineAnnealingLR

### Shared training details
- Loss: `CrossEntropyLoss(label_smoothing=0.1)` — handles noisy financial labels
- Class imbalance fix: `WeightedRandomSampler` (not loss weighting — prevents overcorrection)
- Optimizer: Adam with weight_decay=1e-4 (phase 1), 1e-3 (phase 2)

---

## Key Features

### Grad-CAM Visualization
Implements Gradient-weighted Class Activation Mapping (Brim & Flann 2022) to show *which part of the chart* the model is looking at when making a prediction. Red/yellow = high attention, blue = low attention.

![Grad-CAM Example](assets/gradcam_resnet18.png)

### Temperature Scaling (Calibration)
Post-hoc calibration using a single temperature parameter `T` optimized on the validation set. Fixes model overconfidence without retraining — dividing logits by `T > 1` softens probabilities to better reflect true accuracy.

### Confidence-Threshold Simulation
BUY/SELL decision system that only acts when model confidence exceeds a threshold. The threshold is auto-optimized by sweeping from 0.35 to 0.95 and finding the point that maximizes `accuracy × √coverage` on BUY/SELL signals only (SIDEWAYS predictions are never acted on).

### Per-Ticker Analysis
Breaks down model accuracy separately for each of the 4 assets — verifies that the model generalizes across US equities, commodities, and Indian markets rather than overfitting to one asset's patterns.

---

## Project Structure

```
market-pattern-cnn/
│
├── README.md
├── market_cnn_v5_demo.ipynb      # Main notebook — all stages in one file
│
├── stage1_data_collection.py     # Download OHLC data via yfinance
├── stage2_image_generation.py    # Generate candlestick chart images
│
├── data/
│   ├── raw/                      # CSV files (SPY.csv, AAPL.csv, GLD.csv, RELIANCE.NS.csv)
│   └── images/                   # Generated chart images
│       ├── UP/
│       ├── DOWN/
│       └── SIDEWAYS/
│
├── models/                       # Saved model checkpoints (.pth files)
│   ├── vgg16.pth
│   └── resnet18.pth
│
└── assets/                       # Images for README
    ├── gradcam_resnet18.png
    ├── gradcam_vgg16.png
    ├── per_ticker_accuracy.png
    └── threshold_plot.png
```

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/market-pattern-cnn.git
cd market-pattern-cnn
```

### 2. Install dependencies
```bash
pip install yfinance mplfinance torch torchvision scikit-learn matplotlib seaborn pandas numpy opencv-python
```

### 3. Run Stage 1 — Download data
```bash
python stage1_data_collection.py
```

### 4. Run Stage 2 — Generate images
```bash
python stage2_image_generation.py
```
This takes ~10–15 minutes and generates ~8,873 PNG images in `data/images/`.

### 5. Train and evaluate
Open `market_cnn_v5_demo.ipynb` in **Google Colab** (recommended — free T4 GPU).

Upload `data/images/` as a zip to Google Drive, then run all cells in order.

---

## Running the Demo

The notebook includes a **live demo cell** (Cell 11) that lets you upload any candlestick chart image and get a real-time prediction with Grad-CAM overlay:

1. Run all cells up to Cell 10
2. Run Cell 11
3. Upload any PNG candlestick chart (from TradingView, Yahoo Finance, or your own `data/images/` folder)
4. The model outputs: predicted direction, class probabilities, confidence score, BUY/SELL/HOLD decision, and a Grad-CAM overlay

---

## Research Background

This project is grounded in published literature on visual chart analysis:

| Paper | Finding | Relevance |
|---|---|---|
| Kim & Kim (2019), *PLOS ONE* | Candlestick charts achieve 91% accuracy as CNN input, outperforming bar/line charts | Justifies candlestick format choice |
| Brim & Flann (2022), *PLOS ONE* | CNN on candlestick images alone outperformed S&P 500 index; Grad-CAM reveals attention shift to recent candles | Validates CNN-only approach; Grad-CAM methodology |
| Hoseinzade & Haratizadeh (2019), *Expert Systems* | Diverse multi-ticker training gives CNN richer feature representations | Justifies 4-ticker diversity (US equity, tech, gold, Indian market) |
| Mersal et al. (2025), *PeerJ CS* | Sliding-window candlestick sub-charts validated for CNN trend prediction | Directly validates Stage 2 image generation methodology |
| Lin et al. (2026), *Journal of Forecasting* | VGG16 with mixed chart images outperforms plain CNNs across all prediction horizons | Justifies VGG16 model choice |
| Huang & Chu (2024), *SSRN* | Color candlestick charts significantly enhance CNN accuracy vs black-and-white | Justifies colored candle rendering (green/red) |

---

## Limitations & Future Work

**Current limitations:**
- Labels are based purely on price return — no volume, no market context
- 2% threshold for Up/Down is fixed — optimal threshold may vary by asset and market regime
- Training data ends 2024 — model has not seen post-2024 market conditions

**Planned improvements:**
- Add volume bars and RSI/MACD overlay panels to images (Lin et al. 2026 showed mixed images outperform plain candlesticks)
- Attention mechanism on top of CNN (Huang & Chu 2024 showed attention-augmented CNN outperforms standard CNN)
- Walk-forward validation to test for temporal leakage
- Expand to more diverse assets and longer history

---

## Disclaimer

This project is for **educational and research purposes only**. No part of this system constitutes financial advice. The simulated decision system does not involve real trades or real money.

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Author

**Aditya C.S**
B.Tech Computer Science, PES University, Bengaluru
[LinkedIn](https://linkedin.com/in/AdityaCS) · [Email](mailto:adithyacs27@gmail.com)
