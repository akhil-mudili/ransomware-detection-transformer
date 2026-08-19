# Transformer-Based Early-Stage Ransomware Detection Using Behavioral Sequence Modeling

This repository contains the code, model weights, and results for the paper:

**"Transformer-Based Early-Stage Ransomware Detection Using Behavioral Sequence Modeling"**  
Akhil Mudili — University of Galway

---

## Overview

This project proposes a Transformer-based model for early-stage ransomware detection using API call sequences extracted from dynamic analysis (Cuckoo Sandbox). Detection is evaluated at four partial observation windows (25%, 50%, 75%, 100%) to simulate real-world early detection scenarios.

Three baselines are evaluated under identical conditions: Random Forest, CNN, and LSTM.

---

## Dataset

- **2,860 samples** — 1,405 ransomware (19 families) + 1,455 benign
- API call sequences capped at 3,000 calls per sample
- Vocabulary size: 287 unique API calls
- Source: Cuckoo Sandbox dynamic analysis reports
- Split: 70% train / 15% validation / 15% test (stratified)

---

## Models

| Model | Type |
|---|---|
| Random Forest | Baseline |
| CNN | Baseline |
| LSTM | Baseline |
| **Transformer** | **Proposed** |

---

## Repository Structure

```
├── data/
│   └── vocabulary.pkl          # API call vocabulary mapping
├── notebooks/
│   ├── results_visualizations_final.ipynb
│   └── fig1–fig9 PNGs          # Result figures
├── results/
│   ├── lstm_results.json
│   ├── transformer_results.json
│   └── live_detection/         # Live detection simulation results
├── src/
│   ├── preprocessing.py        # Data loading and sequence windowing
│   ├── dataset_analysis.py     # Dataset statistics and exploration
│   └── models/
│       ├── transformer_model.py
│       ├── lstm_model_local.py
│       ├── cnn_baseline.py
│       ├── rf_baseline.py
│       ├── live_detection_dashboard.py
│       ├── live_detection_all_families.py
│       ├── transformer_window_25/50/75/100.pt
│       └── lstm_window_25/50/75/100.pt
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

**Preprocessing:**
```bash
python src/preprocessing.py
```

**Train / evaluate a model:**
```bash
python src/models/transformer_model.py
python src/models/lstm_model_local.py
python src/models/cnn_baseline.py
python src/models/rf_baseline.py
```

**Live detection dashboard:**
```bash
python src/models/live_detection_dashboard.py
```

---

## Results

The Transformer achieves strong detection performance across all observation windows, with the lowest false negative rate at early windows (25–50%), making it effective for early-stage ransomware detection before significant damage occurs.

Full results and visualisations are available in `notebooks/results_visualizations_final.ipynb`.

---

## License

This project is for academic research purposes.
