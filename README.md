# AIMS-FL: Federated Learning for ITS Network Slice Impact Classification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository extends the [AIMS framework](https://github.com/saraivacode/AIMS) with **federated learning** and **security experiments** for classifying the impact level of network slicing policies on ITS (Intelligent Transportation Systems) applications.

The system embeds distributed impact classification across the network slice lifecycle (Prepare, Commissioning, Operation, Decommissioning). Local controllers at Roadside Units (RSUs) train models on edge monitoring data and transmit only model updates to a central aggregator — no raw data leaves the edge. The classifier maps multidimensional monitoring signals (RTT, PDR, temporal derivatives) into three actionable impact levels (**Low**, **Medium**, **High**) to trigger graduated management responses.

### Relationship to the AIMS Framework

| Layer | Repository | Description |
|---|---|---|
| **Centralized ML** | [saraivacode/AIMS](https://github.com/saraivacode/AIMS) | Random Forest, CatBoost, TabNet (4-class impact classification) |
| **Federated Learning** | This repo (Chapter 9) | DNN, LSTM, GRU with FedAvg/FedProx over IID and Non-IID data |
| **FL Security** | This repo (Chapter 10) | Poisoning vulnerability assessment with Byzantine-robust defenses |

The federated learning module (`code/federated/`) is based on the modular FL package from the AIMS framework, adapted for the 3-class ITS impact classification task and extended with security experiments.

---

## Repository Structure

```
its-fl-slice-impact/
├── code/
│   ├── federatedLearning_ITS.py      # CLI entry point
│   ├── generate_paper_charts.py      # Publication-ready figure generation
│   └── federated/                    # Modular FL package
│       ├── __init__.py               # Package exports
│       ├── fl_config.py              # Configuration & ExperimentConfig
│       ├── fl_data.py                # Data loading, scaling, IID/Non-IID partitioning
│       ├── fl_models.py              # DNN, LSTM, GRU model builders (Keras)
│       ├── fl_server.py              # FL simulation engine (FedAvg, FedProx, Krum, TrimmedMean)
│       ├── fl_security.py            # Label-flip attack & gradient scaling
│       ├── fl_centralized.py         # Centralized training baselines
│       ├── fl_results.py             # Results manager (JSON, CSV, LaTeX)
│       ├── fl_visualizations.py      # Convergence & comparison plots
│       └── fl_main.py                # Experiment orchestrator (all suites)
├── data/
│   └── raw_full.csv                  # Dataset (5,093 samples, 28 features)
├── results/                          # Experiment outputs (per version)
├── requirements.txt
└── README.md
```

---

## Architecture

### FL Simulation

The framework uses a **manual FL simulation loop** (no Ray dependency), giving full control over the training process:

1. **Global model** initialized on the server
2. Each **communication round**:
   - Server distributes global weights to all clients
   - Clients train locally for `local_epochs` epochs
   - (Optional) Malicious clients apply label-flip + gradient scaling
   - Server aggregates updates using the selected strategy
   - Server evaluates global model on the test set
3. Per-round metrics collected: accuracy, F1, precision, recall, confusion matrix

### Models

| Model | Architecture | Parameters |
|---|---|---|
| **DNN** | Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Dense(32) → Dense(3) | L2=0.001 |
| **LSTM** | LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.2) → Dense(3) | Input: (1, features) |
| **GRU** | GRU(64) → Dropout(0.3) → GRU(32) → Dropout(0.2) → Dense(3) | Input: (1, features) |

All models use Adam optimizer with sparse categorical crossentropy and class-weighted training.

### Aggregation Strategies

| Strategy | Description | Defense Capability |
|---|---|---|
| **FedAvg** | Weighted average proportional to dataset size | None |
| **FedProx** | FedAvg + proximal term (mu=0.1) to reduce client drift | Moderate |
| **Krum** | Byzantine-robust: selects update closest to others (m=0, no privileged info) | Strong vs. amplified updates |
| **Trimmed Mean** | Coordinate-wise trimmed average (beta=0.34, effectively median for n=3) | Strongest for n=3 |

---

## Experiments

### Chapter 9 — Federated Learning (Standard)

15 experiments: 12 federated + 3 centralized baselines.

| # | Model | Distribution | Strategy | Rounds | Local Epochs |
|---|-------|-------------|----------|--------|--------------|
| 1–4 | DNN | IID / Non-IID | FedAvg / FedProx | 10 | 5 |
| 5–8 | LSTM | IID / Non-IID | FedAvg / FedProx | 10 | 5 |
| 9–12 | GRU | IID / Non-IID | FedAvg / FedProx | 10 | 5 |
| 13–15 | DNN / LSTM / GRU | Centralized | — | — | 30 (early stop) |

### Chapter 10 — Security Experiments

#### Phase 1a: Label-Flip Poisoning Vulnerability

24 configs: 2 distributions × 3 attack levels × 4 strategies. Model fixed to GRU.

| Attack | Description | Malicious Client |
|---|---|---|
| **None** | Baseline (no attack) | — |
| **20% flip** | 20% of High(2) labels flipped to Low(0) | Client 0 |
| **50% flip** | 50% of High(2) labels flipped to Low(0) | Client 0 |

#### Phase 1b: Model Poisoning (Gradient Scaling)

32 configs: 2 distributions × 4 scale factors × 4 strategies.

| Scale | Description |
|---|---|
| **1x** | Control (label-flip only, 100% H→L) |
| **5x** | Moderate amplification |
| **10x** | Strong amplification |
| **20x** | Extreme amplification |

#### Additional Security Suites

| Suite | Configs | Description |
|---|---|---|
| **Krum Rerun** | 8 | Krum with `num_malicious_clients=0` (no privileged info) |
| **Sensitivity** | 8 | `local_epochs=1` vs default=5 at representative scale factors |

### Non-IID Data Allocation

The class-conditional partitioning reflects progressive congestion along the 650m urban corridor:

| Impact Level | RSU 0 (downstream, saturated) | RSU 1 (middle, mixed) | RSU 2 (entry, pre-congestion) |
|---|---|---|---|
| Low | 15% | 30% | 55% |
| Medium | 25% | 50% | 25% |
| High | 55% | 25% | 20% |

---

## Usage

All commands are run from the `code/` directory:

```bash
cd code
```

### Chapter 9: Standard FL Experiments

```bash
# Full experiment suite (12 FL + 3 centralized)
python federatedLearning_ITS.py

# Quick validation (DNN only, 3 FL + 1 centralized)
python federatedLearning_ITS.py --quick
```

### Chapter 10: Security Experiments

```bash
# Phase 1a: Poisoning vulnerability (24 configs)
python federatedLearning_ITS.py --security

# Phase 1a: Quick validation (4 configs)
python federatedLearning_ITS.py --security-quick

# Phase 1b: Gradient scaling (32 configs)
python federatedLearning_ITS.py --security-phase1b

# Phase 1b: Quick validation (4 configs)
python federatedLearning_ITS.py --security-phase1b-quick

# Phase 1b: Krum rerun with m=0 (8 configs)
python federatedLearning_ITS.py --security-phase1b-krum-rerun

# Sensitivity analysis: local_epochs=1 (8 configs)
python federatedLearning_ITS.py --security-sensitivity-epochs
```

### Generate Publication Figures

After results are available in `results/v1/paper_artifacts/`:

```bash
python generate_paper_charts.py
```

---

## Outputs

Results are saved to `results/<version>/`:

```
results/<version>/
├── paper_artifacts/
│   ├── summary_table.csv               # Comparative results table
│   ├── summary_table.tex               # LaTeX-formatted table
│   ├── security_summary.csv            # Security experiment results
│   ├── security_summary.tex            # Security LaTeX table
│   ├── centralized_results.json        # Centralized baseline metrics
│   ├── <config_name>.json              # Per-experiment detailed results
│   ├── <config>_confusion_matrix.json  # Per-experiment confusion matrices
│   ├── convergence_comparison.png      # Accuracy convergence curves
│   ├── strategy_comparison.png         # FedAvg vs FedProx on Non-IID
│   ├── fl_vs_centralized.png           # FL vs centralized bar chart
│   └── client_distribution_*.png       # IID/Non-IID class distribution
└── client_logs/
    ├── <config>_client_*_round_*_fit.json   # Per-client training metrics
    ├── <config>_client_*_round_*_eval.json  # Per-client evaluation metrics
    ├── <config>_client_*_attack.json        # Attack details (if applicable)
    └── <config>_client_*_round_*_boost.json # Gradient scaling norms
```

---

## Installation

### Prerequisites

- **Python 3.11+**
- GPU recommended but not required (CPU fallback automatic)

### Setup

```bash
git clone https://github.com/saraivacode/its-fl-slice-impact.git
cd its-fl-slice-impact
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.20.0 | Neural network backend |
| `keras` | 3.13.0 | Model building API |
| `scikit-learn` | 1.8.0 | Metrics, preprocessing, splitting |
| `pandas` | 2.3.3 | Data manipulation |
| `numpy` | 2.4.0 | Numerical computing |
| `matplotlib` | 3.10.8 | Visualization |
| `flwr` | 1.25.0 | Flower (available but not used for simulation) |

---

## Reproducibility

All random number generators are seeded deterministically:

- **Global seed**: `42`
- **Client-specific seeds**: `42 + client_id`
- **Stratified 80/20** global train/test split
- **StandardScaler** fit on training set only (no data leakage)
- **Class-weighted training** to handle imbalanced distribution

---

## Dataset

The dataset (`data/raw_full.csv`) contains **5,093** network performance samples from the ITS emulation scenario described in:

> T. do Vale Saraiva *et al.*, "An Application-Driven Framework for Intelligent Transportation Systems Using 5G Network Slicing," *IEEE Trans. Intell. Transp. Syst.*, vol. 22, no. 8, pp. 5247–5260, Aug. 2021. DOI: [10.1109/TITS.2021.3086064](https://doi.org/10.1109/TITS.2021.3086064)

**Scenario**: 650 m urban corridor, 3 RSUs, up to 17 vehicles, 4 application types (Safety, Efficiency, Entertainment, Generic), 3 slicing strategies (Flat Network, Queue-based, Full Slicing).

**Features** (28 total):
- **One-hot encoded** (12): application type (4), slicing approach (3), requirement category (4)
- **Network metrics** (7): `rec_serv`, `env_car`, `rtt`, `ncars`, `pdr`, `bc_rtt`
- **Temporal derivatives** (4): `rtt_change`, `pdr_change`, `rtt_mam`, `pdr_mam`, `rtt_masd`
- **Engineered** (5): log/squared transforms of RTT and PDR

**Target**: `impact_level` — Low (1,883), Medium (2,004), High (1,206).

---

## Security Metrics

In addition to standard ML metrics (accuracy, F1-macro, precision, recall), security experiments track:

| Metric | Description |
|---|---|
| **H→L Rate** | Fraction of true High samples misclassified as Low |
| **H→L Count** | Absolute count of High→Low misclassifications |
| **Delta Accuracy** | Accuracy drop relative to no-attack baseline |
| **Confusion Matrix** | Full 3×3 matrix for per-class analysis |
| **Gradient Norms** | L2 norms before/after scaling (Phase 1b) |

---

## License

This project is licensed under the MIT License.

## Contact

- **Tiago do Vale Saraiva** — [tiago.saraiva@uniriotec.br](mailto:tiago.saraiva@uniriotec.br)
