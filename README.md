# Federated Learning for ITS Network Slice Impact Classification

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flower 1.x](https://img.shields.io/badge/Flower-1.x-green.svg)](https://flower.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository implements a federated learning framework that classifies the impact level of network slicing policies on ITS applications. The system embeds distributed impact classification as an intelligence layer across the network slice lifecycle (Prepare → Commissioning → Operation → Decommissioning).

Local controllers at Roadside Units (RSUs) train models on edge monitoring data and transmit only model updates to a central aggregator — no raw data leaves the edge. The classifier maps multidimensional monitoring signals (RTT, PDR, temporal derivatives) into three actionable impact levels (**Low**, **Medium**, **High**) to trigger graduated management responses.

### Key Findings

- **IID scenarios**: Federated training matches centralized performance (Macro F1 > 0.97), converging to 90% accuracy within the first communication round.
- **Non-IID scenarios**: Accuracy drops ~20 pp across all architectures due to spatial–temporal heterogeneity inherent to vehicular corridors.
- **FedProx (µ=0.1) vs FedAvg**: No statistically meaningful gain (≤0.19 pp), indicating standard proximal regularization is insufficient for the observed degree of heterogeneity.
- **Convergence**: Learning stabilizes within 2–3 rounds under Non-IID, yielding a usable global model with limited backhaul.

---

## Repository Structure

```
its-fl-slice-impact/
├── code/
│   ├── federatedLearning_ITS.py      # Main experiment orchestrator (985 lines)
│   └── generate_paper_charts.py      # Publication-ready figure generation
├── data/
│   └── raw_full.csv                  # Dataset (5,093 samples, 28 features)
├── results/
│   └── v1/
│       ├── paper_artifacts/          # Summary tables, per-experiment JSONs, charts
│       └── client_logs/              # Per-client per-round training metrics
├── requirements.txt
└── README.md
```

---

## Experiment Matrix

15 experiments: 12 federated (3 models × 2 distributions × 2 strategies) + 3 centralized baselines.

| # | Model | Distribution | Strategy | Rounds | Local Epochs |
|---|-------|-------------|----------|--------|--------------|
| 1–4 | DNN | IID / Non-IID | FedAvg / FedProx | 10 | 5 |
| 5–8 | LSTM | IID / Non-IID | FedAvg / FedProx | 10 | 5 |
| 9–12 | GRU | IID / Non-IID | FedAvg / FedProx | 10 | 5 |
| 13–15 | DNN / LSTM / GRU | Centralized | — | — | 30 (early stop, patience=5) |

### Non-IID Data Allocation

The class-conditional partitioning reflects progressive congestion along the corridor:

| Impact Level | RSU 0 (downstream) | RSU 1 (middle) | RSU 2 (entry) |
|---|---|---|---|
| Low | 15% | 30% | 55% |
| Medium | 25% | 50% | 25% |
| High | 55% | 25% | 20% |

---

## Prerequisites

- **Python 3.12** (developed and tested on 3.12.12)
- GPU recommended (experiments ran on Tesla T4 via Google Colab)

### Installation

```bash
git clone https://github.com/saraivacode/its-fl-slice-impact.git
cd its-fl-slice-impact
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### Dependencies

- `flwr` (Flower) — Federated learning framework
- `tensorflow` / `keras` — Neural network models
- `pandas`, `numpy` — Data processing
- `scikit-learn` — Metrics and preprocessing
- `matplotlib`, `seaborn` — Visualization

---

## Usage

### Full Experiment Suite

Runs all 15 experiments with deterministic seeding (seed=42):

```bash
cd code
python federatedLearning_ITS.py
```

### Quick Validation

Runs a reduced subset (DNN only, 3 federated + 1 centralized) for environment testing:

```bash
cd code
python federatedLearning_ITS.py --quick
```

### Generate Publication Figures

After results are available in `results/v1/paper_artifacts/`:

```bash
cd code
python generate_paper_charts.py
```

Produces the four figures used in the paper: performance gap (IID vs Non-IID), efficiency trade-off, convergence analysis, and strategy comparison.

---

## Outputs

Results are saved to `results/v1/paper_artifacts/`:

| File | Description |
|---|---|
| `summary_table.csv` | Comparative results across all 15 experiments |
| `summary_table.tex` | LaTeX-formatted table (paper Table IV) |
| `centralized_results.json` | Centralized baseline metrics per model |
| `<model>_<dist>_<strategy>_e5.json` | Per-experiment results with round-by-round metrics |
| `paper_fig*.png` | Publication-ready figures |
| `client_logs/` | Per-client per-round training metrics |

---

## Reproducibility

All random number generators are seeded deterministically:
- Global seed: `42`
- Client-specific seeds: `42 + client_id`
- Stratified 80/20 train/test split

The FedProx implementation uses a post-epoch weight correction as a practical approximation of the proximal term, leveraging Flower's built-in `FedProx` strategy with `proximal_mu=0.1`.

---

## Dataset

The dataset (`data/raw_full.csv`) contains 5,093 network performance samples from the ITS emulation scenario in:

> T. do Vale Saraiva *et al.*, "An Application-Driven Framework for Intelligent Transportation Systems Using 5G Network Slicing," *IEEE Trans. Intell. Transp. Syst.*, vol. 22, no. 8, pp. 5247–5260, Aug. 2021.

**Scenario**: 650 m urban corridor, 3 RSUs, up to 17 vehicles, 4 application types (Safety, Efficiency, Entertainment, Generic), 3 slicing strategies (Flat Network, Queue-based, Full Slicing).

**Features**: 28 engineered features including RTT, PDR, broadcast RTT, temporal derivatives (change rates, moving averages, statistical moments).

**Target**: Impact level — Low (1,883), Medium (2,004), High (1,206).

---

## License

This project is licensed under the MIT License.

## Contact

- **Tiago do Vale Saraiva** - [tiago.saraiva@uniriotec.br](mailto:tiago.saraiva@uniriotec.br)
