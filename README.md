# its-fl-slice-impact

[![Python 3.8+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Federated Learning for Scalable Network Slice Impact Classification in ITS**

This repository contains the implementation and experimental framework for a distributed intelligence layer designed to support proactive Network Slice (NS) management in Intelligent Transportation Systems (ITS). The project leverages **Federated Learning (FL)** to enable collaborative model training across geographically distributed edge nodes while preserving data privacy and addressing scalability challenges in vehicular environments.

---

## 📖 Overview

Managing network slices in highly dynamic ITS environments requires proactive Quality-of-Service (QoS) monitoring and accurate policy impact assessment. This framework integrates distributed impact classification into the network slice lifecycle:

* **Distributed Intelligence**: Local Controllers (LCs) co-located with Roadside Units (RSUs) operate as federated clients, training models on localized edge monitoring data.
* **Privacy-Preserving**: Only model updates (weights or gradients) are transmitted to a central aggregator, ensuring sensitive vehicular data remains on the edge.
* **Impact-Aware Management**: The system interprets multidimensional monitoring data and maps it to discrete impact levels (Low, Medium, High) to trigger graduated management responses.
* **Lifecycle Support**: Provides intelligence for decision-making across the Preparation, Commissioning, Operation, and Decommissioning phases of a network slice.

---

## 🛠️ Prerequisites

* **Python 3.10+**
* **Dependencies from `requirements.txt`**:  TensorFlow 2.x, Flower (FL framework), pandas, scikit-learn, matplotlib, etc.

### Installation

Install the packages in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Running the Experiments
The core engine is code/federatedLearning_ITS.py, which orchestrates 15 distinct experiments covering multiple neural architectures (DNN, LSTM, GRU) and aggregation strategies (FedAvg, FedProx).

### 1. Execute the Simulation
**Full Experimental Campaign** (~1.5–2h): Runs the complete matrix across IID and Non-IID data distributions.

  ```bash
  python code/federatedLearning_ITS.py
  ```

**Quick Validation** (~20–30 min): Runs a reduced subset of experiments for testing purposes.
- **Quick run** (~20–30 min, reduced subset):
  ```bash
  python code/federatedLearning_ITS.py --quick
  ```

### 2. Generate Visualizations
Once the results are populated in `results/v1/paper_artifacts/`, run `code/generate_paper_charts.py` once `results/v1/paper_artifacts` is populated::

```bash
python code/generate_paper_charts.py
```

---

## Outputs
- Results are saved to `results/<version>/paper_artifacts/`.
- Summaries in CSV/TeX (`summary_table.csv`, `summary_table.tex`) and per-experiment JSONs.
- Convergence and comparison plots are stored in the same directory.

## 📂 Repository Structure
* `code/federatedLearning_ITS.py`: Main orchestrator for federated and centralized training.

* `code/generate_paper_charts.py`: Visualization suite for processing JSON/CSV logs into publication-ready figures (builds charts from `summary_table.csv`).

* `data/raw_full.csv`: Dataset containing 5,093 network performance samples across four application types: Safety, Efficiency, Entertainment, and Generic.

* `results/`: Directory for generated TeX tables, CSV summaries, and PNG/EPS figures.

---

## 📊 Key Findings
* **IID Performance**: Achieves a Macro F1-score exceeding **0.97**, matching centralized baselines without data centralization.

* **Convergence Efficiency**: Demonstrates rapid learning, reaching near-final accuracy within **2 to 3 communication rounds**.

* **Statistical Heterogeneity**: Quantifies the impact of Non-IID data, where **FedProx** provides superior stability compared to standard FedAvg.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Federal University of State of Rio de Janeiro (UNIRIO)
- Dataset based on [saraivacode/framework_its_sdn](https://github.com/saraivacode/framework_its_sdn), which uses:
  - [Mininet-WiFi Emulator](https://github.com/intrig-unicamp/mininet-wifi)
  - [Ryu SDN Controller](https://osrg.github.io/ryu/)
  - [SUMO Mobility Simulator](https://sumo.dlr.de/docs/Installing.html)


## Contact

- **Tiago do Vale Saraiva** - [tiago.saraiva@uniriotec.br](mailto:tiago.saraiva@uniriotec.br)


