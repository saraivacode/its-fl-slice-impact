# Federated Learning for Scalable Network Slice Impact Classification in ITS

Brief guide to run.

## Prerequisites
- Python 3.10+
- Dependencies from `requirements.txt` (TensorFlow, Flower, pandas, etc.).

Install the packages in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the simulation
The main script is `code/federatedLearning_ITS.py`.

- **Full run** (~1.5–2h, all 15 experiments):
  ```bash
  python code/federatedLearning_ITS.py
  ```

- **Quick run** (~20–30 min, reduced subset):
  ```bash
  python code/federatedLearning_ITS.py --quick
  ```

## Outputs
- Results are saved to `results/<version>/paper_artifacts/`.
- Summaries in CSV/TeX (`summary_table.csv`, `summary_table.tex`) and per-experiment JSONs.
- Convergence and comparison plots are stored in the same directory.

## Generate paper charts
Run `code/generate_paper_charts.py` once `results/v1/paper_artifacts` is populated:

```bash
python code/generate_paper_charts.py
```

This produces the `paper_fig*_v3.png` figures in the results directory.

## Quick structure
- `code/federatedLearning_ITS.py`: orchestrates federated and centralized experiments.
- `code/generate_paper_charts.py`: builds charts from `summary_table.csv` and JSONs.
- `data/raw_full.csv`: dataset used by the simulation.
- `results/`: outputs generated after running the scripts
