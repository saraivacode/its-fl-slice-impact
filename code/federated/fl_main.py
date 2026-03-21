#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Main Orchestrator
======================================

Top-level entry point for running the complete FL experiment suite.
Coordinates data loading, partitioning, FL simulation, centralized
baselines, security experiments, results saving, and visualization.

Experiment Suites:
    Chapter 9 (Full): 3 models x 2 distributions x 2 strategies = 12 FL + 3 centralized
    Chapter 9 (Quick): DNN only, 3 FL + 1 centralized
    Chapter 10 Security Phase 1a: 2 dist x 3 attacks x 4 strategies = 24 configs
    Chapter 10 Security Phase 1b: 2 dist x 4 scales x 4 strategies = 32 configs
    Chapter 10 Krum Rerun: 2 dist x 4 scales x 1 strategy = 8 configs
    Chapter 10 Sensitivity: 2 dist x 4 strategies x 1 scale = 8 configs

Usage:
    python federatedLearning_ITS.py                        # Full suite
    python federatedLearning_ITS.py --quick                # Quick test
    python federatedLearning_ITS.py --security             # Phase 1a
    python federatedLearning_ITS.py --security-phase1b     # Phase 1b
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf

from .fl_centralized import train_centralized
from .fl_config import ExperimentConfig, FLDefaults
from .fl_data import load_and_prepare, partition_iid, partition_non_iid
from .fl_models import configure_tf
from .fl_results import FLResultsManager
from .fl_server import run_fl_simulation
from .fl_visualizations import (
    plot_client_distribution,
    plot_convergence,
    plot_fl_vs_centralized,
    plot_strategy_comparison,
)

_DEFAULTS = FLDefaults()


def _get_project_paths():
    """Get absolute paths independent of execution location."""
    code_dir = Path(__file__).resolve().parent.parent
    project_root = code_dir.parent
    return project_root / "results"


def _get_csv_path():
    """Get path to the AIMS dataset."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "aims_dataset.csv"


def _load_and_split(seed: int = 42):
    """Load AIMS dataset, preprocess, and split into train/test."""
    from sklearn.model_selection import train_test_split

    X, y, class_weights, features = load_and_prepare(_get_csv_path(), random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    print(f"  Global split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test, class_weights


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================================
# Chapter 9: Standard FL Experiments
# =========================================================================

def run_full_suite(base_dir_name: str = "v1") -> tuple:
    """
    Full experiment suite: 12 federated + 3 centralized baselines.

    3 models (DNN, LSTM, GRU) x 2 distributions (IID, NonIID) x 2 strategies (FedAvg, FedProx).
    """
    results_root = _get_project_paths()
    base_dir = results_root / base_dir_name

    print("=" * 70)
    print("AIMS Federated Learning - Full Suite")
    print(f"Output Directory: {base_dir}")
    print("=" * 70)

    client_log_dir = base_dir / "client_logs"
    paper_dir = base_dir / "paper_artifacts"
    client_log_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    results_mgr = FLResultsManager(paper_dir)

    # Load and prepare data
    print("\n[Step 1/5] Loading and preparing data...")
    X_train, X_test, y_train, y_test, class_weights = _load_and_split()

    # Build experiment configs
    models = ["dnn", "lstm", "gru"]
    distributions = ["iid", "noniid"]
    strategies = ["fedavg", "fedprox"]

    # Create partitions
    print("\n[Step 2/5] Partitioning data for FL clients...")
    partitions = {}
    for dist in distributions:
        print(f"\n  {dist.upper()} partitioning:")
        if dist == "iid":
            partitions[dist] = partition_iid(X_train, y_train, _DEFAULTS.NUM_CLIENTS)
        else:
            partitions[dist] = partition_non_iid(X_train, y_train, _DEFAULTS.NUM_CLIENTS)
        plot_client_distribution(partitions[dist], dist, paper_dir)

    # Run federated experiments
    print("\n[Step 3/5] Running federated experiments...")
    experiments = [
        ExperimentConfig(m, _DEFAULTS.NUM_CLIENTS, _DEFAULTS.NUM_ROUNDS,
                         _DEFAULTS.LOCAL_EPOCHS, _DEFAULTS.BATCH_SIZE, d, s)
        for m in models for d in distributions for s in strategies
    ]

    fl_results = []
    for i, cfg in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"  Experiment {i+1}/{len(experiments)}: {cfg.to_string()}")
        print(f"{'='*60}")
        try:
            result = run_fl_simulation(
                config=cfg,
                client_partitions=partitions[cfg.distribution],
                X_test=X_test, y_test=y_test,
                class_weights=class_weights,
                log_dir=str(client_log_dir),
            )
            fl_results.append(result)
            results_mgr.save_experiment(result)
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback; traceback.print_exc()

    # Centralized baselines
    print(f"\n{'='*60}")
    print("[Step 4/5] Training centralized baselines...")
    print(f"{'='*60}")

    centralized_results = []
    for model in models:
        try:
            result = train_centralized(
                model_type=model, X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test, class_weights=class_weights,
            )
            centralized_results.append(result)
            results_mgr.save_experiment(result)
        except Exception as e:
            print(f"    Centralized {model} FAILED: {e}")

    # Summary and visualizations
    print(f"\n{'='*60}")
    print("[Step 5/5] Generating summary and visualizations...")
    print(f"{'='*60}")

    if fl_results:
        results_mgr.save_summary_table(fl_results, centralized_results)
        plot_convergence(fl_results, paper_dir)
        plot_strategy_comparison(fl_results, paper_dir)
        if centralized_results:
            results_mgr.save_centralized_results(centralized_results)
            plot_fl_vs_centralized(fl_results, centralized_results, paper_dir)

    _print_final_summary(fl_results, centralized_results, len(experiments), len(models), paper_dir)
    return fl_results, centralized_results


def run_quick_test(base_dir_name: str = "v1") -> list:
    """Quick test: DNN only, 3 FL + 1 centralized."""
    results_root = _get_project_paths()
    base_dir = results_root / base_dir_name

    print("=" * 70)
    print("AIMS Federated Learning - Quick Test")
    print(f"Output Directory: {base_dir}")
    print("=" * 70)

    client_log_dir = base_dir / "client_logs"
    paper_dir = base_dir / "paper_artifacts"
    client_log_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    results_mgr = FLResultsManager(paper_dir)

    X_train, X_test, y_train, y_test, class_weights = _load_and_split()

    partitions = {
        "iid": partition_iid(X_train, y_train, _DEFAULTS.NUM_CLIENTS),
        "noniid": partition_non_iid(X_train, y_train, _DEFAULTS.NUM_CLIENTS),
    }

    experiments = [
        ExperimentConfig("dnn", 3, 10, 3, 32, "iid", "fedavg"),
        ExperimentConfig("dnn", 3, 10, 3, 32, "noniid", "fedavg"),
        ExperimentConfig("dnn", 3, 10, 3, 32, "noniid", "fedprox", 0.1),
    ]

    fl_results = []
    for i, cfg in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"  Quick Test {i+1}/{len(experiments)}: {cfg.to_string()}")
        print(f"{'='*60}")
        result = run_fl_simulation(
            config=cfg, client_partitions=partitions[cfg.distribution],
            X_test=X_test, y_test=y_test, class_weights=class_weights,
            log_dir=str(client_log_dir),
        )
        fl_results.append(result)

    centralized = [train_centralized("dnn", X_train, y_train, X_test, y_test, class_weights, epochs=20)]

    results_mgr.save_summary_table(fl_results, centralized)
    if fl_results:
        plot_convergence(fl_results, paper_dir)
        plot_strategy_comparison(fl_results, paper_dir)

    return fl_results


# =========================================================================
# Chapter 10: Security Experiments
# =========================================================================

def _run_security_suite(
    experiments: List[ExperimentConfig],
    suite_name: str,
    base_dir_name: str,
) -> List[Optional[dict]]:
    """Generic runner for security experiment suites."""
    results_root = _get_project_paths()
    base_dir = results_root / base_dir_name

    print("=" * 70)
    print(f"ITS FL Security Experiments - {suite_name}")
    print(f"Output Directory: {base_dir}")
    print("=" * 70)

    client_log_dir = base_dir / "client_logs"
    paper_dir = base_dir / "paper_artifacts"
    client_log_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    results_mgr = FLResultsManager(paper_dir)

    X_train, X_test, y_train, y_test, class_weights = _load_and_split()

    # Determine which distributions we need
    dists_needed = set(cfg.distribution for cfg in experiments)
    partitions = {}
    for dist in dists_needed:
        print(f"\n  {dist.upper()} partitioning:")
        if dist == "iid":
            partitions[dist] = partition_iid(X_train, y_train, _DEFAULTS.NUM_CLIENTS)
        else:
            partitions[dist] = partition_non_iid(X_train, y_train, _DEFAULTS.NUM_CLIENTS)

    print(f"\nTotal experiments: {len(experiments)}")

    all_results = []
    baselines = {}

    for i, cfg in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"  Experiment {i+1}/{len(experiments)}: {cfg.to_string()}")
        print(f"{'='*60}")

        try:
            result = run_fl_simulation(
                config=cfg, client_partitions=partitions[cfg.distribution],
                X_test=X_test, y_test=y_test, class_weights=class_weights,
                log_dir=str(client_log_dir),
            )
            all_results.append(result)
            results_mgr.save_experiment(result)

            # Track baselines for delta computation
            if cfg.attack_type == "none":
                key = f"{cfg.distribution.upper()}_none_{cfg.strategy.upper()}"
                baselines[key] = result["final_metrics"]["accuracy"]
            elif cfg.scale_factor == 1.0:
                key = f"{cfg.distribution.upper()}_scale1_{cfg.strategy.upper()}"
                baselines[key] = result["final_metrics"]["accuracy"]
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results.append(None)

    # Generate security summary
    results_mgr.save_security_summary(all_results, baselines)

    print("\n" + "=" * 70)
    print(f"{suite_name} complete! Results saved in: {paper_dir}")
    print("=" * 70)

    return all_results


def run_security_experiments(base_dir_name: str = "v3") -> list:
    """
    Phase 1a: FL poisoning vulnerability assessment.
    24 configs: 2 distributions x 3 attack levels x 4 strategies.
    Model fixed to GRU.
    """
    model = "gru"
    malicious = [0]
    distributions = ["iid", "noniid"]
    strategies = ["fedavg", "fedprox", "krum", "trimmed_mean"]
    attacks = [("none", 0.0), ("label_flip", 0.2), ("label_flip", 0.5)]

    experiments = []
    for dist in distributions:
        for atk_type, atk_frac in attacks:
            for strat in strategies:
                mal = malicious if atk_type != "none" else []
                experiments.append(ExperimentConfig(
                    model_type=model, num_clients=3, num_rounds=10,
                    local_epochs=5, batch_size=32, distribution=dist,
                    strategy=strat, fedprox_mu=0.1, attack_type=atk_type,
                    attack_fraction=atk_frac, malicious_clients=mal,
                ))

    return _run_security_suite(experiments, "Phase 1a: Poisoning Vulnerability", base_dir_name)


def run_security_quick_test(base_dir_name: str = "v3_quick") -> list:
    """Quick security test: 4 configs to validate pipeline."""
    experiments = [
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "fedavg"),
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "fedavg",
                         attack_type="label_flip", attack_fraction=0.5, malicious_clients=[0]),
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "krum",
                         attack_type="label_flip", attack_fraction=0.5, malicious_clients=[0]),
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "trimmed_mean",
                         attack_type="label_flip", attack_fraction=0.5, malicious_clients=[0]),
    ]
    return _run_security_suite(experiments, "Phase 1a Quick Validation", base_dir_name)


def run_security_phase1b(base_dir_name: str = "v3b") -> list:
    """
    Phase 1b: Model poisoning with gradient scaling.
    32 configs: 2 distributions x 4 scale factors x 4 strategies.
    100% label-flip + gradient amplification.
    """
    model = "gru"
    malicious = [0]
    distributions = ["iid", "noniid"]
    strategies = ["fedavg", "fedprox", "krum", "trimmed_mean"]
    scale_factors = [1.0, 5.0, 10.0, 20.0]

    experiments = []
    for dist in distributions:
        for scale in scale_factors:
            for strat in strategies:
                experiments.append(ExperimentConfig(
                    model_type=model, num_clients=3, num_rounds=10,
                    local_epochs=5, batch_size=32, distribution=dist,
                    strategy=strat, fedprox_mu=0.1, attack_type="label_flip",
                    attack_fraction=1.0, malicious_clients=malicious, scale_factor=scale,
                ))

    return _run_security_suite(experiments, "Phase 1b: Model Poisoning (Gradient Scaling)", base_dir_name)


def run_security_phase1b_quick(base_dir_name: str = "v3b_quick") -> list:
    """Quick Phase 1b test: 4 configs to verify gradient scaling."""
    experiments = [
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "fedavg",
                         attack_type="label_flip", attack_fraction=1.0,
                         malicious_clients=[0], scale_factor=1.0),
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "fedavg",
                         attack_type="label_flip", attack_fraction=1.0,
                         malicious_clients=[0], scale_factor=10.0),
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "krum",
                         attack_type="label_flip", attack_fraction=1.0,
                         malicious_clients=[0], scale_factor=10.0),
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "trimmed_mean",
                         attack_type="label_flip", attack_fraction=1.0,
                         malicious_clients=[0], scale_factor=10.0),
    ]
    return _run_security_suite(experiments, "Phase 1b Quick Validation", base_dir_name)


def run_security_phase1b_krum_rerun(base_dir_name: str = "v3b_krum_fixed") -> list:
    """Rerun 8 Krum experiments with num_malicious_clients=0 (no privileged info)."""
    model = "gru"
    malicious = [0]
    distributions = ["iid", "noniid"]
    scale_factors = [1.0, 5.0, 10.0, 20.0]

    experiments = []
    for dist in distributions:
        for scale in scale_factors:
            experiments.append(ExperimentConfig(
                model_type=model, num_clients=3, num_rounds=10,
                local_epochs=5, batch_size=32, distribution=dist,
                strategy="krum", attack_type="label_flip", attack_fraction=1.0,
                malicious_clients=malicious, scale_factor=scale,
            ))

    return _run_security_suite(experiments, "Phase 1b Krum Rerun (m=0)", base_dir_name)


def run_security_sensitivity_epochs(base_dir_name: str = "v3b_sensitivity_epochs") -> list:
    """
    Sensitivity analysis: local_epochs=1 vs default=5.
    8 configs: 2 distributions x 4 strategies at representative scale factors.
    """
    model = "gru"
    malicious = [0]
    local_epochs = 1  # KEY CHANGE

    configs = [
        ("iid", "fedavg", 5.0),
        ("iid", "fedprox", 5.0),
        ("iid", "krum", 10.0),
        ("iid", "trimmed_mean", 10.0),
        ("noniid", "fedavg", 5.0),
        ("noniid", "fedprox", 5.0),
        ("noniid", "krum", 10.0),
        ("noniid", "trimmed_mean", 10.0),
    ]

    experiments = []
    for dist, strat, scale in configs:
        experiments.append(ExperimentConfig(
            model_type=model, num_clients=3, num_rounds=10,
            local_epochs=local_epochs, batch_size=32, distribution=dist,
            strategy=strat, attack_type="label_flip", attack_fraction=1.0,
            malicious_clients=malicious, scale_factor=scale,
        ))

    return _run_security_suite(experiments, "Sensitivity Analysis: local_epochs=1", base_dir_name)


# =========================================================================
# Helpers
# =========================================================================

def _print_final_summary(fl_results, centralized_results, total_fl, total_cent, output_dir):
    """Print final experiment summary."""
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Experiments completed: {len(fl_results)}/{total_fl}")
    print(f"  Centralized baselines: {len(centralized_results)}/{total_cent}")
    print(f"  Results saved in: {output_dir}")

    if fl_results:
        print("\n  Best FL results:")
        best = max(fl_results, key=lambda r: r.get("final_metrics", {}).get("f1_macro", 0))
        fm = best.get("final_metrics", {})
        print(f"    {best['config_name']}: "
              f"Acc={fm.get('accuracy', 0):.4f}, F1={fm.get('f1_macro', 0):.4f}")

    print("=" * 70)
