#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITS FL Framework - Federated Learning Simulation Orchestrator
==============================================================

Implements FedAvg, FedProx, Krum, and Trimmed Mean aggregation strategies
using a manual simulation loop (no Ray/Flower dependency for simulation),
with integrated support for poisoning attacks (label-flip + gradient scaling).

Aggregation Strategies:
- FedAvg: weighted average of model parameters proportional to dataset size
- FedProx: FedAvg + proximal term to reduce client drift (mu=0.1)
- Krum: Byzantine-robust aggregation selecting the update closest to others
- Trimmed Mean: coordinate-wise trimmed average discarding extremes
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix as sk_confusion_matrix,
    f1_score, precision_score, recall_score,
)

from .fl_config import ExperimentConfig
from .fl_data import split_client_data
from .fl_models import create_model
from .fl_security import apply_gradient_scaling, apply_label_flip_attack


# =========================================================================
# Aggregation strategies
# =========================================================================

def _fedavg_aggregate(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
) -> List[np.ndarray]:
    """FedAvg: weighted average of model parameters proportional to dataset size."""
    total = sum(client_sizes)
    avg_weights = []
    for layer_idx in range(len(client_weights[0])):
        layer_sum = np.zeros_like(client_weights[0][layer_idx])
        for c_idx, weights in enumerate(client_weights):
            layer_sum += weights[layer_idx] * (client_sizes[c_idx] / total)
        avg_weights.append(layer_sum)
    return avg_weights


def _krum_aggregate(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
    num_malicious: int = 0,
) -> List[np.ndarray]:
    """
    Krum aggregation (Blanchard et al., 2017).

    Selects the client update that is closest to the other updates
    (minimum sum of distances to nearest neighbors).

    With num_malicious=0 and n=3 clients, uses (n-0-2)=1 nearest neighbor
    per client, giving proper distance-based scores without privileged
    knowledge of the adversary.

    Parameters
    ----------
    client_weights : list of list of np.ndarray
        Model weights from each client.
    client_sizes : list of int
        Number of training samples per client (unused by Krum).
    num_malicious : int
        Assumed number of malicious clients (0 = no privileged info).

    Returns
    -------
    list of np.ndarray
        Selected client's weights (single best update).
    """
    n = len(client_weights)
    # num_to_keep = n - num_malicious - 2 (at least 1)
    k = max(1, n - num_malicious - 2)

    # Flatten each client's weights into a single vector
    flat_weights = []
    for cw in client_weights:
        flat_weights.append(np.concatenate([w.flatten() for w in cw]))

    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(flat_weights[i] - flat_weights[j])
            distances[i][j] = d
            distances[j][i] = d

    # For each client, compute sum of distances to k nearest neighbors
    scores = []
    for i in range(n):
        sorted_dists = np.sort(distances[i])  # first element is 0 (self)
        # Sum the k smallest non-self distances
        score = np.sum(sorted_dists[1:k + 1])
        scores.append(score)

    # Select the client with smallest score
    selected = int(np.argmin(scores))
    return client_weights[selected]


def _trimmed_mean_aggregate(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
    beta: float = 0.34,
) -> List[np.ndarray]:
    """
    Trimmed Mean aggregation (Yin et al., 2018).

    For each parameter coordinate, sorts the client values and trims
    the top and bottom beta fraction, then averages the remaining.

    With beta=0.34 and 3 clients: floor(0.34*3) = 1 trim per tail,
    leaving exactly the median value (strongest defense for n=3).

    Parameters
    ----------
    client_weights : list of list of np.ndarray
        Model weights from each client.
    client_sizes : list of int
        Number of training samples per client (unused).
    beta : float
        Fraction to trim from each tail.

    Returns
    -------
    list of np.ndarray
        Trimmed mean of client weights.
    """
    n = len(client_weights)
    trim_count = int(np.floor(beta * n))

    agg_weights = []
    for layer_idx in range(len(client_weights[0])):
        # Stack all client weights for this layer
        stacked = np.stack([cw[layer_idx] for cw in client_weights], axis=0)
        # Sort along the client axis
        sorted_vals = np.sort(stacked, axis=0)
        # Trim top and bottom
        if trim_count > 0 and 2 * trim_count < n:
            trimmed = sorted_vals[trim_count:n - trim_count]
        else:
            trimmed = sorted_vals
        # Average remaining
        agg_weights.append(np.mean(trimmed, axis=0))

    return agg_weights


def _select_aggregation(strategy_name: str):
    """Return the aggregation function for the given strategy."""
    strategies = {
        "fedavg": _fedavg_aggregate,
        "fedprox": _fedavg_aggregate,  # FedProx uses FedAvg aggregation + proximal term
        "krum": _krum_aggregate,
        "trimmed_mean": _trimmed_mean_aggregate,
    }
    key = strategy_name.lower()
    if key not in strategies:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Choose from: {list(strategies.keys())}")
    return strategies[key]


# =========================================================================
# FL Simulation
# =========================================================================

def run_fl_simulation(
    config: ExperimentConfig,
    client_partitions: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weights: Dict[int, float],
    seed: int = 42,
    log_dir: Optional[str] = None,
) -> Dict:
    """
    Run a single federated learning experiment using manual simulation.

    Supports all four aggregation strategies (FedAvg, FedProx, Krum,
    Trimmed Mean) and security features (label-flip + gradient scaling).

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    client_partitions : list
        List of (X, y) tuples, one per client.
    X_test : np.ndarray
        Global test features.
    y_test : np.ndarray
        Global test labels.
    class_weights : dict
        Class weights for imbalanced learning.
    seed : int
        Random seed.
    log_dir : str, optional
        Directory for per-client/per-round JSON logs.

    Returns
    -------
    dict
        Experiment results including per-round metrics and final evaluation.
    """
    config_name = config.to_string()
    print(f"\n  Running FL simulation: {config_name}")
    print(f"    Rounds={config.num_rounds}, LocalEpochs={config.local_epochs}, "
          f"Clients={len(client_partitions)}, Strategy={config.strategy}")

    num_clients = len(client_partitions)
    input_dim = client_partitions[0][0].shape[1]
    use_proximal = config.strategy.lower() == "fedprox"
    aggregate_fn = _select_aggregation(config.strategy)

    # Pre-split each client's data into train/val and apply attacks
    client_data = []
    attack_info = []
    for i, (X_c, y_c) in enumerate(client_partitions):
        X_train, X_val, y_train, y_val = split_client_data(X_c, y_c, test_size=0.2, seed=seed + i)

        # Apply label-flip attack on malicious clients
        if config.attack_type == 'label_flip' and i in config.malicious_clients:
            high_before = int(np.sum(y_train == 2))
            y_train, num_flipped, flip_indices = apply_label_flip_attack(
                y_train, config.attack_fraction,
                source_class=2, target_class=0,
                seed=seed + i,
            )
            print(f"    ATTACK: Client {i} flipped {num_flipped}/{high_before} "
                  f"High->Low labels ({config.attack_fraction*100:.0f}%)")

            atk = {
                "client_id": i, "source_class": 2, "target_class": 0,
                "fraction": config.attack_fraction,
                "total_source_samples": high_before,
                "num_flipped": num_flipped,
                "flip_indices": flip_indices.tolist(),
            }
            attack_info.append(atk)

            if log_dir:
                atk_path = os.path.join(log_dir, f"{config_name}_client_{i}_attack.json")
                with open(atk_path, 'w') as f:
                    json.dump(atk, f)

        client_data.append((X_train, y_train, X_val, y_val))

    # Initialize global model
    global_model = create_model(config.model_type, input_dim)
    global_weights = global_model.get_weights()

    # Prepare test input (reshape for LSTM/GRU)
    X_test_input = X_test
    if config.model_type.upper() in ("LSTM", "GRU"):
        X_test_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Per-round metrics storage
    round_metrics: List[Dict] = []
    start_time = time.time()

    for rnd in range(1, config.num_rounds + 1):
        client_updated_weights = []
        client_sizes = []

        # --- Local training on each client ---
        for c_idx in range(num_clients):
            X_train, y_train, X_val, y_val = client_data[c_idx]

            # Create fresh client model and load global weights
            client_model = create_model(config.model_type, input_dim)
            client_model.set_weights(global_weights)

            # Prepare input shapes
            X_tr = X_train
            if config.model_type.upper() in ("LSTM", "GRU"):
                X_tr = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

            if use_proximal:
                # FedProx: train epoch-by-epoch with proximal correction
                saved_global = [w.copy() for w in global_weights]
                for _ in range(config.local_epochs):
                    client_model.fit(
                        X_tr, y_train, epochs=1,
                        batch_size=config.batch_size,
                        class_weight=class_weights, verbose=0,
                    )
                    new_w = client_model.get_weights()
                    prox_w = [
                        w - config.fedprox_mu * (w - gw)
                        for w, gw in zip(new_w, saved_global)
                    ]
                    client_model.set_weights(prox_w)
            else:
                # Standard local training
                client_model.fit(
                    X_tr, y_train, epochs=config.local_epochs,
                    batch_size=config.batch_size,
                    class_weight=class_weights, verbose=0,
                )

            trained_weights = client_model.get_weights()

            # Apply gradient scaling for malicious clients
            is_malicious = c_idx in config.malicious_clients
            if is_malicious and config.scale_factor > 1.0:
                boosted, orig_norm, boost_norm = apply_gradient_scaling(
                    trained_weights, global_weights, config.scale_factor
                )
                trained_weights = boosted

                print(f"    BOOST: Client {c_idx} round {rnd} "
                      f"scale={config.scale_factor}x, "
                      f"L2 norm: {orig_norm:.4f} -> {boost_norm:.4f}")

                if log_dir:
                    norm_path = os.path.join(
                        log_dir, f"{config_name}_client_{c_idx}_round_{rnd}_boost.json"
                    )
                    with open(norm_path, 'w') as f:
                        json.dump({
                            "round": rnd, "client_id": c_idx,
                            "scale_factor": config.scale_factor,
                            "original_l2_norm": orig_norm,
                            "boosted_l2_norm": boost_norm,
                        }, f)

            client_updated_weights.append(trained_weights)
            client_sizes.append(len(X_train))

            # Save per-client fit metrics
            if log_dir:
                loss_tr, acc_tr = client_model.evaluate(X_tr, y_train, verbose=0)
                fit_path = os.path.join(log_dir, f"{config_name}_client_{c_idx}_round_{rnd}_fit.json")
                with open(fit_path, 'w') as f:
                    json.dump({
                        "round": rnd, "client_id": c_idx,
                        "train_loss": float(loss_tr), "train_accuracy": float(acc_tr),
                    }, f)

            del client_model

        # --- Server aggregation ---
        global_weights = aggregate_fn(client_updated_weights, client_sizes)
        global_model.set_weights(global_weights)

        # --- Server-side evaluation on global test set ---
        loss, accuracy = global_model.evaluate(X_test_input, y_test, verbose=0)
        preds = np.argmax(global_model.predict(X_test_input, verbose=0), axis=1)

        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_per_class = f1_score(y_test, preds, average=None, zero_division=0)

        # Confusion matrix and security metric
        cm = sk_confusion_matrix(y_test, preds, labels=[0, 1, 2])
        total_high = int(cm[2].sum()) if cm.shape[0] > 2 else 0
        high_to_low = int(cm[2][0]) if total_high > 0 else 0
        high_to_low_rate = high_to_low / total_high if total_high > 0 else 0.0

        round_metrics.append({
            "round": rnd,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "f1_per_class": [float(v) for v in f1_per_class],
            "confusion_matrix": cm.tolist(),
            "high_to_low_rate": float(high_to_low_rate),
            "high_to_low_count": high_to_low,
            "total_high": total_high,
        })

        # Save per-round eval metrics for all clients (global model evaluation)
        if log_dir:
            for c_idx in range(num_clients):
                eval_path = os.path.join(
                    log_dir, f"{config_name}_client_{c_idx}_round_{rnd}_eval.json"
                )
                with open(eval_path, 'w') as f:
                    json.dump({
                        "round": rnd, "client_id": c_idx,
                        "loss": float(loss), "accuracy": float(accuracy),
                        "precision": float(prec), "recall": float(rec),
                        "f1": float(f1), "dataset_size": len(y_test),
                        "confusion_matrix": cm.tolist(),
                        "high_to_low_count": high_to_low,
                        "total_high": total_high,
                        "high_to_low_rate": float(high_to_low_rate),
                    }, f)

        if rnd % 5 == 0 or rnd == config.num_rounds or rnd == 1:
            print(f"    Round {rnd:3d}/{config.num_rounds}: "
                  f"Acc={accuracy:.4f}, F1={f1:.4f}, Loss={loss:.4f}"
                  + (f", H2L={high_to_low_rate:.4f}" if config.attack_type != 'none' else ""))

    total_time = time.time() - start_time
    print(f"    Simulation completed in {total_time:.1f}s")

    # Build per-round arrays
    round_accuracies = [{"round": m["round"], "accuracy": m["accuracy"]} for m in round_metrics]
    round_f1s = [{"round": m["round"], "f1_macro": m["f1_macro"]} for m in round_metrics]
    round_losses = [{"round": m["round"], "loss": m["loss"]} for m in round_metrics]

    # Final metrics from last round
    last = round_metrics[-1] if round_metrics else {}
    final_metrics = {
        "accuracy": last.get("accuracy", 0),
        "precision": last.get("precision", 0),
        "recall": last.get("recall", 0),
        "f1_macro": last.get("f1_macro", 0),
        "f1_per_class": last.get("f1_per_class", []),
    }
    print(f"    Final test: Acc={final_metrics['accuracy']:.4f}, "
          f"F1={final_metrics['f1_macro']:.4f}, "
          f"Prec={final_metrics['precision']:.4f}, "
          f"Rec={final_metrics['recall']:.4f}")

    # Convergence metrics
    acc_values = [m["accuracy"] for m in round_metrics]
    conv_85 = _find_convergence_round(acc_values, 0.85)
    conv_90 = _find_convergence_round(acc_values, 0.90)
    stability = float(np.std(acc_values[-5:])) if len(acc_values) >= 5 else (
        float(np.std(acc_values)) if acc_values else 0.0
    )

    # Final confusion matrix
    final_cm = last.get("confusion_matrix", [[0]*3]*3)

    return {
        "config_name": config_name,
        "config": config,
        "model_type": config.model_type,
        "strategy": config.strategy,
        "distribution": config.distribution,
        "num_clients": num_clients,
        "num_rounds": config.num_rounds,
        "local_epochs": config.local_epochs,
        "batch_size": config.batch_size,
        "per_round": {
            "accuracy": round_accuracies,
            "f1_macro": round_f1s,
            "loss": round_losses,
        },
        "round_metrics": round_metrics,
        "final_metrics": final_metrics,
        "convergence": {
            "round_85": conv_85,
            "round_90": conv_90,
            "stability_std": stability,
        },
        "confusion_matrix": final_cm,
        "attack_info": attack_info,
        "total_time": total_time,
    }


def _find_convergence_round(
    values: List[float], threshold: float
) -> Optional[int]:
    """Find the first round where metric >= threshold."""
    for i, v in enumerate(values):
        if v >= threshold:
            return i + 1
    return None
