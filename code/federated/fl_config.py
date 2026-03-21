#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Configuration
==================================================

Centralized configuration constants for all FL experiments.
Extends the AIMS FL module with security experiment support
(label-flip attacks, gradient scaling, Byzantine-robust defenses).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class FLDefaults:
    """Default configuration values for Federated Learning experiments."""

    # FL topology
    NUM_CLIENTS: int = 3
    NUM_ROUNDS: int = 30
    LOCAL_EPOCHS: int = 3
    BATCH_SIZE: int = 32

    # Aggregation strategies
    STRATEGIES: List[str] = field(default_factory=lambda: ["FedAvg", "FedProx"])
    FEDPROX_MU: float = 0.1

    # Data distribution modes
    DISTRIBUTIONS: List[str] = field(default_factory=lambda: ["IID", "NonIID"])

    # Neural network models
    MODEL_TYPES: List[str] = field(default_factory=lambda: ["DNN", "LSTM", "GRU"])

    # DNN architecture
    DNN_HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [128, 64, 32])
    DROPOUT_RATE: float = 0.3
    L2_REG: float = 0.001

    # RNN units (LSTM / GRU)
    RNN_UNITS_1: int = 64
    RNN_UNITS_2: int = 32

    # Classification (4 classes for AIMS impact)
    NUM_CLASSES: int = 4
    CLASS_NAMES: List[str] = field(
        default_factory=lambda: ["Adequate", "Warning", "Severe", "Critical"]
    )

    # Centralized baseline
    CENTRALIZED_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 10

    # Reproducibility
    RANDOM_STATE: int = 42

    # Results sub-directory
    RESULTS_SUBDIR: str = "federated"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model_type: str
    num_clients: int
    num_rounds: int
    local_epochs: int
    batch_size: int
    distribution: str  # 'iid' or 'noniid'
    strategy: str  # 'fedavg', 'fedprox', 'krum', or 'trimmed_mean'
    fedprox_mu: float = 0.1
    attack_type: str = 'none'  # 'none' or 'label_flip'
    attack_fraction: float = 0.0
    malicious_clients: List[int] = field(default_factory=list)
    scale_factor: float = 1.0  # gradient scaling factor (1.0 = no scaling)

    def to_string(self):
        base = f"{self.model_type}_{self.distribution}_{self.strategy}_e{self.local_epochs}"
        if self.attack_type != 'none':
            flip_pct = int(self.attack_fraction * 100)
            mal_ids = ''.join(str(c) for c in sorted(self.malicious_clients))
            base += f"_flip{flip_pct}_m{mal_ids}"
        if self.scale_factor > 1.0:
            base += f"_scale{int(self.scale_factor)}x"
        return base


# Non-IID allocation matrix: rows = clients, cols = classes (0,1,2,3)
# Each column sums to 1.0 -> all data used, no loss.
NON_IID_ALLOCATION = np.array([
    [0.55, 0.30, 0.10, 0.05],   # Client 0: primarily Adequate + Warning
    [0.15, 0.40, 0.40, 0.15],   # Client 1: primarily Warning + Severe
    [0.30, 0.30, 0.50, 0.80],   # Client 2: primarily Severe + Critical
])
# Note: columns sum to [1.0, 1.0, 1.0, 1.0]
