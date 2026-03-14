"""
Enhanced Federated Learning Simulation for ITS Network Slicing Impact Classification

This script implements a federated learning framework to classify the impact level
of network slicing policies on ITS (Intelligent Transportation Systems) applications.
The classification target (impact_level: low/medium/high) indicates how network
conditions affect application requirements based on metrics such as RTT, PDR,
throughput, and derived temporal features.

================================================================================
EXPERIMENTAL CONTEXT
================================================================================

The dataset was collected from a simulation/emulation environment featuring:
- 3 RSUs (Road Side Units) covering a 650m urban corridor
- Up to 17 vehicles simultaneously competing for network resources
- 4 ITS application types: Safety (S), Efficiency (E), Entertainment (E2), Generic (G)
- 3 network slicing strategies: FN (no QoS), FQ (queue-based QoS), FS (full slicing)

The congestion dynamics observed in the experiment show sequential propagation
through the RSU chain (RSU3 → RSU2 → RSU1), which motivates the Non-IID data
distribution strategy based on impact_level.

================================================================================
FEATURES
================================================================================

Data Distribution:
    - IID: Random uniform split across clients (disjoint partitions)
    - Non-IID: Heterogeneous split based on impact_level distribution, reflecting
      the temporal dynamics where different RSUs observe different congestion phases

Aggregation Strategies:
    - FedAvg: Standard federated averaging with weighted aggregation
    - FedProx: Proximal regularization for improved Non-IID robustness (μ=0.1)
    - Krum: Byzantine-robust aggregation (Blanchard et al., 2017)
    - Trimmed Mean: Coordinate-wise trimmed average (Yin et al., 2018)

Models:
    - DNN: Deep Neural Network (Dense layers with dropout)
    - LSTM: Long Short-Term Memory for temporal pattern capture
    - GRU: Gated Recurrent Unit as efficient LSTM alternative

Metrics:
    - Performance: Accuracy, Precision, Recall, F1-Score (macro)
    - Convergence: Rounds to reach 85%/90% accuracy thresholds
    - Stability: Standard deviation of accuracy over last 5 rounds
    - Efficiency: Total training time

Security (Chapter 10):
    - Label-flipping attack: High(2)->Low(0) directed poisoning
    - Configurable attack fraction (20%, 50%) and malicious client selection
    - Security metrics: H->L misclassification rate, confusion matrix, per-class metrics

Baselines:
    - Centralized training for direct comparison with federated approaches

================================================================================
NON-IID DISTRIBUTION RATIONALE
================================================================================

Based on experimental logs, congestion propagates sequentially:
    - RSU 0 (last to congest): Receives overflow, saturated conditions → 55% HIGH
    - RSU 1 (middle point): Receives redirections, mixed conditions → 50% MEDIUM  
    - RSU 2 (entry point): Initial traffic, pre-congestion → 55% LOW

This reflects realistic deployment where RSUs observe different phases of the
network slice lifecycle due to progressive application of slicing policies.

================================================================================
USAGE
================================================================================

Full experiment suite (Chapter 9):
    python federatedLearning_ITS.py

Quick test run (Chapter 9):
    python federatedLearning_ITS.py --quick

Security experiments (Chapter 10 - 24 configs):
    python federatedLearning_ITS.py --security

Security quick test (Chapter 10 - 4 configs):
    python federatedLearning_ITS.py --security-quick

================================================================================
OUTPUT
================================================================================

Results are saved to 'results/<version>/' directory:
    - paper_artifacts/summary_table.csv    : Comparative results table
    - paper_artifacts/summary_table.tex    : LaTeX-formatted table for paper
    - paper_artifacts/centralized_results.json : Baseline centralized results
    - paper_artifacts/convergence_comparison.png : Convergence curves
    - paper_artifacts/strategy_comparison.png : FedAvg vs FedProx on Non-IID
    - paper_artifacts/[config_name].json   : Per-experiment detailed results
    - client_logs/                         : Per-client per-round metrics

================================================================================
EXPERIMENT MATRIX
================================================================================

| #   | Model | Distribution | Strategy | Rounds | Local Epochs |
|-----|-------|--------------|----------|--------|--------------|
| 1-4 | DNN   | IID/Non-IID  | FedAvg/FedProx | 10 | 5 |
| 5-8 | LSTM  | IID/Non-IID  | FedAvg/FedProx | 10 | 5 |
| 9-12| GRU   | IID/Non-IID  | FedAvg/FedProx | 10 | 5 |
| 13  | DNN   | Centralized  | -        | -  | 30 |
| 14  | LSTM  | Centralized  | -        | -  | 30 |
| 15  | GRU   | Centralized  | -        | -  | 30 |

Total: 15 experiments (12 federated + 3 centralized baselines)

================================================================================
DEPENDENCIES
================================================================================

- flwr (Flower): Federated learning framework
- tensorflow/keras: Neural network models
- pandas, numpy: Data processing
- scikit-learn: Metrics and preprocessing
- matplotlib: Visualization

================================================================================
CHANGELOG
================================================================================

Version 1.0 (Production):
- All data cardinality issues resolved (parameter order fix)
- train_time properly saved in JSON and aggregated in summary
- FedProx using fl.server.strategy.FedProx with proximal_mu
- Non-IID uses 100% data with disjoint partitions by class
- GPU/CPU auto-configuration with memory growth
- Validated with quick test (IID: 97.65%, Non-IID FedProx: 78.92%)

================================================================================
"""


import json
import multiprocessing
import os
import time
from typing import List, Optional
from dataclasses import dataclass, asdict, field

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, GRU, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix as sk_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.callbacks import EarlyStopping

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# =============================================================================
# Paths
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "raw_full.csv")

# =============================================================================
# Configuration
# =============================================================================

def set_global_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def configure_system_resources():
    """
    Automatically configures CPU or GPU and prevents OOM errors
    in multiprocessing scenarios (Flower).
    """
    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth: use only what is needed, not all at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # (Optional) If multiple GPUs are available, visibility can be configured here
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✅ GPU Detected! {len(gpus)} Physical, {len(logical_gpus)} Logical")
            print("   Memory Growth enabled to support multiple clients.")
        except RuntimeError as e:
            # Memory must be set before GPUs are initialized
            print(f"⚠️ Error configuring GPU: {e}")
    else:
        print("ℹ️ No GPU detected. Running on CPU (default).")


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
    attack_fraction: float = 0.0  # fraction of source_class labels to flip
    malicious_clients: List[int] = field(default_factory=list)
    scale_factor: float = 1.0  # gradient scaling factor for model poisoning (1.0 = no scaling)

    def to_string(self):
        base = f"{self.model_type}_{self.distribution}_{self.strategy}_e{self.local_epochs}"
        if self.attack_type != 'none':
            flip_pct = int(self.attack_fraction * 100)
            mal_ids = ''.join(str(c) for c in sorted(self.malicious_clients))
            base += f"_flip{flip_pct}_m{mal_ids}"
        if self.scale_factor > 1.0:
            base += f"_scale{int(self.scale_factor)}x"
        return base


@dataclass
class ExperimentResults:
    """Results from a single experiment."""
    config: ExperimentConfig
    rounds: List[int]
    accuracy: List[float]
    precision: List[float]
    recall: List[float]
    f1: List[float]
    train_time_per_round: List[float]
    convergence_round_90: Optional[int]
    convergence_round_85: Optional[int]
    final_accuracy: float
    final_f1: float
    stability_std: float
    total_time: float


# =============================================================================
# Data Distribution
# =============================================================================

def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize impact_level to integers 0, 1, 2."""
    df = df.copy()
    if df['impact_level'].dtype == object:
        df['impact_level'] = df['impact_level'].map({'low': 0, 'medium': 1, 'high': 2})
    df['impact_level'] = df['impact_level'].astype(int)
    return df


def load_and_preprocess_data_iid(client_id: int, num_clients: int, data_path: str = DATA_PATH):
    """Load data with IID distribution (random disjoint split)."""
    print(f"Loading IID data for RSU {client_id}...")
    df = pd.read_csv(data_path)
    df = _normalize_labels(df)
    
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    chunk = len(shuffled) // num_clients
    start = client_id * chunk
    end = start + chunk if client_id < num_clients - 1 else len(shuffled)
    client_data = shuffled.iloc[start:end].copy()
    
    print(f"  RSU {client_id} IID: {len(client_data)} samples")
    return _prepare_features_labels(client_data, client_id)


def load_and_preprocess_data_noniid(client_id: int, num_clients: int, data_path: str = DATA_PATH):
    """
    Load data with Non-IID distribution based on impact_level.
    Uses ALL data with no loss.
    """
    print(f"Loading Non-IID data for RSU {client_id}...")
    df = pd.read_csv(data_path)
    df = _normalize_labels(df)
    
    counts = df['impact_level'].value_counts().sort_index()
    print(f"  Dataset: {len(df)} samples (low={counts.get(0,0)}, med={counts.get(1,0)}, high={counts.get(2,0)})")
    
    # Allocation: columns sum to 1.0 → all data used
    allocation = {
        0: [0.15, 0.25, 0.55],  # RSU0: high-impact heavy
        1: [0.30, 0.50, 0.25],  # RSU1: medium-impact heavy
        2: [0.55, 0.25, 0.20],  # RSU2: low-impact heavy
    }
    
    # Shuffle each class
    level_dfs = {
        lvl: df[df['impact_level'] == lvl].sample(frac=1, random_state=42).reset_index(drop=True)
        for lvl in [0, 1, 2]
    }
    
    client_samples = []
    for lvl in [0, 1, 2]:
        level_df = level_dfs[lvl]
        n = len(level_df)
        if n == 0:
            continue
        
        frac = allocation[client_id][lvl]
        start_frac = sum(allocation[c][lvl] for c in range(client_id))
        start_idx = int(start_frac * n)
        end_idx = int((start_frac + frac) * n) if client_id < num_clients - 1 else n
        
        selected = level_df.iloc[start_idx:end_idx].copy()
        if len(selected) > 0:
            client_samples.append(selected)
            lvl_name = {0: 'low', 1: 'med', 2: 'high'}[lvl]
            print(f"  RSU {client_id} {lvl_name}: {len(selected)} ({len(selected)/n*100:.0f}%)")
    
    if not client_samples:
        return load_and_preprocess_data_iid(client_id, num_clients, data_path)
    
    client_data = pd.DataFrame(pd.concat(client_samples, ignore_index=True))
    client_data = client_data.sample(frac=1, random_state=42+client_id).reset_index(drop=True)
    
    print(f"  RSU {client_id} total: {len(client_data)} samples")
    return _prepare_features_labels(client_data, client_id)


def _prepare_features_labels(client_data: pd.DataFrame, client_id: int):
    """
    Prepare features and labels.
    
    IMPORTANT: Returns (X_train, y_train, X_test, y_test) - note the order!
    This matches the ITSRsuClient constructor parameter order.
    """
    drop_cols = ['time', 'impact_level']
    features = client_data.drop(columns=[c for c in drop_cols if c in client_data.columns])
    labels = client_data['impact_level'].astype(int)
    
    # Normalize numeric features
    num_cols = ['rec_serv', 'env_car', 'rtt', 'ncars', 'pdr', 'bc_rtt', 'rtt_change', 
                'pdr_change', 'rtt_mam', 'pdr_mam', 'rtt_masd', 'log_rtt', 'log_pdr',
                'log2_rtt', 'log2_pdr', 'rtt_sqrd', 'pdr_sqrd']
    num_cols = [c for c in num_cols if c in features.columns]
    
    if num_cols:
        scaler = StandardScaler()
        features[num_cols] = scaler.fit_transform(features[num_cols].astype(float))
    
    features = features.astype('float32')
    
    # Split
    min_count = labels.value_counts().min() if len(labels.value_counts()) > 0 else 0
    stratify = labels if min_count >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=stratify
    )
    
    # Convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    
    # Log info
    dist = client_data['impact_level'].value_counts().sort_index()
    print(f"  RSU {client_id}: {len(X_train)} train, {len(X_test)} test | "
          f"Impact: L={dist.get(0,0)} M={dist.get(1,0)} H={dist.get(2,0)}")
    
    # Return in order that matches ITSRsuClient constructor
    # Constructor: (config, X_train, y_train, X_test, y_test, client_id)
    return X_train, y_train, X_test, y_test


# =============================================================================
# Poisoning Attack
# =============================================================================

def apply_label_flip_attack(y_train: np.ndarray, fraction: float,
                            source_class: int = 2, target_class: int = 0,
                            seed: int = 42) -> tuple:
    """
    Label-flipping attack: flip a fraction of source_class labels to target_class.

    In the ITS context, High(2)->Low(0) flipping suppresses lifecycle decisions:
    the model learns to classify degraded slices as adequate.

    Returns (y_modified, num_flipped, flip_indices).
    """
    assert 0.0 < fraction <= 1.0, f"Fraction must be in (0, 1], got {fraction}"
    assert source_class != target_class, \
        f"source_class and target_class must differ, got {source_class} == {target_class}"
    y_modified = y_train.copy()

    source_indices = np.where(y_modified == source_class)[0]
    total_source = len(source_indices)

    if total_source == 0:
        print(f"  WARNING: No samples of class {source_class} to flip")
        return y_modified, 0, np.array([], dtype=int)

    rng = np.random.RandomState(seed)
    num_to_flip = int(np.ceil(fraction * total_source))
    flip_indices = rng.choice(source_indices, size=num_to_flip, replace=False)

    y_modified[flip_indices] = target_class

    # Assertions
    actual_flipped = total_source - np.sum(y_modified == source_class)
    assert actual_flipped == num_to_flip, \
        f"Expected {num_to_flip} flips, got {actual_flipped}"
    assert np.sum(y_modified == target_class) == np.sum(y_train == target_class) + num_to_flip, \
        "Target class count mismatch after flipping"

    return y_modified, num_to_flip, flip_indices


# =============================================================================
# Models
# =============================================================================

def create_model(model_type: str, input_shape: int, num_classes: int = 3):
    """Create neural network model."""
    model = Sequential()
    
    if model_type == "dnn":
        model.add(Input(shape=(input_shape,)))
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
    elif model_type == "lstm":
        model.add(Input(shape=(1, input_shape)))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        
    elif model_type == "gru":
        model.add(Input(shape=(1, input_shape)))
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(GRU(32))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# =============================================================================
# Flower Client
# =============================================================================

class ITSRsuClient(fl.client.NumPyClient):
    """
    Flower client for ITS RSU.
    
    Constructor parameter order is (config, X_train, y_train, X_test, y_test, client_id)
    This must match the order returned by _prepare_features_labels.
    """
    
    def __init__(self, config: ExperimentConfig, X_train, y_train, X_test, y_test, client_id: int, log_dir: str):
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id
        self.log_dir = log_dir
        self.current_round = 0
        self.global_weights = None
        
        # Validate data shapes
        assert len(self.X_train) == len(self.y_train), \
            f"Train data mismatch: X={len(self.X_train)}, y={len(self.y_train)}"
        assert len(self.X_test) == len(self.y_test), \
            f"Test data mismatch: X={len(self.X_test)}, y={len(self.y_test)}"
        
        print(f"  Client {client_id} initialized: X_train={X_train.shape}, y_train={y_train.shape}, "
              f"X_test={X_test.shape}, y_test={y_test.shape}")
        
        self.model = create_model(config.model_type, X_train.shape[1])
        os.makedirs("client_results", exist_ok=True)
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.current_round = config.get("round", self.current_round + 1)
        self.model.set_weights(parameters)

        # Store global weights for delta computation (needed for FedProx and gradient scaling)
        self.global_weights = [w.copy() for w in parameters]

        # Prepare data
        X_train = self.X_train
        y_train = self.y_train

        # Reshape for recurrent models
        if self.config.model_type in ["lstm", "gru"]:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

        # Final validation before training
        assert len(X_train) == len(y_train), \
            f"Training data mismatch in fit(): X={len(X_train)}, y={len(y_train)}"

        start = time.time()

        if self.config.strategy == "fedprox" and self.global_weights:
            self._train_fedprox(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, epochs=self.config.local_epochs,
                          batch_size=self.config.batch_size, verbose=0)

        train_time = time.time() - start

        # Evaluate on training data (pre-boost, reflects actual local training)
        loss, acc = self.model.evaluate(X_train, y_train, verbose=0)

        # Save training metrics (pre-boost)
        self._save_fit_metrics(train_time, loss, acc)

        # Apply gradient scaling for model poisoning (after metrics, before return)
        is_malicious = self.client_id in self.config.malicious_clients
        if is_malicious and self.config.scale_factor > 1.0:
            trained_weights = self.model.get_weights()
            boosted_weights = [
                gw + self.config.scale_factor * (tw - gw)
                for tw, gw in zip(trained_weights, self.global_weights)
            ]
            self.model.set_weights(boosted_weights)

            # Compute and log L2 norms for post-hoc analysis
            original_norm = sum(
                float(np.linalg.norm(tw - gw))
                for tw, gw in zip(trained_weights, self.global_weights)
            )
            boosted_norm = original_norm * self.config.scale_factor
            print(f"  BOOST: Client {self.client_id} round {self.current_round} "
                  f"scale={self.config.scale_factor}x, "
                  f"L2 norm: {original_norm:.4f} -> {boosted_norm:.4f}")

            # Persist norms to JSON for post-hoc analysis
            norm_log = os.path.join(
                self.log_dir,
                f"{self.config.to_string()}_client_{self.client_id}_round_{self.current_round}_boost.json"
            )
            with open(norm_log, 'w') as f:
                json.dump({
                    "round": self.current_round,
                    "client_id": self.client_id,
                    "scale_factor": self.config.scale_factor,
                    "original_l2_norm": original_norm,
                    "boosted_l2_norm": boosted_norm,
                }, f)

        return self.model.get_weights(), len(self.X_train), {
            "loss": float(loss), "accuracy": float(acc), "train_time": float(train_time)
        }
    
    def _train_fedprox(self, X_train, y_train):
        """
        FedProx training with post-epoch proximal updates.
        w ← w - μ(w - w_global) after each local epoch.
        """
        mu = self.config.fedprox_mu
        for _ in range(self.config.local_epochs):
            self.model.fit(X_train, y_train, epochs=1, 
                          batch_size=self.config.batch_size, verbose=0)
            # Proximal update
            new_w = self.model.get_weights()
            prox_w = [w - mu * (w - gw) for w, gw in zip(new_w, self.global_weights)]
            self.model.set_weights(prox_w)
    
    def _save_fit_metrics(self, train_time, loss, acc):
        """Save training metrics to JSON."""
        filename = f"{self.config.to_string()}_client_{self.client_id}_round_{self.current_round}_fit.json"
        filename = os.path.join(self.log_dir, filename)
        with open(filename, 'w') as f:
            json.dump({
                "round": self.current_round,
                "client_id": self.client_id,
                "train_time": float(train_time),
                "train_loss": float(loss),
                "train_accuracy": float(acc)
            }, f)
    
    def evaluate(self, parameters, config):
        self.current_round = config.get("round", self.current_round)
        self.model.set_weights(parameters)

        X_test = self.X_test
        y_test = self.y_test

        if self.config.model_type in ["lstm", "gru"]:
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        preds = np.argmax(self.model.predict(X_test, verbose=0), axis=1)

        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)

        # Confusion matrix and per-class metrics
        cm = sk_confusion_matrix(y_test, preds, labels=[0, 1, 2])
        per_class_prec = precision_score(y_test, preds, average=None, labels=[0, 1, 2], zero_division=0)
        per_class_rec = recall_score(y_test, preds, average=None, labels=[0, 1, 2], zero_division=0)
        per_class_f1 = f1_score(y_test, preds, average=None, labels=[0, 1, 2], zero_division=0)

        # Security metric: High->Low misclassification rate
        total_high = int(cm[2].sum())
        high_to_low = int(cm[2][0]) if total_high > 0 else 0
        high_to_low_rate = high_to_low / total_high if total_high > 0 else 0.0

        # Save eval metrics
        filename = f"{self.config.to_string()}_client_{self.client_id}_round_{self.current_round}_eval.json"
        filename = os.path.join(self.log_dir, filename)
        with open(filename, 'w') as f:
            json.dump({
                "round": self.current_round, "client_id": self.client_id,
                "loss": float(loss), "accuracy": float(acc),
                "precision": float(prec), "recall": float(rec), "f1": float(f1),
                "dataset_size": len(self.X_test),
                "confusion_matrix": cm.tolist(),
                "per_class_precision": per_class_prec.tolist(),
                "per_class_recall": per_class_rec.tolist(),
                "per_class_f1": per_class_f1.tolist(),
                "high_to_low_count": high_to_low,
                "total_high": total_high,
                "high_to_low_rate": float(high_to_low_rate),
            }, f)

        return loss, len(self.X_test), {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
            "high_to_low_rate": float(high_to_low_rate),
        }


# =============================================================================
# Centralized Training
# =============================================================================

def run_centralized_training(model_type: str, num_epochs: int = 30, data_path: str = DATA_PATH):
    """Centralized baseline."""
    print(f"\n{'='*50}\nCENTRALIZED: {model_type.upper()}\n{'='*50}")
    
    df = pd.read_csv(data_path)
    df = _normalize_labels(df)
    
    drop_cols = ['time', 'impact_level']
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    labels = df['impact_level'].astype(int)
    
    num_cols = ['rec_serv', 'env_car', 'rtt', 'ncars', 'pdr', 'bc_rtt', 'rtt_change',
                'pdr_change', 'rtt_mam', 'pdr_mam', 'rtt_masd', 'log_rtt', 'log_pdr',
                'log2_rtt', 'log2_pdr', 'rtt_sqrd', 'pdr_sqrd']
    num_cols = [c for c in num_cols if c in features.columns]
    if num_cols:
        features[num_cols] = StandardScaler().fit_transform(features[num_cols].astype(float))
    features = features.astype('float32')
    
    X_train, X_test, y_train, y_test = train_test_split(
        features.values, labels.values, test_size=0.2, random_state=42, stratify=labels
    )
    
    model = create_model(model_type, X_train.shape[1])
    
    if model_type in ["lstm", "gru"]:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    start = time.time()

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=32,
                       validation_data=(X_test, y_test), callbacks=[es], verbose=1)
    total_time = time.time() - start
    
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    results = {
        "model_type": model_type,
        "epochs_trained": len(history.history['loss']),
        "accuracy": float(history.history['val_accuracy'][-1]),
        "precision": float(precision_score(y_test, preds, average='macro', zero_division=0)),
        "recall": float(recall_score(y_test, preds, average='macro', zero_division=0)),
        "f1": float(f1_score(y_test, preds, average='macro', zero_division=0)),
        "total_time": total_time
    }
    
    print(f"Result: Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}, Time={total_time:.1f}s")
    return results


# =============================================================================
# Server/Client Processes
# =============================================================================

def weighted_average(metrics):
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {k: sum(m[k] * n for n, m in metrics) / total for k in metrics[0][1].keys()}


def start_client(config: ExperimentConfig, client_id: int, log_dir: str):
    set_global_seeds(42 + client_id)
    configure_system_resources()
    try:
        if config.distribution == "iid":
            X_train, y_train, X_test, y_test = load_and_preprocess_data_iid(client_id, config.num_clients)
        else:
            X_train, y_train, X_test, y_test = load_and_preprocess_data_noniid(client_id, config.num_clients)
        
        # Validate before creating client
        assert len(X_train) == len(y_train), f"Client {client_id}: Train mismatch X={len(X_train)}, y={len(y_train)}"
        assert len(X_test) == len(y_test), f"Client {client_id}: Test mismatch X={len(X_test)}, y={len(y_test)}"

        # Apply poisoning attack if this client is malicious
        if config.attack_type == 'label_flip' and client_id in config.malicious_clients:
            high_before = int(np.sum(y_train == 2))
            y_train, num_flipped, flip_indices = apply_label_flip_attack(
                y_train, config.attack_fraction,
                source_class=2, target_class=0,
                seed=42 + client_id
            )
            print(f"  ATTACK: Client {client_id} flipped {num_flipped}/{high_before} "
                  f"High->Low labels ({config.attack_fraction*100:.0f}%)")

            # Log attack details for post-hoc analysis
            attack_log = os.path.join(log_dir, f"{config.to_string()}_client_{client_id}_attack.json")
            with open(attack_log, 'w') as f:
                json.dump({
                    "client_id": client_id,
                    "source_class": 2, "target_class": 0,
                    "fraction": config.attack_fraction,
                    "total_source_samples": high_before,
                    "num_flipped": num_flipped,
                    "flip_indices": flip_indices.tolist(),
                }, f)

        client = ITSRsuClient(config, X_train, y_train, X_test, y_test, client_id, log_dir)
        fl.client.start_numpy_client(server_address="127.0.0.1:8085", client=client)
    except Exception as e:
        print(f"Client {client_id} error: {e}")
        import traceback; traceback.print_exc()


def start_server(config: ExperimentConfig):
    set_global_seeds(42)
    # Configure GPU for server proccess (if available)
    configure_system_resources()

    print(f"Server: {config.to_string()} ({config.strategy.upper()})")
    
    if config.strategy == "fedprox":
        strategy = fl.server.strategy.FedProx(
            min_fit_clients=config.num_clients,
            min_available_clients=config.num_clients,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=config.fedprox_mu,
        )
    elif config.strategy == "krum":
        strategy = fl.server.strategy.Krum(
            min_fit_clients=config.num_clients,
            min_available_clients=config.num_clients,
            num_malicious_clients=len(config.malicious_clients),
            num_clients_to_keep=0,  # classical Krum (select single best)
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    elif config.strategy == "trimmed_mean":
        strategy = fl.server.strategy.FedTrimmedAvg(
            min_fit_clients=config.num_clients,
            min_available_clients=config.num_clients,
            beta=0.34,  # floor(0.34*3)=1: trims 1 value per tail -> coordinate-wise median
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            min_fit_clients=config.num_clients,
            min_available_clients=config.num_clients,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
    fl.server.start_server(
        server_address="0.0.0.0:8085",
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy
    )


# =============================================================================
# Experiment Runner
# =============================================================================

def run_federated_experiment(config: ExperimentConfig, log_dir: str) -> Optional[ExperimentResults]:
    print(f"\n{'='*60}\nExperiment: {config.to_string()}\n{'='*60}\n")
    
    start_total = time.time()
    
    try:
        server = multiprocessing.Process(target=start_server, args=(config,))
        server.start()
        time.sleep(3)
        
        clients = []
        for i in range(config.num_clients):
            p = multiprocessing.Process(target=start_client, args=(config, i, log_dir))
            clients.append(p)
            p.start()
            time.sleep(0.5)
        
        for p in clients:
            p.join(timeout=600)
        server.join(timeout=60)
        
        for p in clients + [server]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        
        return aggregate_results(config, time.time() - start_total, log_dir)
    except Exception as e:
        print(f"Experiment error: {e}")
        return None


def aggregate_results(config: ExperimentConfig, total_time: float, log_dir: str) -> ExperimentResults:
    """Aggregate results from client JSON files."""
    rounds, accuracy, precision, recall, f1, train_times = [], [], [], [], [], []
    
    for r in range(1, config.num_rounds + 1):
        eval_metrics = []
        fit_metrics = []
        
        for c in range(config.num_clients):
            # Read eval metrics
            eval_file = os.path.join(log_dir, f"{config.to_string()}_client_{c}_round_{r}_eval.json")
            if os.path.exists(eval_file):
                with open(eval_file) as f:
                    eval_metrics.append(json.load(f))
            
            # Read fit metrics (train_time)
            fit_file = os.path.join(log_dir, f"{config.to_string()}_client_{c}_round_{r}_fit.json")
            if os.path.exists(fit_file):
                with open(fit_file) as f:
                    fit_metrics.append(json.load(f))
        
        if eval_metrics:
            total_size = sum(m.get("dataset_size", 1) for m in eval_metrics)
            rounds.append(r)
            
            for name, lst in [("accuracy", accuracy), ("precision", precision),
                              ("recall", recall), ("f1", f1)]:
                weighted = sum(m.get(name, 0) * m.get("dataset_size", 1) for m in eval_metrics)
                lst.append(weighted / total_size if total_size > 0 else 0)
            
            # Aggregate train time (average across clients)
            if fit_metrics:
                avg_time = np.mean([m.get("train_time", 0) for m in fit_metrics])
                train_times.append(avg_time)
            else:
                train_times.append(0)
    
    conv_90 = next((r for r, a in zip(rounds, accuracy) if a >= 0.90), None)
    conv_85 = next((r for r, a in zip(rounds, accuracy) if a >= 0.85), None)
    stability = float(np.std(accuracy[-5:])) if len(accuracy) >= 5 else float(np.std(accuracy)) if accuracy else 0.0
    
    return ExperimentResults(
        config=config, rounds=rounds, accuracy=accuracy, precision=precision,
        recall=recall, f1=f1, train_time_per_round=train_times,
        convergence_round_90=conv_90, convergence_round_85=conv_85,
        final_accuracy=accuracy[-1] if accuracy else 0.0,
        final_f1=f1[-1] if f1 else 0.0,
        stability_std=stability, total_time=total_time
    )


# =============================================================================
# Visualization
# =============================================================================

def generate_summary_table(results: List[ExperimentResults], centralized: dict) -> pd.DataFrame:
    data = []
    for r in results:
        if r:
            avg_train = np.mean(r.train_time_per_round) if r.train_time_per_round else 0
            data.append({
                "Model": r.config.model_type.upper(),
                "Dist": r.config.distribution.upper(),
                "Strategy": r.config.strategy.upper(),
                "Acc": f"{r.final_accuracy:.4f}",
                "F1": f"{r.final_f1:.4f}",
                "@85%": r.convergence_round_85 or "-",
                "@90%": r.convergence_round_90 or "-",
                "σ": f"{r.stability_std:.4f}",
                "Avg Train": f"{avg_train:.2f}s",
                "Total": f"{r.total_time:.1f}s"
            })
    
    for m, res in centralized.items():
        data.append({
            "Model": m.upper(), "Dist": "CENTRAL", "Strategy": "-",
            "Acc": f"{res['accuracy']:.4f}", "F1": f"{res['f1']:.4f}",
            "@85%": "-", "@90%": "-", "σ": "-", "Avg Train": "-",
            "Total": f"{res['total_time']:.1f}s"
        })
    
    return pd.DataFrame(data)


def plot_convergence(results: List[ExperimentResults], output_dir: str = "."):
    valid = [r for r in results if r and r.accuracy]
    if not valid:
        print("No valid results to plot convergence")
        return
    
    models = sorted(set(r.config.model_type for r in valid))
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        for r in [x for x in valid if x.config.model_type == model]:
            label = f"{r.config.distribution}-{r.config.strategy}"
            ax.plot(r.rounds, r.accuracy, marker='o', label=label, markersize=4)
        
        ax.set_title(f'{model.upper()}')
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=0.90, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/convergence_comparison.png")


def plot_strategy_comparison(results: List[ExperimentResults], output_dir: str = "."):
    """Compare FedAvg vs FedProx on Non-IID data."""
    noniid = [r for r in results if r and r.config.distribution == 'noniid' and r.accuracy]
    if len(noniid) < 2:
        print("Not enough Non-IID results for strategy comparison")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for r in noniid:
        label = f"{r.config.model_type.upper()}-{r.config.strategy.upper()}"
        ax.plot(r.rounds, r.accuracy, marker='o', label=label, markersize=4)
    
    ax.set_title('FedAvg vs FedProx on Non-IID Data')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='85%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/strategy_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/strategy_comparison.png")


# =============================================================================
# Main Functions
# =============================================================================

def get_project_paths():
    """
    Helper to get absolute paths independent of execution location.
    Assumes structure:
    /project_root
      /code/script.py
      /results
      /data
    """
    # Absolute path to the script (.py)
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path) # .../code
    project_root = os.path.dirname(script_dir) # .../its_slicing_federated_learning
    
    # Define the results directory (sibling of 'code')
    results_root = os.path.join(project_root, "results")
    
    return results_root

def run_full_suite(base_dir_name="full_suite_results"):
    """Full experiment suite (~1.5-2h)."""
    # Get absolute path to the results folder
    results_root = get_project_paths()
    base_dir = os.path.join(results_root, base_dir_name)

    print("="*70)
    print("ITS Federated Learning - Full Suite (v7)")
    print(f"Output Directory: {base_dir}")
    print("="*70)
    
    client_results_dir = os.path.join(base_dir, "client_logs")  # Ex: ../results/client_logs
    paper_results_dir = os.path.join(base_dir, "paper_artifacts") # Ex: ../results/paper_artifacts

    os.makedirs(client_results_dir, exist_ok=True)
    os.makedirs(paper_results_dir, exist_ok=True)
    
    models = ["dnn", "lstm", "gru"]
    distributions = ["iid", "noniid"]
    strategies = ["fedavg", "fedprox"]
    
    experiments = [
        ExperimentConfig(m, 3, 10, 5, 32, d, s, 0.1)
        for m in models for d in distributions for s in strategies
    ]
    
    # Federated
    print("\n" + "="*50 + "\nFederated Experiments\n" + "="*50)
    all_results = []
    for cfg in experiments:
        result = run_federated_experiment(cfg, log_dir=client_results_dir)
        if result:
            all_results.append(result)
            file_path = os.path.join(paper_results_dir, f"{cfg.to_string()}.json")
            with open(file_path, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)

    # Centralized
    print("\n" + "="*50 + "\nCentralized Baselines\n" + "="*50)
    centralized = {m: run_centralized_training(m, 30) for m in models}
    
    centralized_path = os.path.join(paper_results_dir, "centralized_results.json")
    with open(centralized_path, 'w') as f:
        json.dump(centralized, f, indent=2)
    
    # Generate outputs
    summary = generate_summary_table(all_results, centralized)
    summary.to_csv(os.path.join(paper_results_dir, "summary_table.csv"), index=False)
    
    print("\n" + "="*50 + "\nSummary\n" + "="*50)
    print(summary.to_string(index=False))
    
    tex_path = os.path.join(paper_results_dir, "summary_table.tex")
    with open(tex_path, 'w') as f:
        f.write(summary.to_latex(index=False, escape=False))
    
    plot_convergence(all_results, paper_results_dir)
    plot_strategy_comparison(all_results, paper_results_dir)
    
    print("\n" + "="*70)
    print(f"Complete! Results saved in: {paper_results_dir}")
    print("="*70)
    
    return all_results, centralized

def run_quick_test(base_dir_name = "quick_test_results"):
    """Quick test (~20-25 min)."""
    # Get absolute path to the results folder
    results_root = get_project_paths()
    base_dir = os.path.join(results_root, base_dir_name)
    print("="*70)
    print("ITS Federated Learning - Quick Test")
    print(f"Output Directory: {base_dir}")
    print("="*70)
    
    client_results_dir = os.path.join(base_dir, "client_logs")
    paper_results_dir = os.path.join(base_dir, "paper_artifacts")

    os.makedirs(client_results_dir, exist_ok=True)
    os.makedirs(paper_results_dir, exist_ok=True)
    
    experiments = [
        ExperimentConfig("dnn", 3, 10, 3, 32, "iid", "fedavg"),
        ExperimentConfig("dnn", 3, 10, 3, 32, "noniid", "fedavg"),
        ExperimentConfig("dnn", 3, 10, 3, 32, "noniid", "fedprox", 0.1),
    ]
    
    print("\n" + "="*50 + "\nFederated Experiments (Running First)\n" + "="*50)
    all_results = []
    for cfg in experiments:
        result = run_federated_experiment(cfg, log_dir=client_results_dir)
        if result:
            all_results.append(result)

    print("\n" + "="*50 + "\nCentralized Baselines (Running Last)\n" + "="*50)
    centralized = {"dnn": run_centralized_training("dnn", 20)}
    
    # Generate summary table
    summary = generate_summary_table(all_results, centralized)
    summary.to_csv(os.path.join(paper_results_dir, "quick_summary.csv"), index=False)

    print("\n" + "="*50 + "\nQuick Test Summary\n" + "="*50)
    print(summary.to_string(index=False))
    
    if all_results:
        plot_convergence(all_results, paper_results_dir)
        plot_strategy_comparison(all_results, paper_results_dir)
    
    return all_results


# =============================================================================
# Security Experiments (Chapter 10 - Poisoning Vulnerability)
# =============================================================================

def collect_confusion_matrix(config: ExperimentConfig, log_dir: str) -> np.ndarray:
    """Collect aggregated confusion matrix from the final round across all clients."""
    final_round = config.num_rounds
    cm_total = np.zeros((3, 3), dtype=int)

    for c in range(config.num_clients):
        eval_file = os.path.join(
            log_dir,
            f"{config.to_string()}_client_{c}_round_{final_round}_eval.json"
        )
        if os.path.exists(eval_file):
            with open(eval_file) as f:
                data = json.load(f)
            if "confusion_matrix" in data:
                cm_total += np.array(data["confusion_matrix"])

    return cm_total


def generate_security_summary(all_results: list, output_dir: str,
                              baselines: dict = None):
    """Generate security-focused summary table and comparison."""
    rows = []
    for entry in all_results:
        if entry is None:
            continue
        result, cm = entry
        cfg = result.config

        total_high = int(cm[2].sum()) if cm[2].sum() > 0 else 0
        h2l = int(cm[2][0]) if total_high > 0 else 0
        h2l_rate = h2l / total_high if total_high > 0 else 0.0

        attack_label = "none"
        if cfg.attack_type == 'label_flip':
            attack_label = f"flip_{int(cfg.attack_fraction * 100)}%"

        rows.append({
            "Distribution": cfg.distribution.upper(),
            "Attack": attack_label,
            "Strategy": cfg.strategy.upper(),
            "Accuracy": f"{result.final_accuracy:.4f}",
            "F1": f"{result.final_f1:.4f}",
            "H2L_Rate": f"{h2l_rate:.4f}",
            "H2L_Count": h2l,
            "Total_High": total_high,
            "Confusion_Matrix": cm.tolist(),
        })

    df = pd.DataFrame(rows)

    # Compute delta_acc and defense_recovery if baselines available
    if baselines:
        delta_acc_col = []
        for _, row in df.iterrows():
            dist = row["Distribution"]
            baseline_key = f"{dist}_none_FEDAVG"
            if baseline_key in baselines:
                baseline_acc = baselines[baseline_key]
                delta = baseline_acc - float(row["Accuracy"])
                delta_acc_col.append(f"{delta:+.4f}")
            else:
                delta_acc_col.append("-")
        df["Delta_Acc"] = delta_acc_col

    # Save
    summary_path = os.path.join(output_dir, "security_summary.csv")
    df_save = df.drop(columns=["Confusion_Matrix"])
    df_save.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # LaTeX table
    tex_path = os.path.join(output_dir, "security_summary.tex")
    with open(tex_path, 'w') as f:
        f.write(df_save.to_latex(index=False, escape=False))
    print(f"Saved: {tex_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SECURITY EXPERIMENT RESULTS")
    print("=" * 80)
    print(df_save.to_string(index=False))
    print("=" * 80)

    # Save per-experiment confusion matrices
    for entry in all_results:
        if entry is None:
            continue
        result, cm = entry
        cm_path = os.path.join(output_dir, f"{result.config.to_string()}_confusion_matrix.json")
        with open(cm_path, 'w') as f:
            json.dump({
                "config": result.config.to_string(),
                "confusion_matrix": cm.tolist(),
                "labels": ["Low", "Medium", "High"],
            }, f, indent=2)

    return df


def run_security_experiments(base_dir_name="v2"):
    """
    Phase 1 security experiments: FL poisoning vulnerability assessment.

    Runs 24 configurations: 2 distributions x 3 attack levels x 4 strategies.
    Model fixed to GRU (best trade-off from Chapter 9).
    """
    results_root = get_project_paths()
    base_dir = os.path.join(results_root, base_dir_name)

    print("=" * 70)
    print("ITS FL Security Experiments - Phase 1: Poisoning Vulnerability")
    print(f"Output Directory: {base_dir}")
    print("=" * 70)

    client_results_dir = os.path.join(base_dir, "client_logs")
    paper_results_dir = os.path.join(base_dir, "paper_artifacts")
    os.makedirs(client_results_dir, exist_ok=True)
    os.makedirs(paper_results_dir, exist_ok=True)

    # Fixed parameters
    model = "gru"
    num_clients = 3
    num_rounds = 10
    local_epochs = 5
    batch_size = 32
    malicious = [0]  # RSU 0 (55% High in Non-IID)

    distributions = ["iid", "noniid"]
    strategies = ["fedavg", "fedprox", "krum", "trimmed_mean"]
    attacks = [
        ("none", 0.0),
        ("label_flip", 0.2),
        ("label_flip", 0.5),
    ]

    # Build experiment matrix
    experiments = []
    for dist in distributions:
        for atk_type, atk_frac in attacks:
            for strat in strategies:
                mal = malicious if atk_type != "none" else []
                cfg = ExperimentConfig(
                    model_type=model,
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    distribution=dist,
                    strategy=strat,
                    fedprox_mu=0.1,
                    attack_type=atk_type,
                    attack_fraction=atk_frac,
                    malicious_clients=mal,
                )
                experiments.append(cfg)

    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Model: {model.upper()}, Clients: {num_clients}, "
          f"Rounds: {num_rounds}, Local epochs: {local_epochs}")
    print(f"Malicious client(s): {malicious}")
    print(f"Strategies: {strategies}")
    print(f"Attacks: {attacks}\n")

    # Run experiments
    all_results = []
    baselines = {}

    for i, cfg in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}: {cfg.to_string()}")
        print(f"{'='*60}")

        result = run_federated_experiment(cfg, log_dir=client_results_dir)
        if result:
            cm = collect_confusion_matrix(cfg, client_results_dir)
            all_results.append((result, cm))

            # Save per-experiment JSON
            result_data = asdict(result)
            result_data["confusion_matrix_aggregated"] = cm.tolist()
            file_path = os.path.join(paper_results_dir, f"{cfg.to_string()}.json")
            with open(file_path, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)

            # Track baselines for delta computation
            if cfg.attack_type == "none":
                key = f"{cfg.distribution.upper()}_none_{cfg.strategy.upper()}"
                baselines[key] = result.final_accuracy
        else:
            all_results.append(None)

    # Generate summary
    generate_security_summary(all_results, paper_results_dir, baselines)

    print("\n" + "=" * 70)
    print(f"Complete! Results saved in: {paper_results_dir}")
    print("=" * 70)

    return all_results


def run_security_quick_test(base_dir_name="v2_quick"):
    """Quick security test: 4 configs to validate pipeline."""
    results_root = get_project_paths()
    base_dir = os.path.join(results_root, base_dir_name)

    print("=" * 70)
    print("ITS FL Security - Quick Validation Test")
    print(f"Output Directory: {base_dir}")
    print("=" * 70)

    client_results_dir = os.path.join(base_dir, "client_logs")
    paper_results_dir = os.path.join(base_dir, "paper_artifacts")
    os.makedirs(client_results_dir, exist_ok=True)
    os.makedirs(paper_results_dir, exist_ok=True)

    experiments = [
        # Baseline: no attack, FedAvg, IID
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "fedavg"),
        # Attack: 50% flip, FedAvg, IID
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "fedavg",
                         attack_type="label_flip", attack_fraction=0.5,
                         malicious_clients=[0]),
        # Attack: 50% flip, Krum, IID
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "krum",
                         attack_type="label_flip", attack_fraction=0.5,
                         malicious_clients=[0]),
        # Attack: 50% flip, Trimmed Mean, IID
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "trimmed_mean",
                         attack_type="label_flip", attack_fraction=0.5,
                         malicious_clients=[0]),
    ]

    all_results = []
    for i, cfg in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Quick Test {i+1}/{len(experiments)}: {cfg.to_string()}")
        print(f"{'='*60}")

        result = run_federated_experiment(cfg, log_dir=client_results_dir)
        if result:
            cm = collect_confusion_matrix(cfg, client_results_dir)
            all_results.append((result, cm))
            result_data = asdict(result)
            result_data["confusion_matrix_aggregated"] = cm.tolist()
            file_path = os.path.join(paper_results_dir, f"{cfg.to_string()}.json")
            with open(file_path, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
        else:
            all_results.append(None)

    generate_security_summary(all_results, paper_results_dir)

    print("\n" + "=" * 70)
    print(f"Quick test complete! Results saved in: {paper_results_dir}")
    print("=" * 70)

    return all_results


def run_security_phase1b(base_dir_name="v2b"):
    """
    Phase 1b security experiments: Model poisoning with gradient scaling.

    Tests whether amplifying the malicious update (gradient scaling) can overcome
    the natural resilience demonstrated in Phase 1a (label-flipping alone).

    Configuration:
    - Label-flip: 100% High→Low on Client 0 (maximize malicious content)
    - Scale factors: 1× (control), 5×, 10×, 20×
    - Strategies: FedAvg (no defense), Krum, Trimmed Mean
    - FedProx removed (Phase 1a confirmed no value: ≤0.19pp vs FedAvg)

    Total: 2 distributions × 4 scale factors × 3 strategies = 24 experiments.

    Note on defenses:
    - Krum operates with knowledge of num_malicious_clients=1 (optimistic for defense)
    - Trimmed Mean with beta=0.34 and 3 clients is effectively coordinate-wise median
      (strongest possible defense in this configuration)
    """
    results_root = get_project_paths()
    base_dir = os.path.join(results_root, base_dir_name)

    print("=" * 70)
    print("ITS FL Security Experiments - Phase 1b: Model Poisoning (Gradient Scaling)")
    print(f"Output Directory: {base_dir}")
    print("=" * 70)

    client_results_dir = os.path.join(base_dir, "client_logs")
    paper_results_dir = os.path.join(base_dir, "paper_artifacts")
    os.makedirs(client_results_dir, exist_ok=True)
    os.makedirs(paper_results_dir, exist_ok=True)

    # Fixed parameters (same as Phase 1a)
    model = "gru"
    num_clients = 3
    num_rounds = 10
    local_epochs = 5
    batch_size = 32
    malicious = [0]  # RSU 0 (55% High in Non-IID)

    distributions = ["iid", "noniid"]
    strategies = ["fedavg", "krum", "trimmed_mean"]
    scale_factors = [1.0, 5.0, 10.0, 20.0]

    # Build experiment matrix
    experiments = []
    for dist in distributions:
        for scale in scale_factors:
            for strat in strategies:
                cfg = ExperimentConfig(
                    model_type=model,
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    distribution=dist,
                    strategy=strat,
                    attack_type="label_flip",
                    attack_fraction=1.0,  # 100% High→Low
                    malicious_clients=malicious,
                    scale_factor=scale,
                )
                experiments.append(cfg)

    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Model: {model.upper()}, Clients: {num_clients}, "
          f"Rounds: {num_rounds}, Local epochs: {local_epochs}")
    print(f"Malicious client(s): {malicious}")
    print(f"Strategies: {strategies}")
    print(f"Scale factors: {scale_factors}")
    print(f"Attack: 100% label-flip High→Low + gradient scaling\n")

    # Run experiments
    all_results = []
    baselines = {}

    for i, cfg in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}: {cfg.to_string()}")
        print(f"{'='*60}")

        result = run_federated_experiment(cfg, log_dir=client_results_dir)
        if result:
            cm = collect_confusion_matrix(cfg, client_results_dir)
            all_results.append((result, cm))

            # Save per-experiment JSON
            result_data = asdict(result)
            result_data["confusion_matrix_aggregated"] = cm.tolist()
            file_path = os.path.join(paper_results_dir, f"{cfg.to_string()}.json")
            with open(file_path, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)

            # Track scale=1 as baselines for delta computation
            if cfg.scale_factor == 1.0:
                key = f"{cfg.distribution.upper()}_scale1_{cfg.strategy.upper()}"
                baselines[key] = result.final_accuracy
        else:
            all_results.append(None)

    # Generate summary
    generate_security_summary(all_results, paper_results_dir, baselines)

    print("\n" + "=" * 70)
    print(f"Phase 1b complete! Results saved in: {paper_results_dir}")
    print("=" * 70)

    return all_results


def run_security_phase1b_quick(base_dir_name="v2b_quick"):
    """
    Quick validation test for Phase 1b: 4 configs to verify gradient scaling works.

    Expected results:
    1. scale1x FedAvg: similar to Phase 1a (label-flip alone has no effect)
    2. scale10x FedAvg: measurable accuracy drop and H→L rate increase
    3. scale10x Krum: should recover (reject amplified update via L2 distance)
    4. scale10x Trimmed Mean: should be near-perfect (median discards extremes)
    """
    results_root = get_project_paths()
    base_dir = os.path.join(results_root, base_dir_name)

    print("=" * 70)
    print("ITS FL Security - Phase 1b Quick Validation")
    print(f"Output Directory: {base_dir}")
    print("=" * 70)

    client_results_dir = os.path.join(base_dir, "client_logs")
    paper_results_dir = os.path.join(base_dir, "paper_artifacts")
    os.makedirs(client_results_dir, exist_ok=True)
    os.makedirs(paper_results_dir, exist_ok=True)

    experiments = [
        # Control: 100% flip, no scaling (should match Phase 1a resilience)
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "fedavg",
                         attack_type="label_flip", attack_fraction=1.0,
                         malicious_clients=[0], scale_factor=1.0),
        # Attack: 100% flip + 10x scaling on FedAvg (expect degradation)
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "fedavg",
                         attack_type="label_flip", attack_fraction=1.0,
                         malicious_clients=[0], scale_factor=10.0),
        # Defense: Krum should detect and reject amplified update
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "krum",
                         attack_type="label_flip", attack_fraction=1.0,
                         malicious_clients=[0], scale_factor=10.0),
        # Defense: Trimmed Mean (median) should discard extreme values
        ExperimentConfig("gru", 3, 10, 5, 32, "iid", "trimmed_mean",
                         attack_type="label_flip", attack_fraction=1.0,
                         malicious_clients=[0], scale_factor=10.0),
    ]

    all_results = []
    for i, cfg in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Quick Test {i+1}/{len(experiments)}: {cfg.to_string()}")
        print(f"{'='*60}")

        result = run_federated_experiment(cfg, log_dir=client_results_dir)
        if result:
            cm = collect_confusion_matrix(cfg, client_results_dir)
            all_results.append((result, cm))
            result_data = asdict(result)
            result_data["confusion_matrix_aggregated"] = cm.tolist()
            file_path = os.path.join(paper_results_dir, f"{cfg.to_string()}.json")
            with open(file_path, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
        else:
            all_results.append(None)

    generate_security_summary(all_results, paper_results_dir)

    print("\n" + "=" * 70)
    print(f"Phase 1b quick test complete! Results saved in: {paper_results_dir}")
    print("=" * 70)

    return all_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Fix for Colab/Jupyter multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    set_global_seeds(42)
    tf.get_logger().setLevel('ERROR')

    # Configure system resources (CPU or GPU) to ensure centralized baseline is run on GPU (if available)
    configure_system_resources()

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--security":
        run_security_experiments(base_dir_name="v2")
    elif len(sys.argv) > 1 and sys.argv[1] == "--security-quick":
        run_security_quick_test(base_dir_name="v2_quick")
    elif len(sys.argv) > 1 and sys.argv[1] == "--security-phase1b":
        run_security_phase1b(base_dir_name="v2b")
    elif len(sys.argv) > 1 and sys.argv[1] == "--security-phase1b-quick":
        run_security_phase1b_quick(base_dir_name="v2b_quick")
    elif len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test(base_dir_name="v1")
    else:
        run_full_suite(base_dir_name="v1")
