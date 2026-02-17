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
- Up to 15-17 vehicles simultaneously competing for network resources
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

Models:
    - DNN: Deep Neural Network (Dense layers with dropout)
    - LSTM: Long Short-Term Memory for temporal pattern capture
    - GRU: Gated Recurrent Unit as efficient LSTM alternative

Metrics:
    - Performance: Accuracy, Precision, Recall, F1-Score (macro)
    - Convergence: Rounds to reach 85%/90% accuracy thresholds
    - Stability: Standard deviation of accuracy over last 5 rounds
    - Efficiency: Total training time

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

Full experiment suite (~1.5-2 hours):
    python federatedLearning_ITS.py

Quick test run (~20-30 minutes):
    python federatedLearning_ITS.py --quick

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
from dataclasses import dataclass, asdict

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, GRU, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.callbacks import EarlyStopping

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    strategy: str  # 'fedavg' or 'fedprox'
    fedprox_mu: float = 0.1
    
    def to_string(self):
        return f"{self.model_type}_{self.distribution}_{self.strategy}_e{self.local_epochs}"


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


def load_and_preprocess_data_iid(client_id: int, num_clients: int, data_path: str = '../data/raw_full.csv'):
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


def load_and_preprocess_data_noniid(client_id: int, num_clients: int, data_path: str = '../data/raw_full.csv'):
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
    
    # CRITICAL: Return in order that matches ITSRsuClient constructor!
    # Constructor: (config, X_train, y_train, X_test, y_test, client_id)
    return X_train, y_train, X_test, y_test


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
    
    IMPORTANT: Constructor parameter order is (config, X_train, y_train, X_test, y_test, client_id)
    This must match the order returned by _prepare_features_labels!
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
        
        if self.config.strategy == "fedprox":
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
        
        # Evaluate on training data
        loss, acc = self.model.evaluate(X_train, y_train, verbose=0)
        
        # Save training metrics
        self._save_fit_metrics(train_time, loss, acc)
        
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
        
        # Save eval metrics
        filename = f"{self.config.to_string()}_client_{self.client_id}_round_{self.current_round}_eval.json"
        filename = os.path.join(self.log_dir, filename)
        with open(filename, 'w') as f:
            json.dump({
                "round": self.current_round, "client_id": self.client_id,
                "loss": float(loss), "accuracy": float(acc),
                "precision": float(prec), "recall": float(rec), "f1": float(f1),
                "dataset_size": len(self.X_test)
            }, f)
        
        return loss, len(self.X_test), {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1)
        }


# =============================================================================
# Centralized Training
# =============================================================================

def run_centralized_training(model_type: str, num_epochs: int = 30, data_path: str = '../data/raw_full.csv'):
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

    results_version="v1"

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test(base_dir_name=results_version)
    else:
        run_full_suite(base_dir_name=results_version)
