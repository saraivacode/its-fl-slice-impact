#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITS FL Framework - Data Module
===============================

Handles data loading from the pre-processed ITS dataset (raw_full.csv),
StandardScaler normalization, and IID/Non-IID partitioning for FL clients.

The dataset contains 5,093 samples with 29 features + 1 target (impact_level)
already one-hot encoded and feature-engineered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .fl_config import NON_IID_ALLOCATION


# Numeric columns to scale (non-OHE features)
NUMERIC_COLS = [
    'rec_serv', 'env_car', 'rtt', 'ncars', 'pdr', 'bc_rtt',
    'rtt_change', 'pdr_change', 'rtt_mam', 'pdr_mam', 'rtt_masd',
    'log_rtt', 'log_pdr', 'log2_rtt', 'log2_pdr', 'rtt_sqrd', 'pdr_sqrd',
]


def load_and_prepare(
    csv_path: Path,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, float], List[str]]:
    """
    Load the ITS dataset, normalize labels, scale features, and split into
    global train/test sets.

    Parameters
    ----------
    csv_path : Path
        Path to raw_full.csv.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray
        Scaled training feature matrix (float32).
    X_test : np.ndarray
        Scaled test feature matrix (float32).
    y_train : np.ndarray
        Training labels (0=Low, 1=Medium, 2=High).
    y_test : np.ndarray
        Test labels.
    class_weight_dict : dict
        Class weights for imbalanced learning.
    feature_names : list[str]
        Names of the features.
    """
    # Step 1: Load
    df = pd.read_csv(csv_path)
    print(f"  Loaded dataset: {len(df)} samples, {len(df.columns)} columns")

    # Step 2: Normalize labels to integers
    if df['impact_level'].dtype == object:
        df['impact_level'] = df['impact_level'].map({'low': 0, 'medium': 1, 'high': 2})
    df['impact_level'] = df['impact_level'].astype(int)

    # Step 3: Separate features and labels
    drop_cols = ['time', 'impact_level']
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    labels = df['impact_level']

    print(f"  Impact label distribution: {labels.value_counts().sort_index().to_dict()}")

    features = features.astype('float32')
    feature_names = list(features.columns)

    # Step 4: Split FIRST, then scale (avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state, stratify=labels,
    )

    # Step 5: Scale numeric columns (fit on train only)
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    if num_cols:
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols].astype(float))
        X_test[num_cols] = scaler.transform(X_test[num_cols].astype(float))

    # Convert to numpy
    X_train = X_train.values.astype('float32')
    X_test = X_test.values.astype('float32')
    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)

    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

    print(f"  Global split: {len(X_train)} train, {len(X_test)} test")
    print(f"  Feature matrix: {X_train.shape[1]} features")

    return X_train, X_test, y_train, y_test, class_weight_dict, feature_names


def partition_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data into IID (random disjoint) subsets for each client.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    num_clients : int
        Number of FL clients.
    seed : int
        Random seed.

    Returns
    -------
    list of (X_client, y_client) tuples.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    chunks = np.array_split(indices, num_clients)

    partitions = []
    for i, idx in enumerate(chunks):
        partitions.append((X[idx], y[idx]))
        dist = pd.Series(y[idx]).value_counts().sort_index().to_dict()
        print(f"    Client {i} (IID): {len(idx)} samples, dist={dist}")

    return partitions


def partition_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data into Non-IID subsets based on class distribution.

    Uses the NON_IID_ALLOCATION matrix to assign different proportions
    of each class to each client, reflecting realistic RSU deployments
    where different road segments observe different traffic impact levels.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    num_clients : int
        Number of FL clients.
    seed : int
        Random seed.

    Returns
    -------
    list of (X_client, y_client) tuples.
    """
    rng = np.random.RandomState(seed)
    allocation = NON_IID_ALLOCATION[:num_clients]
    unique_classes = sorted(np.unique(y))

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls_idx, cls in enumerate(unique_classes):
        cls_mask = np.where(y == cls)[0]
        cls_indices = rng.permutation(cls_mask)
        n = len(cls_indices)

        if cls_idx >= allocation.shape[1]:
            # If more classes than allocation columns, distribute equally
            chunks = np.array_split(cls_indices, num_clients)
            for c in range(num_clients):
                client_indices[c].extend(chunks[c].tolist())
            continue

        # Assign proportionally based on allocation matrix
        start = 0
        for c in range(num_clients):
            frac = allocation[c, cls_idx]
            count = int(round(frac * n))
            if c == num_clients - 1:
                end = n
            else:
                end = min(start + count, n)
            client_indices[c].extend(cls_indices[start:end].tolist())
            start = end

    partitions = []
    for i in range(num_clients):
        idx = np.array(client_indices[i])
        rng.shuffle(idx)
        partitions.append((X[idx], y[idx]))
        dist = pd.Series(y[idx]).value_counts().sort_index().to_dict()
        print(f"    Client {i} (NonIID): {len(idx)} samples, dist={dist}")

    return partitions


def split_client_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a client's data into train and validation sets (stratified).

    Returns (X_train, X_val, y_train, y_val).
    """
    min_count = pd.Series(y).value_counts().min() if len(y) > 0 else 0
    stratify = y if min_count >= 2 else None

    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
