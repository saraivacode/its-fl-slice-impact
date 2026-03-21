#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Security Module (Poisoning Attacks)
=====================================================

Implements label-flipping attacks and gradient scaling for FL poisoning
vulnerability assessment.

Attack Model:
- Label-flip: Critical(3) -> Adequate(0) directed poisoning
- Gradient scaling: amplify malicious update by scale_factor before aggregation

In the ITS context, Critical->Adequate flipping suppresses lifecycle decisions:
the model learns to classify severely degraded slices as adequate.
"""

from __future__ import annotations

import numpy as np


def apply_label_flip_attack(
    y_train: np.ndarray,
    fraction: float,
    source_class: int = 3,
    target_class: int = 0,
    seed: int = 42,
) -> tuple:
    """
    Label-flipping attack: flip a fraction of source_class labels to target_class.

    Parameters
    ----------
    y_train : np.ndarray
        Training labels.
    fraction : float
        Fraction of source_class labels to flip (0, 1].
    source_class : int
        Class to flip from (default: 3 = Critical).
    target_class : int
        Class to flip to (default: 0 = Adequate).
    seed : int
        Random seed.

    Returns
    -------
    tuple of (y_modified, num_flipped, flip_indices)
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


def apply_gradient_scaling(
    trained_weights: list,
    global_weights: list,
    scale_factor: float,
) -> tuple:
    """
    Gradient scaling attack: amplify the malicious update delta.

    new_weights = global_weights + scale_factor * (trained_weights - global_weights)

    Parameters
    ----------
    trained_weights : list of np.ndarray
        Weights after local training.
    global_weights : list of np.ndarray
        Global weights before local training.
    scale_factor : float
        Amplification factor (>1 amplifies, 1 = no scaling).

    Returns
    -------
    tuple of (boosted_weights, original_norm, boosted_norm)
    """
    boosted_weights = [
        gw + scale_factor * (tw - gw)
        for tw, gw in zip(trained_weights, global_weights)
    ]

    original_norm = sum(
        float(np.linalg.norm(tw - gw))
        for tw, gw in zip(trained_weights, global_weights)
    )
    boosted_norm = original_norm * scale_factor

    return boosted_weights, original_norm, boosted_norm
