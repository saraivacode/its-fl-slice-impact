#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITS FL Framework - Federated Learning Package
===============================================

Provides federated learning capabilities for the ITS network slicing
impact classification framework, supporting DNN, LSTM, and GRU models
with FedAvg, FedProx, Krum, and Trimmed Mean aggregation strategies
over IID and Non-IID data distributions.

Security features include label-flipping attacks and gradient scaling
for poisoning vulnerability assessment.
"""

from .fl_main import (
    run_full_suite,
    run_quick_test,
    run_security_experiments,
    run_security_quick_test,
    run_security_phase1b,
    run_security_phase1b_quick,
    run_security_phase1b_krum_rerun,
    run_security_sensitivity_epochs,
)

__all__ = [
    "run_full_suite",
    "run_quick_test",
    "run_security_experiments",
    "run_security_quick_test",
    "run_security_phase1b",
    "run_security_phase1b_quick",
    "run_security_phase1b_krum_rerun",
    "run_security_sensitivity_epochs",
]
