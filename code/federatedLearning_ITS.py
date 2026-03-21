#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Federated Learning for ITS Network Slicing Impact Classification

This script is the entry point for the federated learning framework that
classifies the impact level of network slicing policies on ITS applications.
The classification target (impact_level: low/medium/high) indicates how
network conditions affect application requirements.

Based on the AIMS FL framework (modular federated/ package), extended with:
- Byzantine-robust aggregation: Krum, Trimmed Mean
- Security experiments: label-flipping attacks, gradient scaling
- Manual FL simulation (no Ray dependency) for full control

================================================================================
USAGE
================================================================================

Full experiment suite (Chapter 9):
    python federatedLearning_ITS.py

Quick test run (Chapter 9):
    python federatedLearning_ITS.py --quick

Security experiments (Chapter 10 - Phase 1a, 24 configs):
    python federatedLearning_ITS.py --security

Security quick test (Chapter 10 - 4 configs):
    python federatedLearning_ITS.py --security-quick

Security Phase 1b (gradient scaling, 32 configs):
    python federatedLearning_ITS.py --security-phase1b

Security Phase 1b quick (4 configs):
    python federatedLearning_ITS.py --security-phase1b-quick

Security Phase 1b Krum rerun (8 configs):
    python federatedLearning_ITS.py --security-phase1b-krum-rerun

Security sensitivity analysis (8 configs):
    python federatedLearning_ITS.py --security-sensitivity-epochs

================================================================================
"""

import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure the code directory is in the path
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
import tensorflow as tf

from federated.fl_main import (
    set_seeds,
    run_full_suite,
    run_quick_test,
    run_security_experiments,
    run_security_quick_test,
    run_security_phase1b,
    run_security_phase1b_quick,
    run_security_phase1b_krum_rerun,
    run_security_sensitivity_epochs,
)
from federated.fl_models import configure_tf


if __name__ == "__main__":
    set_seeds(42)
    tf.get_logger().setLevel('ERROR')
    configure_tf()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    else:
        cmd = None

    if cmd == "--security":
        run_security_experiments(base_dir_name="v3")
    elif cmd == "--security-quick":
        run_security_quick_test(base_dir_name="v3_quick")
    elif cmd == "--security-phase1b":
        run_security_phase1b(base_dir_name="v3b")
    elif cmd == "--security-phase1b-quick":
        run_security_phase1b_quick(base_dir_name="v3b_quick")
    elif cmd == "--security-phase1b-krum-rerun":
        run_security_phase1b_krum_rerun(base_dir_name="v3b_krum_fixed")
    elif cmd == "--security-sensitivity-epochs":
        run_security_sensitivity_epochs(base_dir_name="v3b_sensitivity_epochs")
    elif cmd == "--quick":
        run_quick_test(base_dir_name="v1")
    else:
        run_full_suite(base_dir_name="v1")
