#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITS FL Framework - Results Manager
====================================

Manages saving, aggregating, and formatting FL experiment results
with JSON outputs, summary tables (CSV + LaTeX), and metadata.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class FLResultsManager:
    """
    Manages FL experiment results with JSON format.

    Parameters
    ----------
    output_dir : Path
        Directory where results will be saved.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, result: Dict, filename: Optional[str] = None) -> Path:
        """Save a single experiment result as JSON."""
        if filename is None:
            filename = f"{result['config_name']}.json"

        filepath = self.output_dir / filename

        output = {
            "model_info": {
                "model_name": result.get("config_name", "Unknown"),
                "model_type": result.get("model_type", "Unknown"),
                "library": _get_tf_version(),
            },
            "run_info": {
                "framework_version": "ITS-FL 2.0.0",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "system": _get_system_info(),
            },
            "fl_config": {
                "mode": result.get("mode", "federated"),
                "strategy": result.get("strategy", "-"),
                "distribution": result.get("distribution", "-"),
                "num_clients": result.get("num_clients", "-"),
                "num_rounds": result.get("num_rounds", "-"),
                "local_epochs": result.get("local_epochs", "-"),
                "batch_size": result.get("batch_size", "-"),
            },
            "security": {
                "attack_type": _get_config_attr(result, "attack_type", "none"),
                "attack_fraction": _get_config_attr(result, "attack_fraction", 0.0),
                "malicious_clients": _get_config_attr(result, "malicious_clients", []),
                "scale_factor": _get_config_attr(result, "scale_factor", 1.0),
                "attack_info": result.get("attack_info", []),
            },
            "metrics": {
                "final": result.get("final_metrics", {}),
                "per_round": result.get("per_round", {}),
                "convergence": result.get("convergence", {}),
                "history": result.get("history", {}),
            },
            "confusion_matrix": result.get("confusion_matrix", []),
            "performance": {
                "total_time": result.get("total_time", 0),
                "epochs_trained": result.get("epochs_trained"),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"    Saved: {filepath}")
        return filepath

    def save_summary_table(
        self,
        fl_results: List[Dict],
        centralized_results: List[Dict],
    ) -> tuple:
        """Create and save a summary comparison table as CSV and LaTeX."""
        rows = []

        for r in fl_results:
            fm = r.get("final_metrics", {})
            conv = r.get("convergence", {})
            rows.append({
                "Model": r.get("model_type", "").upper(),
                "Distribution": r.get("distribution", "").upper(),
                "Strategy": r.get("strategy", "").upper(),
                "Accuracy": f"{fm.get('accuracy', 0):.4f}",
                "Precision": f"{fm.get('precision', 0):.4f}",
                "Recall": f"{fm.get('recall', 0):.4f}",
                "F1-Macro": f"{fm.get('f1_macro', 0):.4f}",
                "@85%": conv.get("round_85", "-") or "-",
                "@90%": conv.get("round_90", "-") or "-",
                "Stability": f"{conv.get('stability_std', 0):.4f}",
                "Time (s)": f"{r.get('total_time', 0):.1f}",
            })

        for r in centralized_results:
            fm = r.get("final_metrics", {})
            rows.append({
                "Model": r.get("model_type", "").upper(),
                "Distribution": "CENTRAL",
                "Strategy": "-",
                "Accuracy": f"{fm.get('accuracy', 0):.4f}",
                "Precision": f"{fm.get('precision', 0):.4f}",
                "Recall": f"{fm.get('recall', 0):.4f}",
                "F1-Macro": f"{fm.get('f1_macro', 0):.4f}",
                "@85%": "-",
                "@90%": "-",
                "Stability": "-",
                "Time (s)": f"{r.get('total_time', 0):.1f}",
            })

        df = pd.DataFrame(rows)

        csv_path = self.output_dir / "summary_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved summary CSV: {csv_path}")

        tex_path = self.output_dir / "summary_table.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False, escape=False))
        print(f"    Saved summary LaTeX: {tex_path}")

        return csv_path, tex_path

    def save_centralized_results(self, centralized_results: List[Dict]) -> Path:
        """Save all centralized baseline results in a single JSON."""
        filepath = self.output_dir / "centralized_results.json"
        data = {r["model_type"]: r for r in centralized_results}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"    Saved centralized results: {filepath}")
        return filepath

    def save_security_summary(
        self,
        all_results: List[Dict],
        baselines: Optional[Dict] = None,
    ) -> Path:
        """Generate and save security-focused summary table."""
        rows = []
        for result in all_results:
            if result is None:
                continue

            cfg = result.get("config")
            fm = result.get("final_metrics", {})
            cm = np.array(result.get("confusion_matrix", [[0]*3]*3))

            total_high = int(cm[2].sum()) if cm.shape[0] > 2 and cm[2].sum() > 0 else 0
            h2l = int(cm[2][0]) if total_high > 0 else 0
            h2l_rate = h2l / total_high if total_high > 0 else 0.0

            attack_label = "none"
            if cfg and cfg.attack_type == 'label_flip':
                attack_label = f"flip_{int(cfg.attack_fraction * 100)}%"

            scale_label = ""
            if cfg and cfg.scale_factor > 1.0:
                scale_label = f"{int(cfg.scale_factor)}x"

            rows.append({
                "Distribution": (cfg.distribution if cfg else "").upper(),
                "Attack": attack_label,
                "Scale": scale_label,
                "Strategy": (cfg.strategy if cfg else "").upper(),
                "Accuracy": f"{fm.get('accuracy', 0):.4f}",
                "F1": f"{fm.get('f1_macro', 0):.4f}",
                "H2L_Rate": f"{h2l_rate:.4f}",
                "H2L_Count": h2l,
                "Total_High": total_high,
            })

        df = pd.DataFrame(rows)

        # Compute delta_acc if baselines available
        if baselines:
            delta_col = []
            for _, row in df.iterrows():
                dist = row["Distribution"]
                baseline_key = f"{dist}_none_FEDAVG"
                if baseline_key in baselines:
                    delta = baselines[baseline_key] - float(row["Accuracy"])
                    delta_col.append(f"{delta:+.4f}")
                else:
                    delta_col.append("-")
            df["Delta_Acc"] = delta_col

        summary_path = self.output_dir / "security_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\n    Saved: {summary_path}")

        tex_path = self.output_dir / "security_summary.tex"
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(df.to_latex(index=False, escape=False))
        print(f"    Saved: {tex_path}")

        print("\n" + "=" * 80)
        print("SECURITY EXPERIMENT RESULTS")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        # Save per-experiment confusion matrices
        for result in all_results:
            if result is None:
                continue
            cm = result.get("confusion_matrix", [])
            cm_path = self.output_dir / f"{result['config_name']}_confusion_matrix.json"
            with open(cm_path, 'w') as f:
                json.dump({
                    "config": result['config_name'],
                    "confusion_matrix": cm if isinstance(cm, list) else np.array(cm).tolist(),
                    "labels": ["Low", "Medium", "High"],
                }, f, indent=2)

        return summary_path


def _get_config_attr(result: Dict, attr: str, default: Any = None) -> Any:
    """Get attribute from result's config object or return default."""
    cfg = result.get("config")
    if cfg and hasattr(cfg, attr):
        return getattr(cfg, attr)
    return default


def _get_tf_version() -> str:
    try:
        import tensorflow as tf
        return f"tensorflow v{tf.__version__}"
    except ImportError:
        return "tensorflow (not installed)"


def _get_system_info() -> Dict[str, Any]:
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            info["gpu"] = tf.config.list_physical_devices("GPU")[0].name
    except Exception:
        pass
    return info
