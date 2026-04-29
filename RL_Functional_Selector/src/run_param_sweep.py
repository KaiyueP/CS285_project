"""
Run multiple training configurations and save each run separately.

Usage:
  python run_param_sweep.py --gscdb-root ../../GSCDB --base-out-dir ../models/sweeps

Optional custom config file:
  python run_param_sweep.py --gscdb-root ../../GSCDB --base-out-dir ../models/sweeps --config sweep_config.json

Config JSON format:
{
  "runs": [
    {"name": "r1", "warmup_supervised": 2000, "steps": 3000, "lr": 0.003},
    {"name": "r2", "warmup_supervised": 6000, "steps": 1000, "lr": 0.002}
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .train_reaction_reinforce import train
except ImportError:
    from train_reaction_reinforce import train


def _default_runs() -> List[Dict[str, Any]]:
    # Starter grid you can edit.
    return [
        {
            "name": "w2000_s3000_lr0.01_bs128_e0.02",
            "warmup_supervised": 2000,
            "steps": 3000,
            "lr": 0.01,
            "batch_size": 128,
            "entropy_coef": 0.02,
            "baseline_momentum": 0.95,
            "hidden": [256, 128],
            "test_fraction": 0.15,
            "seed": 42,
            "max_reactions": None,
            "log_interval": 500,
        },
        {
            "name": "w6000_s1000_lr0.003_bs256_e0.005",
            "warmup_supervised": 6000,
            "steps": 1000,
            "lr": 0.003,
            "batch_size": 256,
            "entropy_coef": 0.005,
            "baseline_momentum": 0.95,
            "hidden": [256, 128],
            "test_fraction": 0.15,
            "seed": 42,
            "max_reactions": None,
            "log_interval": 500,
        },
    ]


def _load_runs(config_path: Path | None) -> List[Dict[str, Any]]:
    if config_path is None:
        return _default_runs()
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    runs = cfg.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Config must contain non-empty 'runs' list.")
    return runs


def main(argv=None):
    p = argparse.ArgumentParser(description="Run multiple RL parameter settings and save each run.")
    p.add_argument("--gscdb-root", type=str, required=True)
    p.add_argument("--base-out-dir", type=str, required=True)
    p.add_argument("--config", type=str, default=None, help="Optional JSON file specifying runs.")
    args = p.parse_args(argv)

    gscdb_root = Path(args.gscdb_root)
    base_out_dir = Path(args.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(Path(args.config) if args.config else None)

    index_rows = []
    for run in runs:
        name = str(run.get("name") or f"run_{len(index_rows)+1}")
        out_dir = base_out_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        train(
            gscdb_root=gscdb_root,
            out_dir=out_dir,
            max_reactions=run.get("max_reactions"),
            test_fraction=float(run.get("test_fraction", 0.15)),
            steps=int(run.get("steps", 3000)),
            seed=int(run.get("seed", 42)),
            hidden=tuple(run.get("hidden", [256, 128])),
            lr=float(run.get("lr", 0.01)),
            entropy_coef=float(run.get("entropy_coef", 0.02)),
            baseline_momentum=float(run.get("baseline_momentum", 0.95)),
            warmup_supervised=int(run.get("warmup_supervised", 2000)),
            batch_size=int(run.get("batch_size", 128)),
            log_interval=int(run.get("log_interval", 500)),
            reward_mode=str(run.get("reward_mode", "absolute")),
            select_best_by=str(run.get("select_best_by", "mae")),
            emae_steps=int(run.get("emae_steps", 0)),
        )

        meta_path = out_dir / "reinforce_meta.json"
        hist_path = out_dir / "training_history.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        index_rows.append(
            {
                "name": name,
                "out_dir": str(out_dir),
                "test_mae_energy_greedy_final": meta.get("test_mae_energy_greedy_final"),
                "test_mae_energy_oracle": meta.get("test_mae_energy_oracle"),
                "test_mae_energy_uniform_random": meta.get("test_mae_energy_uniform_random"),
                "n_train": meta.get("n_train"),
                "n_test": meta.get("n_test"),
                "warmup_supervised": run.get("warmup_supervised", 2000),
                "steps": run.get("steps", 3000),
                "lr": run.get("lr", 0.01),
                "batch_size": run.get("batch_size", 128),
                "entropy_coef": run.get("entropy_coef", 0.02),
                "seed": run.get("seed", 42),
                "meta_path": str(meta_path),
                "history_path": str(hist_path),
            }
        )

    index_csv = base_out_dir / "sweep_index.csv"
    if index_rows:
        with open(index_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)

    print(f"Completed {len(index_rows)} runs.")
    print(f"Index written to: {index_csv}")


if __name__ == "__main__":
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P(__file__).resolve().parent))
    main()

