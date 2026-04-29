"""
Compare multiple training runs by plotting MAE / regret / top3_hit on one figure.

Example:
  python compare_runs.py \
    --run ../models/reaction_reinforce \
    --run ../models/reaction_reinforce_mae_tuned \
    --run ../models/reaction_reinforce_regret \
    --max-cum-update 5000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_run(run_dir: Path) -> Tuple[dict, List[dict]]:
    meta_path = run_dir / "reinforce_meta.json"
    hist_path = run_dir / "training_history.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    if not hist_path.is_file():
        raise FileNotFoundError(f"Missing history file: {hist_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(hist_path, encoding="utf-8") as f:
        hist = json.load(f)
    recs = hist.get("records") or []
    if not recs:
        raise ValueError(f"No records in {hist_path}")
    return meta, recs


def _arr(recs: List[dict], key: str, fallback: str | None = None) -> np.ndarray:
    out = []
    for r in recs:
        v = r.get(key)
        if v is None and fallback is not None:
            v = r.get(fallback)
        out.append(np.nan if v is None else float(v))
    return np.array(out, dtype=float)


def _clip_by_cum(cum: np.ndarray, max_cum: int | None) -> np.ndarray:
    if max_cum is None:
        return np.ones_like(cum, dtype=bool)
    return cum <= max_cum


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare multiple RL run histories.")
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run directory containing reinforce_meta.json and training_history.json. Repeat for multiple runs.",
    )
    p.add_argument("--labels", nargs="*", default=None, help="Optional labels aligned with --run order.")
    p.add_argument("--output-dir", type=str, default=None, help="Where to save plots/CSV (default: first run / comparisons)")
    p.add_argument("--max-cum-update", type=int, default=None, help="Only use points with cum_update <= this.")
    args = p.parse_args(argv)

    run_dirs = [Path(r) for r in args.run]
    labels = args.labels if args.labels else [d.name for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit("If provided, --labels must match number of --run entries.")

    out_dir = Path(args.output_dir) if args.output_dir else run_dirs[0] / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_data: List[Dict] = []
    for label, run_dir in zip(labels, run_dirs):
        meta, recs = _load_run(run_dir)
        cum = _arr(recs, "cum_update")
        m = _clip_by_cum(cum, args.max_cum_update)
        if not np.any(m):
            raise SystemExit(f"No points <= max-cum-update for run: {run_dir}")
        run_data.append(
            {
                "label": label,
                "dir": run_dir,
                "meta": meta,
                "cum": cum[m],
                "test_mae": _arr(recs, "test_mae_energy", fallback="test_mae_hartree")[m],
                "train_mae": _arr(recs, "train_mae_energy")[m],
                "test_regret": _arr(recs, "test_regret")[m],
                "test_top3": _arr(recs, "test_top3_hit")[m],
                "test_greedy": _arr(recs, "test_greedy")[m],
            }
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    # ---- Figure 1: core metrics ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    title = "Run comparison"
    if args.max_cum_update is not None:
        title += f" (cum_update ≤ {args.max_cum_update})"
    fig.suptitle(title, fontsize=13)

    # Test MAE
    ax = axes[0, 0]
    for d in run_data:
        ax.plot(d["cum"], d["test_mae"], marker="o", markersize=3, linewidth=1.3, label=d["label"])
    ax.set_title("Test MAE")
    unit = run_data[0]["meta"].get("energy_unit", "kcal/mol")
    ax.set_ylabel(f"MAE ({unit})")
    ax.set_xlabel("Cumulative update")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Test regret
    ax = axes[0, 1]
    any_reg = False
    for d in run_data:
        y = d["test_regret"]
        if np.isfinite(y).any():
            any_reg = True
            ax.plot(d["cum"], y, marker="o", markersize=3, linewidth=1.3, label=d["label"])
    ax.set_title("Test regret (lower better)")
    ax.set_ylabel("Regret")
    ax.set_xlabel("Cumulative update")
    ax.grid(True, alpha=0.3)
    if any_reg:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No regret metric in these runs", ha="center", va="center", transform=ax.transAxes)

    # Test top3 hit
    ax = axes[1, 0]
    any_top3 = False
    for d in run_data:
        y = d["test_top3"]
        if np.isfinite(y).any():
            any_top3 = True
            ax.plot(d["cum"], y, marker="s", markersize=3, linewidth=1.3, label=d["label"])
    ax.set_title("Test top3 hit (higher better)")
    ax.set_ylabel("Hit rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Cumulative update")
    ax.grid(True, alpha=0.3)
    if any_top3:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No top3 metric in these runs", ha="center", va="center", transform=ax.transAxes)

    # Test greedy accuracy
    ax = axes[1, 1]
    for d in run_data:
        ax.plot(d["cum"], d["test_greedy"], marker=".", markersize=3, linewidth=1.3, label=d["label"])
    ax.set_title("Test greedy accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Cumulative update")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    suffix = f"_first{args.max_cum_update}" if args.max_cum_update is not None else ""
    fig_path = out_dir / f"run_comparison_overview{suffix}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---- Figure 2: train vs test MAE ----
    fig2, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for d in run_data:
        x = d["cum"]
        ax.plot(x, d["test_mae"], linewidth=1.5, label=f"{d['label']} (test)")
        if np.isfinite(d["train_mae"]).any():
            ax.plot(x, d["train_mae"], linestyle="--", linewidth=1.1, alpha=0.8, label=f"{d['label']} (train)")
    ax.set_title("Train vs Test MAE")
    ax.set_xlabel("Cumulative update")
    ax.set_ylabel(f"MAE ({unit})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig2_path = out_dir / f"run_comparison_mae{suffix}.png"
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    # ---- Summary CSV ----
    rows = []
    for d in run_data:
        row = {
            "label": d["label"],
            "run_dir": str(d["dir"]),
            "reward_mode": d["meta"].get("reward_mode"),
            "select_best_by": d["meta"].get("select_best_by"),
            "final_test_mae": float(d["test_mae"][-1]),
            "final_train_mae": float(d["train_mae"][-1]) if np.isfinite(d["train_mae"][-1]) else np.nan,
            "final_test_regret": float(d["test_regret"][-1]) if np.isfinite(d["test_regret"][-1]) else np.nan,
            "final_test_top3_hit": float(d["test_top3"][-1]) if np.isfinite(d["test_top3"][-1]) else np.nan,
            "final_test_greedy": float(d["test_greedy"][-1]),
            "meta_final_test_mae": d["meta"].get("test_mae_energy_greedy_final"),
            "meta_final_test_regret": d["meta"].get("test_regret_final"),
            "meta_final_test_top3_hit": d["meta"].get("test_top3_hit_final"),
        }
        rows.append(row)

    csv_path = out_dir / f"run_comparison_summary{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:")
    print(f"  {fig_path}")
    print(f"  {fig2_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P(__file__).resolve().parent))
    main()

