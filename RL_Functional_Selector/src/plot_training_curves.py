"""
Plot metrics saved by train_reaction_reinforce.py in training_history.json.

Generates a multi-panel figure explaining training dynamics. Run after training:

  python plot_training_curves.py --history ../models/reaction_reinforce/training_history.json
  python plot_training_curves.py --history .../training_history.json --max-cum-update 7000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HARTREE_TO_KCAL = 627.509


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot curves from training_history.json")
    p.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to training_history.json (default: ../models/reaction_reinforce/training_history.json)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for PNG files (default: same folder as history file)",
    )
    p.add_argument(
        "--max-cum-update",
        type=int,
        default=None,
        help="Only plot points with cum_update <= this (e.g. 7000 for first 7000 cumulative steps).",
    )
    args = p.parse_args(argv)

    hist_path = Path(args.history) if args.history else None
    if hist_path is None:
        here = Path(__file__).resolve().parent
        hist_path = here.parent / "models" / "reaction_reinforce" / "training_history.json"
    if not hist_path.is_file():
        raise SystemExit(f"History file not found: {hist_path}\nTrain first to create training_history.json.")

    out_dir = Path(args.output_dir) if args.output_dir else hist_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(hist_path, encoding="utf-8") as f:
        data = json.load(f)

    recs = data.get("records") or []
    if not recs:
        raise SystemExit("No records in history file.")

    oracle = float(data["te_mae_oracle"])
    rand_mae = float(data["te_mae_uniform_random"])
    wu = int(data.get("warmup_supervised") or 0)

    cum = np.array([r["cum_update"] for r in recs])
    phases = np.array([r["phase"] for r in recs])
    test_g = np.array([r["test_greedy"] for r in recs])
    train_g = np.array([r["train_greedy"] for r in recs])
    test_p = np.array([r["test_prob_on_best"] for r in recs])
    test_mae = np.array([r["test_mae_hartree"] for r in recs])
    baselines = [r.get("baseline") for r in recs]
    bl = np.array([b if b is not None else np.nan for b in baselines])

    max_cum = args.max_cum_update
    if max_cum is not None:
        m = cum <= max_cum
        if not np.any(m):
            raise SystemExit(f"No records with cum_update <= {max_cum}.")
        cum = cum[m]
        phases = phases[m]
        test_g = test_g[m]
        train_g = train_g[m]
        test_p = test_p[m]
        test_mae = test_mae[m]
        bl = bl[m]

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    title = "Functional selector training metrics"
    if max_cum is not None:
        title += f" (cum_update ≤ {max_cum})"
    fig.suptitle(title, fontsize=13)

    # --- Panel 1: accuracy-style metrics ---
    ax = axes[0, 0]
    ax.plot(cum, test_g, "b-o", markersize=3, label="Test greedy accuracy", linewidth=1.2)
    ax.plot(cum, train_g, "g--s", markersize=2, label="Train greedy accuracy", linewidth=1, alpha=0.8)
    if wu > 0 and (max_cum is None or wu <= max_cum):
        ax.axvline(wu, color="gray", linestyle=":", linewidth=1.5, label="Warmup → REINFORCE")
    ax.set_xlabel("Cumulative update (warmup 1..W, then W+1..W+RL)")
    ax.set_ylabel("Fraction correct (0–1)")
    ax.set_title("Greedy top-1 matches a best functional")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, max(0.55, float(test_g.max()) * 1.15))
    if max_cum is not None:
        ax.set_xlim(left=0, right=max_cum)

    # --- Panel 2: MAE Hartree + kcal twin ---
    ax = axes[0, 1]
    ax.plot(cum, test_mae, "darkred", marker="o", markersize=3, linewidth=1.2, label="Test MAE (greedy)")
    ax.axhline(oracle, color="green", linestyle="--", linewidth=1.5, label=f"Oracle MAE ({oracle:.4g} Ha)")
    ax.axhline(rand_mae, color="orange", linestyle="--", linewidth=1.5, label=f"Uniform random (~{rand_mae:.4g} Ha)")
    if wu > 0 and (max_cum is None or wu <= max_cum):
        ax.axvline(wu, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Cumulative update")
    ax.set_ylabel("MAE (Hartree)")
    ax.set_title("Energy error vs reference (predicted functional)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    if max_cum is not None:
        ax.set_xlim(left=0, right=max_cum)

    # --- Panel 3: probability on optimal functionals ---
    ax = axes[1, 0]
    ax.plot(cum, test_p, "purple", marker="o", markersize=3, linewidth=1.2)
    if wu > 0 and (max_cum is None or wu <= max_cum):
        ax.axvline(wu, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Cumulative update")
    ax.set_ylabel("Total prob on best DFAs (0–1)")
    ax.set_title("Policy mass on tied-lowest-error functionals (test)")
    ax.grid(True, alpha=0.3)
    if max_cum is not None:
        ax.set_xlim(left=0, right=max_cum)

    # --- Panel 4: REINFORCE baseline (reward scale) ---
    ax = axes[1, 1]
    rl_mask = phases == "reinforce"
    if np.any(rl_mask):
        ax.plot(cum[rl_mask], bl[rl_mask], "k-", marker=".", markersize=2, linewidth=1, label="EMA batch-mean reward")
        ax.set_xlabel("Cumulative update")
        ax.set_ylabel("Baseline (~negative typical |error|)")
        ax.set_title("REINFORCE reward baseline (training only)")
        ax.legend(loc="best", fontsize=8)
        if max_cum is not None:
            ax.set_xlim(left=0, right=max_cum)
    else:
        ax.text(0.5, 0.5, "No REINFORCE phase logged", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    suffix = f"_first{max_cum}" if max_cum is not None else ""
    out_main = out_dir / f"training_curves_overview{suffix}.png"
    fig.savefig(out_main, dpi=150)
    plt.close(fig)

    # Second figure: MAE in kcal/mol only (often more readable)
    fig2, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(cum, test_mae * HARTREE_TO_KCAL, "darkred", marker="o", markersize=3, label="Test MAE (greedy)")
    ax.axhline(oracle * HARTREE_TO_KCAL, color="green", linestyle="--", label="Oracle")
    ax.axhline(rand_mae * HARTREE_TO_KCAL, color="orange", linestyle="--", label="Uniform random")
    if wu > 0 and (max_cum is None or wu <= max_cum):
        ax.axvline(wu, color="gray", linestyle=":", label="Warmup end")
    ax.set_xlabel("Cumulative update")
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("Reaction energy error vs reference — kcal/mol")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if max_cum is not None:
        ax.set_xlim(left=0, right=max_cum)
    out_mae = out_dir / f"training_mae_kcal_mol{suffix}.png"
    fig2.savefig(out_mae, dpi=150)
    plt.close(fig2)

    print(f"Saved:\n  {out_main}\n  {out_mae}")


if __name__ == "__main__":
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P(__file__).resolve().parent))
    main()
