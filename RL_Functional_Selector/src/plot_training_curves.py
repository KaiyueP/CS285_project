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


def _tight_ylim(*series: np.ndarray, pad_frac: float = 0.12, floor: float = 0.0, ceil: float = 1.0) -> tuple[float, float]:
    values = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return floor, ceil
    lo = float(values.min())
    hi = float(values.max())
    span = max(hi - lo, 0.02)
    lo = max(floor, lo - pad_frac * span)
    hi = min(ceil, hi + pad_frac * span)
    if hi - lo < 0.05:
        mid = 0.5 * (lo + hi)
        lo = max(floor, mid - 0.025)
        hi = min(ceil, mid + 0.025)
    return lo, hi


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
    unit = data.get("energy_unit", "kcal/mol")
    wu = int(data.get("warmup_supervised") or 0)
    emae_n = int(data.get("emae_steps") or 0)

    cum = np.array([r["cum_update"] for r in recs])
    phases = np.array([r["phase"] for r in recs])
    test_g = np.array([r["test_greedy"] for r in recs])
    train_g = np.array([r["train_greedy"] for r in recs])
    test_p = np.array([r["test_prob_on_best"] for r in recs])
    test_regret = np.array([r.get("test_regret", np.nan) for r in recs], dtype=float)
    test_top3 = np.array([r.get("test_top3_hit", np.nan) for r in recs], dtype=float)
    train_mae = np.array([r.get("train_mae_energy", np.nan) for r in recs], dtype=float)
    test_mae = np.array([r.get("test_mae_energy", r.get("test_mae_hartree")) for r in recs], dtype=float)
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
        test_regret = test_regret[m]
        test_top3 = test_top3[m]
        train_mae = train_mae[m]
        test_mae = test_mae[m]
        bl = bl[m]

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    font = {
        "title": 13,
        "panel_title": 13,
        "label": 13,
        "tick": 13,
        "legend": 10,
        "annotation": 10,
    }
    plt.rcParams.update(
        {
            "font.size": font["tick"],
            "axes.titlesize": font["panel_title"],
            "axes.labelsize": font["label"],
            "xtick.labelsize": font["tick"],
            "ytick.labelsize": font["tick"],
            "legend.fontsize": font["legend"],
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.06,
        h_pad=0.06,
        wspace=0.05,
        hspace=0.05,
    )
    # title = "Functional selector training metrics"
    # if max_cum is not None:
    #     title += f" (cum_update ≤ {max_cum})"
    # fig.suptitle(title, fontsize=font["title"])

    # --- Panel 1: accuracy-style metrics ---
    ax = axes[0, 0]
    ax.plot(cum, test_g, "b-o", markersize=3, label="Test greedy accuracy", linewidth=1.2)
    ax.plot(cum, train_g, "g--s", markersize=2, label="Train greedy accuracy", linewidth=1, alpha=0.8)
    if wu > 0 and (max_cum is None or wu <= max_cum):
        ax.axvline(wu, color="gray", linestyle=":", linewidth=1.5, label="CE warmup end")
    if emae_n > 0 and (max_cum is None or wu + emae_n <= max_cum):
        ax.axvline(wu + emae_n, color="olive", linestyle=":", linewidth=1.5, label="EMAE end → REINFORCE")
    ax.set_xlabel("Cumulative update")
    ax.set_ylabel("Fraction correct")
    ax.set_title("Greedy top-1")
    ax.legend(loc="best", fontsize=font["legend"])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(*_tight_ylim(test_g, train_g, floor=0.0, ceil=1.0))
    if max_cum is not None:
        ax.set_xlim(left=0, right=max_cum)

    # --- Panel 2: MAE in dataset units ---
    ax = axes[0, 1]
    ax.plot(cum, test_mae, "darkred", marker="o", markersize=3, linewidth=1.2, label="Test MAE (greedy)")
    ax.axhline(oracle, color="green", linestyle="--", linewidth=1.5, label=f"Oracle MAE ({oracle:.4g} {unit})")
    ax.axhline(rand_mae, color="orange", linestyle="--", linewidth=1.5, label=f"Uniform random (~{rand_mae:.4g} {unit})")
    if wu > 0 and (max_cum is None or wu <= max_cum):
        ax.axvline(wu, color="gray", linestyle=":", linewidth=1.5)
    if emae_n > 0 and (max_cum is None or wu + emae_n <= max_cum):
        ax.axvline(wu + emae_n, color="olive", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Cumulative update")
    ax.set_ylabel(f"MAE ({unit})")
    ax.set_title("Energy error")
    ax.legend(loc="best", fontsize=font["legend"])
    ax.grid(True, alpha=0.3)
    if max_cum is not None:
        ax.set_xlim(left=0, right=max_cum)
    # Annotate latest test MAE so the exact number is visible on the plot.
    latest_te_mae = float(test_mae[-1])
    latest_tr_mae = float(train_mae[-1]) if np.isfinite(train_mae[-1]) else float("nan")
    mae_text = f"Latest test MAE: {latest_te_mae:.4f} {unit}"
    if np.isfinite(latest_tr_mae):
        mae_text += f"\nLatest train MAE: {latest_tr_mae:.4f} {unit}"
    ax.text(
        0.98,
        0.02,
        mae_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=font["annotation"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
    )

    # --- Panel 3: RL quality metrics ---
    ax = axes[1, 0]
    plotted_any = False
    if np.isfinite(test_regret).any():
        ax.plot(cum, test_regret, "purple", marker="o", markersize=3, linewidth=1.2, label="Test regret (lower better)")
        plotted_any = True
    if np.isfinite(test_top3).any():
        ax.plot(cum, test_top3, "teal", marker="s", markersize=3, linewidth=1.2, label="Test top3 hit (higher better)")
        plotted_any = True
    # keep previous metric for continuity
    ax.plot(cum, test_p, "gray", marker=".", markersize=2, linewidth=1.0, alpha=0.7, label="Test prob on best")
    if wu > 0 and (max_cum is None or wu <= max_cum):
        ax.axvline(wu, color="gray", linestyle=":", linewidth=1.5)
    if emae_n > 0 and (max_cum is None or wu + emae_n <= max_cum):
        ax.axvline(wu + emae_n, color="olive", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Cumulative update")
    ax.set_ylabel("Metric value")
    ax.set_title("RL quality metrics")
    if plotted_any:
        ax.legend(loc="best", fontsize=font["legend"])
    ax.grid(True, alpha=0.3)
    if max_cum is not None:
        ax.set_xlim(left=0, right=max_cum)
    ax.set_ylim(*_tight_ylim(test_regret, test_top3, test_p, floor=0.0, ceil=1.0))

    # --- Panel 4: REINFORCE baseline (reward scale) ---
    ax = axes[1, 1]
    rl_mask = phases == "reinforce"
    if np.any(rl_mask):
        ax.plot(cum[rl_mask], bl[rl_mask], "k-", marker=".", markersize=2, linewidth=1, label="EMA batch-mean reward")
        ax.set_xlabel("Cumulative update")
        ax.set_ylabel("Baseline")
        ax.set_title("REINFORCE reward baseline")
        ax.legend(loc="best", fontsize=font["legend"])
        if max_cum is not None:
            ax.set_xlim(left=0, right=max_cum)
    else:
        ax.text(0.5, 0.5, "No REINFORCE phase logged", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    suffix = f"_first{max_cum}" if max_cum is not None else ""
    out_main = out_dir / f"training_curves_overview{suffix}.png"
    fig.savefig(out_main, dpi=150)
    plt.close(fig)

    # Second figure: MAE only (same unit as dataset)
    fig2, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.plot(cum, test_mae, "darkred", marker="o", markersize=3, label="Test MAE (greedy)")
    ax.axhline(oracle, color="green", linestyle="--", label="Oracle")
    ax.axhline(rand_mae, color="orange", linestyle="--", label="Uniform random")
    if wu > 0 and (max_cum is None or wu <= max_cum):
        ax.axvline(wu, color="gray", linestyle=":", label="CE warmup end")
    if emae_n > 0 and (max_cum is None or wu + emae_n <= max_cum):
        ax.axvline(wu + emae_n, color="olive", linestyle=":", label="EMAE end")
    ax.set_xlabel("Cumulative update", fontsize=16)
    ax.set_ylabel(f"MAE ({unit})", fontsize=16)
    ax.set_title("Reaction energy error vs reference", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    if max_cum is not None:
        ax.set_xlim(left=0, right=max_cum)
    latest_te_mae = float(test_mae[-1])
    latest_tr_mae = float(train_mae[-1]) if np.isfinite(train_mae[-1]) else float("nan")
    latest_te_regret = float(test_regret[-1]) if np.isfinite(test_regret[-1]) else float("nan")
    latest_te_top3 = float(test_top3[-1]) if np.isfinite(test_top3[-1]) else float("nan")
    mae_text = f"Latest test MAE: {latest_te_mae:.4f} {unit}"
    if np.isfinite(latest_tr_mae):
        mae_text += f"\nLatest train MAE: {latest_tr_mae:.4f} {unit}"
    if np.isfinite(latest_te_regret):
        mae_text += f"\nLatest test regret: {latest_te_regret:.4f}"
    if np.isfinite(latest_te_top3):
        mae_text += f"\nLatest test top3 hit: {latest_te_top3:.4f}"
    ax.text(
        0.98,
        0.02,
        mae_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
    )
    out_mae = out_dir / f"training_mae_energy{suffix}.png"
    fig2.savefig(out_mae, dpi=150)
    plt.close(fig2)

    print(f"Saved:\n  {out_main}\n  {out_mae}")


if __name__ == "__main__":
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P(__file__).resolve().parent))
    main()
