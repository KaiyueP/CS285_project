"""
Train REINFORCE policy on GSCDB reactions.

Training recipe (improved vs single-sample REINFORCE):
1) Optional supervised warm-start: cross-entropy to the true best functional (argmin error).
2) Batched REINFORCE with advantages centered per batch (lower variance).
3) Logs greedy hit-rate and mean probability mass on tied-best functionals (smoother signal).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .reaction_dataset import build_reaction_arrays, normalize_state
    from .reinforce_policy import ReinforcePolicyAgent
except ImportError:
    from reaction_dataset import build_reaction_arrays, normalize_state
    from reinforce_policy import ReinforcePolicyAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _project_defaults():
    root = Path(__file__).resolve().parents[2]
    return {
        "gscdb": root / "GSCDB",
        "out": Path(__file__).resolve().parents[1] / "models",
    }


def eval_greedy_accuracy(agent: ReinforcePolicyAgent, X: np.ndarray, errors: np.ndarray) -> float:
    correct = 0
    for i in range(len(X)):
        a = agent.greedy_action(X[i])
        if errors[i, a] <= errors[i].min() + 1e-12:
            correct += 1
    return correct / max(1, len(X))


def eval_mean_prob_on_best(agent: ReinforcePolicyAgent, X: np.ndarray, errors: np.ndarray) -> float:
    """Average total policy probability placed on functionals tied for lowest error (0–1)."""
    total = 0.0
    for i in range(len(X)):
        p = agent.action_probs(X[i])
        emin = errors[i].min()
        mask = errors[i] <= emin + 1e-12
        total += float(p[mask].sum())
    return total / max(1, len(X))


def eval_mae_under_greedy(agent: ReinforcePolicyAgent, X: np.ndarray, errors: np.ndarray) -> float:
    """
    Mean absolute error (reaction energy) vs reference when using the greedy-predicted functional.

    errors[i, j] = |E_ij - E_ref_i| already, so MAE = mean_i errors[i, greedy(i)].
    Units: Hartree.
    """
    if len(X) == 0:
        return float("nan")
    vals = []
    for i in range(len(X)):
        a = agent.greedy_action(X[i])
        vals.append(float(errors[i, a]))
    return float(np.mean(vals))


def eval_mae_oracle(errors: np.ndarray) -> float:
    """MAE if we always picked a functional with lowest |E - E_ref| (best possible among listed DFAs)."""
    if len(errors) == 0:
        return float("nan")
    return float(np.mean(errors.min(axis=1)))


def eval_mae_uniform_random(errors: np.ndarray, rng: np.random.Generator) -> float:
    """MAE when choosing a functional uniformly at random (Monte Carlo estimate)."""
    n, k = errors.shape
    if n == 0:
        return float("nan")
    j = rng.integers(0, k, size=n)
    return float(np.mean(errors[np.arange(n), j]))


def train(
    gscdb_root: Path,
    out_dir: Path,
    max_reactions: Optional[int],
    test_fraction: float,
    steps: int,
    seed: int,
    hidden: tuple,
    lr: float,
    entropy_coef: float,
    baseline_momentum: float,
    warmup_supervised: int,
    batch_size: int,
    log_interval: int,
):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    X, errors, functional_names, reaction_ids, _ = build_reaction_arrays(
        gscdb_root, max_reactions=max_reactions, skip_missing_xyz=True
    )
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_test = max(1, int(n * test_fraction))
    te = idx[:n_test]
    tr = idx[n_test:]
    if len(tr) < 10:
        tr = idx
        te = idx[: min(n_test, n)]

    mean = X[tr].mean(axis=0, keepdims=True)
    std = X[tr].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    X_tr = normalize_state(X[tr], mean, std)
    X_te = normalize_state(X[te], mean, std)
    err_tr = errors[tr]
    err_te = errors[te]

    te_mae_oracle = eval_mae_oracle(err_te)
    te_mae_rand = eval_mae_uniform_random(err_te, rng)
    logger.info(
        "Energy MAE vs reference on TEST (Hartree): oracle=%.6g  uniform_random=%.6g",
        te_mae_oracle,
        te_mae_rand,
    )

    n_actions = len(functional_names)
    best_tr = np.argmin(err_tr, axis=1)

    agent = ReinforcePolicyAgent(
        state_dim=X.shape[1],
        n_actions=n_actions,
        hidden_layers=hidden,
        learning_rate=lr,
        entropy_coef=entropy_coef,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    history: dict = {
        "te_mae_oracle": te_mae_oracle,
        "te_mae_uniform_random": te_mae_rand,
        "warmup_supervised": warmup_supervised,
        "reinforce_steps": steps,
        "records": [],
    }

    def _record(
        phase: str,
        step_in_phase: int,
        tr_acc: float,
        te_acc: float,
        te_p: float,
        te_mae: float,
        baseline_val: Optional[float],
    ) -> None:
        cum = step_in_phase
        if phase == "reinforce":
            cum = warmup_supervised + step_in_phase
        history["records"].append(
            {
                "phase": phase,
                "step_in_phase": step_in_phase,
                "cum_update": cum,
                "train_greedy": tr_acc,
                "test_greedy": te_acc,
                "test_prob_on_best": te_p,
                "test_mae_hartree": te_mae,
                "baseline": baseline_val,
            }
        )

    # --- Supervised warm-start (strong signal; fixes flat greedy acc from random init) ---
    if warmup_supervised > 0:
        logger.info(
            "Supervised warm-start: %d steps, batch=%d (predict argmin-error functional)",
            warmup_supervised,
            batch_size,
        )
        for u in range(warmup_supervised):
            bi = np.random.choice(len(tr), size=min(batch_size, len(tr)), replace=False)
            agent.supervised_cross_entropy_batch(X_tr[bi], best_tr[bi])
            if (u + 1) % log_interval == 0 or u == 0:
                tr_acc = eval_greedy_accuracy(agent, X_tr, err_tr)
                te_acc = eval_greedy_accuracy(agent, X_te, err_te)
                te_p = eval_mean_prob_on_best(agent, X_te, err_te)
                te_mae = eval_mae_under_greedy(agent, X_te, err_te)
                logger.info(
                    "warmup %d/%d  train_greedy=%.4f  test_greedy=%.4f  test_prob_on_best=%.4f  "
                    "test_mae_energy=%.6g",
                    u + 1,
                    warmup_supervised,
                    tr_acc,
                    te_acc,
                    te_p,
                    te_mae,
                )
                _record("warmup", u + 1, tr_acc, te_acc, te_p, te_mae, None)

    baseline = 0.0
    best_te_acc = -1.0

    # --- Batched REINFORCE ---
    logger.info(
        "REINFORCE: %d steps, batch=%d, entropy_coef=%s, baseline_momentum=%.3f",
        steps,
        batch_size,
        entropy_coef,
        baseline_momentum,
    )

    for t in range(steps):
        bi = np.random.choice(len(tr), size=min(batch_size, len(tr)), replace=False)
        states = X_tr[bi]
        actions = np.zeros(len(bi), dtype=np.int64)
        rewards = np.zeros(len(bi), dtype=np.float64)
        for j, idx_i in enumerate(bi):
            a, _, _ = agent.sample_action(states[j])
            actions[j] = a
            rewards[j] = -float(err_tr[idx_i, a])

        batch_mean_r = float(rewards.mean())
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * batch_mean_r
        # Batch-centered advantages (zero mean per batch → lower-variance REINFORCE)
        adv = rewards - rewards.mean()

        agent.reinforce_batch(states, actions, adv)

        if (t + 1) % log_interval == 0 or t == 0:
            tr_acc = eval_greedy_accuracy(agent, X_tr, err_tr)
            te_acc = eval_greedy_accuracy(agent, X_te, err_te)
            te_p = eval_mean_prob_on_best(agent, X_te, err_te)
            te_mae = eval_mae_under_greedy(agent, X_te, err_te)
            logger.info(
                "step %d  baseline=%.5f  train_greedy=%.4f  test_greedy=%.4f  test_prob_on_best=%.4f  "
                "test_mae_energy=%.6g",
                t + 1,
                baseline,
                tr_acc,
                te_acc,
                te_p,
                te_mae,
            )
            if te_acc > best_te_acc:
                best_te_acc = te_acc
                agent.save(out_dir / "reinforce_policy_best.pkl")
            _record("reinforce", t + 1, tr_acc, te_acc, te_p, te_mae, baseline)

    agent.save(out_dir / "reinforce_policy_final.pkl")

    final_te_mae = eval_mae_under_greedy(agent, X_te, err_te)
    logger.info(
        "Final TEST energy MAE (greedy predicted functional): %.6g Hartree  "
        "(oracle lower bound %.6g, uniform-random ~%.6g)",
        final_te_mae,
        te_mae_oracle,
        te_mae_rand,
    )

    meta = {
        "functional_names": functional_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "state_dim": int(X.shape[1]),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "reaction_ids_train": [reaction_ids[j] for j in tr.tolist()],
        "reaction_ids_test": [reaction_ids[j] for j in te.tolist()],
        "seed": seed,
        "steps": steps,
        "warmup_supervised": warmup_supervised,
        "batch_size": batch_size,
        "test_mae_energy_greedy_final": final_te_mae,
        "test_mae_energy_oracle": te_mae_oracle,
        "test_mae_energy_uniform_random": te_mae_rand,
    }
    with open(out_dir / "reinforce_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    hist_path = out_dir / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Wrote %s, meta.json, and %s", out_dir / "reinforce_policy_final.pkl", hist_path)


def main(argv=None):
    d = _project_defaults()
    p = argparse.ArgumentParser(description="Train REINFORCE functional selector on GSCDB reactions.")
    p.add_argument(
        "--gscdb-root",
        type=str,
        default=str(d["gscdb"]),
        help="Directory containing GSCDB Info/, Analysis/, xyz_files/ (not the literal path/to/GSCDB).",
    )
    p.add_argument("--out-dir", type=str, default=str(d["out"] / "reaction_reinforce"))
    p.add_argument("--max-reactions", type=int, default=None)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--baseline-momentum", type=float, default=0.95)
    p.add_argument(
        "--warmup-supervised",
        type=int,
        default=2000,
        help="SGD steps of cross-entropy to true best functional before REINFORCE (0 to disable).",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--log-interval", type=int, default=500)
    args = p.parse_args(argv)

    train(
        gscdb_root=Path(args.gscdb_root),
        out_dir=Path(args.out_dir),
        max_reactions=args.max_reactions,
        test_fraction=args.test_fraction,
        steps=args.steps,
        seed=args.seed,
        hidden=tuple(args.hidden),
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        baseline_momentum=args.baseline_momentum,
        warmup_supervised=args.warmup_supervised,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
