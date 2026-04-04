"""
Rank DFAs for one reaction from geometry (reactant/product XYZ) using a trained REINFORCE policy.

Scores = softmax policy probabilities (higher = more recommended). Also prints oracle order
by absolute error when GSCDB Reaction_Energies.csv and reaction id are available.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from .reaction_stoichiometry import load_reaction_table
    from .reinforce_policy import ReinforcePolicyAgent
    from .xyz_features import reaction_state_vector, state_from_xyz_path_lists
    from .reaction_dataset import normalize_state
except ImportError:
    from reaction_stoichiometry import load_reaction_table
    from reinforce_policy import ReinforcePolicyAgent
    from xyz_features import reaction_state_vector, state_from_xyz_path_lists
    from reaction_dataset import normalize_state

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_meta_and_policy(
    model_path: Path,
    meta_path: Optional[Path],
) -> Tuple[ReinforcePolicyAgent, np.ndarray, np.ndarray, List[str]]:
    if meta_path is None:
        meta_path = model_path.parent / "reinforce_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    mean = np.array(meta["mean"], dtype=np.float64)
    std = np.array(meta["std"], dtype=np.float64)
    names = list(meta["functional_names"])
    agent = ReinforcePolicyAgent.load(model_path)
    return agent, mean, std, names


def oracle_errors_for_reaction(
    gscdb_root: Path,
    reaction_id: str,
    functional_order: List[str],
) -> Optional[Tuple[np.ndarray, float]]:
    eval_csv = gscdb_root / "Info" / "DatasetEval.csv"
    rxe_csv = gscdb_root / "Analysis" / "Reaction_Energies.csv"
    _, _, rxe = load_reaction_table(eval_csv, rxe_csv)
    row = rxe.loc[rxe["Reaction"] == reaction_id]
    if row.empty:
        return None
    ref = float(row["Reference"].values[0])
    missing = [c for c in functional_order if c not in row.columns]
    if missing:
        return None
    vals = row[functional_order].values.astype(np.float64).flatten()
    err = np.abs(vals - ref)
    return err, ref


def print_ranking(
    functional_names: List[str],
    scores: np.ndarray,
    title: str,
    errors: Optional[np.ndarray] = None,
) -> None:
    order = np.argsort(-scores)
    print(title)
    print(f"{'rank':>4}  {'functional':<22}  {'score':>12}" + ("  abs_err" if errors is not None else ""))
    for r, j in enumerate(order, start=1):
        line = f"{r:4d}  {functional_names[j]:<22}  {scores[j]:12.6f}"
        if errors is not None:
            line += f"  {errors[j]:.6f}"
        print(line)


def main(argv=None):
    root = _project_root()
    p = argparse.ArgumentParser(description="Rank functionals for a reaction (REINFORCE policy).")
    p.add_argument(
        "--model",
        type=str,
        default=str(root / "RL_Functional_Selector" / "models" / "reaction_reinforce" / "reinforce_policy_final.pkl"),
    )
    p.add_argument("--meta", type=str, default=None)
    p.add_argument("--gscdb-root", type=str, default=str(root / "GSCDB"))

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--reaction-id", type=str, help="GSCDB reaction id, e.g. 3B-69_1")
    g.add_argument("--reactant-xyz", nargs="+", help="Paths to reactant .xyz files")
    p.add_argument("--product-xyz", nargs="+", help="Required with --reactant-xyz")

    args = p.parse_args(argv)
    model_path = Path(args.model)
    meta_path = Path(args.meta) if args.meta else None
    agent, mean, std, functional_names = load_meta_and_policy(model_path, meta_path)

    gscdb = Path(args.gscdb_root)
    xyz_dir = gscdb / "xyz_files"

    if args.reaction_id:
        eval_csv = gscdb / "Info" / "DatasetEval.csv"
        rxe_csv = gscdb / "Analysis" / "Reaction_Energies.csv"
        specs, _, _ = load_reaction_table(eval_csv, rxe_csv)
        spec = next((s for s in specs if s.reaction_id == args.reaction_id), None)
        if spec is None:
            raise SystemExit(f"Unknown reaction id: {args.reaction_id}")
        raw = reaction_state_vector(spec, xyz_dir)
        if raw is None:
            raise SystemExit("Could not build state (missing XYZ for a species in this reaction).")
        oracle = oracle_errors_for_reaction(gscdb, args.reaction_id, functional_names)
    else:
        if not args.product_xyz:
            p.error("--product-xyz is required with --reactant-xyz")
        raw = state_from_xyz_path_lists(args.reactant_xyz, args.product_xyz)
        if raw is None:
            raise SystemExit("Failed to read one or more XYZ files.")
        oracle = None

    s = normalize_state(raw.reshape(1, -1), mean, std)[0]
    probs = agent.action_probs(s)

    print_ranking(functional_names, probs, "\nREINFORCE policy scores (best = highest probability):\n")
    if oracle is not None:
        oerr, ref = oracle
        order_truth = np.argsort(oerr)
        print("\nOracle by absolute error vs reference (best = smallest error):")
        print(f"(reference reaction energy = {ref:.8f})\n")
        print(f"{'rank':>4}  {'functional':<22}  {'abs_err':>12}")
        for r, j in enumerate(order_truth, start=1):
            print(f"{r:4d}  {functional_names[j]:<22}  {oerr[j]:12.6f}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
