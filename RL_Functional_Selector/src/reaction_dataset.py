"""Build state tensors and per-functional absolute errors for all reactions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from .reaction_stoichiometry import ReactionSpec, load_reaction_table
    from .xyz_features import reaction_state_vector, state_feature_dim
except ImportError:
    from reaction_stoichiometry import ReactionSpec, load_reaction_table
    from xyz_features import reaction_state_vector, state_feature_dim

logger = logging.getLogger(__name__)


def ensure_gscdb_root(gscdb_root: str | Path) -> Path:
    """Resolve path and verify GSCDB layout; raise FileNotFoundError with hints if invalid."""
    root = Path(gscdb_root).expanduser()
    try:
        root = root.resolve()
    except OSError:
        pass

    if not root.is_dir():
        raise FileNotFoundError(
            f"GSCDB path is not a directory (or does not exist): {gscdb_root}\n\n"
            "Do not use the docs placeholder 'path/to/GSCDB'. Use your real tree, e.g.\n"
            "  from RL_Functional_Selector/src:  ../../GSCDB\n"
            "  WSL:  /mnt/e/course/CS285/project/GSCDB"
        )

    eval_csv = root / "Info" / "DatasetEval.csv"
    rxe_csv = root / "Analysis" / "Reaction_Energies.csv"
    xyz_dir = root / "xyz_files"
    missing = [p for p in (eval_csv, rxe_csv, xyz_dir) if not p.exists()]
    if missing:
        hint = ""
        nested = root / "GSCDB"
        if nested.is_dir():
            ne, nr, nx = nested / "Info" / "DatasetEval.csv", nested / "Analysis" / "Reaction_Energies.csv", nested / "xyz_files"
            if ne.is_file() and nr.is_file() and nx.is_dir():
                hint = (
                    f"\n\nHint: GSCDB data is inside a subdirectory. Try:\n  --gscdb-root {nested}"
                    "\n  (from RL_Functional_Selector/src that is often: ../../GSCDB)"
                )
        raise FileNotFoundError(
            "Folder does not look like GSCDB (need Info/DatasetEval.csv, "
            "Analysis/Reaction_Energies.csv, xyz_files/). Missing:\n  "
            + "\n  ".join(str(m) for m in missing)
            + f"\n\nYou gave: {root}"
            + hint
        )
    return root


def build_reaction_arrays(
    gscdb_root: str | Path,
    max_reactions: Optional[int] = None,
    skip_missing_xyz: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[ReactionSpec]]:
    """
    Returns:
        X_raw: (N, state_dim) unnormalized features (fit mean/std on train only)
        errors: (N, n_func) absolute error vs reference per functional
        functional_names, reaction_ids, specs (aligned rows)
    """
    gscdb_root = ensure_gscdb_root(gscdb_root)
    logger.info("GSCDB root: %s", gscdb_root)
    eval_csv = gscdb_root / "Info" / "DatasetEval.csv"
    rxe_csv = gscdb_root / "Analysis" / "Reaction_Energies.csv"
    xyz_dir = gscdb_root / "xyz_files"

    specs, functional_names, rxe_df = load_reaction_table(eval_csv, rxe_csv)
    ref = rxe_df["Reference"].values.astype(np.float64)
    func_vals = rxe_df[list(functional_names)].values.astype(np.float64)
    errors = np.abs(func_vals - ref[:, None])

    X_list: List[np.ndarray] = []
    err_list: List[np.ndarray] = []
    ids: List[str] = []
    kept_specs: List[ReactionSpec] = []

    for i, spec in enumerate(specs):
        if max_reactions is not None and len(X_list) >= max_reactions:
            break
        x = reaction_state_vector(spec, xyz_dir)
        if x is None:
            if skip_missing_xyz:
                logger.debug("Skip %s: missing/invalid XYZ for some species", spec.reaction_id)
                continue
            raise RuntimeError(f"Missing XYZ for reaction {spec.reaction_id}")
        X_list.append(x)
        err_list.append(errors[i])
        ids.append(spec.reaction_id)
        kept_specs.append(spec)

    if not X_list:
        raise RuntimeError("No reactions with complete XYZ; check GSCDB paths.")

    X = np.stack(X_list, axis=0)
    E = np.stack(err_list, axis=0)

    return X, E, functional_names, ids, kept_specs


def normalize_state(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def feature_dim() -> int:
    return state_feature_dim()
