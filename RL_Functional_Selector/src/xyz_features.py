"""Fixed-size feature vectors from XYZ geometries (one molecule)."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .reaction_stoichiometry import ReactionSpec

logger = logging.getLogger(__name__)

# Elements we count explicitly (extend if needed for your subsets)
_ELEM_ORDER = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
)


def extract_xyz_features(xyz_path: str | Path) -> Optional[np.ndarray]:
    """
    Return a 1D float vector:
      [log1p(n_atoms), rg, max_dist, min_nonzero_dist, count_elem_0/K, ...]
    """
    xyz_path = Path(xyz_path)
    try:
        lines = xyz_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        logger.warning("Cannot read %s: %s", xyz_path, e)
        return None

    if len(lines) < 3:
        return None

    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None

    coords: List[np.ndarray] = []
    symbols: List[str] = []
    for i in range(2, min(2 + n_atoms, len(lines))):
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))

    if not coords:
        return None

    coords_arr = np.stack(coords, axis=0)
    n = float(len(coords_arr))
    com = coords_arr.mean(axis=0)
    dists = np.linalg.norm(coords_arr - com, axis=1)
    rg = float(np.sqrt(np.mean(dists**2)))
    pd = np.linalg.norm(coords_arr[:, None, :] - coords_arr[None, :, :], axis=-1)
    triu = np.triu_indices(len(coords_arr), k=1)
    pair_d = pd[triu]
    max_d = float(pair_d.max()) if len(pair_d) else 0.0
    pos = pair_d[pair_d > 1e-8]
    min_d = float(pos.min()) if len(pos) else 0.0

    counts = Counter(symbols)
    K = len(_ELEM_ORDER)
    elem_frac = np.zeros(K, dtype=np.float64)
    for j, el in enumerate(_ELEM_ORDER):
        if counts.get(el):
            elem_frac[j] = counts[el] / n

    feat = np.concatenate(
        [
            np.array(
                [np.log1p(n), rg, max_d, min_d, n],
                dtype=np.float64,
            ),
            elem_frac,
        ]
    )
    return feat


def aggregate_side_features(
    molecule_ids: List[str],
    xyz_dir: Path,
    coeffs: Optional[List[float]] = None,
    require_all: bool = True,
) -> Optional[np.ndarray]:
    """
    Mean-pool per-molecule features on one side (reactants or products).
    If coeffs is given, use abs(coeff) as weights for the mean.
    """
    if not molecule_ids:
        return np.zeros(_side_feature_dim(), dtype=np.float64)

    vecs = []
    weights = []
    for i, mid in enumerate(molecule_ids):
        p = xyz_dir / f"{mid}.xyz"
        v = extract_xyz_features(p)
        if v is None:
            if require_all:
                return None
            continue
        vecs.append(v)
        w = abs(float(coeffs[i])) if coeffs is not None else 1.0
        weights.append(w)

    if not vecs:
        return None if require_all else np.zeros(_side_feature_dim(), dtype=np.float64)

    V = np.stack(vecs, axis=0)
    w = np.array(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    return (V * w[:, None]).sum(axis=0)


def _side_feature_dim() -> int:
    return 5 + len(_ELEM_ORDER)


def reaction_state_vector(spec: "ReactionSpec", xyz_dir: str | Path) -> Optional[np.ndarray]:
    """Concatenate [reactant_agg | product_agg | n_react | n_prod]."""
    xyz_dir = Path(xyz_dir)
    r_ids = [t.molecule_id for t in spec.reactants]
    r_c = [t.coefficient for t in spec.reactants]
    p_ids = [t.molecule_id for t in spec.products]
    p_c = [t.coefficient for t in spec.products]

    fr = aggregate_side_features(r_ids, xyz_dir, r_c, require_all=True)
    fp = aggregate_side_features(p_ids, xyz_dir, p_c, require_all=True)
    if fr is None or fp is None:
        return None
    meta = np.array(
        [len(r_ids), len(p_ids), len(spec.terms)],
        dtype=np.float64,
    )
    return np.concatenate([fr, fp, meta], axis=0)


def state_feature_dim() -> int:
    return _side_feature_dim() * 2 + 3


def aggregate_paths_features(
    xyz_paths: List[str | Path],
    weights: Optional[List[float]] = None,
) -> Optional[np.ndarray]:
    """Pool features from explicit XYZ paths (user-provided reactants/products)."""
    if not xyz_paths:
        return np.zeros(_side_feature_dim(), dtype=np.float64)
    vecs: List[np.ndarray] = []
    ws: List[float] = []
    for i, p in enumerate(xyz_paths):
        v = extract_xyz_features(p)
        if v is None:
            return None
        vecs.append(v)
        if weights is not None:
            ws.append(abs(float(weights[i])))
        else:
            ws.append(1.0)
    V = np.stack(vecs, axis=0)
    w = np.array(ws, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    return (V * w[:, None]).sum(axis=0)


def state_from_xyz_path_lists(
    reactant_xyz: List[str | Path],
    product_xyz: List[str | Path],
    reactant_weights: Optional[List[float]] = None,
    product_weights: Optional[List[float]] = None,
) -> Optional[np.ndarray]:
    fr = aggregate_paths_features(reactant_xyz, reactant_weights)
    fp = aggregate_paths_features(product_xyz, product_weights)
    if fr is None or fp is None:
        return None
    meta = np.array(
        [len(reactant_xyz), len(product_xyz), len(reactant_xyz) + len(product_xyz)],
        dtype=np.float64,
    )
    return np.concatenate([fr, fp, meta], axis=0)
