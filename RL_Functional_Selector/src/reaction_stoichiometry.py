"""
Parse GSCDB DatasetEval stoichiometry into reactant / product molecule IDs.

Convention (matches typical energy-difference bookkeeping in GSCDB):
  Δ = sum_i c_i E_i  with c_i > 0  ->  "products" side
                          c_i < 0  ->  "reactants" side
So reactants are molecules whose stoichiometric coefficient is negative.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MoleculeRole:
    """One species in the reaction linear combination."""

    molecule_id: str
    coefficient: float
    role: str  # "reactant" | "product"


@dataclass
class ReactionSpec:
    reaction_id: str
    dataset: str
    reference: float
    terms: List[MoleculeRole] = field(default_factory=list)

    @property
    def reactants(self) -> List[MoleculeRole]:
        return [t for t in self.terms if t.role == "reactant"]

    @property
    def products(self) -> List[MoleculeRole]:
        return [t for t in self.terms if t.role == "product"]


def parse_stoichiometry(stoich_str: str) -> List[Tuple[float, str]]:
    """
    Parse the quoted CSV field: "1,mol_a,-1,mol_b,..."
    Returns list of (coefficient, molecule_id).
    """
    s = stoich_str.strip().strip('"')
    if not s:
        return []
    parts = s.split(",")
    out: List[Tuple[float, str]] = []
    i = 0
    while i + 1 < len(parts):
        coeff_raw = parts[i].strip()
        mol_id = parts[i + 1].strip()
        try:
            c = float(coeff_raw)
        except ValueError:
            try:
                c = float(ast.literal_eval(coeff_raw))
            except (SyntaxError, ValueError):
                logger.warning("Bad coefficient %r in stoichiometry", coeff_raw)
                i += 2
                continue
        out.append((c, mol_id))
        i += 2
    return out


def terms_from_stoichiometry(stoich_str: str) -> List[MoleculeRole]:
    terms: List[MoleculeRole] = []
    for c, mid in parse_stoichiometry(stoich_str):
        if c < 0:
            role = "reactant"
        elif c > 0:
            role = "product"
        else:
            continue
        terms.append(MoleculeRole(molecule_id=mid, coefficient=c, role=role))
    return terms


def load_reaction_table(
    dataset_eval_csv: str | Path,
    reaction_energies_csv: str | Path,
) -> Tuple[List[ReactionSpec], List[str], pd.DataFrame]:
    """
    Merge DatasetEval definitions with per-functional reaction energies + reference.

    Returns:
        specs: ordered list of ReactionSpec
        functional_names: column names for DFAs (same order as error matrix)
        reaction_df: aligned DataFrame with columns [reaction_id, dataset, reference, *functionals]
    """
    dataset_eval_csv = Path(dataset_eval_csv)
    reaction_energies_csv = Path(reaction_energies_csv)

    eval_df = pd.read_csv(dataset_eval_csv)
    if "Stoichiometry" not in eval_df.columns:
        raise ValueError(f"{dataset_eval_csv} must contain Stoichiometry column")

    rxe = pd.read_csv(reaction_energies_csv)
    meta = {"Reaction", "Dataset", "Reference"}
    functional_names = [c for c in rxe.columns if c not in meta]

    rxe = rxe.set_index("Reaction", drop=False)
    specs: List[ReactionSpec] = []

    for _, row in eval_df.iterrows():
        rid = str(row["Reaction"])
        if rid not in rxe.index:
            logger.debug("Skipping %s: not in Reaction_Energies.csv", rid)
            continue
        ref_eval = float(row["Reference"])
        ref_rx = float(rxe.loc[rid, "Reference"])
        if abs(ref_eval - ref_rx) > 1e-3:
            logger.warning(
                "Reference mismatch for %s: DatasetEval=%s Reaction_Energies=%s (using Reaction_Energies)",
                rid,
                ref_eval,
                ref_rx,
            )
        terms = terms_from_stoichiometry(str(row["Stoichiometry"]))
        specs.append(
            ReactionSpec(
                reaction_id=rid,
                dataset=str(row["Dataset"]),
                reference=ref_rx,
                terms=terms,
            )
        )

    aligned = rxe.loc[[s.reaction_id for s in specs]].reset_index(drop=True)
    return specs, functional_names, aligned


def write_roles_manifest(spec: ReactionSpec, path: str | Path) -> None:
    """Optional: save human-readable reactant/product list for one reaction."""
    path = Path(path)
    lines = [
        f"reaction_id: {spec.reaction_id}",
        f"dataset: {spec.dataset}",
        "",
        "reactants (coefficient < 0 in DatasetEval):",
    ]
    for t in spec.reactants:
        lines.append(f"  {t.coefficient:+.6g}  {t.molecule_id}")
    lines.append("")
    lines.append("products (coefficient > 0):")
    for t in spec.products:
        lines.append(f"  {t.coefficient:+.6g}  {t.molecule_id}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
