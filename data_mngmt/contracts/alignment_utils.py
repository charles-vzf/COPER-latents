"""COPER ↔ MDP alignment: ICUSTAY_ID overlap for latent / tabular joins."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_mortality_pickle_details(mortality_pickle_path: Path) -> dict[str, Any] | None:
    """Return the first tuple element (metadata dict) from a COPER mortality pickle."""
    with open(mortality_pickle_path, "rb") as f:
        details = pickle.load(f)[0]
    return details if isinstance(details, dict) else None


def _icustay_col(df: pd.DataFrame) -> str:
    for c in ("icustayid", "ICUSTAY_ID", "icustay_id"):
        if c in df.columns:
            return c
    raise KeyError(f"No icustay id column in {list(df.columns)[:25]}")


def alignment_summary_coper_mdp(
    mortality_pickle_path: Path,
    mdp_cohort_csv: Path,
) -> dict[str, Any]:
    """Summarize how ICUSTAY_ID sets from the COPER pickle overlap the MDP cohort table.

    The COPER bundle stores ``details['icustay_id']['train'|'val'|'test']`` (same row order as ``X_*``).
    The MDP cohort has one row per RL decision step; we use **unique** ICU stays in that CSV.
    """
    details = load_mortality_pickle_details(mortality_pickle_path)
    if details is None:
        return {"error": "pickle details is not a dict"}

    icu = details.get("icustay_id")
    if not icu:
        return {"error": "no details['icustay_id']; rebuild pickle with benchmark_episode_root set"}

    def _valid(arr: np.ndarray) -> set[int]:
        a = np.asarray(arr).reshape(-1).astype(np.int64)
        return set(int(x) for x in a if x >= 0)

    c_train, c_val, c_test = _valid(icu["train"]), _valid(icu["val"]), _valid(icu["test"])
    c_all = c_train | c_val | c_test

    df = pd.read_csv(mdp_cohort_csv)
    col = _icustay_col(df)
    mdp_unique = set(df[col].dropna().astype(int).unique().tolist())

    def _ov(s: set[int]) -> dict[str, int]:
        inter = s & mdp_unique
        return {
            "n_coper": len(s),
            "n_mdp_cohort_unique": len(mdp_unique),
            "n_intersection": len(inter),
            "coper_in_mdp_frac": float(len(inter) / len(s)) if s else 0.0,
        }

    return {
        "temporal_alignment": (
            "COPER: 1 h × 48 h IHM tensors (mimic3-benchmarks). MDP: one transition per RL row; "
            "unified build defaults to 1 h RL blocs (sepsis_cohort --bloc-interval-hours). "
            "Join trajectories on ICUSTAY_ID; MDP has multiple rows per stay (one per bloc)."
        ),
        "coper_unique_icustay": {
            "train": len(c_train),
            "val": len(c_val),
            "test": len(c_test),
            "all_splits_union": len(c_all),
        },
        "overlap_by_split": {
            "train": _ov(c_train),
            "val": _ov(c_val),
            "test": _ov(c_test),
            "all_splits": _ov(c_all),
        },
        "note": (
            "Join COPER rows to MDP rows on ICUSTAY_ID; MDP has multiple rows per stay (bloc). "
            "Latent vectors align to COPER row index i ↔ icustay_id[i]."
        ),
    }


def write_alignment_json(
    path: Path,
    mortality_pickle_path: Path,
    mdp_cohort_csv: Path | None,
) -> Path | None:
    if mdp_cohort_csv is None or not mdp_cohort_csv.is_file():
        return None
    summ = alignment_summary_coper_mdp(mortality_pickle_path, mdp_cohort_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, default=str)
        f.write("\n")
    return path
