"""Wrap ``icu_sepsis_helpers.build.build_mimic_params`` for CLI / notebook use."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from data_mngmt import coper_root


def _ensure_icu_sepsis_paths() -> None:
    root = coper_root()
    for sub in ("icu_sepsis/icu_sepsis", "icu_sepsis/icu_sepsis_helpers"):
        p = str((root / sub).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)


def build_mdp_from_sepsis_cohort(
    cohort_csv: Path,
    out_dir: Path,
    *,
    n_states: int = 750,
    n_action_levels: int = 5,
    threshold: int = 20,
    seed: int = 0,
    outcome_column: str = "mortality_90d",
) -> Path:
    """Build MDPParameters under ``out_dir`` from a sepsis RL-style CSV.

    The CSV must match the schema expected by ``create_rl_dataset`` (AI Clinician–like
    columns: SOFA, fluids, vasopressors, ``bloc``, ``mortality_90d``, etc.).
    """
    _ensure_icu_sepsis_paths()
    import icu_sepsis  # noqa: F401 — gym registry
    from icu_sepsis_helpers.build import build_mimic_params

    cohort_csv = Path(cohort_csv).resolve()
    out_dir = Path(out_dir).resolve()
    if not cohort_csv.is_file():
        raise FileNotFoundError(cohort_csv)

    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Building MDP from %s -> %s", cohort_csv, out_dir)
    build_mimic_params(
        str(cohort_csv),
        str(out_dir),
        n_states,
        n_action_levels,
        threshold,
        seed=seed,
        outcome_column=outcome_column,
    )
    return out_dir
