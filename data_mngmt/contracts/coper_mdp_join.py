"""Join COPER mortality pickle rows to MDP cluster-state occupancy from the RL cohort CSV.

Uses the same ``create_rl_dataset`` / KMeans labelling as ``icu_sepsis_helpers.build.build_mimic_params`` when
given identical hyperparameters (``n_states``, ``n_action_levels``, ``seed``, ``outcome_column``).

There is **no per-timestep alignment** between IHM tensors and sepsis blocs; the join is **per ICU stay**:
for each COPER sample we attach the **empirical distribution of discrete cluster states** visited across all RL
rows for that ``icustayid`` in the cohort table.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_mngmt.contracts.alignment_utils import _icustay_col, load_mortality_pickle_details

log = logging.getLogger(__name__)


def read_mdp_metadata_n_states(mdp_params_dir: Path) -> int | None:
    """Return ``n_states`` from ``mdp_params_dir/metadata.json`` if present."""
    p = Path(mdp_params_dir) / "metadata.json"
    if not p.is_file():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        return int(meta["n_states"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def cohort_csv_with_mdp_states(
    cohort_csv: Path,
    *,
    n_states: int,
    n_action_levels: int = 5,
    seed: int = 0,
    outcome_column: str = "mortality_inhospital",
    n_clustering: int = 32,
    ratio_clustering: float = 0.25,
    max_iter_kmeans: int = 10_000,
    init_kmeans: str = "k-means++",
) -> pd.DataFrame:
    """Re-run ``create_rl_dataset`` in a temp dir and align ``icustayid`` row-wise to ``mimic_rl_table``.

    Row order matches the input CSV (same as ``create_rl_dataset`` internal processing).
    """
    cohort_csv = Path(cohort_csv).resolve()
    raw = pd.read_csv(cohort_csv)
    icu_col = _icustay_col(raw)

    from data_mngmt.pipeline.build_mdp import _ensure_icu_sepsis_paths

    _ensure_icu_sepsis_paths()
    import icu_sepsis  # noqa: F401 — gym registry for downstream

    from icu_sepsis_helpers.mdp_creation.create_rl_table import create_rl_dataset

    with tempfile.TemporaryDirectory(prefix="coper_mdp_join_") as td:
        td_path = Path(td)
        create_rl_dataset(
            cohort_csv,
            td_path,
            n_states,
            n_action_levels,
            seed=seed,
            ratio_clustering=ratio_clustering,
            max_iter=max_iter_kmeans,
            init=init_kmeans,
            n_clustering=n_clustering,
            outcome_column=outcome_column,
        )
        rl = pd.read_csv(td_path / "mimic_rl_table.csv")

    if len(rl) != len(raw):
        raise ValueError(
            f"RL table rows ({len(rl)}) != cohort rows ({len(raw)}); check cohort CSV vs create_rl_dataset."
        )
    out = rl.copy()
    out[icu_col] = raw[icu_col].values
    return out


def _histogram_for_stay(states: np.ndarray, out_dim: int) -> np.ndarray:
    """Normalized occupancy; ``states`` are integer cluster ids in ``[0, out_dim)``."""
    states = np.asarray(states, dtype=np.int64).reshape(-1)
    if states.size == 0:
        q = np.ones(out_dim, dtype=np.float64) / out_dim
        return q
    bc = np.bincount(states, minlength=out_dim).astype(np.float64)
    if bc.shape[0] > out_dim:
        bc = bc[:out_dim]
    s = float(bc.sum())
    if s <= 0:
        return np.ones(out_dim, dtype=np.float64) / out_dim
    q = bc / s
    return q


def state_targets_for_coper_splits(
    mortality_pickle_path: Path,
    cohort_with_states: pd.DataFrame,
    *,
    n_states_out: int,
    icu_col: str | None = None,
) -> dict[str, Any]:
    """For each COPER row (same order as ``load_xy_split``), build a target distribution over ``n_states_out`` MDP labels.

    ``n_states_out`` should match the KMeans cluster count used to build ``cohort_with_states`` (typically
    ``metadata.json`` → ``n_states`` from the same MDP build). If you instead train against a gym env with a
    different observation size, pad or slice outside this helper.

    Returns a dict with per-split ``q`` arrays ``(n_samples, n_states_out)`` and ``icustay_ids`` aligned with
    COPER tensors, plus ``missing`` counts where a pickle ``icustay_id`` had no RL rows.
    """
    details = load_mortality_pickle_details(Path(mortality_pickle_path))
    if details is None or not isinstance(details.get("icustay_id"), dict):
        raise ValueError("Pickle must contain details['icustay_id'] with train/val/test arrays.")

    if icu_col is None:
        icu_col = _icustay_col(cohort_with_states)

    grp = cohort_with_states.groupby(icu_col, sort=False)["state"]

    out: dict[str, Any] = {"splits": {}, "icu_col": icu_col, "n_states_out": int(n_states_out)}

    for split in ("train", "val", "test"):
        icu_arr = np.asarray(details["icustay_id"][split]).reshape(-1)
        qs = []
        missing = 0
        for i in range(icu_arr.size):
            iid = int(icu_arr[i])
            if iid < 0:
                qs.append(np.ones(n_states_out, dtype=np.float64) / n_states_out)
                missing += 1
                continue
            try:
                st = grp.get_group(iid).to_numpy()
            except KeyError:
                qs.append(np.ones(n_states_out, dtype=np.float64) / n_states_out)
                missing += 1
                continue
            qs.append(_histogram_for_stay(st, n_states_out))

        q_mat = np.stack(qs, axis=0).astype(np.float32)
        q_mat /= np.clip(q_mat.sum(axis=1, keepdims=True), 1e-8, None)
        out["splits"][split] = {
            "q_target": q_mat,
            "icustay_ids": icu_arr.astype(np.int64),
            "n_missing_or_empty_stays": missing,
        }

    return out


def per_bloc_state_matrix(
    cohort_with_states: pd.DataFrame,
    icu_ids: np.ndarray,
    *,
    n_blocs: int = 48,
    icu_col: str | None = None,
    bloc_col: str | None = None,
    state_col: str = "state",
    max_state_exclusive: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each ``ICUSTAY_ID`` row in ``icu_ids`` (order preserved), build per-bloc MDP cluster states.

    Uses **hourly (1 h) bloc convention**: bloc values ``1 .. n_blocs`` must each appear at least once
    for the stay to be marked valid. If multiple RL rows share the same bloc, the **first** row is used.

    Args:
        cohort_with_states: RL table with at least ``icu_col``, ``bloc_col``, ``state_col``.
        icu_ids: shape ``(N,)`` int64 stay ids (use ``-1`` for padding / unknown; those rows stay invalid).
        n_blocs: number of blocs to require (default ``48`` for first 48h with 1h blocs).
        max_state_exclusive: if set, reject stays where any state is outside ``[0, max_state_exclusive)``.

    Returns:
        ``states`` of shape ``(N, n_blocs)`` int64 with ``-1`` for invalid / incomplete rows.
        ``valid`` shape ``(N,)`` bool — ``True`` iff every bloc ``1..n_blocs`` is present with a finite state.
    """
    icu_ids = np.asarray(icu_ids, dtype=np.int64).reshape(-1)
    n = int(icu_ids.shape[0])
    if icu_col is None:
        icu_col = _icustay_col(cohort_with_states)
    if bloc_col is None:
        bloc_col = next((c for c in ("bloc", "BLOC") if c in cohort_with_states.columns), None)
    if bloc_col is None:
        raise KeyError("cohort_with_states must contain a bloc column (bloc / BLOC)")

    states = np.full((n, int(n_blocs)), -1, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    grp = cohort_with_states.groupby(icu_col, sort=False)

    for i in range(n):
        iid = int(icu_ids[i])
        if iid < 0:
            continue
        try:
            g = grp.get_group(iid)
        except KeyError:
            continue
        b = pd.to_numeric(g[bloc_col], errors="coerce")
        st = pd.to_numeric(g[state_col], errors="coerce")
        ok_row = b.notna() & st.notna()
        if not bool(ok_row.any()):
            continue

        row_states: list[int] = []
        filled = True
        for t in range(1, int(n_blocs) + 1):
            sel = g.loc[ok_row & (b == t)]
            if len(sel) == 0:
                filled = False
                break
            si = int(pd.to_numeric(sel[state_col].iloc[0], errors="coerce"))
            if not np.isfinite(si):
                filled = False
                break
            if max_state_exclusive is not None and not (0 <= si < int(max_state_exclusive)):
                filled = False
                break
            row_states.append(si)
        if filled:
            states[i] = np.asarray(row_states, dtype=np.int64)
            valid[i] = True

    return states, valid


def default_unified_cohort_paths(
    repo_root: Path | None = None,
    *,
    slug: str | None = None,
) -> tuple[Path, Path]:
    """Return ``(mdp_cohort_csv, mdp_params_dir)`` under ``data_mngmt/generated/unified/<slug>/``."""
    from data_mngmt import coper_root
    from data_mngmt.pipeline.unified_build import UnifiedBuildParams, build_slug

    repo = Path(repo_root or coper_root()).resolve()
    slug = slug or build_slug(UnifiedBuildParams())
    work = repo / "data_mngmt" / "generated" / "unified" / slug
    cohort = work / f"mdp_cohort_{slug}.csv"
    params = work / f"mdp_params_{slug}"
    return cohort.resolve(), params.resolve()
