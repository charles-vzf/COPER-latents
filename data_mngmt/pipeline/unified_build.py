"""Unified MIMIC build: PhysioNet CSVs → mimic3-benchmarks IHM → COPER pickle + ICU-Sepsis MDP.

**MDP source table (AI Clinician / mimic_sepsis schema):** one row per ICU stay time **bloc**
(default **1 h** per row via ``sepsis_cohort --bloc-interval-hours``, matching COPER timestep; configurable). Must include
``icustayid`` (or ``ICUSTAY_ID`` / ``icustay_id``), ``bloc``, and the clinical columns consumed by
``icu_sepsis_helpers.mdp_creation.create_rl_table`` (vitals, labs, ``SOFA``, ``SIRS``, fluids
``input_*`` / ``output_*``, ``max_dose_vaso``, etc.). ``run_unified_build`` adds ``mortality_inhospital``
from PhysioNet ``ADMISSIONS`` before calling ``build_mdp``.
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_mngmt import coper_root, load_paths
from data_mngmt.contracts.alignment_utils import write_alignment_json
from data_mngmt.contracts.pipeline_contract import (
    MDP_OUTCOME_COLUMN_DEFAULT,
    default_pipeline_contract,
    rl_table_contents_summary,
)
from data_mngmt.layout import coper_mortality_pickle_path, generated_root
from data_mngmt.mimic.coper_mimic3_export import (
    default_mimic3_repo,
    export_mortality_pickle_for_coper,
)
from data_mngmt.mimic.mimic_physionet_pipeline import (
    build_stem_to_icustay,
    ensure_benchmark_episodes,
    ensure_in_hospital_mortality_task,
    filter_ihm_listfiles_to_icustays,
)
from data_mngmt.mimic.sepsis_icustays import inhospital_mortality_by_icustay, sepsis_icustay_ids
from data_mngmt.pipeline.build_mdp import build_mdp_from_sepsis_cohort

log = logging.getLogger(__name__)

MDP_OUTCOME_COL = MDP_OUTCOME_COLUMN_DEFAULT
# Canonical name under ``paths.json`` → ``icu_sepsis_csv_tables_dir`` (published after each MDP build).
MIMIC_DATASET_TABLE_PUBLISHED = "mimic_dataset_table.csv"
MDP_COHORT_PREPARED_PUBLISHED = "mdp_cohort_prepared.csv"


def _summarize_mortality_pickle(pickle_path: Path) -> dict[str, Any]:
    with open(pickle_path, "rb") as f:
        details, X_train, y_train, X_val, y_val, X_test, y_test, _ = pickle.load(f)

    def split_stats(y: np.ndarray, name: str) -> dict[str, Any]:
        y = np.asarray(y).reshape(-1)
        pos = int((y >= 0.5).sum())
        n = y.shape[0]
        return {
            "split": name,
            "n": n,
            "positive": pos,
            "rate": float(pos / n) if n else 0.0,
        }

    rows = [
        split_stats(y_train, "train"),
        split_stats(y_val, "val"),
        split_stats(y_test, "test"),
    ]
    out: dict[str, Any] = {
        "details": details,
        "splits": rows,
        "X_train_shape": tuple(X_train.shape),
        "X_val_shape": tuple(X_val.shape),
        "X_test_shape": tuple(X_test.shape),
    }
    if isinstance(details, dict):
        icu = details.get("icustay_id")
        if icu:
            out["icustay_id"] = {
                "train_n": int(np.asarray(icu["train"]).size),
                "val_n": int(np.asarray(icu["val"]).size),
                "test_n": int(np.asarray(icu["test"]).size),
                "train_valid": int((np.asarray(icu["train"]) >= 0).sum()),
                "val_valid": int((np.asarray(icu["val"]) >= 0).sum()),
                "test_valid": int((np.asarray(icu["test"]) >= 0).sum()),
            }
        if details.get("icustay_id_stem_missing"):
            out["icustay_id_stem_missing"] = details["icustay_id_stem_missing"]
    return out


def _summarize_sepsis_cohort_csv(csv_path: Path, max_rows_sample: int = 50_000) -> dict[str, Any]:
    """Lightweight stats for MDP input CSV (full read may be large)."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, nrows=max_rows_sample)
    out: dict[str, Any] = {
        "path": str(csv_path),
        "rows_read": len(df),
        "columns": list(df.columns),
    }
    if "mortality_90d" in df.columns:
        out["mortality_90d_rate"] = float(df["mortality_90d"].mean())
    if "mortality_inhospital" in df.columns:
        out["mortality_inhospital_rate"] = float(df["mortality_inhospital"].mean())
    if "bloc" in df.columns:
        out["bloc_min"] = int(df["bloc"].min())
        out["bloc_max"] = int(df["bloc"].max())
    return out


def _mdp_rl_bloc_cache_tag(h: float) -> str:
    """Stable suffix for cache filenames (e.g. 4 → ``4``, 1.5 → ``1p5``)."""
    h = float(h)
    if abs(h - round(h)) < 1e-9:
        return str(int(round(h)))
    return f"{h:g}".replace(".", "p")


@dataclass
class UnifiedBuildParams:
    sepsis_cohort: bool = True
    timestep_minutes: int = 60
    horizon_hours: int = 48
    build_mdp: bool = True
    force_rebuild_benchmark: bool = False
    #: Explicit RL cohort CSV path (only non–Postgres input; no reuse of workdir or published caches).
    cohort_csv: Path | None = None
    #: When ``build_mdp`` and no valid ``cohort_csv``, resolve the RL table from Postgres unless a
    #: non-empty ``mimic_dataset_table_src_bloc<N>h.csv`` snapshot already exists (see
    #: ``mdp_force_rebuild_source_table``). Does not read ``icu_sepsis_csv_tables_dir`` as input.
    mdp_rebuild_table_from_db: bool = True
    mdp_skip_preprocess: bool = False
    #: If True, always run Postgres preprocess + ``sepsis_cohort`` even when a matching snapshot exists.
    mdp_force_rebuild_source_table: bool = False
    mdp_mimic_sepsis_workdir: Path | None = None
    mdp_n_states: int = 750
    mdp_n_action_levels: int = 5
    mdp_threshold: int = 20
    mdp_seed: int = 0
    #: After a successful MDP build, copy ``dynamics.npz`` + sidecars into
    #: ``icu_sepsis/.../envs/assets`` so ``gym.make("Sepsis/ICU-Sepsis-v2")`` uses them.
    publish_gym_env_assets: bool = False
    #: Hours per RL table row when rebuilding from Postgres: passed to vendored
    #: ``sepsis_cohort.py --bloc-interval-hours`` (aggregation window). Also recorded in
    #: ``pipeline_contract``. For a custom ``cohort_csv``, set this to match the CSV’s spacing.
    mdp_rl_bloc_interval_hours: float = 1.0


def build_slug(p: UnifiedBuildParams) -> str:
    tag = "sepsis" if p.sepsis_cohort else "all"
    return f"{tag}-{p.timestep_minutes}m-h{p.horizon_hours}ihm"


# Default ``build_data`` / notebook parameters → COPER pickle stem (sync ``paths.json`` → ``mimic3_mortality``).
DEFAULT_UNIFIED_SLUG = build_slug(UnifiedBuildParams())


def _icu_sepsis_tables_dir(repo: Path) -> Path:
    raw = load_paths().get("icu_sepsis_csv_tables_dir")
    if not raw:
        raise KeyError("paths.json: missing icu_sepsis_csv_tables_dir")
    p = Path(raw)
    return p if p.is_absolute() else (repo / p).resolve()


def published_mimic_dataset_table(repo: Path) -> Path:
    """``icu_sepsis_csv_tables_dir/mimic_dataset_table.csv`` (MIMIC-rebuilt cohort, stable name)."""
    return _icu_sepsis_tables_dir(repo) / MIMIC_DATASET_TABLE_PUBLISHED


def publish_icu_sepsis_csv_tables(
    repo: Path,
    source_rl_csv: Path,
    prepared_mdp_csv: Path | None,
) -> dict[str, str]:
    """Copy RL + prepared MDP CSVs into ``paths.json`` → ``icu_sepsis_csv_tables_dir``."""
    tdir = _icu_sepsis_tables_dir(repo)
    tdir.mkdir(parents=True, exist_ok=True)
    main = tdir / MIMIC_DATASET_TABLE_PUBLISHED
    shutil.copy2(source_rl_csv, main)
    meta: dict[str, str] = {"mimic_dataset_table": str(main)}
    if prepared_mdp_csv is not None and Path(prepared_mdp_csv).is_file():
        prep = tdir / MDP_COHORT_PREPARED_PUBLISHED
        shutil.copy2(prepared_mdp_csv, prep)
        meta["mdp_cohort_prepared"] = str(prep)
    note = tdir / "PUBLISHED_FROM_UNIFIED_BUILD.txt"
    note.write_text(
        f"source_rl={source_rl_csv}\nprepared={prepared_mdp_csv}\n",
        encoding="utf-8",
    )
    log.info("Published ICU-Sepsis CSV tables under %s", tdir)
    return meta


def ensure_mdp_source_csv(repo: Path, work: Path, p: UnifiedBuildParams) -> Path | None:
    """Resolve RL table: explicit ``cohort_csv`` if the file exists, else Postgres rebuild when enabled.

    When rebuilding from Postgres, a per-bloc snapshot ``work/mimic_dataset_table_src_bloc<N>h.csv`` is
    written after a successful run. On later runs, if that file already exists (non-empty) and
    ``mdp_force_rebuild_source_table`` is False, it is **reused** and the Postgres pipeline is skipped.

    Does **not** load MDP input from ``icu_sepsis_csv_tables_dir/mimic_dataset_table.csv`` as a build input;
    that path is a publish target only.
    """
    if p.cohort_csv is not None:
        cand = Path(p.cohort_csv)
        cand = cand if cand.is_absolute() else (repo / cand).resolve()
        if cand.is_file():
            return cand
        log.warning("cohort_csv missing (%s); will use Postgres if mdp_rebuild_table_from_db=True", cand)

    if not p.build_mdp:
        return None

    bloc_h = float(p.mdp_rl_bloc_interval_hours)
    staged = work / f"mimic_dataset_table_src_bloc{_mdp_rl_bloc_cache_tag(bloc_h)}h.csv"

    if not p.mdp_rebuild_table_from_db:
        log.warning(
            "build_mdp=True but no valid cohort_csv and mdp_rebuild_table_from_db=False. "
            "Set an existing cohort_csv or mdp_rebuild_table_from_db=True (Postgres).",
        )
        return None

    if (
        staged.is_file()
        and staged.stat().st_size > 0
        and not p.mdp_force_rebuild_source_table
    ):
        log.info(
            "Reusing existing RL source snapshot %s (delete file or set "
            "mdp_force_rebuild_source_table=True to rebuild from Postgres)",
            staged,
        )
        return staged

    msw = p.mdp_mimic_sepsis_workdir or (work / "mimic_sepsis_build")
    msw = Path(msw)
    msw = msw if msw.is_absolute() else (repo / msw).resolve()

    from data_mngmt.sepsis_rl.mimic_sepsis_run import build_rl_cohort_csv_from_mimic_db

    built = build_rl_cohort_csv_from_mimic_db(
        msw,
        skip_preprocess=p.mdp_skip_preprocess,
        bloc_interval_hours=bloc_h,
    )
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, staged)
    log.info("Wrote MDP source table snapshot %s (from %s)", staged, built)
    return staged


def prepare_mdp_cohort_csv(
    source_csv: Path,
    physionet_csv_dir: Path,
    out_csv: Path,
    *,
    sepsis_ids: set[int] | None,
) -> Path:
    """Filter optional sepsis cohort and attach in-hospital mortality (MIMIC ADMISSIONS)."""
    df = pd.read_csv(source_csv)
    col = None
    for c in ("icustayid", "ICUSTAY_ID", "icustay_id"):
        if c in df.columns:
            col = c
            break
    if col is None:
        raise KeyError(
            f"Need icustay id column (icustayid / ICUSTAY_ID) in {source_csv}"
        )
    ih = inhospital_mortality_by_icustay(physionet_csv_dir)
    icu_series = df[col].astype(int)
    if sepsis_ids is not None:
        df = df.loc[icu_series.isin(sepsis_ids)].copy()
        icu_series = df[col].astype(int)
    df[MDP_OUTCOME_COL] = icu_series.map(lambda i: int(ih.get(int(i), 0))).astype(int)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Wrote MDP cohort CSV %s (%d rows)", out_csv, len(df))
    return out_csv


def run_unified_build(
    params: UnifiedBuildParams | None = None,
    *,
    physionet_csv_dir: Path | None = None,
    mimic3_benchmarks_repo: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run PhysioNet → benchmarks → COPER pickle; optionally MDP from RL-style cohort CSV."""
    repo = Path(repo_root or coper_root()).resolve()
    p = params or UnifiedBuildParams()
    phys = Path(physionet_csv_dir or load_paths()["physionet_mimic_root"])
    phys = phys if phys.is_absolute() else (repo / phys).resolve()
    if not phys.is_dir():
        raise FileNotFoundError(f"physionet_mimic_root not a directory: {phys}")

    mimic3 = Path(mimic3_benchmarks_repo or default_mimic3_repo()).resolve()
    slug = build_slug(p)
    work = (generated_root(repo) / "unified" / slug).resolve()
    work.mkdir(parents=True, exist_ok=True)
    benchmark_root = work / "root"
    ihm_root = work / "in-hospital-mortality"

    if p.horizon_hours != 48:
        log.warning(
            "horizon_hours=%s: upstream mimic3-benchmarks create_in_hospital_mortality "
            "uses a fixed 48 h window; export still uses period_length=%s in the reader.",
            p.horizon_hours,
            float(p.horizon_hours),
        )

    timestep_hours = float(p.timestep_minutes) / 60.0

    ensure_benchmark_episodes(
        phys,
        mimic3,
        benchmark_root,
        force=p.force_rebuild_benchmark,
    )
    ensure_in_hospital_mortality_task(
        mimic3,
        benchmark_root,
        ihm_root,
        force=p.force_rebuild_benchmark,
    )

    sepsis_ids: set[int] | None = None
    if p.sepsis_cohort:
        sepsis_ids = sepsis_icustay_ids(phys)
        stem_map = build_stem_to_icustay(benchmark_root)
        if len(stem_map) == 0:
            raise RuntimeError(
                "No benchmark stay stems were found under benchmark_root "
                f"({benchmark_root}). This usually means benchmark episodes are missing/"
                "incomplete. Re-run with force_rebuild_benchmark=True (or delete the "
                "workdir root) so extract_episodes/split_train_and_test are regenerated."
            )
        filter_ihm_listfiles_to_icustays(ihm_root, stem_map, sepsis_ids)

    pickle_path = coper_mortality_pickle_path(repo, slug=slug)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {
        "build_slug": slug,
        "sepsis_cohort": bool(p.sepsis_cohort),
        "timestep_minutes": int(p.timestep_minutes),
        "timestep_hours": timestep_hours,
        "horizon_hours": float(p.horizon_hours),
        "label": "in_hospital_mortality",
        "label_description": (
            "Binary in-hospital mortality (IHM) from mimic3-benchmarks listfiles; "
            "aligned with MIMIC ADMISSIONS.HOSPITAL_EXPIRE_FLAG for MDP cohort column "
            f"{MDP_OUTCOME_COL}."
        ),
        "physionet_mimic_root": str(phys),
        "benchmark_workdir": str(work),
    }
    coper_info = export_mortality_pickle_for_coper(
        mimic3_benchmarks_repo=mimic3,
        ihm_data_root=ihm_root,
        out_path=pickle_path,
        timestep=timestep_hours,
        period_length=float(p.horizon_hours),
        extra_details=extra,
        benchmark_episode_root=benchmark_root,
    )

    out: dict[str, Any] = {
        "slug": slug,
        "physionet_mimic_root": str(phys),
        "mimic3_benchmarks_repo": str(mimic3),
        "benchmark_root": str(benchmark_root),
        "ihm_root": str(ihm_root),
        "coper_pickle": coper_info,
        "pipeline_contract": default_pipeline_contract(
            timestep_minutes=p.timestep_minutes,
            horizon_hours=p.horizon_hours,
            mdp_outcome_column=MDP_OUTCOME_COL,
            mdp_bloc_hours=p.mdp_rl_bloc_interval_hours,
        ).as_dict(),
        "rl_table_contents_summary": rl_table_contents_summary(),
    }

    out["coper_summary"] = _summarize_mortality_pickle(pickle_path)

    cohort_src = ensure_mdp_source_csv(repo, work, p)
    out["mdp_source_csv"] = str(cohort_src) if cohort_src else None
    if p.build_mdp:
        if cohort_src is None or not cohort_src.is_file():
            log.warning("build_mdp=True but no MDP source CSV; skipping MDP.")
        else:
            mdp_csv = work / f"mdp_cohort_{slug}.csv"
            prepare_mdp_cohort_csv(cohort_src, phys, mdp_csv, sepsis_ids=sepsis_ids)
            n_mdp = len(pd.read_csv(mdp_csv))
            mdp_dir = work / f"mdp_params_{slug}"
            if n_mdp == 0:
                log.warning(
                    "Skipping MDP build: prepared cohort CSV has 0 rows "
                    "(check sepsis filter vs cohort icustay ids)."
                )
            else:
                build_mdp_from_sepsis_cohort(
                    mdp_csv,
                    mdp_dir,
                    n_states=p.mdp_n_states,
                    n_action_levels=p.mdp_n_action_levels,
                    threshold=p.mdp_threshold,
                    seed=p.mdp_seed,
                    outcome_column=MDP_OUTCOME_COL,
                )
            if n_mdp > 0:
                out["mdp"] = {
                    "cohort_prepared": str(mdp_csv),
                    "params_dir": str(mdp_dir),
                    "outcome_column": MDP_OUTCOME_COL,
                }
                out["mdp_cohort_summary"] = _summarize_sepsis_cohort_csv(mdp_csv)
                align_path = work / f"coper_mdp_alignment_{slug}.json"
                p_align = write_alignment_json(align_path, pickle_path, mdp_csv)
                if p_align is not None:
                    out["coper_mdp_alignment_json"] = str(p_align)
                try:
                    out["icu_sepsis_csv_tables_published"] = publish_icu_sepsis_csv_tables(
                        repo, cohort_src, mdp_csv
                    )
                except Exception as e:
                    log.warning("Could not publish to icu_sepsis_csv_tables_dir: %s", e)

                if p.publish_gym_env_assets:
                    from data_mngmt.tools.publish_icu_sepsis_env_assets import (
                        publish_mdp_params_to_pkg_assets,
                    )

                    try:
                        dest = publish_mdp_params_to_pkg_assets(mdp_dir)
                        out["gym_env_assets_published"] = [str(x) for x in dest]
                    except Exception as e:
                        log.warning("publish_gym_env_assets failed: %s", e)

    meta_path = work / "unified_build.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
        f.write("\n")
    out["meta_written"] = str(meta_path)
    return out
