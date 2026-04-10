"""COPER ↔ MDP dataset contract: time grids, features, and join semantics.

The unified build produces **two parallel views** on a **shared sepsis ICU cohort**
(when ``sepsis_cohort=True``): mimic3-benchmarks IHM tensors for COPER, and an
AI Clinician–style RL table for the tabular MDP. They are aligned on **ICUSTAY_ID**
and **in-hospital mortality** for the outcome used in MDP clustering; they are **not**
aligned on per-timestep clinical features (different variable sets and usually
different clock binning).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Unified build default: one RL row per 1 h bloc (matches COPER timestep; see mimic_sepsis_run).
MDP_RL_BLOC_INTERVAL_HOURS_DEFAULT = 1.0

# mimic3-benchmarks in-hospital mortality task (COPER pickle export).
COPER_IHM_TIMESTEP_HOURS_DEFAULT = 1.0
COPER_IHM_HORIZON_HOURS_DEFAULT = 48.0

# Outcome column written by unified build before ``build_mdp`` (from ADMISSIONS).
MDP_OUTCOME_COLUMN_DEFAULT = "mortality_inhospital"


@dataclass(frozen=True)
class PipelineContract:
    """Serializable summary for ``unified_build.json`` → ``pipeline_contract``."""

    coper_timestep_hours: float
    coper_horizon_hours: float
    coper_task: str
    coper_feature_source: str
    mdp_rl_table_schema: str
    mdp_bloc_interval_hours_assumed: float
    mdp_outcome_column: str
    mdp_state_features_note: str
    join_key: str
    temporal_alignment_note: str
    coper_vs_mdp_features_note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "coper_timestep_hours": self.coper_timestep_hours,
            "coper_horizon_hours": self.coper_horizon_hours,
            "coper_task": self.coper_task,
            "coper_feature_source": self.coper_feature_source,
            "mdp_rl_table_schema": self.mdp_rl_table_schema,
            "mdp_bloc_interval_hours_assumed": self.mdp_bloc_interval_hours_assumed,
            "mdp_outcome_column": self.mdp_outcome_column,
            "mdp_state_features_note": self.mdp_state_features_note,
            "join_key": self.join_key,
            "temporal_alignment_note": self.temporal_alignment_note,
            "coper_vs_mdp_features_note": self.coper_vs_mdp_features_note,
        }


def default_pipeline_contract(
    *,
    timestep_minutes: int,
    horizon_hours: int,
    mdp_outcome_column: str = MDP_OUTCOME_COLUMN_DEFAULT,
    mdp_bloc_hours: float = MDP_RL_BLOC_INTERVAL_HOURS_DEFAULT,
) -> PipelineContract:
    th = float(timestep_minutes) / 60.0
    return PipelineContract(
        coper_timestep_hours=th,
        coper_horizon_hours=float(horizon_hours),
        coper_task="mimic3-benchmarks in_hospital_mortality",
        coper_feature_source=(
            "YerevaNN mimic3-benchmarks IHM tensors: discretizer + normalizer channels "
            "(mask + value columns; not the same names as the RL table)."
        ),
        mdp_rl_table_schema=(
            "AI Clinician / microsoft.mimic_sepsis MIMICtable: one row per temporal bloc "
            f"(stock pipeline ~{mdp_bloc_hours:g} h per bloc); columns include icustayid, bloc, "
            "vitals, labs, SOFA, SIRS, fluids (input_4hourly, …), outputs, max_dose_vaso, … "
            "(see icu_sepsis_helpers.mdp_creation.create_rl_table)."
        ),
        mdp_bloc_interval_hours_assumed=mdp_bloc_hours,
        mdp_outcome_column=mdp_outcome_column,
        mdp_state_features_note=(
            "MDP discrete states come from KMeans on normalized RL-table columns "
            "(SOFA and other cols in create_rl_table); SOFA is used for MDP state / sofa_scores in dynamics."
        ),
        join_key="ICUSTAY_ID (details['icustay_id'] in COPER pickle ↔ icustayid in RL table)",
        temporal_alignment_note=(
            f"COPER grid: {th:g} h steps × {horizon_hours:g} h window from ICU admission (IHM task). "
            f"MDP transitions: one per RL row; RL bloc spacing is {mdp_bloc_hours:g} h "
            "(unified default matches COPER 1 h timestep). "
            "Latent↔MDP matching is at stay level (and index alignment for COPER rows), not step-by-step same clock."
        ),
        coper_vs_mdp_features_note=(
            "COPER does not use the raw SOFA column from the RL table; it uses benchmark tensor channels. "
            "The MDP uses SOFA (and other vitals/labs) from the RL table for clustering and bundled sofa_scores."
        ),
    )


def rl_table_contents_summary() -> str:
    """Short description of what ``mimic_dataset_table.csv`` / ``MIMICtable.csv`` contains."""
    return (
        "Wide per-bloc table: one row per (icustayid, bloc) from mimic_sepsis, with demographics, "
        "vitals, labs, SOFA, SIRS, fluid/vasopressor amounts, etc. The unified build copies or builds "
        "this file, then writes ``mdp_cohort_<slug>.csv`` with the same rows (sepsis-filtered) plus "
        f"``{MDP_OUTCOME_COLUMN_DEFAULT}`` from MIMIC ADMISSIONS. "
        "``icu_sepsis_helpers`` then builds ``mimic_rl_table.csv`` (bloc, state, action, outcome_y) from that."
    )
