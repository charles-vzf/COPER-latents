# COPER + MDP: data, granularity, alignment

This document defines the **contract** between COPER (in-hospital mortality) tensors and the ICU-Sepsis MDP RL table, so we do not confuse windows, time steps, and clinical variables.

## What `build_data` / `run_unified_build` produce (defaults)

With **MIMIC-III CSV** (`paths.json` → `physionet_mimic_root`), **mimic3-benchmarks** (vendored), and an **RL table** obtained either via **Postgres + `mimic_sepsis`** (default path) or an explicit **`cohort_csv`** (no automatic reuse of older CSVs under `generated/unified/` nor of `icu_sepsis_csv_tables_dir` as input):

| Output | Role |
|--------|------|
| `data_mngmt/generated/mortality_coper_<slug>.data` | COPER pickle: in-hospital mortality tensors (train/val/test), **1 h** step, **48 h** window from stay start in the benchmark task. |
| `data_mngmt/generated/unified/<slug>/mdp_cohort_<slug>.csv` | MDP cohorts: RL table rows filtered to sepsis + column **`mortality_inhospital`** (ADMISSIONS). |
| `.../mdp_params_<slug>/` | `dynamics.npz`, etc., via `icu_sepsis_helpers.build_mimic_params`. |
| `.../coper_mdp_alignment_<slug>.json` | **ICUSTAY_ID** overlap COPER ↔ MDP. |
| `unified_build.json` | Includes **`pipeline_contract`** (grids, schemas, alignment notes). |

## RL table (`mimic_dataset_table.csv` / `MIMICtable.csv`)

- **Typical origin**: **microsoft/mimic_sepsis** pipeline (PostgreSQL MIMIC) → `preprocess.py` + `sepsis_cohort.py` → large file such as **`MIMICtable.csv`** (see `data_mngmt/sepsis_rl/mimic_sepsis_run.py`).
- **One row** = one **time block** of a stay (`icustayid`, `bloc`, …). In this repo’s **unified build default**, blocks are spaced **1 h** apart (`mdp_rl_bloc_interval_hours` → `sepsis_cohort --bloc-interval-hours`), matching COPER’s timestep. You can set a larger interval (e.g. 4 h) if the upstream script supports it.
- **Columns** (non-exhaustive): demographics, vitals, labs, **SOFA**, **SIRS**, inputs (`input_4hourly`, `input_total`, …), outputs, `max_dose_vaso`, etc. — consumed by `icu_sepsis_helpers/mdp_creation/create_rl_table.py` (lists `colbin` / `colnorm` / `collog`). Column names are historical; values are per chosen bloc window.
- The unified build **filters** the sepsis cohort and **adds** `mortality_inhospital` to match the COPER in-hospital mortality label.

## COPER (in-hospital mortality) — 1 h step / 48 h window

- Source: **mimic3-benchmarks** `in_hospital_mortality` (discretizer + normalizer).
- **Time step**: `timestep_minutes / 60` (default **1 h**).
- **Window**: `horizon_hours` (default **48 h**; upstream in-hospital mortality script is calibrated for 48 h).
- **Channels**: benchmark schema (values + masks), **not** the same list as the RL table (no identical “SOFA” column on the tensor side; MDP SOFA lives in the RL table).

## MDP — transitions per RL row

- Discrete states come from **KMeans** on normalized table features (`create_rl_dataset`).
- **SOFA**: present in the RL table, used for clustering and aggregated **sofa_scores** in dynamics.
- **Reward**: terminal (survival / death), defined in `create_matrices.rl_table_to_unnormalized_matrices` and build parameters `r_survive` / `r_death`.

## Alignment “same trajectory”

- **Cohort**: same **sepsis** subset + same **in-hospital mortality** for the MDP label as for the in-hospital mortality listfiles (unified build).
- **Temporal granularity**: with default **1 h** RL blocs, bloc **spacing** matches COPER’s 1 h bins; feature sets and preprocessing still differ (COPER uses benchmark tensors; MDP uses the RL table).
- **Join**: on **`ICUSTAY_ID`**; COPER latents align to **row index** in the pickle; the MDP has **several rows per stay** (blocks). See `alignment_utils.py`.

## Code reference files

- `data_mngmt/contracts/pipeline_contract.py` — constants and serialized `pipeline_contract` dict.
- `data_mngmt/pipeline/unified_build.py` — unified build orchestration.
- `data_mngmt/sepsis_rl/mimic_sepsis_run.py` — RL table construction from the database.
- `data_mngmt/layout.py` — paths `generated/` vs `vendor/` vs `legacy/`.
- `icu_sepsis_helpers/mdp_creation/create_rl_table.py` — MDP schema from CSV.

## MDP granularity (RL row time step)

RL granularity: **`UnifiedBuildParams.mdp_rl_bloc_interval_hours`** (default **1**) is passed to **`sepsis_cohort.py --bloc-interval-hours`** on a Postgres rebuild; a **snapshot** `mimic_dataset_table_src_bloc<N>h.csv` is written under the unified workdir. **Subsequent runs reuse** that snapshot if it exists (non-empty), unless **`mdp_force_rebuild_source_table`** is True.
