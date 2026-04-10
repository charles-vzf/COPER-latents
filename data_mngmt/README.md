# `data_mngmt` — MIMIC → COPER + ICU-Sepsis MDP

## Directory layout

| Path | Role |
|------|------|
| **`generated/`** | **Reproducible** outputs from MIMIC: COPER pickle `mortality_coper_<slug>.data`, `unified/<slug>/`, `mimic_sepsis_work/`, published tables when `paths.json` → `icu_sepsis_csv_tables_dir` is set. |
| **`legacy/`** | **Frozen** reference data for comparison; **not** read by `run_unified_build` except via `cohort_csv` or manual extraction. |
| **`vendor/mimic3_benchmarks/`** | [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks): in-hospital mortality extraction from PhysioNet. |
| **`vendor/mimic_sepsis_upstream/`** | [microsoft/mimic_sepsis](https://github.com/microsoft/mimic_sepsis): Postgres → RL table (**MIMICtable**). |

## Python modules (subpackages)

- **`pipeline/`** — `unified_build`, `build_data`, `build_mdp`
- **`mimic/`** — PhysioNet + mimic3-benchmarks + COPER pickle export + `sepsis_icustays`
- **`sepsis_rl/`** — vendoring + running mimic_sepsis (Postgres)
- **`contracts/`** — `pipeline_contract`, `alignment_utils`
- **`tools/`** — `dataset_compare`, `publish_icu_sepsis_env_assets`
- **`layout.py`** — paths for `generated/` / `vendor/` / `legacy/`

Typical imports: `from data_mngmt.pipeline.unified_build import UnifiedBuildParams, run_unified_build`.

## Useful CLIs

| Command | Role |
|---------|------|
| `python -m data_mngmt` | Unified build (same as `notebooks/build_data.ipynb`). |
| `python -m data_mngmt.mimic.mimic3_benchmarks_vendor` | Clone mimic3-benchmarks into `vendor/`. |
| `python -m data_mngmt.sepsis_rl.mimic_sepsis_vendor` | Clone / patch mimic_sepsis into `vendor/mimic_sepsis_upstream/`. |
| `python -m data_mngmt.tools.publish_icu_sepsis_env_assets` | Copy `dynamics.npz` + sidecars into `icu_sepsis/.../envs/assets`. |
| *(notebook)* `notebooks/icu_sepsis_demo.ipynb` | Gymnasium demo on **`generated/unified/<slug>/mdp_params_<slug>/`** (default) vs packaged assets; complements `build_data.ipynb`. |
| `python -m data_mngmt.sepsis_rl.ai_clinician_stack` | AI Clinician stack + optional unified build. |
