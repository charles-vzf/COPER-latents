# MDP row spacing (RL table)

- **Knob:** `UnifiedBuildParams.mdp_rl_bloc_interval_hours` (CLI `--mdp-rl-bloc-interval-hours`, notebook `MDP_RL_BLOC_INTERVAL_HOURS`).
- **Postgres rebuild:** passed to vendored `vendor/mimic_sepsis_upstream/sepsis_cohort.py` as `--bloc-interval-hours` (integer hours: 1, 2, 4, …). It sets the aggregation window in the “data combination” loops (`timestep` in that script).
- **Snapshot + reuse:** each Postgres-backed run writes `data_mngmt/generated/unified/<slug>/mimic_dataset_table_src_bloc<N>h.csv`. **Later runs reuse** that file if present (non-empty) and **`mdp_force_rebuild_source_table`** is false—so Postgres is skipped unless you delete the snapshot or force a rebuild. Use **`cohort_csv`** for an explicit RL-only input without Postgres.
- **Published** `icu_sepsis_csv_tables_dir/mimic_dataset_table.csv` is a **publish target** after a successful MDP build, not an input source for `ensure_mdp_source_csv`.

Column names such as `input_4hourly` / `output_4hourly` are unchanged; values are per chosen window.
