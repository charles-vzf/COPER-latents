"""CLI: unified PhysioNet → COPER + ICU-Sepsis MDP build."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from data_mngmt.pipeline.unified_build import UnifiedBuildParams, run_unified_build

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Build unified MIMIC artifacts from PhysioNet CSVs via mimic3-benchmarks: "
            "COPER IHM pickle (explicit filename mortality_coper_<slug>.data) and "
            "optional ICU-Sepsis MDP params (IHM label column for terminal outcomes)."
        )
    )
    parser.add_argument(
        "--physionet",
        type=Path,
        default=None,
        help="MIMIC-III CSV folder (default: paths.json → physionet_mimic_root)",
    )
    parser.add_argument(
        "--mimic3-repo",
        type=Path,
        default=None,
        help="mimic3-benchmarks clone (default: vendored data_mngmt/vendor/mimic3_benchmarks or MIMIC3_BENCHMARKS_REPO)",
    )
    parser.add_argument(
        "--sepsis-cohort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict IHM listfiles (and MDP cohort rows) to sepsis-flagged ICUSTAY_IDs",
    )
    parser.add_argument(
        "--timestep-minutes",
        type=int,
        default=60,
        help="Discretizer bin width in minutes (60 → 1 h step, 48 h → 48 bins with horizon 48)",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=48,
        help="IHM prediction window (hours). Upstream IHM creation is 48 h; other values log a warning.",
    )
    parser.add_argument(
        "--no-mdp",
        action="store_true",
        help="Skip MDP build (COPER pickle only)",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=None,
        help="Explicit RL cohort CSV (otherwise reuse unified slug snapshot or Postgres rebuild; not read from published tables dir)",
    )
    parser.add_argument(
        "--no-mdp-rebuild-table-from-db",
        action="store_true",
        help=(
            "Do not call mimic_sepsis/Postgres; MDP requires --cohort-csv to an existing RL table "
            "(no reuse of workdir or published CSV caches as inputs)"
        ),
    )
    parser.add_argument(
        "--mdp-skip-preprocess",
        action="store_true",
        help="Reuse existing processed_files/ in mimic_sepsis workdir (Postgres preprocess skipped)",
    )
    parser.add_argument(
        "--mdp-force-rebuild-source-table",
        action="store_true",
        help=(
            "Ignore existing generated/unified/<slug>/mimic_dataset_table_src_bloc* snapshot; "
            "always run mimic_sepsis preprocess + sepsis_cohort against Postgres"
        ),
    )
    parser.add_argument(
        "--mdp-mimic-sepsis-workdir",
        type=Path,
        default=None,
        help="Working dir for preprocess/sepsis_cohort (default: <unified>/mimic_sepsis_build)",
    )
    parser.add_argument(
        "--force-rebuild-benchmark",
        action="store_true",
        help="Delete and rebuild benchmark_root + in-hospital-mortality under generated/unified/<slug>/",
    )
    parser.add_argument(
        "--mdp-n-states",
        type=int,
        default=750,
    )
    parser.add_argument(
        "--mdp-action-levels",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mdp-threshold",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--mdp-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--mdp-rl-bloc-interval-hours",
        type=float,
        default=1.0,
        help=(
            "Hours per RL row: passed to vendored sepsis_cohort --bloc-interval-hours when rebuilding "
            "from Postgres; also recorded in pipeline_contract. Integer hours (e.g. 1, 2, 4). Default 1 matches COPER."
        ),
    )
    parser.add_argument(
        "--publish-gym-env-assets",
        action="store_true",
        help=(
            "After MDP build, copy dynamics.npz + metadata.json + admissible_actions.txt into "
            "icu_sepsis/.../envs/assets (Sepsis/ICU-Sepsis-v2 default load path)"
        ),
    )
    args = parser.parse_args(argv)

    params = UnifiedBuildParams(
        sepsis_cohort=args.sepsis_cohort,
        timestep_minutes=args.timestep_minutes,
        horizon_hours=args.horizon_hours,
        build_mdp=not args.no_mdp,
        force_rebuild_benchmark=args.force_rebuild_benchmark,
        cohort_csv=args.cohort_csv,
        mdp_rebuild_table_from_db=(False if args.no_mdp else not args.no_mdp_rebuild_table_from_db),
        mdp_skip_preprocess=args.mdp_skip_preprocess,
        mdp_force_rebuild_source_table=args.mdp_force_rebuild_source_table,
        mdp_mimic_sepsis_workdir=args.mdp_mimic_sepsis_workdir,
        mdp_n_states=args.mdp_n_states,
        mdp_n_action_levels=args.mdp_action_levels,
        mdp_threshold=args.mdp_threshold,
        mdp_seed=args.mdp_seed,
        publish_gym_env_assets=args.publish_gym_env_assets,
        mdp_rl_bloc_interval_hours=args.mdp_rl_bloc_interval_hours,
    )
    try:
        out = run_unified_build(
            params,
            physionet_csv_dir=args.physionet,
            mimic3_benchmarks_repo=args.mimic3_repo,
        )
    except Exception as e:
        log.exception("build_data failed: %s", e)
        return 1
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
