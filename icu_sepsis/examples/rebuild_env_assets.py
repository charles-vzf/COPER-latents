#!/usr/bin/env python3
"""Rebuild ICU-Sepsis tabular dynamics and install them for ``Sepsis/ICU-Sepsis-v2``.

The Gymnasium env loads ``dynamics.npz`` + ``metadata.json`` + ``admissible_actions.txt``
from ``icu_sepsis/icu_sepsis/icu_sepsis/envs/assets`` unless you pass a custom
``MDPParameters`` to ``gym.make``.

**Option A — copy only** (MDP already built, e.g. by ``run_unified_build``)::

    python -m data_mngmt.tools.publish_icu_sepsis_env_assets \\
        --mdp-dir data_mngmt/generated/unified/sepsis-60m-h48ihm/mdp_params_sepsis-60m-h48ihm

**Option B — build from cohort CSV then copy** (from repository root, venv active)::

    python icu_sepsis/examples/rebuild_env_assets.py \\
        -i data_mngmt/generated/icu_sepsis_csv_tables/mimic_dataset_table.csv \\
        --out-dir data_mngmt/mdp_params_custom \\
        --outcome-column mortality_inhospital

Use ``--outcome-column mortality_90d`` if your table matches the original demo cohort.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repository root: icu_sepsis/examples → parents[2] == COPER/
_COPER_ROOT = Path(__file__).resolve().parents[2]
if str(_COPER_ROOT) not in sys.path:
    sys.path.insert(0, str(_COPER_ROOT))
for _sub in ("icu_sepsis/icu_sepsis", "icu_sepsis/icu_sepsis_helpers"):
    _p = str((_COPER_ROOT / _sub).resolve())
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=(
            "Build ICU-Sepsis MDP from a cohort CSV and copy dynamics into package envs/assets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="AI Clinician–style RL cohort CSV (e.g. mimic_dataset_table.csv)",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=_COPER_ROOT / "data_mngmt" / "mimic_mdp_params_rebuild",
        help="Where to write MDPParameters before copying into package assets",
    )
    p.add_argument("-s", "--n-states", type=int, default=750)
    p.add_argument("-a", "--n-action-levels", type=int, default=5)
    p.add_argument("-t", "--threshold", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--outcome-column",
        type=str,
        default="mortality_inhospital",
        help="Outcome column in the CSV (unified MIMIC build uses mortality_inhospital)",
    )
    p.add_argument(
        "--no-publish",
        action="store_true",
        help="Only build under --out-dir; do not overwrite package envs/assets",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print copy targets only during publish (MDP build still runs)",
    )
    args = p.parse_args(argv)

    from icu_sepsis_helpers.build import build_mimic_params
    from data_mngmt.tools.publish_icu_sepsis_env_assets import publish_mdp_params_to_pkg_assets

    inp = Path(args.input_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not inp.is_file():
        logging.error("Input CSV not found: %s", inp)
        return 1

    logging.info("Building MDP parameters: %s -> %s", inp, out_dir)
    build_mimic_params(
        str(inp),
        str(out_dir),
        args.n_states,
        args.n_action_levels,
        args.threshold,
        seed=args.seed,
        outcome_column=args.outcome_column,
    )

    if args.no_publish:
        logging.info("Skipping publish (--no-publish). Install manually with:")
        logging.info(
            "  python -m data_mngmt.tools.publish_icu_sepsis_env_assets --mdp-dir %s",
            out_dir,
        )
        return 0

    publish_mdp_params_to_pkg_assets(out_dir, dry_run=args.dry_run)
    logging.info(
        "Done. `gym.make('Sepsis/ICU-Sepsis-v2')` will load dynamics from package assets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
