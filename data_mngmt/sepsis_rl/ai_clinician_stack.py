"""End-to-end: vendor/run [microsoft/mimic_sepsis](https://github.com/microsoft/mimic_sepsis), then unified COPER + MDP.

1. **PostgreSQL MIMIC-III** (schema ``mimiciii``): ``preprocess.py`` extracts ``processed_files/``.
2. **sepsis_cohort.py** (configurable bloc hours, default **1 h** in unified build): writes ``MIMICtable.csv`` with ``--save_intermediate``.
3. **PhysioNet CSV tree** + mimic3-benchmarks: ``run_unified_build`` exports the COPER IHM pickle and builds MDP from the table copy.

COPER tensors still come from **PhysioNet + mimic3-benchmarks**, not from the database extract; the RL table from step 2 supplies the **MDP** cohort aligned with the same sepsis definition when you use the unified sepsis filter.

Python deps for preprocess/sepsis_cohort are listed in the repo root ``requirements.txt``.

See also ``data_mngmt/DATASET_PIPELINE.md`` and ``data_mngmt/pipeline_contract.py`` for COPER vs MDP alignment.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from data_mngmt.pipeline.unified_build import UnifiedBuildParams, run_unified_build
from data_mngmt.sepsis_rl.mimic_sepsis_run import (
    MIMICTABLE_NAME,
    build_rl_cohort_csv_from_mimic_db,
    default_workdir,
)
from data_mngmt.sepsis_rl.mimic_sepsis_vendor import (
    DEFAULT_GIT_URL,
    HTTPS_GIT_URL,
    vendor_mimic_sepsis,
)

log = logging.getLogger(__name__)


def run_ai_clinician_table_only(
    workdir: Path | None = None,
    *,
    skip_preprocess: bool = False,
    save_intermediate: bool = True,
    bloc_interval_hours: float = 1.0,
    pg_user: str | None = None,
    pg_password: str | None = None,
    pg_host: str | None = None,
    pg_port: int | None = None,
    pg_dbname: str | None = None,
) -> dict[str, str]:
    """Vendor (if needed), preprocess, sepsis_cohort, copy ``MIMICtable`` → RL cohort CSV path."""
    workdir = Path(workdir or default_workdir()).resolve()
    cohort_csv = build_rl_cohort_csv_from_mimic_db(
        workdir,
        skip_preprocess=skip_preprocess,
        save_intermediate=save_intermediate,
        bloc_interval_hours=bloc_interval_hours,
        username=pg_user,
        password=pg_password,
        host=pg_host,
        port=pg_port,
        dbname=pg_dbname,
    )
    return {
        "workdir": str(workdir),
        "mimic_table_csv": str(workdir / "MIMICtable.csv"),
        "cohort_csv_for_mdp": str(cohort_csv),
    }


def run_full_coper_mdp_stack(
    workdir: Path | None = None,
    *,
    skip_preprocess: bool = False,
    vendor_force: bool = False,
    unified: UnifiedBuildParams | None = None,
    pg_user: str | None = None,
    pg_password: str | None = None,
    pg_host: str | None = None,
    pg_port: int | None = None,
    pg_dbname: str | None = None,
) -> dict:
    """Build AI Clinician table from DB, then ``run_unified_build`` with that cohort CSV."""
    if vendor_force:
        vendor_mimic_sepsis(force=True)
    table_info = run_ai_clinician_table_only(
        workdir=workdir,
        skip_preprocess=skip_preprocess,
        pg_user=pg_user,
        pg_password=pg_password,
        pg_host=pg_host,
        pg_port=pg_port,
        pg_dbname=pg_dbname,
    )
    base = unified or UnifiedBuildParams()
    p = replace(base, cohort_csv=Path(table_info["cohort_csv_for_mdp"]))
    out = run_unified_build(params=p)
    out["ai_clinician_table"] = table_info
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="microsoft/mimic_sepsis + unified COPER/MDP build",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("vendor", help="Clone mimic_sepsis into data_mngmt/vendor/mimic_sepsis_upstream/")
    v.add_argument("--force", action="store_true")
    v.add_argument("--https", action="store_true", help="HTTPS instead of SSH clone")

    t = sub.add_parser("table", help="Preprocess DB + sepsis_cohort → MIMICtable + cohort CSV copy")
    t.add_argument("--workdir", type=Path, default=None)
    t.add_argument("--skip-preprocess", action="store_true")
    t.add_argument("--pg-user", default=None)
    t.add_argument("--pg-password", default=None)
    t.add_argument("--pg-host", default=None)
    t.add_argument("--pg-port", type=int, default=None)
    t.add_argument("--pg-dbname", default=None)

    f = sub.add_parser(
        "full",
        help="table + run_unified_build (PhysioNet COPER pickle + MDP from generated cohort)",
    )
    f.add_argument("--workdir", type=Path, default=None)
    f.add_argument("--skip-preprocess", action="store_true")
    f.add_argument("--vendor-force", action="store_true")
    f.add_argument("--no-mdp", action="store_true")
    f.add_argument("--force-rebuild-benchmark", action="store_true")
    f.add_argument("--pg-user", default=None)
    f.add_argument("--pg-password", default=None)
    f.add_argument("--pg-host", default=None)
    f.add_argument("--pg-port", type=int, default=None)
    f.add_argument("--pg-dbname", default=None)

    args = p.parse_args(argv)

    try:
        if args.cmd == "vendor":
            vendor_mimic_sepsis(
                git_url=HTTPS_GIT_URL if args.https else DEFAULT_GIT_URL,
                force=args.force,
            )
        elif args.cmd == "table":
            info = run_ai_clinician_table_only(
                workdir=args.workdir,
                skip_preprocess=args.skip_preprocess,
                pg_user=args.pg_user,
                pg_password=args.pg_password,
                pg_host=args.pg_host,
                pg_port=args.pg_port,
                pg_dbname=args.pg_dbname,
            )
            print(json.dumps(info, indent=2))
        elif args.cmd == "full":
            u = UnifiedBuildParams(
                build_mdp=not args.no_mdp,
                force_rebuild_benchmark=args.force_rebuild_benchmark,
            )
            out = run_full_coper_mdp_stack(
                workdir=args.workdir,
                skip_preprocess=args.skip_preprocess,
                vendor_force=args.vendor_force,
                unified=u,
                pg_user=args.pg_user,
                pg_password=args.pg_password,
                pg_host=args.pg_host,
                pg_port=args.pg_port,
                pg_dbname=args.pg_dbname,
            )
            print(json.dumps(out, indent=2, default=str))
    except Exception as e:
        log.exception("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
