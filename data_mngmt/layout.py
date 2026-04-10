"""Canonical paths under ``data_mngmt/`` (MIMIC-derived outputs vs legacy reference).

- ``generated/`` — artifacts produced from PhysioNet / Postgres / benchmark CSVs:
  COPER pickles, ``unified/<slug>/``, published RL tables (via ``paths.json``), ``mimic_sepsis`` work.
- ``legacy/`` — frozen bundles for comparison (e.g. ICU-Sepsis tarball), **not** used by the unified build.
- ``vendor/mimic3_benchmarks/`` — clone of [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks):
  ``extract_subjects``, ``create_in_hospital_mortality``, normalizers, in-hospital mortality readers.
- ``vendor/mimic_sepsis_upstream/`` — patched copy of [microsoft/mimic_sepsis](https://github.com/microsoft/mimic_sepsis):
  ``preprocess.py`` (SQL → ``processed_files/``) + ``sepsis_cohort.py`` (RL block aggregation / MIMICtable).
"""
from __future__ import annotations

from pathlib import Path

from data_mngmt import coper_root


def generated_root(repo: Path | None = None) -> Path:
    """``<repo>/data_mngmt/generated`` (pickles COPER, unified workdirs, mimic_sepsis work, etc.)."""
    root = Path(repo or coper_root()).resolve()
    return (root / "data_mngmt" / "generated").resolve()


def coper_mortality_pickle_path(repo: Path | None = None, *, slug: str) -> Path:
    """COPER in-hospital mortality pickle path: ``generated/mortality_coper_<slug>.data``."""
    return (generated_root(repo) / f"mortality_coper_{slug}.data").resolve()


def vendor_mimic3_benchmarks(repo: Path | None = None) -> Path:
    return (Path(repo or coper_root()).resolve() / "data_mngmt" / "vendor" / "mimic3_benchmarks").resolve()


def vendor_mimic_sepsis(repo: Path | None = None) -> Path:
    return (Path(repo or coper_root()).resolve() / "data_mngmt" / "vendor" / "mimic_sepsis_upstream").resolve()


def legacy_root(repo: Path | None = None) -> Path:
    return (Path(repo or coper_root()).resolve() / "data_mngmt" / "legacy").resolve()
