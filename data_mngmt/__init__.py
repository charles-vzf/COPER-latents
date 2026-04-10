"""Data management: resolve paths from ``code/COPER/paths.json`` (single source of truth)."""
from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "coper_root",
    "load_paths",
    "resolve_path",
    "postgres_login_path",
    "load_postgres_login",
    "mimic3_benchmarks_repo",
    "mortality_pickle_path",
    "mortality_pickle_legacy_path",
    "icu_sepsis_csv_archive_path",
    "icu_sepsis_csv_legacy_path",
    "icu_sepsis_csv_tables_dir",
    "physionet_mimic_root",
]


def coper_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_paths() -> dict:
    p = coper_root() / "paths.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(key: str) -> Path:
    data = load_paths()
    if key not in data:
        raise KeyError(f"Missing key {key!r} in paths.json")
    raw = data[key]
    if raw is None or raw == "":
        raise ValueError(f"Empty path for {key!r} in paths.json")
    path = Path(raw)
    return path if path.is_absolute() else (coper_root() / path).resolve()


def postgres_login_path() -> Path:
    """Path to JSON credentials with keys: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD."""
    return resolve_path("postgres_login_json")


def load_postgres_login(*, strict: bool = True) -> dict[str, str]:
    """Load PostgreSQL credentials from ``paths.json`` -> ``postgres_login_json``.

    Expected JSON fields:
      - PGHOST
      - PGPORT
      - PGDATABASE
      - PGUSER
      - PGPASSWORD
    """
    p = postgres_login_path()
    if not p.is_file():
        if strict:
            raise FileNotFoundError(
                f"Postgres login JSON missing: {p}. "
                "Set paths.json -> postgres_login_json and create the file "
                "(see postgres_login.dummy.json template at repo root)."
            )
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    required = ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD")
    missing = [k for k in required if not str(data.get(k, "")).strip()]
    if missing and strict:
        raise ValueError(f"Missing Postgres login keys in {p}: {missing}")
    return {k: str(data.get(k, "")).strip() for k in required}


def mimic3_benchmarks_repo() -> Path:
    """Vendored tree under ``data_mngmt/vendor/mimic3_benchmarks`` or ``MIMIC3_BENCHMARKS_REPO``."""
    from data_mngmt.mimic.coper_mimic3_export import default_mimic3_repo

    return default_mimic3_repo()


def mortality_pickle_path() -> Path:
    return resolve_path("mimic3_mortality")


def mortality_pickle_legacy_path() -> Path:
    return resolve_path("mimic3_mortality_legacy")


def _resolve_legacy_or_archive_tarball() -> Path:
    data = load_paths()
    raw = data.get("icu_sepsis_csv_legacy") or data.get("icu_sepsis_csv_archive")
    if not raw:
        raise KeyError(
            "paths.json: set icu_sepsis_csv_legacy or icu_sepsis_csv_archive to the optional tarball path"
        )
    path = Path(raw)
    return path if path.is_absolute() else (coper_root() / path).resolve()


def icu_sepsis_csv_archive_path() -> Path:
    """Optional ICU-Sepsis CSV bundle tarball (``icu_sepsis_csv_legacy`` or ``icu_sepsis_csv_archive``)."""
    return _resolve_legacy_or_archive_tarball()


def icu_sepsis_csv_legacy_path() -> Path:
    return _resolve_legacy_or_archive_tarball()


def icu_sepsis_csv_tables_dir() -> Path:
    return resolve_path("icu_sepsis_csv_tables_dir")


def physionet_mimic_root() -> Path:
    return resolve_path("physionet_mimic_root")
