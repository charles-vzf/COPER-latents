"""Derive a sepsis ICU stay set from MIMIC-III PhysioNet CSV tables (no SQL)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _resolve_mimic_table(physionet_csv_dir: Path, stem: str) -> Path:
    for name in (f"{stem}.csv", f"{stem}.csv.gz"):
        p = physionet_csv_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Missing {stem}.csv (or .csv.gz) under {physionet_csv_dir}"
    )


def icd9_codes_indicate_sepsis(icd9: str) -> bool:
    """Heuristic ICD-9 flags (bacteremia / explicit severe sepsis / septic shock).

    This is **not** full Angus criteria; it is a practical filter for research builds.
    Extend the rule set if you need stricter epidemiology.
    """
    s = str(icd9).strip().upper().replace(".", "")
    if not s:
        return False
    if s.startswith("038"):
        return True
    if s.startswith("99592") or s == "99592":
        return True
    if s.startswith("78552") or s == "78552":
        return True
    if s in ("7907", "6758", "77181", "67020", "67024", "65830", "65930", "65150", "65940"):
        return True
    return False


def sepsis_hadm_ids_from_diagnoses(physionet_csv_dir: Path, chunksize: int = 400_000) -> set[int]:
    path = _resolve_mimic_table(physionet_csv_dir, "DIAGNOSES_ICD")
    hadm: set[int] = set()
    reader = pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=["HADM_ID", "ICD9_CODE"],
        dtype={"HADM_ID": "Int64", "ICD9_CODE": "string"},
        low_memory=False,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["HADM_ID"])
        mask = chunk["ICD9_CODE"].fillna("").map(icd9_codes_indicate_sepsis)
        if mask.any():
            hadm.update(chunk.loc[mask, "HADM_ID"].astype(int).unique().tolist())
    log.info("Sepsis-related HADM_ID count (from DIAGNOSES_ICD): %d", len(hadm))
    return hadm


def sepsis_icustay_ids(physionet_csv_dir: Path) -> set[int]:
    """ICUSTAY_IDs with at least one sepsis-flagged ICD-9 on the same HADM_ID."""
    hadm_sepsis = sepsis_hadm_ids_from_diagnoses(physionet_csv_dir)
    icu_path = _resolve_mimic_table(physionet_csv_dir, "ICUSTAYS")
    icu = pd.read_csv(
        icu_path,
        usecols=["ICUSTAY_ID", "HADM_ID"],
        dtype={"ICUSTAY_ID": "Int64", "HADM_ID": "Int64"},
    )
    icu = icu.dropna(subset=["ICUSTAY_ID", "HADM_ID"])
    icu = icu[icu["HADM_ID"].astype(int).isin(hadm_sepsis)]
    out = set(icu["ICUSTAY_ID"].astype(int).unique().tolist())
    log.info("Sepsis ICUSTAY_ID count: %d", len(out))
    return out


def inhospital_mortality_by_icustay(physionet_csv_dir: Path) -> dict[int, int]:
    """Map ICUSTAY_ID -> HOSPITAL_EXPIRE_FLAG (0/1) via HADM_ID."""
    icu_path = _resolve_mimic_table(physionet_csv_dir, "ICUSTAYS")
    adm_path = _resolve_mimic_table(physionet_csv_dir, "ADMISSIONS")
    icu = pd.read_csv(
        icu_path,
        usecols=["ICUSTAY_ID", "HADM_ID"],
        dtype={"ICUSTAY_ID": "Int64", "HADM_ID": "Int64"},
    )
    adm = pd.read_csv(
        adm_path,
        usecols=["HADM_ID", "HOSPITAL_EXPIRE_FLAG"],
        dtype={"HADM_ID": "Int64", "HOSPITAL_EXPIRE_FLAG": "Int64"},
    )
    icu = icu.dropna(subset=["ICUSTAY_ID", "HADM_ID"])
    m = icu.merge(adm, on="HADM_ID", how="left")
    m["HOSPITAL_EXPIRE_FLAG"] = m["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int).clip(0, 1)
    return dict(zip(m["ICUSTAY_ID"].astype(int), m["HOSPITAL_EXPIRE_FLAG"]))
