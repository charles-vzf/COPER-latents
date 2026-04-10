"""Run YerevaNN/mimic3-benchmarks extraction from PhysioNet MIMIC-III CSV folders."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _env_with_repo(mimic3_repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    extra = str(mimic3_repo.resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = extra if not prev else f"{extra}{os.pathsep}{prev}"
    return env


def run_subprocess_mimic3(
    mimic3_repo: Path,
    module: str,
    args: list[str | Path],
    *,
    env: dict[str, str] | None = None,
) -> None:
    cmd = [sys.executable, "-m", module, *[str(a) for a in args]]
    log.info("Running: %s (cwd=%s)", " ".join(cmd), mimic3_repo)
    subprocess.run(cmd, check=True, cwd=str(mimic3_repo), env=env or _env_with_repo(mimic3_repo))


def ensure_benchmark_episodes(
    physionet_csv_dir: Path,
    mimic3_repo: Path,
    benchmark_root: Path,
    *,
    force: bool = False,
) -> Path:
    """Steps 1–4 of mimic3-benchmarks README; writes ``benchmark_root`` (…/root)."""
    train_pt = benchmark_root / "train"
    if not force and train_pt.is_dir() and any(train_pt.iterdir()):
        # Reuse only if episodic outputs look complete enough for downstream IHM.
        # A stale/partial root (e.g., interrupted extract_episodes) can still have
        # subject folders but no *_timeseries.csv files.
        has_timeseries = any(benchmark_root.glob("train/*/*_timeseries.csv")) or any(
            benchmark_root.glob("test/*/*_timeseries.csv")
        )
        if has_timeseries:
            log.info("Reusing existing benchmark_root: %s", benchmark_root)
            return benchmark_root
        log.warning(
            "benchmark_root exists but has no episode timeseries files. Rebuilding: %s",
            benchmark_root,
        )

    benchmark_root.parent.mkdir(parents=True, exist_ok=True)
    if force and benchmark_root.exists():
        import shutil

        shutil.rmtree(benchmark_root)

    env = _env_with_repo(mimic3_repo)
    run_subprocess_mimic3(
        mimic3_repo,
        "mimic3benchmark.scripts.extract_subjects",
        [physionet_csv_dir, benchmark_root],
        env=env,
    )
    run_subprocess_mimic3(
        mimic3_repo,
        "mimic3benchmark.scripts.validate_events",
        [benchmark_root],
        env=env,
    )
    run_subprocess_mimic3(
        mimic3_repo,
        "mimic3benchmark.scripts.extract_episodes_from_subjects",
        [benchmark_root],
        env=env,
    )
    run_subprocess_mimic3(
        mimic3_repo,
        "mimic3benchmark.scripts.split_train_and_test",
        [benchmark_root],
        env=env,
    )
    return benchmark_root


def ensure_in_hospital_mortality_task(
    mimic3_repo: Path,
    benchmark_root: Path,
    ihm_root: Path,
    *,
    force: bool = False,
) -> Path:
    """create_in_hospital_mortality + split_train_val."""
    list_train = ihm_root / "train_listfile.csv"
    if not force and list_train.is_file():
        # Guard against partial/failed runs that leave an empty listfile (header only).
        try:
            n_lines = sum(1 for _ in list_train.open("r", encoding="utf-8"))
        except Exception:
            n_lines = 0
        if n_lines > 1:
            log.info("Reusing existing IHM task dir: %s", ihm_root)
            return ihm_root
        log.warning(
            "IHM listfile exists but has no samples (%s lines). Rebuilding IHM task.",
            n_lines,
        )

    ihm_root.parent.mkdir(parents=True, exist_ok=True)
    if force and ihm_root.exists():
        import shutil

        shutil.rmtree(ihm_root)

    env = _env_with_repo(mimic3_repo)
    run_subprocess_mimic3(
        mimic3_repo,
        "mimic3benchmark.scripts.create_in_hospital_mortality",
        [benchmark_root, ihm_root],
        env=env,
    )
    run_subprocess_mimic3(
        mimic3_repo,
        "mimic3models.split_train_val",
        [ihm_root],
        env=env,
    )
    return ihm_root


_STAY_RE = re.compile(r"^(\d+)_episode(\d+)_timeseries\.csv$")


def build_stem_to_icustay(benchmark_root: Path) -> dict[str, int]:
    """Map IHM stay filename stem (as in listfile) -> ICUSTAY_ID."""
    out: dict[str, int] = {}
    skipped_empty = 0
    skipped_invalid = 0
    for part in ("train", "test"):
        base = benchmark_root / part
        if not base.is_dir():
            continue
        for subj_dir in base.iterdir():
            if not subj_dir.is_dir() or not subj_dir.name.isdigit():
                continue
            sid = subj_dir.name
            for ep_csv in subj_dir.glob("episode*.csv"):
                if "timeseries" in ep_csv.name:
                    continue
                m = re.match(r"episode(\d+)\.csv$", ep_csv.name)
                if not m:
                    continue
                k = int(m.group(1))
                stem = f"{sid}_episode{k}_timeseries.csv"
                try:
                    head = pd.read_csv(ep_csv, nrows=4)
                except pd.errors.EmptyDataError:
                    skipped_empty += 1
                    continue

                icu: int | None = None
                if "Icustay" in head.columns and not head.empty:
                    col = pd.to_numeric(head["Icustay"], errors="coerce").dropna()
                    if not col.empty:
                        icu = int(col.iloc[0])
                elif not head.empty:
                    idx = pd.read_csv(ep_csv, index_col=0, nrows=1)
                    if len(idx.index) > 0:
                        try:
                            icu = int(idx.index[0])
                        except (TypeError, ValueError):
                            icu = None

                if icu is None:
                    skipped_invalid += 1
                    continue
                out[stem] = icu
    log.info(
        "Mapped %d stay stems -> ICUSTAY_ID (skipped empty=%d invalid=%d)",
        len(out),
        skipped_empty,
        skipped_invalid,
    )
    return out


def filter_ihm_listfiles_to_icustays(
    ihm_root: Path,
    stem_to_icu: dict[str, int],
    keep_ids: set[int] | None,
) -> dict[str, int]:
    """Subset train/val/test listfiles to stays whose ICUSTAY_ID is in ``keep_ids``.

    If ``keep_ids`` is None, no filtering (returns zeros).
    Returns counts of kept rows per listfile basename.
    """
    if keep_ids is None:
        return {}

    counts: dict[str, int] = {}
    for name in ("train_listfile.csv", "val_listfile.csv", "test_listfile.csv"):
        path = ihm_root / name
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue
        header = lines[0]
        kept: list[str] = [header]
        for line in lines[1:]:
            stem = line.split(",")[0].strip()
            icu = stem_to_icu.get(stem)
            if icu is None or icu not in keep_ids:
                continue
            kept.append(line)
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        counts[name] = len(kept) - 1
        log.info("Filtered %s: %d stays", name, counts[name])
    return counts
