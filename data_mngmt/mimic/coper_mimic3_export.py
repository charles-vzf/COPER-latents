"""Export COPER-style mortality pickle from mimic3-benchmarks IHM tensors."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np

from data_mngmt import coper_root
from data_mngmt.mimic.mimic3_benchmarks_vendor import default_vendor_dest


def _prepend_sys_path(repo: Path) -> None:
    s = str(repo.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def icustay_ids_for_ihm_listfiles(
    benchmark_episode_root: Path,
    ihm_data_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Map each IHM listfile row (same order as InHospitalMortalityReader) to ICUSTAY_ID.

    ``benchmark_episode_root`` is the ``root`` folder containing ``train/`` and ``test/`` subject
    episode CSVs (see mimic3-benchmarks README). Missing stems get ``-1``.
    """
    from data_mngmt.mimic.mimic_physionet_pipeline import build_stem_to_icustay

    stem_map = build_stem_to_icustay(Path(benchmark_episode_root).resolve())
    missing_total = 0
    out: dict[str, np.ndarray] = {}
    miss_split: dict[str, int] = {}
    for split, fname in (
        ("train", "train_listfile.csv"),
        ("val", "val_listfile.csv"),
        ("test", "test_listfile.csv"),
    ):
        path = Path(ihm_data_root) / fname
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            out[split] = np.array([], dtype=np.int64)
            miss_split[split] = 0
            continue
        ids: list[int] = []
        miss = 0
        for line in lines[1:]:
            stem = line.split(",")[0].strip()
            icu = stem_map.get(stem)
            if icu is None:
                miss += 1
                ids.append(-1)
            else:
                ids.append(int(icu))
        out[split] = np.array(ids, dtype=np.int64)
        miss_split[split] = miss
        missing_total += miss
    return out, miss_split


def export_mortality_pickle_for_coper(
    *,
    mimic3_benchmarks_repo: Path,
    ihm_data_root: Path,
    out_path: Path,
    timestep: float = 1.0,
    period_length: float = 48.0,
    extra_details: dict | None = None,
    benchmark_episode_root: Path | None = None,
) -> dict:
    """Stack train/val/test arrays like ``export_mortality_pickle_for_coper.py`` in mimic3-benchmarks.

    ``period_length`` is passed to ``InHospitalMortalityReader`` (hours of context).
    ``timestep`` is the discretizer bin width in hours (e.g. 1.0 → 48 bins for 48h).

    Requires a matching normalizer state file under
    ``mimic3models/in_hospital_mortality/ihm_ts{timestep}....`` inside the benchmarks repo.

    If ``benchmark_episode_root`` is set (directory with ``train/`` and ``test/`` episode CSVs from
    mimic3-benchmarks ``extract_episodes`` + ``split_train_and_test``), ``details['icustay_id']`` is
    filled with ``train``/``val``/``test`` arrays aligned row-wise with ``X_*`` (same order as
    listfiles). Use ``-1`` if a stay stem cannot be mapped. Enables joins with the MDP RL cohort
    table on ``icustayid`` / ``ICUSTAY_ID``.

    If ``None``, tries ``mimic3_benchmarks_repo / "data" / "root"`` when that directory exists.
    """
    if period_length <= 0 or timestep <= 0:
        raise ValueError("period_length and timestep must be positive")

    repo = mimic3_benchmarks_repo.resolve()
    data_root = ihm_data_root.resolve()
    for name in ("train_listfile.csv", "val_listfile.csv", "test_listfile.csv"):
        if not (data_root / name).is_file():
            raise FileNotFoundError(
                f"Missing {data_root / name}. Run the mimic3-benchmarks "
                f"in-hospital-mortality creation scripts first (see mimic3benchmark/scripts/create_in_hospital_mortality.py)."
            )
    _prepend_sys_path(repo)

    from mimic3benchmark.readers import InHospitalMortalityReader
    from mimic3models import common_utils
    from mimic3models.preprocessing import Discretizer, Normalizer

    def load_split(reader: InHospitalMortalityReader, discretizer, normalizer):
        n = reader.get_number_of_examples()
        ret = common_utils.read_chunk(reader, n)
        data = ret["X"]
        ts = ret["t"]
        labels = np.array(ret["y"], dtype=np.float32)
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if normalizer is not None:
            data = [normalizer.transform(X) for X in data]
        X = np.stack(data, axis=0).astype(np.float32)
        y = labels.astype(np.float32).reshape(-1)
        return X, y

    train_reader = InHospitalMortalityReader(
        dataset_dir=str(data_root / "train"),
        listfile=str(data_root / "train_listfile.csv"),
        period_length=float(period_length),
    )
    val_reader = InHospitalMortalityReader(
        dataset_dir=str(data_root / "train"),
        listfile=str(data_root / "val_listfile.csv"),
        period_length=float(period_length),
    )
    test_reader = InHospitalMortalityReader(
        dataset_dir=str(data_root / "test"),
        listfile=str(data_root / "test_listfile.csv"),
        period_length=float(period_length),
    )

    discretizer = Discretizer(
        timestep=float(timestep),
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
    )
    _ = discretizer.transform(train_reader.read_example(0)["X"])
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(",")
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)
    state_name = "ihm_ts{}.input_str-previous.start_time-zero.normalizer".format(timestep)
    state_path = repo / "mimic3models" / "in_hospital_mortality" / state_name
    if not state_path.is_file():
        raise FileNotFoundError(
            f"Normalizer state not found: {state_path}\n"
            f"Use timestep supported by your mimic3-benchmarks checkout (often 1.0), "
            f"or train/save a normalizer as in mimic3models."
        )
    normalizer.load_params(str(state_path))

    X_train, y_train = load_split(train_reader, discretizer, normalizer)
    X_val, y_val = load_split(val_reader, discretizer, normalizer)
    X_test, y_test = load_split(test_reader, discretizer, normalizer)

    n_steps = X_train.shape[1]
    implied = period_length / timestep
    if abs(n_steps - implied) > 0.51:
        # rounding / reader behavior
        pass

    details = {
        "format": "mimic3-benchmarks in-hospital-mortality + discretizer/normalizer",
        "timestep": timestep,
        "period_length": period_length,
        "horizon_mode": "first_n_hours",
        "shapes": {
            "train": X_train.shape,
            "val": X_val.shape,
            "test": X_test.shape,
        },
    }
    if extra_details:
        details.update(extra_details)

    ep_root = benchmark_episode_root
    if ep_root is None:
        cand = repo / "data" / "root"
        if cand.is_dir():
            ep_root = cand
    if ep_root is not None and Path(ep_root).is_dir():
        try:
            icu_map, miss = icustay_ids_for_ihm_listfiles(Path(ep_root), data_root)
            details["icustay_id"] = icu_map
            details["icustay_id_stem_missing"] = miss
            details["benchmark_episode_root"] = str(Path(ep_root).resolve())
        except Exception as e:
            details["icustay_id_error"] = str(e)

    if n_steps != 48:
        details["coper_warning"] = (
            f"time dimension is {n_steps}; default COPER mimic training expects seq_len=48. "
            "Adjust model / period_hours / timestep or retrain."
        )

    payload = (details, X_train, y_train, X_val, y_val, X_test, y_test, None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    return {
        "out_path": str(out_path),
        "details": details,
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
    }


def default_mimic3_repo() -> Path:
    """Resolve mimic3-benchmarks tree: env ``MIMIC3_BENCHMARKS_REPO``, else vendored copy.

    One-time setup::

        python -m data_mngmt.mimic.mimic3_benchmarks_vendor
        pip install -r data_mngmt/vendor/mimic3_benchmarks/requirements.txt
    """
    env = os.environ.get("MIMIC3_BENCHMARKS_REPO")
    if env:
        p = Path(env).expanduser()
        return p if p.is_absolute() else (coper_root() / p).resolve()
    root = default_vendor_dest(coper_root())
    if not (root / "mimic3benchmark").is_dir():
        raise FileNotFoundError(
            "mimic3-benchmarks not found. Run:\n"
            "  python -m data_mngmt.mimic.mimic3_benchmarks_vendor\n"
            "  pip install -r data_mngmt/vendor/mimic3_benchmarks/requirements.txt\n"
            "Or set MIMIC3_BENCHMARKS_REPO to a local clone."
        )
    return root.resolve()


def default_ihm_data_root(mimic3_repo: Path | None = None) -> Path:
    root = mimic3_repo or default_mimic3_repo()
    return (root / "data" / "in-hospital-mortality").resolve()
