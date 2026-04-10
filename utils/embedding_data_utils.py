"""Data and latent collection helpers for embedding visualization notebooks."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def load_xy_split(pickle_path, split: str):
    with open(pickle_path, "rb") as f:
        _details, X_train, y_train, X_val, y_val, X_test, y_test, _ = pickle.load(f)
    if split == "train":
        return X_train, y_train
    if split == "val":
        return X_val, y_val
    if split == "test":
        return X_test, y_test
    raise ValueError(split)


def load_icustay_ids_split(pickle_path, split: str) -> np.ndarray:
    """ICUSTAY_ID per row in ``load_xy_split`` order; ``-1`` if missing in pickle details."""
    with open(pickle_path, "rb") as f:
        details, X_train, y_train, X_val, y_val, X_test, y_test, _ = pickle.load(f)
    if not isinstance(details, dict):
        raise ValueError("mortality pickle details must be a dict with icustay_id splits")
    icu = details.get("icustay_id")
    if not icu:
        raise ValueError(
            "Pickle has no details['icustay_id']; rebuild with unified export "
            "(benchmark_episode_root set) so RL-table joins are possible."
        )
    if split == "train":
        return np.asarray(icu["train"], dtype=np.int64).reshape(-1)
    if split == "val":
        return np.asarray(icu["val"], dtype=np.int64).reshape(-1)
    if split == "test":
        return np.asarray(icu["test"], dtype=np.int64).reshape(-1)
    raise ValueError(split)


def sofa_per_icustay_from_mdp_cohort(
    cohort_csv: Path | str,
    *,
    horizon_hours: float = 48.0,
    bloc_interval_hours: float = 1.0,
    agg: str = "max",
) -> pd.Series:
    """Aggregate **real** SOFA from the sepsis RL / MDP cohort CSV (first ``horizon_hours`` of ICU).

    Rows are one bloc per ``bloc_interval_hours``. Bloc indices are assumed 1-based hours when
    ``bloc_interval_hours == 1`` (unified default). For 2 h blocs, ``max_bloc`` scales accordingly.
    """
    path = Path(cohort_csv)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    icu_col = None
    for c in ("icustayid", "ICUSTAY_ID", "icustay_id"):
        if c in df.columns:
            icu_col = c
            break
    if icu_col is None:
        raise KeyError(f"No ICU stay id column in {path}")
    sofa_col = next((c for c in ("SOFA", "sofa", "sofa_score") if c in df.columns), None)
    if sofa_col is None:
        raise KeyError(f"No SOFA column in {path} (expected SOFA / sofa / sofa_score)")
    if "bloc" not in df.columns:
        raise KeyError(f"No bloc column in {path}")
    max_bloc = int(np.ceil(float(horizon_hours) / float(bloc_interval_hours)))
    b = pd.to_numeric(df["bloc"], errors="coerce")
    s = pd.to_numeric(df[sofa_col], errors="coerce")
    mask = (b >= 1) & (b <= max_bloc)
    sub = df.loc[mask].copy()
    sub["_b"] = b[mask]
    sub["_s"] = s[mask]
    sub[icu_col] = pd.to_numeric(sub[icu_col], errors="coerce").astype("Int64")
    sub = sub.dropna(subset=[icu_col, "_s"])
    g = sub.groupby(sub[icu_col].astype(np.int64), sort=False)
    if agg == "max":
        out = g["_s"].max()
    elif agg == "mean":
        out = g["_s"].mean()
    elif agg == "last":
        out = sub.sort_values("_b").groupby(sub[icu_col].astype(np.int64), sort=False)["_s"].last()
    else:
        raise ValueError("agg must be max, mean, or last")
    out.name = f"SOFA_{agg}_first_{horizon_hours:g}h"
    return out


def sofa_values_for_icustay_rows(
    sofa_per_icu: pd.Series,
    icustay_ids: np.ndarray,
) -> np.ndarray:
    """Map ``icustay_ids`` (same order as tensor rows) to SOFA; NaN for invalid / missing stays."""
    ids = np.asarray(icustay_ids, dtype=np.int64).reshape(-1)
    out = np.full(ids.shape[0], np.nan, dtype=np.float64)
    valid = ids >= 0
    if valid.any():
        mapped = sofa_per_icu.reindex(ids[valid])
        out[valid] = pd.to_numeric(mapped, errors="coerce").to_numpy(dtype=np.float64)
    return out


def tensors_to_loader(X_np, y_np, max_samples, batch_size, random_state: int, device):
    """Build a loader over rows of ``X_np``; returns ``(loader, row_indices)`` into the original arrays."""
    X_np = np.asarray(X_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)
    n = X_np.shape[0]
    idx = np.arange(n, dtype=np.int64)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_np, y_np = X_np[idx], y_np[idx]
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y
            T = x.shape[1]
            self.tp = torch.linspace(0, 1, T, device=x.device)

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, i):
            return {"X": self.x[i], "y": self.y[i], "tp": self.tp}

    ds = _DS(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False), idx


@torch.no_grad()
def collect_latents(model, loader, latent_before_classifier):
    zs, ys = [], []
    for batch in loader:
        X, y = batch["X"], batch["y"]
        tp = batch["tp"][0]
        z = latent_before_classifier(model, X, [tp], [tp], [tp])
        zs.append(z.cpu().numpy())
        ys.append(y.cpu().numpy())
    Z = np.concatenate(zs, axis=0)
    y = np.concatenate(ys, axis=0)
    return Z, y
