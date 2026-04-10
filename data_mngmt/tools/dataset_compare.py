"""Summaries for side-by-side comparison: COPER mortality tensors vs packaged ICU-Sepsis MDP."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np

from data_mngmt import coper_root, mortality_pickle_path


def _summarize_mortality_pickle(pickle_path: Path) -> dict[str, Any]:
    with open(pickle_path, "rb") as f:
        details, X_train, y_train, X_val, y_val, X_test, y_test, _ = pickle.load(f)

    def split_stats(y: np.ndarray, name: str) -> dict[str, Any]:
        y = np.asarray(y).reshape(-1)
        pos = int((y >= 0.5).sum())
        n = y.shape[0]
        return {
            "split": name,
            "n": n,
            "positive": pos,
            "rate": float(pos / n) if n else 0.0,
        }

    return {
        "details": details,
        "splits": [
            split_stats(y_train, "train"),
            split_stats(y_val, "val"),
            split_stats(y_test, "test"),
        ],
        "X_train_shape": tuple(X_train.shape),
        "X_val_shape": tuple(X_val.shape),
        "X_test_shape": tuple(X_test.shape),
    }


def _entropy_discrete(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = p[p > eps]
    return float(-(p * np.log(p + eps)).sum())


def default_mdp_assets_dir() -> Path:
    return (
        coper_root()
        / "icu_sepsis"
        / "icu_sepsis"
        / "icu_sepsis"
        / "envs"
        / "assets"
    ).resolve()


def summarize_coper_mortality_dataset(pickle_path: Path | None = None) -> dict[str, Any]:
    """Stats for the COPER / mimic3-benchmarks in-hospital mortality pickle."""
    p = Path(pickle_path or mortality_pickle_path())
    if not p.is_file():
        return {
            "artifact": str(p),
            "exists": False,
            "task": "in-hospital mortality (IHM)",
            "typical_setup": "first 48h of ICU stay, 1h discretization (stock normalizer)",
            "error": "pickle missing — run python -m data_mngmt or notebooks/build_data.ipynb or set paths.json → mimic3_mortality",
        }
    s = _summarize_mortality_pickle(p)
    details = s.get("details") or {}
    return {
        "artifact": str(p),
        "exists": p.is_file(),
        "task": "in-hospital mortality (IHM)",
        "typical_setup": "first 48h of ICU stay, 1h discretization (stock normalizer)",
        "details": details,
        "splits": s.get("splits"),
        "X_train_shape": s.get("X_train_shape"),
        "X_val_shape": s.get("X_val_shape"),
        "X_test_shape": s.get("X_test_shape"),
    }


def summarize_icu_mdp_params_dir(params_dir: Path) -> dict[str, Any]:
    """Same stats as packaged env assets, but for a freshly built MDPParameters folder."""
    root = Path(params_dir)
    meta_path = root / "metadata.json"
    npz_path = root / "dynamics.npz"
    if not npz_path.is_file():
        return {"artifact": str(npz_path), "exists": False, "error": "dynamics.npz missing"}

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    data = np.load(npz_path)
    d_0 = np.asarray(data["d_0"], dtype=np.float64)
    tx = np.asarray(data["tx_mat"])
    n_states, n_actions = tx.shape[0], tx.shape[1]
    pos = d_0 > 1e-12
    return {
        "artifact": str(npz_path),
        "exists": True,
        "task": "ICU-Sepsis tabular MDP (custom build directory)",
        "typical_setup": str(root),
        "metadata": meta,
        "n_states_mdp": int(n_states),
        "n_actions_mdp": int(n_actions),
        "d_0_entropy_nat": _entropy_discrete(d_0),
        "d_0_support_states": int(pos.sum()),
        "d_0_top5_mass": float(np.sort(d_0)[-5:].sum()) if d_0.size else 0.0,
        "gamma_env_default": 1.0,
    }


def summarize_packaged_icu_mdp(assets_dir: Path | None = None) -> dict[str, Any]:
    """Stats from bundled ``dynamics.npz`` (tabular MDP used by Gymnasium env)."""
    root = Path(assets_dir or default_mdp_assets_dir())
    meta_path = root / "metadata.json"
    npz_path = root / "dynamics.npz"
    if not npz_path.is_file():
        return {"artifact": str(npz_path), "exists": False, "error": "dynamics.npz missing"}

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    data = np.load(npz_path)
    d_0 = np.asarray(data["d_0"], dtype=np.float64)
    tx = np.asarray(data["tx_mat"])
    n_states, n_actions = tx.shape[0], tx.shape[1]
    pos = d_0 > 1e-12
    return {
        "artifact": str(npz_path),
        "exists": True,
        "task": "ICU-Sepsis tabular MDP (sepsis cohort, upstream RL table)",
        "typical_setup": (
            "discrete decision epochs along full ICU trajectories in the RL table; "
            "clinical alignment (e.g. 4h steps) is defined when building that table, "
            "not stored as hours inside dynamics.npz"
        ),
        "metadata": meta,
        "n_states_mdp": int(n_states),
        "n_actions_mdp": int(n_actions),
        "d_0_entropy_nat": _entropy_discrete(d_0),
        "d_0_support_states": int(pos.sum()),
        "d_0_top5_mass": float(np.sort(d_0)[-5:].sum()) if d_0.size else 0.0,
        "gamma_env_default": 1.0,
    }


def _icu_base_env(env) -> Any:
    e = env.unwrapped
    for _ in range(8):
        if hasattr(e, "_expert_policy"):
            return e
        nxt = getattr(e, "env", None)
        if nxt is None:
            break
        e = nxt
    raise RuntimeError("Could not find ICUSepsisEnv with _expert_policy")


def sample_mdp_episode_lengths(
    n_episodes: int = 2_000,
    *,
    seed: int = 0,
    policy: Literal["random", "expert"] = "random",
    env_id: str = "Sepsis/ICU-Sepsis-v2",
    return_lengths: bool = False,
) -> dict[str, Any]:
    """Monte Carlo distribution of episode lengths (steps until terminal or TimeLimit)."""
    import gymnasium as gym

    import icu_sepsis  # noqa: F401 — register env

    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    base = _icu_base_env(env) if policy == "expert" else None
    lengths: list[int] = []
    for _ in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        steps = 0
        while not done:
            if policy == "random":
                a = env.action_space.sample()
            else:
                pol = base._expert_policy
                dist = np.asarray(pol[int(s)], dtype=np.float64)
                dist = dist / (dist.sum() + 1e-12)
                a = int(rng.choice(np.arange(dist.size), p=dist))
            s, _r, term, trunc, _ = env.step(a)
            done = term or trunc
            steps += 1
        lengths.append(steps)
    env.close()
    arr = np.array(lengths, dtype=np.int32)
    out = {
        "n_episodes": n_episodes,
        "policy": policy,
        "env_id": env_id,
        "length_mean": float(arr.mean()),
        "length_std": float(arr.std()),
        "length_median": float(np.median(arr)),
        "length_p90": float(np.percentile(arr, 90)),
        "length_max": int(arr.max()),
        "truncated_fraction": float((arr >= env.spec.max_episode_steps).mean()),
    }
    if return_lengths:
        out["lengths"] = arr
    return out


def compare_mortality_pickles(
    path_a: Path,
    path_b: Path,
) -> dict[str, Any]:
    """Lightweight comparison of two COPER-style mortality pickles (shapes + label rates)."""
    sa = _summarize_mortality_pickle(Path(path_a))
    sb = _summarize_mortality_pickle(Path(path_b))
    return {
        "a": str(path_a),
        "b": str(path_b),
        "a_shapes": {
            "train": sa.get("X_train_shape"),
            "val": sa.get("X_val_shape"),
            "test": sa.get("X_test_shape"),
        },
        "b_shapes": {
            "train": sb.get("X_train_shape"),
            "val": sb.get("X_val_shape"),
            "test": sb.get("X_test_shape"),
        },
        "a_splits": sa.get("splits"),
        "b_splits": sb.get("splits"),
        "a_details": sa.get("details"),
        "b_details": sb.get("details"),
    }


def comparison_summary_table(
    coper: dict[str, Any] | None = None,
    mdp: dict[str, Any] | None = None,
) -> dict[str, tuple[str, str]]:
    """Side-by-side strings for a small markdown or DataFrame display."""
    coper = coper or summarize_coper_mortality_dataset()
    mdp = mdp or summarize_packaged_icu_mdp()
    c_task = coper.get("task", "—")
    m_task = mdp.get("task", "—")
    c_setup = str(coper.get("typical_setup", "—"))
    m_setup = str(mdp.get("typical_setup", "—"))

    c_shape = str(coper.get("X_train_shape", "N/A")) if coper.get("exists") else "missing file"
    m_shape = (
        f"{mdp.get('n_states_mdp', '?')} states × {mdp.get('n_actions_mdp', '?')} actions"
        if mdp.get("exists")
        else "missing dynamics"
    )

    return {
        "Representation": ("Dense time series tensors (per stay)", "Discrete states + tabular dynamics"),
        "Benchmark / cohort": (c_task, m_task),
        "Typical time framing": (c_setup, m_setup),
        "Train-scale summary": (c_shape, m_shape),
    }
