"""Per-MDP-state COPER mortality scores for ``coper_next`` rewards."""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _cache_path(cache_dir: Path, n_states: int) -> Path:
    return cache_dir / f"coper_mortality_n{n_states}.npz"


def _resolve_models_dir(repo: Path) -> Path:
    marker = repo / "results" / "latest_mimic_compare_models_dir.txt"
    if marker.is_file():
        return Path(marker.read_text(encoding="utf-8").strip())
    runs = repo / "results" / "runs"
    cand = sorted(runs.glob("*_mimic_compare/models"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cand:
        return cand[0]
    raise FileNotFoundError(
        "No COPER bundle directory found. Run COPERvsTRANSFORMER_mortality.ipynb or set "
        "results/latest_mimic_compare_models_dir.txt."
    )


def _pick_coper_bundle(models_dir: Path) -> Path:
    for stem in (
        "coper_1node_drop0_s1_e10",
        "coper_2node_drop0_s1_e10",
        "coper_1node_drop0_s1_e3",
    ):
        bundle = models_dir / f"{stem}.pt"
        if bundle.is_file():
            return bundle
    matches = sorted(models_dir.glob("coper_*_drop0_s1_e10.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No COPER bundle under {models_dir}")


def _resolve_mortality_pickle(repo: Path) -> Path | None:
    from data_mngmt import mortality_pickle_path

    for candidate in (
        mortality_pickle_path(),
        repo / "data_mngmt/legacy/mortality_for_coper_mdp_legacy.data",
    ):
        if candidate.is_file():
            return candidate
    return None


@torch.no_grad()  # type: ignore[misc]
def _coper_predict_all(model, X: np.ndarray, *, device: str) -> np.ndarray:
    import torch

    model = model.to(device)
    model.eval()
    out_chunks: list[np.ndarray] = []
    bs = 128
    Xf = X.astype(np.float32, copy=False)
    for start in range(0, Xf.shape[0], bs):
        xb = torch.from_numpy(Xf[start : start + bs]).to(device)
        tp = torch.linspace(0, 1, xb.shape[1], device=device, dtype=xb.dtype)
        logits = model(xb, [tp], [tp], [tp])
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        out_chunks.append(logits.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out_chunks, axis=0)


def _build_from_pickle_alignment(
    *,
    repo: Path,
    pickle_path: Path,
    models_dir: Path,
    n_states: int,
    device: str,
) -> np.ndarray | None:
    """Patient-level COPER preds aggregated by MDP state (needs ``icustay_id`` in pickle)."""
    try:
        from data_mngmt.contracts.coper_mdp_join import (
            cohort_csv_with_mdp_states,
            default_unified_cohort_paths,
            per_bloc_state_matrix,
        )
        from utils.embedding_data_utils import load_icustay_ids_split, load_xy_split
    except ImportError:
        return None

    try:
        icu_tr = load_icustay_ids_split(pickle_path, "train")
    except ValueError:
        return None

    n_tr = n_states - 3
    cohort_csv, _ = default_unified_cohort_paths(repo, slug="sepsis-60m-h48ihm")
    if not cohort_csv.is_file():
        return None

    cohort = cohort_csv_with_mdp_states(
        cohort_csv=cohort_csv,
        n_states=n_tr,
        n_action_levels=5,
        seed=0,
    )
    X_tr, _ = load_xy_split(pickle_path, "train")
    X_va, _ = load_xy_split(pickle_path, "val")
    X_te, _ = load_xy_split(pickle_path, "test")
    icu_va = load_icustay_ids_split(pickle_path, "val")
    icu_te = load_icustay_ids_split(pickle_path, "test")
    X_all = np.concatenate([X_tr, X_va, X_te], axis=0)
    icu_all = np.concatenate([icu_tr, icu_va, icu_te], axis=0)
    state_mat, valid = per_bloc_state_matrix(
        cohort, icu_all, n_blocs=X_all.shape[1], max_state_exclusive=n_tr
    )
    if int(valid.sum()) < 32:
        return None

    from utils.load_coper_bundle import load_coper_from_bundle

    bundle = _pick_coper_bundle(models_dir)
    model, _ = load_coper_from_bundle(bundle, repo, device=device)
    patient_probs = _coper_predict_all(model, X_all[valid], device=device)
    state_mat = state_mat[valid]

    scores = np.zeros(n_states, dtype=np.float64)
    for s in range(n_tr):
        visited = (state_mat == s).any(axis=1)
        if visited.any():
            scores[s] = float(patient_probs[visited].mean())
    fill = float(np.nanmean(scores[:n_tr]))
    for s in range(n_tr):
        if not np.isfinite(scores[s]):
            scores[s] = fill
    return scores


def _build_quantile_match(
    *,
    repo: Path,
    pickle_path: Path,
    models_dir: Path,
    sofa_scores: np.ndarray,
    n_states: int,
    device: str,
) -> np.ndarray:
    """Map patient-level COPER outputs to states via SOFA rank matching (packaged MDP fallback)."""
    from utils.embedding_data_utils import load_xy_split
    from utils.load_coper_bundle import load_coper_from_bundle

    n_tr = n_states - 3
    sofa_tr = np.asarray(sofa_scores, dtype=np.float64).reshape(-1)[:n_tr]
    X_parts = [load_xy_split(pickle_path, split)[0] for split in ("train", "val", "test")]
    X_all = np.concatenate(X_parts, axis=0)
    bundle = _pick_coper_bundle(models_dir)
    model, _ = load_coper_from_bundle(bundle, repo, device=device)
    patient_probs = _coper_predict_all(model, X_all, device=device)

    state_order = np.argsort(sofa_tr)
    patient_order = np.argsort(patient_probs)
    ranks = np.linspace(0, len(patient_probs) - 1, num=n_tr)
    scores = np.zeros(n_states, dtype=np.float64)
    for rank_idx, s in enumerate(state_order):
        j = int(round(ranks[rank_idx]))
        scores[s] = float(patient_probs[patient_order[j]])
    return scores


def build_coper_mortality_per_state(
    *,
    repo: Path,
    n_states: int,
    sofa_scores: np.ndarray,
    cache_dir: Path | None = None,
    device: str | None = None,
    force_rebuild: bool = False,
) -> np.ndarray:
    """Return length-``n_states`` COPER mortality scores (transient filled, absorbing zeros)."""
    if torch is None:
        raise ImportError("PyTorch is required to build COPER state scores.")

    cache_dir = cache_dir or (repo / "results" / "demo_outputs" / "reward_functions")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, n_states)
    if cache_file.is_file() and not force_rebuild:
        data = np.load(cache_file)
        scores = np.asarray(data["scores"], dtype=np.float64).reshape(-1)
        if scores.shape[0] == n_states:
            return scores

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = _resolve_models_dir(repo)
    pickle_path = _resolve_mortality_pickle(repo)
    if pickle_path is None:
        raise FileNotFoundError(
            "No mortality pickle found. Set paths.json mimic3_mortality or add "
            "data_mngmt/legacy/mortality_for_coper_mdp_legacy.data."
        )

    scores = _build_from_pickle_alignment(
        repo=repo,
        pickle_path=pickle_path,
        models_dir=models_dir,
        n_states=n_states,
        device=device,
    )
    if scores is None:
        warnings.warn(
            "Could not align mortality pickle to MDP cohort; using SOFA-quantile-matched "
            "COPER patient predictions for per-state scores.",
            stacklevel=2,
        )
        scores = _build_quantile_match(
            repo=repo,
            pickle_path=pickle_path,
            models_dir=models_dir,
            sofa_scores=sofa_scores,
            n_states=n_states,
            device=device,
        )

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.shape[0] != n_states:
        raise ValueError(f"Built coper_scores length {scores.shape[0]} != n_states {n_states}")
    np.savez(cache_file, scores=scores)
    return scores
