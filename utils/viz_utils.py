"""Visualization helpers for COPER latent embedding comparisons."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.embedding_data_utils import (
    load_icustay_ids_split,
    sofa_per_icustay_from_mdp_cohort,
    sofa_values_for_icustay_rows,
)


def plot_scatter_2d(Z2: np.ndarray, y: np.ndarray, title: str, ax) -> None:
    """Plot a 2D embedding with mortality labels."""
    y = y.astype(int)
    ax.scatter(Z2[y == 0, 0], Z2[y == 0, 1], s=6, alpha=0.35, label="survive")
    ax.scatter(Z2[y == 1, 0], Z2[y == 1, 1], s=6, alpha=0.55, label="mortality")
    ax.set_title(title)
    ax.legend(markerscale=2)
    ax.set_aspect("equal", adjustable="datalim")


def plot_scatter_2d_continuous(
    Z2: np.ndarray,
    c: np.ndarray,
    title: str,
    ax,
    *,
    cbar_label: str = "SOFA",
    missing_label: str = "No SOFA (not in RL table / invalid ID / no score in window)",
) -> None:
    """2D scatter colored by a continuous variable (NaNs drawn in light grey).

    NaNs usually mean: ``ICUSTAY_ID == -1`` in the pickle, the stay is absent from the MDP
    cohort CSV (e.g. COPER ``all-`` stays not in mimic_sepsis), or every SOFA value in the
    first-hour window was missing so the stay was dropped from the aggregate.
    """
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    mask = np.isfinite(c)
    if (~mask).any():
        ax.scatter(
            Z2[~mask, 0],
            Z2[~mask, 1],
            s=5,
            alpha=0.35,
            c="0.72",
            edgecolors="0.45",
            linewidths=0.15,
            label=missing_label,
        )
    if mask.any():
        sc = ax.scatter(
            Z2[mask, 0],
            Z2[mask, 1],
            s=8,
            alpha=0.55,
            c=c[mask],
            cmap="viridis",
        )
        ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    if (~mask).any():
        ax.legend(loc="best", fontsize=7, markerscale=2)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")


def run_viz(
    *,
    bundle_path: Path,
    repo_root: Path,
    label: str,
    device,
    mortality_pickle: Path,
    split: str,
    max_samples: int,
    batch_size: int,
    random_state: int,
    latents_dir: Path,
    load_coper_from_bundle,
    load_xy_split,
    tensors_to_loader,
    collect_latents,
    save_figure_path: Path | None = None,
    mdp_cohort_csv: Path | str | None = None,
    sofa_agg: str = "max",
    coper_horizon_hours: float | None = None,
    mdp_bloc_interval_hours: float | None = None,
):
    """
    Load model bundle, compute latents, run PCA/UMAP/t-SNE, plot and persist NPZ+meta.

    Dependency functions are injected so notebooks can pass existing loaders
    without creating import cycles.
    """
    if not bundle_path.is_file():
        print(f"SKIP (missing): {bundle_path}")
        return None

    model, meta = load_coper_from_bundle(bundle_path, repo_root, device=device)
    X_np, y_np = load_xy_split(mortality_pickle, split)
    loader, row_idx = tensors_to_loader(X_np, y_np, max_samples, batch_size)
    Z, y = collect_latents(model, loader)
    print(label, "Z", Z.shape, "y", y.shape, "pos_rate", float(y.mean()))

    sofa = None
    icu_sub = None
    sofa_label = f"SOFA ({sofa_agg}, first window)"
    cohort_p = Path(mdp_cohort_csv) if mdp_cohort_csv else None
    if cohort_p is not None and cohort_p.is_file():
        try:
            with open(mortality_pickle, "rb") as f:
                details = pickle.load(f)[0]
            h = float(coper_horizon_hours) if coper_horizon_hours is not None else 48.0
            bh = float(mdp_bloc_interval_hours) if mdp_bloc_interval_hours is not None else 1.0
            if isinstance(details, dict):
                h = float(details.get("horizon_hours", h))
                bh = float(details.get("timestep_hours", bh))
            icu_all = load_icustay_ids_split(mortality_pickle, split)
            icu_sub = icu_all[row_idx]
            per_icu = sofa_per_icustay_from_mdp_cohort(
                cohort_p,
                horizon_hours=h,
                bloc_interval_hours=bh,
                agg=sofa_agg,
            )
            sofa = sofa_values_for_icustay_rows(per_icu, icu_sub)
            n_ok = int(np.isfinite(sofa).sum())
            n_bad_id = int((icu_sub < 0).sum())
            n_no_join = int(((icu_sub >= 0) & ~np.isfinite(sofa)).sum())
            print(
                label,
                "SOFA sidecar:",
                n_ok,
                "/",
                sofa.shape[0],
                "finite (from",
                cohort_p.name,
                f", first {h:g}h × {bh:g}h blocs).",
                f"Gray points: {n_bad_id} invalid ICUSTAY_ID (-1);",
                f"{n_no_join} valid ID but no SOFA row in RL cohort/window.",
            )
        except Exception as e:
            print(label, "SOFA sidecar skipped:", e)
            sofa = None
            icu_sub = None

    pca = PCA(n_components=2, random_state=random_state)
    Z_pca = pca.fit_transform(Z)

    reducer = umap.UMAP(
        n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1
    )
    Z_umap = reducer.fit_transform(Z)

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, max_iter=1000)
    Z_tsne = tsne.fit_transform(Z)

    n_row = 2 if sofa is not None and np.isfinite(sofa).any() else 1
    fig, axes = plt.subplots(n_row, 3, figsize=(14, 4 * n_row))
    if n_row == 1:
        axr0 = axes
    else:
        axr0, axr1 = axes[0], axes[1]
    plot_scatter_2d(Z_pca, y, f"{label} PCA", axr0[0])
    plot_scatter_2d(Z_umap, y, f"{label} UMAP", axr0[1])
    plot_scatter_2d(Z_tsne, y, f"{label} t-SNE", axr0[2])
    if n_row == 2:
        plot_scatter_2d_continuous(Z_pca, sofa, f"{label} PCA ({sofa_label})", axr1[0])
        plot_scatter_2d_continuous(Z_umap, sofa, f"{label} UMAP ({sofa_label})", axr1[1])
        plot_scatter_2d_continuous(Z_tsne, sofa, f"{label} t-SNE ({sofa_label})", axr1[2])
    plt.tight_layout()
    if save_figure_path is not None:
        save_figure_path = Path(save_figure_path)
        save_figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure_path, dpi=150, bbox_inches="tight")
        print("Saved figure", save_figure_path)
    plt.show()

    out_npz = latents_dir / f"latents_{label}_{split}_n{Z.shape[0]}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {"Z": Z, "y": y}
    if sofa is not None:
        save_kw["sofa"] = sofa
    if icu_sub is not None:
        save_kw["icustay_id"] = icu_sub
    np.savez(out_npz, **save_kw)
    with open(out_npz.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved", out_npz, "+ meta json")
    return {
        "label": label,
        "Z": Z,
        "y": y,
        "sofa": sofa,
        "icustay_id": icu_sub,
        "meta": meta,
        "latent_npz": out_npz,
    }
