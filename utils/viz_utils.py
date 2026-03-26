"""Visualization helpers for COPER latent embedding comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_scatter_2d(Z2: np.ndarray, y: np.ndarray, title: str, ax) -> None:
    """Plot a 2D embedding with mortality labels."""
    y = y.astype(int)
    ax.scatter(Z2[y == 0, 0], Z2[y == 0, 1], s=6, alpha=0.35, label="survive")
    ax.scatter(Z2[y == 1, 0], Z2[y == 1, 1], s=6, alpha=0.55, label="mortality")
    ax.set_title(title)
    ax.legend(markerscale=2)
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
    artifacts_dir: Path,
    load_coper_from_bundle,
    load_xy_split,
    tensors_to_loader,
    collect_latents,
    save_figure_path: Path | None = None,
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
    loader = tensors_to_loader(X_np, y_np, max_samples, batch_size)
    Z, y = collect_latents(model, loader)
    print(label, "Z", Z.shape, "y", y.shape, "pos_rate", float(y.mean()))

    pca = PCA(n_components=2, random_state=random_state)
    Z_pca = pca.fit_transform(Z)

    reducer = umap.UMAP(
        n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1
    )
    Z_umap = reducer.fit_transform(Z)

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, max_iter=1000)
    Z_tsne = tsne.fit_transform(Z)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_scatter_2d(Z_pca, y, f"{label} PCA", axes[0])
    plot_scatter_2d(Z_umap, y, f"{label} UMAP", axes[1])
    plot_scatter_2d(Z_tsne, y, f"{label} t-SNE", axes[2])
    plt.tight_layout()
    if save_figure_path is not None:
        save_figure_path = Path(save_figure_path)
        save_figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure_path, dpi=150, bbox_inches="tight")
        print("Saved figure", save_figure_path)
    plt.show()

    out_npz = artifacts_dir / f"latents_{label}_{split}_n{Z.shape[0]}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, Z=Z, y=y)
    with open(out_npz.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved", out_npz, "+ meta json")
    return {"label": label, "Z": Z, "y": y, "meta": meta, "latent_npz": out_npz}
