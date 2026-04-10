"""Standard layout under ``code/COPER/results/`` (single tree for training + exports + plots).

Typical structure::

    results/
      checkpoints/   # .ckpt (gitignored)
      models/          # exported .pt + .json bundles (pt ignored, json kept)
      predictions/     # Predictions_*.npz (gitignored)
      latents/         # latents_*.npz caches (gitignored)
      logs/            # training logs (gitignored)
      tables/          # CSV summaries (tracked)
      figures/         # PNG/SVG (tracked); PDF can be ignored
      traces/          # small run manifests
      runs/            # optional dated sessions: runs/YYYY-MM-DD_slug/…

Notebook-generated artifacts (PNGs, ad-hoc CSVs, demo checkpoints) should go under
``results/demo_outputs/<notebook_slug>/{figures,tables,models,latents,...}/`` via
``notebook_demo_dir`` (not under ``notebooks/demo_outputs``).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

SUBDIRS = (
    "checkpoints",
    "models",
    "predictions",
    "latents",
    "logs",
    "tables",
    "figures",
    "traces",
)


def coper_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def results_root(repo: Path | None = None) -> Path:
    return (repo or coper_repo_root()) / "results"


def ensure_results_subdirs(base: Path) -> None:
    """Create standard subdirectories under ``base`` (usually ``results/`` or a run folder)."""
    for name in SUBDIRS:
        (base / name).mkdir(parents=True, exist_ok=True)


def dated_session_dir(repo: Path | None = None, slug: str = "run") -> Path:
    """``results/runs/YYYY-MM-DD_<slug>/`` with standard subdirs created."""
    root = results_root(repo) / "runs" / f"{date.today().isoformat()}_{slug}"
    ensure_results_subdirs(root)
    return root


def global_models_dir(repo: Path | None = None) -> Path:
    """Shared export directory when not using a dated session (``results/models``)."""
    p = results_root(repo) / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def demo_outputs_root(repo: Path | None = None, *, mkdir: bool = True) -> Path:
    """``results/demo_outputs/`` — root for per-notebook export trees."""
    p = results_root(repo) / "demo_outputs"
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p


def table_name_dated(stem: str, suffix: str = ".csv") -> str:
    """``stem_YYYY-MM-DD.csv`` for versioned table outputs."""
    return f"{stem}_{date.today().isoformat()}{suffix}"


def notebook_demo_dir(
    subdir: str,
    notebook_slug: str,
    repo: Path | None = None,
    *,
    mkdir: bool = True,
) -> Path:
    """Directory for one notebook's outputs: ``results/demo_outputs/<notebook_slug>/<subdir>/``.

    ``subdir`` must be a standard bucket (e.g. ``figures``, ``tables``, ``models``, ``latents``).
    """
    if subdir not in SUBDIRS:
        raise ValueError(f"subdir must be one of {SUBDIRS}, got {subdir!r}")
    base = results_root(repo) / "demo_outputs"
    if mkdir:
        base.mkdir(parents=True, exist_ok=True)
    p = base / notebook_slug / subdir
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p
