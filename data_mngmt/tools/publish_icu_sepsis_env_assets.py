"""Install rebuilt ICU-Sepsis MDP files into the Gymnasium package ``envs/assets`` directory.

`ICUSepsisEnv` loads `dynamics.npz`, `metadata.json`, and `admissible_actions.txt` from
`icu_sepsis/icu_sepsis/icu_sepsis/envs/assets` when no custom `MDPParameters` are passed.
After `build_mdp_from_sepsis_cohort` or `run_unified_build`, run this module so
`gym.make("Sepsis/ICU-Sepsis-v2")` uses cohort-derived dynamics.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from data_mngmt import coper_root

log = logging.getLogger(__name__)

# Must match what `icu_sepsis.utils.io.MDPParameters` loads from disk (npz + sidecars).
_ASSET_FILENAMES = ("dynamics.npz", "metadata.json", "admissible_actions.txt")


def default_icu_sepsis_env_assets_dir(repo_root: Path | None = None) -> Path:
    """Directory shipped with `icu_sepsis` that `ICUSepsisEnv` reads by default."""
    root = Path(repo_root or coper_root()).resolve()
    return (
        root
        / "icu_sepsis"
        / "icu_sepsis"
        / "icu_sepsis"
        / "envs"
        / "assets"
    ).resolve()


def publish_mdp_params_to_pkg_assets(
    mdp_params_dir: Path,
    *,
    assets_dir: Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Copy dynamics artifacts from a `build_mimic_params` output tree into package `assets/`.

    Returns the list of destination paths written (or that would be written if ``dry_run``).
    """
    src = Path(mdp_params_dir).resolve()
    dest_root = Path(assets_dir or default_icu_sepsis_env_assets_dir()).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"MDP params directory not found: {src}")
    if not dest_root.is_dir():
        raise FileNotFoundError(f"Package assets directory not found: {dest_root}")

    missing = [name for name in _ASSET_FILENAMES if not (src / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {src}: {missing}. "
            "Run build_mdp_from_sepsis_cohort / build_mimic_params first."
        )

    written: list[Path] = []
    for name in _ASSET_FILENAMES:
        s, d = src / name, dest_root / name
        if dry_run:
            log.info("Would copy %s -> %s", s, d)
        else:
            shutil.copy2(s, d)
            log.info("Copied %s -> %s", s, d)
        written.append(d)
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Copy dynamics.npz + metadata.json + admissible_actions.txt from an MDP build "
            "into icu_sepsis/.../envs/assets (default Gymnasium data for Sepsis/ICU-Sepsis-v2)."
        )
    )
    parser.add_argument(
        "--mdp-dir",
        type=Path,
        required=True,
        help="Output directory from build_mimic_params / build_mdp_from_sepsis_cohort",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Override package assets directory (default: icu_sepsis/.../envs/assets under repo root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print copy plan without writing files",
    )
    args = parser.parse_args(argv)
    try:
        publish_mdp_params_to_pkg_assets(
            args.mdp_dir,
            assets_dir=args.assets_dir,
            dry_run=args.dry_run,
        )
    except Exception as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
