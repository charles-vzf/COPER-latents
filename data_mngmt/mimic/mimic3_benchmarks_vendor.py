"""Vendor [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) into ``data_mngmt/vendor/``.

Shallow clone into ``data_mngmt/vendor/mimic3_benchmarks`` so PhysioNet → IHM extraction and
``mimic3models`` normalizers run without a separate ``paths.json`` entry. Override with env
``MIMIC3_BENCHMARKS_REPO`` if needed.

After vendoring, install upstream Python deps from the clone, e.g.::

    pip install -r data_mngmt/vendor/mimic3_benchmarks/requirements.txt
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_GIT_URL = "https://github.com/YerevaNN/mimic3-benchmarks.git"
SSH_GIT_URL = "git@github.com:YerevaNN/mimic3-benchmarks.git"


def default_vendor_dest(repo_root: Path | None = None) -> Path:
    from data_mngmt.layout import vendor_mimic3_benchmarks

    return vendor_mimic3_benchmarks(repo_root)


def vendor_mimic3_benchmarks(
    *,
    dest: Path | None = None,
    git_url: str = DEFAULT_GIT_URL,
    ref: str = "master",
    force: bool = False,
) -> Path:
    """Clone mimic3-benchmarks into ``dest`` (default ``data_mngmt/vendor/mimic3_benchmarks``)."""
    dest = Path(dest or default_vendor_dest()).resolve()
    marker = dest / "mimic3benchmark"
    if marker.is_dir() and not force:
        log.info("Already present at %s (use --force to re-clone)", dest)
        return dest

    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mimic3_benchmarks_vendor_") as tmp:
        clone_root = Path(tmp) / "mimic3-benchmarks"
        log.info("Cloning %s (branch=%s) …", git_url, ref)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, git_url, str(clone_root)],
            check=True,
        )
        rev = subprocess.run(
            ["git", "-C", str(clone_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(clone_root), str(dest))

        (dest / "VENDOR_README.txt").write_text(
            "Vendored from https://github.com/YerevaNN/mimic3-benchmarks for COPER data_mngmt.\n"
            "Refresh with: python -m data_mngmt.mimic.mimic3_benchmarks_vendor --force\n"
            "Install deps: pip install -r requirements.txt (from this directory).\n",
            encoding="utf-8",
        )
        (dest / "SOURCE_REVISION.txt").write_text(
            f"url={git_url}\nref={ref}\ncommit={rev}\n",
            encoding="utf-8",
        )

    log.info("Vendored mimic3-benchmarks into %s", dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Vendor YerevaNN/mimic3-benchmarks into data_mngmt/vendor/mimic3_benchmarks/"
    )
    p.add_argument("--dest", type=Path, default=None, help="Destination directory")
    p.add_argument(
        "--ssh",
        action="store_true",
        help=f"Use SSH clone URL ({SSH_GIT_URL}) instead of HTTPS",
    )
    p.add_argument("--ref", default="master", help="Git branch or tag (default: master)")
    p.add_argument("--force", action="store_true", help="Remove existing dest and re-clone")
    args = p.parse_args(argv)
    url = SSH_GIT_URL if args.ssh else DEFAULT_GIT_URL
    try:
        vendor_mimic3_benchmarks(dest=args.dest, git_url=url, ref=args.ref, force=args.force)
    except subprocess.CalledProcessError as e:
        log.error("git failed: %s", e)
        return 1
    except Exception as e:
        log.exception("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
