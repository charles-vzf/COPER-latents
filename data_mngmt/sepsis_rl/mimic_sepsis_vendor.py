"""Vendor [microsoft/mimic_sepsis](https://github.com/microsoft/mimic_sepsis) into this repo (MIT).

Clone to a temporary directory over SSH/HTTPS, copy scripts + ``ReferenceFiles/``, patch
``preprocess.py`` so PostgreSQL host/db/user are configurable, record the upstream git
revision, then remove the clone.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from data_mngmt import coper_root

log = logging.getLogger(__name__)

DEFAULT_GIT_URL = "git@github.com:microsoft/mimic_sepsis.git"
HTTPS_GIT_URL = "https://github.com/microsoft/mimic_sepsis.git"

_PREPROCESS_CONN_OLD = """parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username", default='USERNAME', help="Username used to access the MIMIC Database", type=str)
parser.add_argument("-p", "--password", default='PASSWORD', help="User's password for MIMIC Database", type=str)
pargs = parser.parse_args()

# Initializing database connection
conn = pg.connect("dbname='mimic' user={0} host='mimic' options='--search_path=mimimciii' password={1}".format(pargs.username,pargs.password))"""

_PREPROCESS_CONN_NEW = """parser = argparse.ArgumentParser()
parser.add_argument(
    "-u", "--username", default=None,
    help="PostgreSQL user (default: PGUSER or mimic)",
    type=str,
)
parser.add_argument(
    "-p", "--password", default=None,
    help="PostgreSQL password (default: PGPASSWORD env)",
    type=str,
)
parser.add_argument(
    "--dbname", default=None,
    help="Database name (default: PGDATABASE or mimic)",
)
parser.add_argument(
    "--host", default=None,
    help="PostgreSQL host (default: PGHOST or localhost)",
)
parser.add_argument(
    "--port", default=None, type=int,
    help="PostgreSQL port (default: PGPORT or 5432)",
)
pargs = parser.parse_args()

# --- Patched by COPER vendor: configurable connection (SQL uses mimiciii.* prefixes) ---
_user = pargs.username or os.environ.get("PGUSER", "mimic")
_pass = pargs.password if pargs.password is not None else os.environ.get("PGPASSWORD", "")
_db = pargs.dbname or os.environ.get("PGDATABASE", "mimic")
_host = pargs.host or os.environ.get("PGHOST", "localhost")
_port = int(pargs.port or os.environ.get("PGPORT", "5432"))

conn = pg.connect(host=_host, port=_port, dbname=_db, user=_user, password=_pass)"""


def default_upstream_dir() -> Path:
    from data_mngmt.layout import vendor_mimic_sepsis

    return vendor_mimic_sepsis()


def _patch_preprocess(preprocess_path: Path) -> None:
    text = preprocess_path.read_text(encoding="utf-8")
    if "Patched by COPER vendor" in text:
        log.info("preprocess.py already patched, skipping")
        return
    if _PREPROCESS_CONN_OLD not in text:
        raise RuntimeError(
            "preprocess.py no longer matches the expected Microsoft template; "
            "update data_mngmt/sepsis_rl/mimic_sepsis_vendor.py patch strings."
        )
    preprocess_path.write_text(text.replace(_PREPROCESS_CONN_OLD, _PREPROCESS_CONN_NEW), encoding="utf-8")
    log.info("Patched preprocess.py for PGHOST/PGDATABASE/PGUSER/PGPASSWORD")


_SEP_COHORT_ARGPARSE_OLD = """parser.add_argument("--process_raw", action='store_true', help="If specified, additionally save trajectories without normalized features")
parser.add_argument("--save_intermediate", action="store_true", help="If specified, save off intermediate tables used to construct final patient table")
pargs = parser.parse_args()

print('Loading processed files created from database using "preprocess.py"')"""

_SEP_COHORT_ARGPARSE_NEW = """parser.add_argument("--process_raw", action='store_true', help="If specified, additionally save trajectories without normalized features")
parser.add_argument("--save_intermediate", action="store_true", help="If specified, save off intermediate tables used to construct final patient table")
parser.add_argument(
    "--bloc-interval-hours",
    type=float,
    default=1.0,
    help="Hours per MDP row (aggregation window for vitals/labs/IO). Must be a positive integer (e.g. 1, 2, 4).",
)
pargs = parser.parse_args()

_bloc = float(pargs.bloc_interval_hours)
BLOC_INTERVAL_HOURS = int(round(_bloc))
if BLOC_INTERVAL_HOURS < 1:
    raise SystemExit("bloc-interval-hours must be >= 1")
if abs(BLOC_INTERVAL_HOURS - _bloc) > 1e-6:
    raise SystemExit("bloc-interval-hours must be a whole number of hours")
print(f'Bloc interval (hours): {BLOC_INTERVAL_HOURS}')

print('Loading processed files created from database using "preprocess.py"')"""

# Microsoft mimic_sepsis main (spacing differs from older local vendored copies).
_SEP_COHORT_TIMESTEP_85_OLD = """timestep = 4 # Resolution of timesteps, in hours
irow = 0 
icustayidlist = np.unique(reformat[:,1]).astype(np.int32)
reformat2 = np.nan*np.ones((reformat.shape[0], 85)) # Output array """

_SEP_COHORT_TIMESTEP_85_NEW = """timestep = BLOC_INTERVAL_HOURS  # Resolution of timesteps, in hours (CLI --bloc-interval-hours)
irow = 0 
icustayidlist = np.unique(reformat[:,1]).astype(np.int32)
reformat2 = np.nan*np.ones((reformat.shape[0], 85))  # Output array"""

_SEP_COHORT_TIMESTEP_86_OLD = """timestep = 4 # Resolution of timesteps, in hours
irow = 0 
icustayidlist = np.unique(reformat[:,1]).astype(np.int32)
reformat2 = np.nan*np.ones((reformat.shape[0], 86)) # Output array"""

_SEP_COHORT_TIMESTEP_86_NEW = """timestep = BLOC_INTERVAL_HOURS  # Resolution of timesteps, in hours (CLI --bloc-interval-hours)
irow = 0 
icustayidlist = np.unique(reformat[:,1]).astype(np.int32)
reformat2 = np.nan*np.ones((reformat.shape[0], 86))  # Output array"""


def _patch_sepsis_cohort_bloc_interval(path: Path) -> None:
    """Inject ``--bloc-interval-hours`` + use ``BLOC_INTERVAL_HOURS`` in aggregation loops."""
    text = path.read_text(encoding="utf-8")
    if "--bloc-interval-hours" in text:
        log.info("sepsis_cohort.py already patched for bloc interval, skipping")
        return
    if _SEP_COHORT_ARGPARSE_OLD not in text:
        raise RuntimeError(
            "sepsis_cohort.py no longer matches the expected Microsoft template; "
            "update data_mngmt/sepsis_rl/mimic_sepsis_vendor.py patch strings."
        )
    text = text.replace(_SEP_COHORT_ARGPARSE_OLD, _SEP_COHORT_ARGPARSE_NEW)
    if _SEP_COHORT_TIMESTEP_85_OLD not in text or _SEP_COHORT_TIMESTEP_86_OLD not in text:
        raise RuntimeError(
            "sepsis_cohort.py timestep blocks no longer match; "
            "update data_mngmt/sepsis_rl/mimic_sepsis_vendor.py patch strings."
        )
    text = text.replace(_SEP_COHORT_TIMESTEP_85_OLD, _SEP_COHORT_TIMESTEP_85_NEW)
    text = text.replace(_SEP_COHORT_TIMESTEP_86_OLD, _SEP_COHORT_TIMESTEP_86_NEW)
    path.write_text(text, encoding="utf-8")
    log.info("Patched sepsis_cohort.py for --bloc-interval-hours")


def vendor_mimic_sepsis(
    *,
    dest: Path | None = None,
    git_url: str = DEFAULT_GIT_URL,
    ref: str = "main",
    force: bool = False,
) -> Path:
    """Clone mimic_sepsis, copy tracked files into ``dest``, patch preprocess, delete clone.

    Returns the destination directory.
    """
    dest = Path(dest or default_upstream_dir()).resolve()
    if dest.exists() and any(dest.iterdir()) and not force:
        log.info("Upstream already present at %s (use force=True to refresh)", dest)
        _patch_preprocess(dest / "preprocess.py")
        _patch_sepsis_cohort_bloc_interval(dest / "sepsis_cohort.py")
        return dest

    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mimic_sepsis_vendor_") as tmp:
        clone_root = Path(tmp) / "mimic_sepsis"
        log.info("Cloning %s (%s) …", git_url, ref)
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

        for name in ("preprocess.py", "sepsis_cohort.py", "LICENSE", "README.md"):
            src = clone_root / name
            if src.is_file():
                shutil.copy2(src, dest / name)

        ref_dir = clone_root / "ReferenceFiles"
        if ref_dir.is_dir():
            shutil.copytree(ref_dir, dest / "ReferenceFiles", dirs_exist_ok=True)

        (dest / "SOURCE_REVISION.txt").write_text(
            f"url={git_url}\nref={ref}\ncommit={rev}\n",
            encoding="utf-8",
        )
        (dest / "VENDOR_README.txt").write_text(
            "Vendored from https://github.com/microsoft/mimic_sepsis (MIT).\n"
            "Do not edit in place except via data_mngmt/mimic_sepsis_vendor.py; re-vendor to refresh.\n",
            encoding="utf-8",
        )

    _patch_preprocess(dest / "preprocess.py")
    _patch_sepsis_cohort_bloc_interval(dest / "sepsis_cohort.py")
    log.info("Vendored mimic_sepsis into %s", dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Vendor microsoft/mimic_sepsis into data_mngmt/vendor/mimic_sepsis_upstream/"
    )
    p.add_argument("--dest", type=Path, default=None, help="Destination directory")
    p.add_argument(
        "--https",
        action="store_true",
        help=f"Use HTTPS clone URL instead of SSH ({DEFAULT_GIT_URL})",
    )
    p.add_argument("--ref", default="main", help="Git branch or tag")
    p.add_argument("--force", action="store_true", help="Remove existing dest and re-clone")
    args = p.parse_args(argv)
    url = HTTPS_GIT_URL if args.https else DEFAULT_GIT_URL
    try:
        vendor_mimic_sepsis(dest=args.dest, git_url=url, ref=args.ref, force=args.force)
    except subprocess.CalledProcessError as e:
        log.error("git failed: %s", e)
        return 1
    except Exception as e:
        log.exception("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
