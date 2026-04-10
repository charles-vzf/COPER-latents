"""Run vendored microsoft/mimic_sepsis preprocess + sepsis_cohort in a working directory."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from data_mngmt import coper_root, load_paths
from data_mngmt.sepsis_rl.mimic_sepsis_vendor import default_upstream_dir, vendor_mimic_sepsis

log = logging.getLogger(__name__)

MIMICTABLE_NAME = "MIMICtable.csv"
RL_COHORT_COPY_NAME = "mimic_dataset_table_from_mimic_sepsis.csv"


def _mimic_sepsis_subprocess_env() -> dict[str, str]:
    """Fresh env dict for child scripts (inherits current process env)."""
    return os.environ.copy()


def _use_pty_for_mimic_scripts() -> bool:
    """Use a pseudo-TTY so ``pyprind`` sees ``isatty()==True`` and draws ProgBars.

    Disable with ``COPER_MIMIC_SEPSIS_PTY=0`` if ``pty.openpty`` fails or you
    prefer plain pipes. Windows is not supported (no pty in this path).
    """
    if sys.platform == "win32":
        return False
    v = os.environ.get("COPER_MIMIC_SEPSIS_PTY", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _run_cmd_with_pty(cmd: list[str], cwd: str, env: dict[str, str]) -> int:
    import pty

    # pyprind only gates on sys.stdout.isatty(); some PTY/Jupyter combos still get False.
    # This matches the library's own escape hatch and avoids spamming "No valid output stream."
    env = dict(env)
    env.setdefault("PYCHARM_HOSTED", "1")

    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
    except Exception:
        os.close(master_fd)
        os.close(slave_fd)
        raise

    os.close(slave_fd)

    def _write_stderr_bytes(chunk: bytes) -> None:
        # Jupyter/IPython replaces sys.stderr with OutStream (no .buffer).
        buf = getattr(sys.stderr, "buffer", None)
        if buf is not None:
            buf.write(chunk)
            buf.flush()
        else:
            sys.stderr.write(chunk.decode("latin-1", errors="replace"))
            sys.stderr.flush()

    def _forward_master_to_stderr() -> None:
        try:
            while True:
                chunk = os.read(master_fd, 65536)
                if not chunk:
                    break
                try:
                    _write_stderr_bytes(chunk)
                except (AttributeError, OSError, TypeError, ValueError):
                    pass
        except OSError:
            pass

    th = threading.Thread(target=_forward_master_to_stderr, daemon=False)
    th.start()
    ret = proc.wait()
    th.join()
    try:
        os.close(master_fd)
    except OSError:
        pass
    return int(ret) if ret is not None else 0


def _run_mimic_script(cmd: list[str], cwd: str, *, env: dict[str, str] | None = None) -> None:
    """Run ``preprocess`` / ``sepsis_cohort`` with real TTY when possible (pyprind bars)."""
    env = env if env is not None else _mimic_sepsis_subprocess_env()
    if _use_pty_for_mimic_scripts():
        try:
            log.info("Running mimic_sepsis script with pseudo-TTY (progress output forwarded to stderr).")
            rc = _run_cmd_with_pty(cmd, cwd, env)
        except OSError as e:
            log.warning(
                "pty unavailable (%s); falling back to subprocess without TTY "
                "(pyprind may print 'No valid output stream').",
                e,
            )
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
            return
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def default_workdir() -> Path:
    data = load_paths()
    raw = data.get("mimic_sepsis_workdir")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (coper_root() / p).resolve()
    return (coper_root() / "data_mngmt" / "generated" / "mimic_sepsis_work").resolve()


def ensure_upstream() -> Path:
    up = default_upstream_dir()
    if not (up / "sepsis_cohort.py").is_file():
        log.info("Vendoring mimic_sepsis …")
        vendor_mimic_sepsis()
    return up


def prepare_workdir(workdir: Path, *, upstream: Path | None = None) -> Path:
    """Ensure ``workdir`` has ``ReferenceFiles/`` (from upstream) for sepsis_cohort."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    up = Path(upstream or ensure_upstream())
    ref_src = up / "ReferenceFiles"
    ref_dst = workdir / "ReferenceFiles"
    if not ref_src.is_dir():
        raise FileNotFoundError(f"Missing {ref_src}; run python -m data_mngmt.sepsis_rl.mimic_sepsis_vendor")
    # Skip re-copy when the workdir was already provisioned (saves time on repeated sepsis_cohort runs).
    if (ref_dst / "sample_and_hold.csv").is_file():
        return workdir
    if ref_dst.exists():
        shutil.rmtree(ref_dst)
    shutil.copytree(ref_src, ref_dst)
    return workdir


def run_preprocess(
    workdir: Path,
    *,
    upstream: Path | None = None,
    username: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
) -> None:
    """Run ``preprocess.py``; writes ``workdir/processed_files/*.csv`` (many hours)."""
    workdir = prepare_workdir(workdir, upstream=upstream)
    up = Path(upstream or ensure_upstream())
    script = up / "preprocess.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    cmd = [sys.executable, str(script)]
    if username:
        cmd.extend(["-u", username])
    if password is not None:
        cmd.extend(["-p", password])
    if dbname:
        cmd.extend(["--dbname", dbname])
    if host:
        cmd.extend(["--host", host])
    if port is not None:
        cmd.extend(["--port", str(port)])

    log.info("Running preprocess (cwd=%s) …", workdir)
    _run_mimic_script(cmd, str(workdir), env=_mimic_sepsis_subprocess_env())


def run_sepsis_cohort(
    workdir: Path,
    *,
    upstream: Path | None = None,
    save_intermediate: bool = True,
    process_raw: bool = False,
    bloc_interval_hours: float = 1.0,
) -> Path:
    """Run ``sepsis_cohort.py``; with ``save_intermediate``, writes ``MIMICtable.csv`` in cwd."""
    workdir = prepare_workdir(workdir, upstream=upstream)
    pf = workdir / "processed_files"
    if not pf.is_dir() or not any(pf.glob("*.csv")):
        raise FileNotFoundError(
            f"Expected CSV extracts under {pf}; run preprocess first."
        )
    up = Path(upstream or ensure_upstream())
    script = up / "sepsis_cohort.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    cmd = [sys.executable, str(script)]
    if save_intermediate:
        cmd.append("--save_intermediate")
    if process_raw:
        cmd.append("--process_raw")
    cmd.extend(["--bloc-interval-hours", str(float(bloc_interval_hours))])

    log.info("Running sepsis_cohort (cwd=%s) …", workdir)
    _run_mimic_script(cmd, str(workdir), env=_mimic_sepsis_subprocess_env())

    table = workdir / MIMICTABLE_NAME
    if save_intermediate and not table.is_file():
        raise FileNotFoundError(
            f"Expected {table} after sepsis_cohort --save_intermediate"
        )
    return table


def build_rl_cohort_csv_from_mimic_db(
    workdir: Path | None = None,
    *,
    skip_preprocess: bool = False,
    save_intermediate: bool = True,
    bloc_interval_hours: float = 1.0,
    username: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
) -> Path:
    """Run vendored ``preprocess.py`` + ``sepsis_cohort.py``; return RL cohort CSV for ``build_mdp``.

    Requires a MIMIC-III **PostgreSQL** database (schema ``mimiciii``). Connection defaults follow
    ``preprocess.py`` patches: ``PGHOST``, ``PGUSER``, ``PGPASSWORD``, ``PGDATABASE``, ``PGPORT``,
    or the explicit kwargs above.

    Output rows are one **clinical bloc** per ``bloc_interval_hours`` (default **1 h** in unified build) per ICU stay
    (``icustayid``) with vitals, labs, SOFA/SIRS, fluid and vasopressor columns used by
    ``icu_sepsis_helpers`` (see ``mdp_creation/create_rl_table.py`` for the exact feature list).
    Pass ``bloc_interval_hours`` (or ``sepsis_cohort.py --bloc-interval-hours``) to change aggregation.
    """
    workdir = Path(workdir or default_workdir()).resolve()
    ensure_upstream()

    pf = workdir / "processed_files"
    if not skip_preprocess:
        run_preprocess(
            workdir,
            username=username,
            password=password,
            host=host,
            port=port,
            dbname=dbname,
        )
    elif not pf.is_dir() or not any(pf.glob("*.csv")):
        raise FileNotFoundError(
            f"skip_preprocess=True but no CSV extracts under {pf}. "
            "Each unified build slug uses its own mimic_sepsis workdir — first MDP run for a new slug "
            "needs preprocess. Set UnifiedBuildParams.mdp_skip_preprocess=False (notebook: "
            "MDP_SKIP_PREPROCESS=False) or run preprocess once, then you may skip on later runs."
        )

    run_sepsis_cohort(
        workdir,
        save_intermediate=save_intermediate,
        bloc_interval_hours=bloc_interval_hours,
    )
    return copy_mimictable_for_mdp(workdir)


def copy_mimictable_for_mdp(workdir: Path, dest: Path | None = None) -> Path:
    """Copy ``MIMICtable.csv`` to a stable RL-cohort filename for ``build_mdp`` / unified build."""
    src = Path(workdir) / MIMICTABLE_NAME
    if not src.is_file():
        raise FileNotFoundError(src)
    out = Path(dest) if dest else (Path(workdir) / RL_COHORT_COPY_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    log.info("Copied AI Clinician table %s -> %s", src, out)
    return out
