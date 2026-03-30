"""Reusable helpers for running and parsing COPER benchmark experiments."""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def patch_paths_json(repo_dir: Path, mimic_pickle: Path) -> dict:
    """Patch repo paths.json to ensure same mimic3_mortality input."""
    p = repo_dir / "paths.json"
    original = _load_json(p)
    updated = dict(original)
    updated["mimic3_mortality"] = str(mimic_pickle)
    _write_json(p, updated)
    return original


def restore_paths_json(repo_dir: Path, original: dict) -> None:
    p = repo_dir / "paths.json"
    _write_json(p, original)


def parse_metrics(stdout: str) -> dict:
    patterns = {
        "test_og_auroc": r"Test-OG AUROC\s*=\s*([0-9]*\.?[0-9]+)",
        "test_og_auprc": r"Test-OG AUPRC\s*=\s*([0-9]*\.?[0-9]+)",
        "test_g_auroc": r"Test-G AUROC\s*=\s*([0-9]*\.?[0-9]+)",
        "test_g_auprc": r"Test-G AUPRC\s*=\s*([0-9]*\.?[0-9]+)",
    }
    out = {}
    for key, pat in patterns.items():
        m = re.findall(pat, stdout)
        out[key] = float(m[-1]) if m else np.nan
    return out


def niters_from_cli_tokens(tokens: list) -> int | None:
    """Read `--niters` from a flat argv-style list."""
    for i, tok in enumerate(tokens):
        if tok == "--niters" and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                return None
    return None


def model_type_from_cli_tokens(tokens: list) -> str:
    """Read `--model-type` from a flat argv-style list (shared_args + extra_args)."""
    for i, tok in enumerate(tokens):
        if tok == "--model-type" and i + 1 < len(tokens):
            return str(tokens[i + 1])
    return "COPER"


def expected_ckpt_path(
    results_root: Path,
    fold: int,
    drop: float,
    random_seed: int,
    second_node: bool,
    model_type: str = "COPER",
) -> Path:
    """Match utils/run_exp.py checkpoint naming (suffix _N2 for --second-node)."""
    drop_tag = f"{drop:g}"
    suffix = "_N2" if second_node else ""
    name = f"{model_type}-mimic-F{fold}_D{drop_tag}_S{random_seed}{suffix}.ckpt"
    return results_root / "checkpoints" / name


def run_one(
    *,
    arch_id: str,
    arch_label: str,
    extra_args: list,
    seed: int,
    drop: float,
    shared_args: list,
    repo: Path,
    results_root: Path,
    fold: int,
    python_executable: str,
) -> dict:
    second_node = "--second-node" in extra_args
    cli_tokens = list(shared_args) + list(extra_args)
    model_type = model_type_from_cli_tokens(cli_tokens)
    niters_val = niters_from_cli_tokens(cli_tokens)
    cmd = [python_executable, "utils/run_exp.py"] + shared_args + extra_args + [
        "--random-seed",
        str(seed),
    ]
    if drop > 0:
        cmd += ["--drop", str(drop)]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    dt = time.time() - t0

    text_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    metrics = parse_metrics(text_out)
    ckpt_path = expected_ckpt_path(
        results_root, fold, drop, seed, second_node, model_type=model_type
    )

    return {
        "arch_id": arch_id,
        "architecture": arch_label,
        "model_type": model_type,
        "niters": niters_val,
        "second_node": second_node,
        "repo_dir": str(repo),
        "seed": seed,
        "drop": drop,
        "return_code": proc.returncode,
        "runtime_sec": dt,
        "ckpt_path": str(ckpt_path),
        "ckpt_exists": ckpt_path.is_file(),
        **metrics,
        "raw_tail": "\n".join(text_out.splitlines()[-120:]),
    }
