"""Load COPER or TRANSFORMER from an exported .pt bundle (see utils/export_coper_checkpoint.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


def _namespace_from_hyperparams(h: dict) -> SimpleNamespace:
    drop = h.get("drop")
    return SimpleNamespace(
        UQ=2,
        model_type=h.get("model_type", "COPER"),
        setting="Train",
        drop=drop,
        niters=120,
        num_labels=1,
        fold=h.get("fold", -1),
        kfold=5,
        patience=10,
        lr=1e-4,
        batch_size=64,
        save="results/checkpoints/",
        load=None,
        random_seed=h.get("random_seed", 2022),
        dataset=h.get("dataset", "mimic"),
        project="perceiver",
        num_latents=h["num_latents"],
        rec_dims=40,
        rec_layers=h["rec_layers"],
        units=h["units"],
        emb_dim=h["emb_dim"],
        mask=False,
        cont_in=h.get("cont_in", True),
        cont_out=h.get("cont_out", False),
        second_node=h.get("second_node", False),
        self_per_cross_attn=h["self_per_cross_attn"],
        latent_heads=h["latent_heads"],
        cross_heads=h["cross_heads"],
        cross_dim_head=h["cross_dim_head"],
        latent_dim_head=h["latent_dim_head"],
        latent_dim=h["latent_dim"],
        ff_dropout=h["ff_dropout"],
        att_dropout=h["att_dropout"],
        ode_dropout=h["ode_dropout"],
        bidirectional=False,
        num_layers=2,
        lstm_dropout=0.1,
        rec_hidden=64,
        gen_hidden=32,
        enc_num_heads=1,
        dec_num_heads=1,
        embed_time=256,
        learn_emb=False,
        kl=False,
        norm=False,
        std=0.01,
        k_iwae=1,
        alpha=5.0,
        poisson=False,
        gen_layers=3,
        gru_units=50,
        linear_classif=True,
        seq_len=h.get("seq_len", 48),
    )


def load_coper_from_bundle(
    bundle_path: Path,
    repo_root: Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Load exported bundle, instantiate **COPER** or **TRANSFORMER** from ``meta.hyperparams.model_type``, load weights.

    Returns (model, meta_dict).
    """
    bundle_path = Path(bundle_path).resolve()
    repo_root = Path(repo_root).resolve()
    r = str(repo_root)
    if r not in sys.path:
        sys.path.insert(0, r)

    from src.coper_model import COPER
    from src.transformer_model import TRANSFORMER

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(bundle_path, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
        meta = raw.get("meta", {})
    else:
        state = raw
        meta = {}

    h = meta.get("hyperparams", {})
    if not h:
        raise ValueError(
            "Bundle missing meta.hyperparams; re-export with utils/export_coper_checkpoint.py"
        )

    args = _namespace_from_hyperparams(h)
    # Robustness: some older bundles may have an incorrect `meta.hyperparams.second_node`.
    # If the checkpoint contains `ode_out.*` weights, we must instantiate the model
    # with `second_node=True` so `load_state_dict(..., strict=True)` succeeds.
    has_ode_out = isinstance(state, dict) and any(
        str(k).startswith("ode_out.") for k in state.keys()
    )
    if has_ode_out:
        args.second_node = True
    n_labels = 1
    input_size = h.get("input_size_mimic", 76)

    mt = str(h.get("model_type", "COPER")).upper()
    if mt == "TRANSFORMER":
        model = TRANSFORMER(
            args,
            n_labels,
            input_size,
            args.num_latents,
            args.latent_dim,
            args.rec_layers,
            args.units,
            nn.Tanh,
            args.cont_in,
            args.cont_out,
            emb_dim=args.emb_dim,
            device=device,
        ).to(device)
    else:
        model = COPER(
            args,
            n_labels,
            input_size,
            args.num_latents,
            args.latent_dim,
            args.rec_layers,
            args.units,
            nn.Tanh,
            args.cont_in,
            args.cont_out,
            emb_dim=args.emb_dim,
            device=device,
        ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta


def load_meta_json(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)
