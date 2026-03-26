"""Count trainable parameters for COPER (MIMIC mortality defaults) without loading checkpoints."""
from __future__ import annotations

import types

import torch
import torch.nn as nn

from src.coper_model import COPER


def count_coper_params_mimic(
    latent_dim: int,
    *,
    emb_dim: int = 32,
    num_latents: int = 48,
    units: int = 128,
    rec_layers: int = 3,
    latent_heads: int = 2,
    cross_heads: int = 1,
    cross_dim_head: int = 128,
    latent_dim_head: int = 128,
    att_dropout: float = 0.5,
    ff_dropout: float = 0.5,
    ode_dropout: float = 0.5,
    self_per_cross_attn: int = 1,
    input_dim: int = 76,
    n_labels: int = 1,
    second_node: bool = False,
    device: torch.device | None = None,
) -> int:
    """
    Matches the COPER stack used in `utils/run_exp.py` for MIMIC (cont_in=True, cont_out=False).
    `latent_dim` is the only structural knob varied in a typical latent-width sweep.
    """
    dev = device or torch.device("cpu")
    cfg = types.SimpleNamespace(
        setting="Train",
        att_dropout=att_dropout,
        ff_dropout=ff_dropout,
        ode_dropout=ode_dropout,
        self_per_cross_attn=self_per_cross_attn,
        latent_heads=latent_heads,
        cross_heads=cross_heads,
        cross_dim_head=cross_dim_head,
        latent_dim_head=latent_dim_head,
        second_node=second_node,
    )
    model = COPER(
        cfg,
        n_labels,
        input_dim,
        num_latents,
        latent_dim,
        rec_layers,
        units,
        nn.Tanh,
        True,
        False,
        emb_dim=emb_dim,
        device=dev,
    ).to(dev)
    return sum(p.numel() for p in model.parameters())
