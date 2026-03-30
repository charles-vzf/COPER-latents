"""
Extract latent embeddings from a trained COPER model (same forward as coper_model.COPER
but stops before the classifier). Does not modify upstream repos.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def latent_before_classifier(
    model: torch.nn.Module,
    data: torch.Tensor,
    time_steps,
    obj_t,
    pred_t,
) -> torch.Tensor:
    """
    Returns tensor of shape (batch, D): last time index after optional second ODE and LayerNorm,
    matching the slice used for classification (COPER: last latent slot; TRANSFORMER: last sequence step).
    """
    setting = model.config.setting
    time_steps = time_steps[0]
    pred_t = pred_t[0]
    # DataLoader collates `tp` into shape (batch, T); the ODE implementation expects a 1D
    # time grid shared across the batch, so we collapse to the first row.
    if torch.is_tensor(time_steps) and time_steps.ndim > 1:
        # This notebook/data loader setup expects identical time grids across the batch.
        if not torch.allclose(time_steps, time_steps[0].expand_as(time_steps)):
            raise RuntimeError(
                "Expected `time_steps` to be identical across the batch, but got "
                f"shape={tuple(time_steps.shape)} with varying rows."
            )
        time_steps = time_steps[0]
    if torch.is_tensor(pred_t) and pred_t.ndim > 1:
        if not torch.allclose(pred_t, pred_t[0].expand_as(pred_t)):
            raise RuntimeError(
                "Expected `pred_t` to be identical across the batch, but got "
                f"shape={tuple(pred_t.shape)} with varying rows."
            )
        pred_t = pred_t[0]
    h = data
    if len(time_steps) == len(pred_t):
        pred_t = []
    if getattr(model, "emb_dim", None) is not None:
        h = model.embed(h)
    if model.cont_in:
        h = model.ode_in(h, time_steps, pred_t, setting)
    h = model.net(h)
    if getattr(model, "second_node", False):
        L = h.shape[1]
        ts = torch.linspace(0.0, 1.0, L, device=h.device, dtype=h.dtype)
        h = model.ode_out(h, ts, [], setting)
    h = model.norm(h)
    return h[:, -1, :]


@torch.no_grad()
def latent_grid(
    model: torch.nn.Module,
    data: torch.Tensor,
    time_steps,
    obj_t,
    pred_t,
) -> torch.Tensor:
    """
    Returns (batch, num_latents, latent_dim) after LayerNorm, before classifier slice.
    """
    setting = model.config.setting
    time_steps = time_steps[0]
    pred_t = pred_t[0]
    # DataLoader collates `tp` into shape (batch, T); the ODE implementation expects a 1D
    # time grid shared across the batch, so we collapse to the first row.
    if torch.is_tensor(time_steps) and time_steps.ndim > 1:
        # This notebook/data loader setup expects identical time grids across the batch.
        if not torch.allclose(time_steps, time_steps[0].expand_as(time_steps)):
            raise RuntimeError(
                "Expected `time_steps` to be identical across the batch, but got "
                f"shape={tuple(time_steps.shape)} with varying rows."
            )
        time_steps = time_steps[0]
    if torch.is_tensor(pred_t) and pred_t.ndim > 1:
        if not torch.allclose(pred_t, pred_t[0].expand_as(pred_t)):
            raise RuntimeError(
                "Expected `pred_t` to be identical across the batch, but got "
                f"shape={tuple(pred_t.shape)} with varying rows."
            )
        pred_t = pred_t[0]
    h = data
    if len(time_steps) == len(pred_t):
        pred_t = []
    if getattr(model, "emb_dim", None) is not None:
        h = model.embed(h)
    if model.cont_in:
        h = model.ode_in(h, time_steps, pred_t, setting)
    h = model.net(h)
    if getattr(model, "second_node", False):
        L = h.shape[1]
        ts = torch.linspace(0.0, 1.0, L, device=h.device, dtype=h.dtype)
        h = model.ode_out(h, ts, [], setting)
    h = model.norm(h)
    return h
