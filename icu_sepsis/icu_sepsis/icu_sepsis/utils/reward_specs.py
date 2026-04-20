"""Alternative tabular reward tensors for ICU-Sepsis (same transitions, different ``r_mat``).

Terminal columns (death, survival, ``s_inf``) are always taken from the packaged
``dynamics.npz`` matrix so the absorbing outcome semantics stay aligned with the
cohort labels. Only transitions among transient states get dense shaping.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# gym.make(..., reward_spec=...)
SURVIVAL_PACKAGED = "packaged"
SOFA_NEXT = "sofa_next"
SOFA_DELTA = "sofa_delta"
SEVERITY_PROXY = "severity_proxy"
DEATH_PROB_DENSE = "death_prob_dense"
COPER_COMPOSITE = "coper_composite"

KNOWN_SPECS = frozenset(
    {
        SURVIVAL_PACKAGED,
        "survival",
        "default",
        SOFA_NEXT,
        SOFA_DELTA,
        SEVERITY_PROXY,
        DEATH_PROB_DENSE,
        COPER_COMPOSITE,
    }
)


def _n_transient(n_states: int) -> int:
    return int(n_states) - 3


def build_reward_matrix(
    tx_mat: np.ndarray,
    sofa_scores: np.ndarray,
    state_cluster_centers: np.ndarray,
    packaged_r_mat: np.ndarray,
    reward_spec: str,
    params: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return a new ``(S, A, S)`` reward tensor.

    Args:
        tx_mat: Transition tensor ``P(s'|s,a)``.
        sofa_scores: Mean SOFA per discrete state (length ``S``).
        state_cluster_centers: Shape ``(S, F)`` cluster centroids (transient + absorbing).
        packaged_r_mat: Rewards shipped with the environment (sparse + terminal).
        reward_spec: One of :data:`KNOWN_SPECS` (aliases ``survival`` / ``default`` = packaged).
        params: Optional floats ``lambda``, ``beta``, ``w_sofa``, ``w_death``, etc.
    """
    params = dict(params or {})
    spec = reward_spec if reward_spec is not None else SURVIVAL_PACKAGED
    if spec in (SURVIVAL_PACKAGED, "survival", "default", ""):
        return np.array(packaged_r_mat, dtype=np.float64, copy=True)

    if spec not in KNOWN_SPECS:
        raise ValueError(
            f"Unknown reward_spec={reward_spec!r}. Expected one of {sorted(KNOWN_SPECS)}."
        )

    tx_mat = np.asarray(tx_mat, dtype=np.float64)
    packaged_r_mat = np.asarray(packaged_r_mat, dtype=np.float64)
    sofa = np.asarray(sofa_scores, dtype=np.float64).reshape(-1)
    n_s, n_a, n_sp = tx_mat.shape
    if packaged_r_mat.shape != (n_s, n_a, n_sp):
        raise ValueError(
            f"Shape mismatch tx_mat {tx_mat.shape} vs r_mat {packaged_r_mat.shape}"
        )
    if len(sofa) != n_s:
        raise ValueError(f"sofa_scores length {len(sofa)} != num states {n_s}")

    n_tr = _n_transient(n_s)
    death = n_s - 3
    r = np.array(packaged_r_mat, copy=True)

    if spec == SOFA_NEXT:
        lam = float(params.get("lambda", 0.05))
        smax = max(float(sofa[:n_tr].max()), 1e-6)
        r[:n_tr, :, :n_tr] = (-lam / smax) * sofa[:n_tr].reshape(1, 1, n_tr)
        return r

    if spec == SOFA_DELTA:
        lam = float(params.get("lambda", 0.05))
        den = float(params.get("sofa_scale", max(float(np.ptp(sofa[:n_tr])), 1e-6)))
        svec = sofa[:n_tr].reshape(n_tr, 1)
        sprime = sofa[:n_tr].reshape(1, n_tr)
        delta_block = (-lam / den) * (sprime - svec)
        r[:n_tr, :, :n_tr] = delta_block[:, np.newaxis, :]
        return r

    if spec == SEVERITY_PROXY:
        lam = float(params.get("lambda", 0.05))
        cc = np.asarray(state_cluster_centers, dtype=np.float64)[:n_tr]
        med = np.median(cc, axis=0)
        hinge = np.clip(cc - med, 0.0, None)
        psi = hinge.mean(axis=1)
        pmax = max(float(psi.max()), 1e-6)
        r[:n_tr, :, :n_tr] = (-lam / pmax) * psi.reshape(1, 1, n_tr)
        return r

    if spec == DEATH_PROB_DENSE:
        beta = float(params.get("beta", 1.0))
        pdeath = tx_mat[:, :, death]
        r[:n_tr, :, :n_tr] = (-beta * pdeath[:n_tr, :, np.newaxis])
        return r

    if spec == COPER_COMPOSITE:
        w_sofa = float(params.get("w_sofa", 0.5))
        w_death = float(params.get("w_death", 0.5))
        r_s = build_reward_matrix(
            tx_mat,
            sofa,
            state_cluster_centers,
            packaged_r_mat,
            SOFA_NEXT,
            params,
        )
        r_d = build_reward_matrix(
            tx_mat,
            sofa,
            state_cluster_centers,
            packaged_r_mat,
            DEATH_PROB_DENSE,
            params,
        )
        r[:n_tr, :, :n_tr] = (
            w_sofa * r_s[:n_tr, :, :n_tr] + w_death * r_d[:n_tr, :, :n_tr]
        )
        return r

    raise AssertionError(f"Unhandled reward_spec {spec!r}")


def episode_survived(final_state: int, state_survival: int) -> bool:
    return int(final_state) == int(state_survival)
