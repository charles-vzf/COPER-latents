"""Logistic regression baselines for MIMIC mortality (tabular / flattened sequences)."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def _as_xy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tr = np.asarray(X_train, dtype=np.float64).reshape(len(X_train), -1)
    y_tr = np.asarray(y_train, dtype=np.int64).reshape(-1)
    X_va = np.asarray(X_val, dtype=np.float64).reshape(len(X_val), -1)
    y_va = np.asarray(y_val, dtype=np.int64).reshape(-1)
    return X_tr, y_tr, X_va, y_va


def _acc_auroc(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    proba = np.asarray(proba, dtype=np.float64).reshape(-1)
    pred = (proba >= 0.5).astype(np.int64)
    acc = float(metrics.accuracy_score(y_true, pred))
    try:
        auroc = float(metrics.roc_auc_score(y_true, proba))
    except ValueError:
        auroc = float("nan")
    return acc, auroc


def fit_logistic_mortality(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    *,
    penalty: str = "l2",
    C: float = 1.0,
    max_iter: int = 5000,
    random_state: int = 0,
) -> dict[str, Any]:
    """Fit ``sklearn.linear_model.LogisticRegression`` on flattened ``X_*``.

    ``penalty='l2'`` (default) or ``'l1'`` (Lasso-style sparsity on weights; solver ``saga``).
    """
    if penalty not in ("l1", "l2"):
        raise ValueError("penalty must be 'l1' or 'l2'")
    X_tr, y_tr, X_va, y_va = _as_xy(X_train, y_train, X_val, y_val)
    # sklearn>=1.8: prefer default L2 (omit penalty) to avoid deprecation noise.
    if penalty == "l2":
        clf = LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=int(max_iter),
            random_state=int(random_state),
            class_weight="balanced",
        )
    else:
        clf = LogisticRegression(
            penalty="l1",
            C=float(C),
            solver="saga",
            max_iter=int(max_iter),
            random_state=int(random_state),
            class_weight="balanced",
        )
    clf.fit(X_tr, y_tr)

    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va = clf.predict_proba(X_va)[:, 1]
    tr_acc, tr_auroc = _acc_auroc(y_tr, p_tr)
    va_acc, va_auroc = _acc_auroc(y_va, p_va)

    out: dict[str, Any] = {
        "model": f"logistic_{penalty}",
        "penalty": penalty,
        "C": float(C),
        "train_accuracy": tr_acc,
        "val_accuracy": va_acc,
        "train_auroc": tr_auroc,
        "val_auroc": va_auroc,
    }

    if X_test is not None and y_test is not None:
        X_te = np.asarray(X_test, dtype=np.float64).reshape(len(X_test), -1)
        y_te = np.asarray(y_test, dtype=np.int64).reshape(-1)
        p_te = clf.predict_proba(X_te)[:, 1]
        te_acc, te_auroc = _acc_auroc(y_te, p_te)
        out["test_accuracy"] = te_acc
        out["test_auroc"] = te_auroc

    out["estimator"] = clf
    return out
