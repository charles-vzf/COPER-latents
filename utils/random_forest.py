"""Random forest baseline for MIMIC mortality (flattened sequences)."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def _flatten(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64).reshape(len(X), -1)


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


def fit_random_forest_mortality(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    *,
    n_estimators: int = 200,
    max_depth: int | None = 16,
    min_samples_leaf: int = 2,
    random_state: int = 0,
    n_jobs: int = -1,
) -> dict[str, Any]:
    X_tr = _flatten(X_train)
    y_tr = np.asarray(y_train, dtype=np.int64).reshape(-1)
    X_va = _flatten(X_val)
    y_va = np.asarray(y_val, dtype=np.int64).reshape(-1)

    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
    )
    clf.fit(X_tr, y_tr)

    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va = clf.predict_proba(X_va)[:, 1]
    tr_acc, tr_auroc = _acc_auroc(y_tr, p_tr)
    va_acc, va_auroc = _acc_auroc(y_va, p_va)

    out: dict[str, Any] = {
        "model": "random_forest",
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "train_accuracy": tr_acc,
        "val_accuracy": va_acc,
        "train_auroc": tr_auroc,
        "val_auroc": va_auroc,
    }

    if X_test is not None and y_test is not None:
        X_te = _flatten(X_test)
        y_te = np.asarray(y_test, dtype=np.int64).reshape(-1)
        p_te = clf.predict_proba(X_te)[:, 1]
        te_acc, te_auroc = _acc_auroc(y_te, p_te)
        out["test_accuracy"] = te_acc
        out["test_auroc"] = te_auroc

    return out
