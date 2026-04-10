"""Lightweight LSTM classifier for MIMIC mortality (sequence tensors)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset


class _MortalityLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
    ):
        super().__init__()
        bi = 2 if bidirectional else 1
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim * bi, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)


def _acc_auroc(y_true: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    proba = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64).reshape(-1)))
    pred = (proba >= 0.5).astype(np.int64)
    acc = float(metrics.accuracy_score(y_true, pred))
    try:
        auroc = float(metrics.roc_auc_score(y_true, proba))
    except ValueError:
        auroc = float("nan")
    return acc, auroc


@torch.no_grad()
def _eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, zs = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        ys.append(yb.numpy())
        zs.append(logits.detach().cpu().numpy())
    y = np.concatenate(ys, axis=0)
    z = np.concatenate(zs, axis=0)
    return y, z


def fit_lstm_mortality(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    *,
    hidden_dim: int = 64,
    num_layers: int = 1,
    bidirectional: bool = True,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 40,
    patience: int = 7,
    device: str | torch.device | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    """Train a small LSTM on ``(N, T, F)`` float tensors; binary logits + BCEWithLogitsLoss."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(random_state))
    if device_t.type == "cuda":
        torch.cuda.manual_seed_all(int(random_state))

    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.float32).reshape(-1)
    X_va = np.asarray(X_val, dtype=np.float32)
    y_va = np.asarray(y_val, dtype=np.float32).reshape(-1)

    input_dim = int(X_tr.shape[-1])
    model = _MortalityLSTM(
        input_dim=input_dim,
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        bidirectional=bool(bidirectional),
        dropout=float(dropout),
    ).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    crit = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    best_state = None
    best_val = float("inf")
    bad = 0
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        yv, zv = _eval_model(model, val_loader, device_t)
        with torch.no_grad():
            v_loss = float(
                crit(
                    torch.from_numpy(zv).float().to(device_t),
                    torch.from_numpy(yv).float().to(device_t),
                )
            )
        if v_loss < best_val - 1e-6:
            best_val = v_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    yt, zt = _eval_model(model, train_loader, device_t)
    yv, zv = _eval_model(model, val_loader, device_t)
    tr_acc, tr_auroc = _acc_auroc(yt, zt)
    va_acc, va_auroc = _acc_auroc(yv, zv)

    out: dict[str, Any] = {
        "model": "lstm",
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "bidirectional": bool(bidirectional),
        "epochs_trained": int(epoch),
        "train_accuracy": tr_acc,
        "val_accuracy": va_acc,
        "train_auroc": tr_auroc,
        "val_auroc": va_auroc,
    }

    if X_test is not None and y_test is not None:
        X_te = np.asarray(X_test, dtype=np.float32)
        y_te = np.asarray(y_test, dtype=np.float32).reshape(-1)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
            batch_size=int(batch_size),
            shuffle=False,
        )
        yte, zte = _eval_model(model, test_loader, device_t)
        te_acc, te_auroc = _acc_auroc(yte, zte)
        out["test_accuracy"] = te_acc
        out["test_auroc"] = te_auroc

    return out
