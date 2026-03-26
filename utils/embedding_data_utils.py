"""Data and latent collection helpers for embedding visualization notebooks."""
from __future__ import annotations

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_xy_split(pickle_path, split: str):
    with open(pickle_path, "rb") as f:
        _details, X_train, y_train, X_val, y_val, X_test, y_test, _ = pickle.load(f)
    if split == "train":
        return X_train, y_train
    if split == "val":
        return X_val, y_val
    if split == "test":
        return X_test, y_test
    raise ValueError(split)


def tensors_to_loader(X_np, y_np, max_samples, batch_size, random_state: int, device):
    X_np = np.asarray(X_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)
    n = X_np.shape[0]
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_np, y_np = X_np[idx], y_np[idx]
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y
            T = x.shape[1]
            self.tp = torch.linspace(0, 1, T, device=x.device)

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, i):
            return {"X": self.x[i], "y": self.y[i], "tp": self.tp}

    ds = _DS(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def collect_latents(model, loader, latent_before_classifier):
    zs, ys = [], []
    for batch in loader:
        X, y = batch["X"], batch["y"]
        tp = batch["tp"][0]
        z = latent_before_classifier(model, X, [tp], [tp], [tp])
        zs.append(z.cpu().numpy())
        ys.append(y.cpu().numpy())
    Z = np.concatenate(zs, axis=0)
    y = np.concatenate(ys, axis=0)
    return Z, y
