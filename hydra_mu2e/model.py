"""
The per-plot CNN classifier

Hydra trains one image classifier per plot type. This is a small stand-in sized for CPU
training on a gpvm: four conv/BN/ReLU/max-pool blocks (128 -> 8 pixels) and a dense head
that keeps the spatial layout, since where a feature sits matters (a PE peak 15% low).
The output is P(BAD).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn


class PlotCNN(nn.Module):
    def __init__(self, size: int = 128, channels=(16, 32, 64, 64)):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        side = size // 2 ** len(channels)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(c_in * side * side, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


@dataclass
class History:
    train_loss: List[float]
    val_acc: List[float]


def train(X: np.ndarray, y: np.ndarray, epochs: int = 15, seed: int = 0,
          val_frac: float = 0.2, batch: int = 32, lr: float = 1e-3):
    """Train a PlotCNN on images X [N,1,H,W] and labels y (1 = BAD). Returns (model, History)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    val, tr = idx[:n_val], idx[n_val:]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    model = PlotCNN(size=X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    hist = History([], [])
    for _ in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(tr))
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        hist.train_loss.append(total / max(len(tr), 1))
        if n_val:
            p = predict(model, X[val])
            hist.val_acc.append(float(((p > 0.5) == (y[val] > 0.5)).mean()))
    return model, hist


@torch.no_grad()
def predict(model: PlotCNN, X: np.ndarray) -> np.ndarray:
    """P(BAD) for each image."""
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))).numpy()
