"""
PyTorch models for arm-specific response prediction.

    PooledMLP  "S-learner": one network takes genes + clinical + the ARM as an input
               feature and outputs one logit. Counterfactual: change the arm input.
    ArmNet     "TARNet-style" multi-task net (Shalit et al., ICML 2017): a SHARED
               trunk learns a representation of the tumour, and one small HEAD per
               arm maps it to that arm's pCR logit. Each patient trains only the head
               of the arm they actually received (the "factual" loss), but every head
               can be evaluated for every patient (counterfactual predictions).

Why a shared trunk? Each arm has only 49–179 HER2-negative patients, far too
few to train separate deep networks. Sharing the representation lets arms "borrow
strength" from each other, the same intuition as hierarchical models in
statistics, while the heads let the effect of each drug depend on the tumour.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class PooledMLP(nn.Module):
    """x = [genes, clinical, arm one-hot] -> 1 logit."""

    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2),                 # input dropout: thousands of noisy genes
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)       # (n,)


class ArmNet(nn.Module):
    """Shared trunk + one head per arm. forward(x) returns logits for ALL arms: shape (n, K)."""

    def __init__(self, n_in: int, n_arms: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        # ModuleList registers each head's parameters with the model (a plain list would not).
        self.heads = nn.ModuleList([nn.Linear(hidden // 2, 1) for _ in range(n_arms)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)                                       # (n, hidden/2)
        return torch.cat([head(h) for head in self.heads], 1)   # (n, K)


def _train(model: nn.Module, loss_fn, tensors: tuple, val_tensors: tuple, lr: float = 1e-3,
           weight_decay: float = 1e-2, max_epochs: int = 300, patience: int = 25,
           batch_size: int = 64, seed: int = 0) -> nn.Module:
    """Generic mini-batch training loop with AdamW and early stopping on validation loss.

    Early stopping is itself a regulariser: training stops once the held-out loss
    stops improving, and the best weights are restored (Prechelt, 1998).
    """
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, wait = np.inf, copy.deepcopy(model.state_dict()), 0
    for _ in range(max_epochs):
        model.train()                              # dropout ON
        for batch in loader:
            opt.zero_grad()                        # reset accumulated gradients
            loss_fn(model, *batch).backward()      # forward + backward
            opt.step()                             # AdamW update
        model.eval()                               # dropout OFF for evaluation
        with torch.no_grad():
            v = loss_fn(model, *val_tensors).item()
        if v < best - 1e-4:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _val_split(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    k = max(1, int(round(frac * n)))
    return idx[k:], idx[:k]


class TorchArmModel:
    """Common interface: fit(G, C, arm, y) and predict_arms(G, C) -> (n, K) probabilities.

    kind = "armnet" (multi-head) or "pooled" (S-learner MLP). `n_seeds` > 1 trains a
    deep ensemble and averages probabilities (Lakshminarayanan et al., 2017).
    """

    def __init__(self, n_arms: int, kind: str = "armnet", hidden: int = 64,
                 dropout: float = 0.5, weight_decay: float = 1e-2, lr: float = 1e-3,
                 n_seeds: int = 1, val_frac: float = 0.2, seed: int = 0):
        self.n_arms, self.kind = n_arms, kind
        self.hp = dict(hidden=hidden, dropout=dropout)
        self.opt = dict(weight_decay=weight_decay, lr=lr)
        self.n_seeds, self.val_frac, self.seed = n_seeds, val_frac, seed
        self.models: list[nn.Module] = []

    def _x(self, G, C, arm=None) -> torch.Tensor:
        parts = [G, C]
        if self.kind == "pooled":
            parts.append(np.eye(self.n_arms, dtype=np.float32)[arm])
        return torch.from_numpy(np.column_stack(parts).astype(np.float32))

    def fit(self, G, C, arm, y) -> TorchArmModel:
        arm = np.asarray(arm)
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        at = torch.from_numpy(arm.astype(np.int64))
        self.models = []
        for s in range(self.n_seeds):
            seed = self.seed + 1000 * s
            set_seed(seed)
            tr, va = _val_split(len(yt), self.val_frac, seed)
            if self.kind == "armnet":
                X = self._x(G, C)
                model = ArmNet(X.shape[1], self.n_arms, **self.hp)

                def loss_fn(m, x, a, yy):
                    # Factual loss: pick each patient's own arm column from the (n, K) logits.
                    logit = m(x).gather(1, a.view(-1, 1)).squeeze(1)
                    return nn.functional.binary_cross_entropy_with_logits(logit, yy)
                tensors = (X[tr], at[tr], yt[tr])
                val = (X[va], at[va], yt[va])
            else:
                X = self._x(G, C, arm)
                model = PooledMLP(X.shape[1], **self.hp)

                def loss_fn(m, x, yy):
                    return nn.functional.binary_cross_entropy_with_logits(m(x), yy)
                tensors, val = (X[tr], yt[tr]), (X[va], yt[va])
            self.models.append(_train(model, loss_fn, tensors, val, seed=seed, **self.opt))
        return self

    @torch.no_grad()
    def predict_arms(self, G, C) -> np.ndarray:
        n = len(G)
        out = np.zeros((n, self.n_arms))
        for m in self.models:
            if self.kind == "armnet":
                out += torch.sigmoid(m(self._x(G, C))).numpy()
            else:
                for k in range(self.n_arms):
                    out[:, k] += torch.sigmoid(m(self._x(G, C, np.full(n, k)))).numpy()
        return out / len(self.models)
