"""
Deep Cox proportional-hazards model in PyTorch (DeepSurv; Katzman et al., 2018).

The Cox model says hazard_i(t) = h0(t) · exp(f(x_i)). It's fitted by maximising
the PARTIAL likelihood, which compares each patient who dies with everyone still
at risk at that time, so the baseline hazard h0 cancels out (Cox, 1972):

    log PL = Σ_{i: event} [ f(x_i) − log Σ_{j: T_j ≥ T_i} exp(f(x_j)) ]

The loss is −log PL / n_events. Sorting patients by time (descending) turns the
risk-set sum into a running log-cumsum-exp, which is O(n) and numerically
stable. Ties are handled with Breslow's approximation (good enough for
continuous survival times).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative mean partial log-likelihood. risk = f(x): shape (n,)."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_risk_set = torch.logcumsumexp(risk, dim=0)        # log Σ_{T_j >= T_i} exp(f_j)
    n_events = event.sum().clamp(min=1.0)
    return -((risk - log_risk_set) * event).sum() / n_events


class DeepCox(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1, bias=False))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LinearCox(nn.Module):
    """Linear Cox model (the classic one), fitted by the same loss: a check on cox_ph_loss."""

    def __init__(self, n_in: int):
        super().__init__()
        self.lin = nn.Linear(n_in, 1, bias=False)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


def fit_cox(model: nn.Module, X, time, event, weight_decay=1e-2, lr=1e-3, max_epochs=500,
            patience=30, val_frac=0.2, seed=0, full_batch=True) -> nn.Module:
    """Train with AdamW on the full batch (risk sets need many patients), with early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(time))
    k = int(round(val_frac * len(time)))
    va, tr = idx[:k], idx[k:]
    X = torch.as_tensor(X, dtype=torch.float32)
    T = torch.as_tensor(time, dtype=torch.float32)
    E = torch.as_tensor(event, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, state, wait = np.inf, copy.deepcopy(model.state_dict()), 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        cox_ph_loss(model(X[tr]), T[tr], E[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = cox_ph_loss(model(X[va]), T[va], E[va]).item()
        if v < best - 1e-4:
            best, state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model
