"""
L2-penalised logistic regression written from scratch with PyTorch autograd.

Why does this exist in the package (and not only in lesson 01)?
  * It's the reference "one linear layer" model. Tests check that it
    reproduces scikit-learn, which validates our understanding of the loss,
    the penalty and autograd before we trust bigger networks.
  * Later lessons reuse `l2_logistic_loss` as a building block.

Equivalence with scikit-learn (see lessons/01_tensors_autograd.md, section 4):
  scikit-learn minimises   ½‖w‖² + C · Σ_i logloss_i
  we minimise              (1/n) Σ_i logloss_i + (λ/2)‖w‖²   with λ = 1/(C·n)
  Dividing the first objective by C·n gives the second, so both have the same minimiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LogRegResult:
    """Fitted parameters plus the loss history (handy for plotting convergence)."""

    coef: np.ndarray          # shape (p,)
    intercept: float
    loss_history: list[float]


def l2_logistic_loss(
    X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor, lam: float
) -> torch.Tensor:
    """Mean binary cross-entropy on logits + (lam/2)·‖w‖². The intercept is not penalised.

    Shapes: X (n, p), y (n,), w (p,), b (1,). Returns a scalar tensor.
    """
    # Defensive shape checks: the (n,) vs (n, 1) broadcasting bug fails here loudly.
    if X.ndim != 2 or y.ndim != 1 or w.ndim != 1:
        raise ValueError(f"expected X (n,p), y (n,), w (p,); got {tuple(X.shape)}, "
                         f"{tuple(y.shape)}, {tuple(w.shape)}")
    if X.shape[0] != y.shape[0] or X.shape[1] != w.shape[0]:
        raise ValueError("shape mismatch between X, y and w")
    logits = X @ w + b                                     # (n,)
    bce = F.binary_cross_entropy_with_logits(logits, y)    # stable; mean over patients
    return bce + 0.5 * lam * torch.sum(w ** 2)


def fit_logreg_gd(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    lr: float = 0.5,
    n_steps: int = 3000,
    tol: float = 1e-12,
) -> LogRegResult:
    """Fit by plain (full-batch) gradient descent using autograd.

    Parameters
    ----------
    X : (n, p) features (ideally standardised; gradient descent converges
        much faster when features share a scale).
    y : (n,) binary labels in {0, 1}.
    C : inverse penalty strength, same meaning as in scikit-learn.
    lr : learning rate (step size).
    n_steps : maximum number of gradient steps.
    tol : stop early when the gradient norm falls below this value.
    """
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    n, p = Xt.shape
    lam = 1.0 / (C * n)

    # Parameters to learn, initialised at zero (the loss is convex, so the
    # starting point does not change the solution, only the time to reach it).
    w = torch.zeros(p, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    history: list[float] = []
    for _ in range(n_steps):
        loss = l2_logistic_loss(Xt, yt, w, b, lam)   # forward
        w.grad, b.grad = None, None                   # gradients accumulate: reset
        loss.backward()                               # backward (autograd)
        with torch.no_grad():                         # update outside the graph
            w -= lr * w.grad
            b -= lr * b.grad
        history.append(loss.item())
        grad_norm = torch.sqrt(w.grad.pow(2).sum() + b.grad.pow(2).sum()).item()
        if grad_norm < tol:
            break

    return LogRegResult(coef=w.detach().numpy().copy(),
                        intercept=float(b.item()),
                        loss_history=history)
