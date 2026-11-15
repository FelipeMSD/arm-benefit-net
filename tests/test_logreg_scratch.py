"""
Tests for armbenefit.models.logreg_scratch (lesson 01).

What each test protects against:
  test_matches_sklearn          -> wrong loss, wrong penalty scaling (λ = 1/(C·n)), or a penalised intercept
  test_autograd_equals_analytic -> misunderstanding of what autograd computes
  test_loss_decreases           -> broken update step (e.g. a sign error, or a missing grad reset)
  test_shape_bug_is_caught      -> the silent (n,) vs (n, 1) broadcasting bug
  test_deterministic            -> hidden randomness that would make results irreproducible

Run with:  pytest -q
"""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from armbenefit.models.logreg_scratch import fit_logreg_gd, l2_logistic_loss


@pytest.fixture
def synthetic():
    """Small sparse-signal dataset: 200 'patients', 10 'genes', 3 informative."""
    rng = np.random.default_rng(0)
    n, p = 200, 10
    X = rng.normal(size=(n, p))
    w_true = np.array([1.5, -2.0, 1.0] + [0.0] * (p - 3))
    y = rng.binomial(1, 1 / (1 + np.exp(-(X @ w_true - 0.5)))).astype(float)
    return X, y


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_matches_sklearn(synthetic, C):
    X, y = synthetic
    ours = fit_logreg_gd(X, y, C=C, lr=0.5, n_steps=20_000)
    ref = LogisticRegression(penalty="l2", C=C, tol=1e-12, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(ours.coef, ref.coef_.ravel(), atol=1e-5)
    assert ours.intercept == pytest.approx(ref.intercept_[0], abs=1e-5)


def test_autograd_equals_analytic(synthetic):
    X, y = synthetic
    Xt, yt = torch.tensor(X), torch.tensor(y)
    lam = 0.01
    torch.manual_seed(1)
    w = torch.randn(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.randn(1, dtype=torch.float64, requires_grad=True)
    l2_logistic_loss(Xt, yt, w, b, lam).backward()
    with torch.no_grad():
        resid = torch.sigmoid(Xt @ w + b) - yt
        grad_w = Xt.T @ resid / len(yt) + lam * w
        grad_b = resid.mean()
    torch.testing.assert_close(w.grad, grad_w)
    torch.testing.assert_close(b.grad.squeeze(), grad_b)


def test_loss_decreases(synthetic):
    X, y = synthetic
    hist = fit_logreg_gd(X, y, n_steps=200, tol=0.0).loss_history
    assert hist[0] == pytest.approx(np.log(2))   # w = b = 0 -> p = 0.5 -> loss = ln 2
    assert all(a >= b - 1e-15 for a, b in zip(hist, hist[1:]))  # monotone (convex, small lr)


def test_shape_bug_is_caught(synthetic):
    X, y = synthetic
    Xt = torch.tensor(X)
    y_col = torch.tensor(y).unsqueeze(1)          # (n, 1): the classic mistake
    w = torch.zeros(X.shape[1], dtype=torch.float64)
    b = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError):
        l2_logistic_loss(Xt, y_col, w, b, lam=0.01)


def test_deterministic(synthetic):
    X, y = synthetic
    a = fit_logreg_gd(X, y, n_steps=500)
    b = fit_logreg_gd(X, y, n_steps=500)
    np.testing.assert_array_equal(a.coef, b.coef)
