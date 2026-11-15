"""Model tests on synthetic trials with a KNOWN arm-specific effect (fast, CI-safe)."""

import numpy as np
import pytest

from armbenefit.models.baselines import ClinicalLR, RidgeT, LGBMS
from armbenefit.models.nets import ArmNet, TorchArmModel


def synthetic_trial(n=900, p=30, seed=0):
    """3 arms. Arm 1 helps only when gene 0 is high, and arm 2 only when gene 1 is high."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, p)).astype(np.float32)
    C = rng.integers(0, 2, size=(n, 3)).astype(np.float32)
    arm = rng.integers(0, 3, n)
    logits = np.column_stack([-1 + 0 * G[:, 0], -1 + 2.0 * G[:, 0], -1 + 2.0 * G[:, 1]])
    true_p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, true_p[np.arange(n), arm]).astype(np.float32)
    return G, C, arm, y, true_p


def test_armnet_output_shape():
    import torch
    net = ArmNet(n_in=10, n_arms=4)
    assert net(torch.zeros(5, 10)).shape == (5, 4)


@pytest.mark.parametrize("kind", ["armnet", "pooled"])
def test_torch_models_recover_benefit(kind):
    G, C, arm, y, true_p = synthetic_trial()
    m = TorchArmModel(n_arms=3, kind=kind, hidden=32, dropout=0.1, weight_decay=1e-3).fit(G, C, arm, y)
    P = m.predict_arms(G, C)
    assert P.shape == (len(G), 3) and np.all((P >= 0) & (P <= 1))
    true_tau = true_p[:, 1] - true_p[:, 0]
    est_tau = P[:, 1] - P[:, 0]
    assert np.corrcoef(true_tau, est_tau)[0, 1] > 0.5     # ranks benefit correctly


def test_torch_model_deterministic():
    G, C, arm, y, _ = synthetic_trial(n=300)
    a = TorchArmModel(3, seed=7).fit(G, C, arm, y).predict_arms(G, C)
    b = TorchArmModel(3, seed=7).fit(G, C, arm, y).predict_arms(G, C)
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("cls", [ClinicalLR, RidgeT, LGBMS])
def test_baselines_interface(cls):
    G, C, arm, y, _ = synthetic_trial(n=400)
    P = cls(3).fit(G, C, arm, y).predict_arms(G, C)
    assert P.shape == (400, 3)


def test_ridge_s_keeps_arm_effects():
    """Regression test: the penalty must not shrink the arm effects to zero (all arms identical)."""
    from armbenefit.models.baselines import RidgeS
    rng = np.random.default_rng(0)
    G = rng.normal(size=(600, 300)).astype(np.float32)          # many noise genes
    C = rng.integers(0, 2, size=(600, 3)).astype(np.float32)
    arm = rng.integers(0, 3, 600)
    y = rng.binomial(1, np.array([0.15, 0.5, 0.3])[arm]).astype(float)
    P = RidgeS(3).fit(G, C, arm, y).predict_arms(G, C)
    assert P[:, 1].mean() - P[:, 0].mean() > 0.15
