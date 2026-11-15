"""Tests for leakage-safe CV and the benefit metrics (synthetic data only, so they run in CI)."""

import numpy as np
import pandas as pd
import pytest

from armbenefit.eval.cv import FoldPreprocessor, outer_folds
from armbenefit.eval.metrics import (benefit_interaction, c_for_benefit, policy_value,
                                     summarise_predictions)


def _fake_clin(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "arm_code": rng.choice(["CTRL", "A", "B"], n),
        "subtype": rng.choice(["TNBC", "HR+HER2-"], n),
        "pcr": rng.integers(0, 2, n),
    }, index=[f"p{i}" for i in range(n)])


def test_folds_partition_patients():
    clin = _fake_clin()
    seen = []
    for tr, te in outer_folds(clin, n_splits=5):
        assert len(set(tr) & set(te)) == 0          # no patient in both sets
        seen.extend(te)
    assert sorted(seen) == list(range(len(clin)))   # every patient tested exactly once


def test_preprocessor_never_sees_test_data():
    """If test rows changed what was learned in fit(), the filter would be leaking."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(100, 50)), columns=[f"g{i}" for i in range(50)])
    train, test = X.iloc[:80], X.iloc[80:].copy()
    pp1 = FoldPreprocessor(top_k=10).fit(train)
    test.iloc[:, :] = test.to_numpy() * 1000          # wildly change the test data
    pp2 = FoldPreprocessor(top_k=10).fit(train)
    assert list(pp1.genes_) == list(pp2.genes_)
    np.testing.assert_allclose(pp1.mean_, pp2.mean_)
    Z = pp1.transform(train)
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-5)  # z-scored on the training fold


def test_summary_perfect_and_random():
    y = np.array([0, 0, 1, 1] * 50)
    assert summarise_predictions(y, y * 0.98 + 0.01, n_boot=50)["auc"] == 1.0
    rng = np.random.default_rng(0)
    assert abs(summarise_predictions(y, rng.uniform(size=len(y)), n_boot=50)["auc"] - 0.5) < 0.15


def _benefit_data(n=2000, true_interaction=True, seed=0):
    """Treatment helps only patients with high x (when true_interaction=True)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    logit = -1 + (1.5 * x * t if true_interaction else 0.5 * t)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return y, t, x, np.zeros(n, dtype=int)


def test_interaction_detects_true_heterogeneity():
    y, t, x, s = _benefit_data(true_interaction=True)
    res = benefit_interaction(y, t, x, s)
    assert res["beta_int"] > 0 and res["p"] < 1e-6


def test_c_for_benefit_informative_vs_noise():
    y, t, x, s = _benefit_data(true_interaction=True)
    assert c_for_benefit(y, t, x, s) > 0.55
    noise = np.random.default_rng(3).normal(size=len(x))
    assert c_for_benefit(y, t, noise, s) == pytest.approx(0.5, abs=0.04)


def test_policy_value_equals_arm_rate_when_recommending_one_arm():
    rng = np.random.default_rng(0)
    arm = rng.choice(["CTRL", "A"], 1000)
    y = rng.binomial(1, np.where(arm == "A", 0.6, 0.2))
    v = policy_value(y, arm, np.full(1000, "A"), np.full(1000, 0.5))
    assert v == pytest.approx(y[arm == "A"].mean())
