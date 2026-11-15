"""The PyTorch Cox loss must reproduce the partial log-likelihood computed by lifelines."""

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter

from armbenefit.models.survival import DeepCox, cox_ph_loss


def _surv_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = rng.exponential(1 / np.exp(X @ np.array([0.8, -0.5, 0.0])))
    c = rng.exponential(2.0, n)
    return X, np.minimum(t, c), (t <= c).astype(float)       # continuous times: no ties


def test_cox_loss_matches_lifelines():
    X, time, event = _surv_data()
    df = pd.DataFrame(X, columns=["a", "b", "c"]).assign(T=time, E=event)
    cph = CoxPHFitter().fit(df, "T", "E")
    beta = torch.tensor(cph.params_.to_numpy(), dtype=torch.float64)
    risk = torch.tensor(X) @ beta
    ours = -cox_ph_loss(risk, torch.tensor(time), torch.tensor(event)).item() * event.sum()
    assert np.isclose(ours, cph.log_likelihood_, rtol=1e-6)


def test_gradient_points_to_truth():
    """At beta = 0, the negative gradient should point towards the true coefficients' signs."""
    X, time, event = _surv_data(n=1000)
    beta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    cox_ph_loss(torch.tensor(X) @ beta, torch.tensor(time), torch.tensor(event)).backward()
    assert -beta.grad[0] > 0 and -beta.grad[1] < 0


def test_deepcox_shape():
    assert DeepCox(5)(torch.zeros(7, 5)).shape == (7,)
