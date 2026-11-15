"""
Evaluation metrics with uncertainty.

  * AUC (discrimination) and Brier score (overall accuracy of probabilities).
  * Calibration intercept and slope (Van Calster et al. 2019): regress the
    outcome on logit(p̂). A slope < 1 means predictions are too extreme (overfitting).
  * Patient-level bootstrap CIs (Efron & Tibshirani 1993): resample patients, recompute.
  * Benefit metrics for treatment-effect heterogeneity:
      - interaction test: is predicted benefit associated with observed benefit?
      - c-for-benefit (van Klaveren et al. 2018).
      - policy value by inverse-probability weighting (Hájek estimator).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration: logit P(y=1) = a + b·logit(p̂). Ideal: a = 0, b = 1."""
    lp = _logit(p)
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(intercept), float(slope)


def bootstrap_ci(fn, *arrays, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI for statistic fn(*arrays) by resampling rows (patients) with replacement."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats.append(fn(*(a[idx] for a in arrays)))
        except ValueError:          # e.g. a resample with one class only
            continue
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarise_predictions(y: np.ndarray, p: np.ndarray, n_boot: int = 1000) -> dict:
    """AUC and Brier with 95% bootstrap CIs, plus calibration intercept and slope."""
    auc = roc_auc_score(y, p)
    a_lo, a_hi = bootstrap_ci(roc_auc_score, y, p, n_boot=n_boot)
    brier = brier_score_loss(y, p)
    b_lo, b_hi = bootstrap_ci(brier_score_loss, y, p, n_boot=n_boot)
    ci, cs = calibration_intercept_slope(y, p)
    return dict(auc=auc, auc_lo=a_lo, auc_hi=a_hi, brier=brier, brier_lo=b_lo, brier_hi=b_hi,
                cal_intercept=ci, cal_slope=cs)


def benefit_interaction(y: np.ndarray, treated: np.ndarray, tau_hat: np.ndarray,
                        strata: np.ndarray) -> dict:
    """Test whether predicted benefit modifies the observed treatment effect.

    Logistic model on patients randomised to arm a or control:
        logit P(pCR) = β0 + β1·treated + β2·τ̂ + β3·treated×τ̂ + γ·stratum
    β3 > 0 means patients predicted to benefit more do benefit more (log-odds
    scale). τ̂ is standardised so β3 is per 1 SD of predicted benefit.
    """
    t = (tau_hat - tau_hat.mean()) / (tau_hat.std() + 1e-12)
    S = pd.get_dummies(pd.Series(strata), drop_first=True, dtype=float).to_numpy()
    X = sm.add_constant(np.column_stack([treated, t, treated * t, S]))
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    b, se = fit.params[3], fit.bse[3]
    return dict(beta_int=float(b), se=float(se), ci_lo=float(b - 1.96 * se),
                ci_hi=float(b + 1.96 * se), p=float(fit.pvalues[3]))


def c_for_benefit(y: np.ndarray, treated: np.ndarray, tau_hat: np.ndarray,
                  strata: np.ndarray, seed: int = 0) -> float:
    """c-for-benefit (van Klaveren et al., J Clin Epidemiol 2018), matched within strata.

    Within each stratum, treated and control patients are sorted by predicted
    benefit and matched 1:1 by rank. A pair's observed benefit is y_t − y_c in
    {−1, 0, 1}, and its predicted benefit is the mean τ̂ of the pair. The
    statistic is the concordance between predicted and observed pair benefit
    (0.5 = no discrimination of benefit).
    """
    rng = np.random.default_rng(seed)
    pred, obs = [], []
    for s in np.unique(strata):
        m = strata == s
        ti, ci = np.where(m & (treated == 1))[0], np.where(m & (treated == 0))[0]
        k = min(len(ti), len(ci))
        if k == 0:
            continue
        ti = rng.choice(ti, k, replace=False) if len(ti) > k else ti
        ci = rng.choice(ci, k, replace=False) if len(ci) > k else ci
        ti, ci = ti[np.argsort(tau_hat[ti])], ci[np.argsort(tau_hat[ci])]
        pred.extend((tau_hat[ti] + tau_hat[ci]) / 2)
        obs.extend(y[ti] - y[ci])
    pred, obs = np.asarray(pred), np.asarray(obs)
    conc = disc = 0.0
    for i in range(len(obs)):
        d_obs = obs[i] - obs[i + 1:]
        d_pred = pred[i] - pred[i + 1:]
        use = d_obs != 0
        conc += np.sum((np.sign(d_obs[use]) == np.sign(d_pred[use])) & (d_pred[use] != 0))
        conc += 0.5 * np.sum(d_pred[use] == 0)
        disc += use.sum()
    return float(conc / disc) if disc else np.nan


def policy_value(y: np.ndarray, arm: np.ndarray, rec: np.ndarray, propensity: np.ndarray) -> float:
    """Hájek IPW estimate of the pCR rate if every patient received the recommended arm.

    Only patients whose randomised arm equals the recommendation contribute, each
    weighted by 1 / P(arm | stratum). Valid when allocation depends only on the
    strata used for the propensity (randomisation within subtype).
    """
    w = (arm == rec) / propensity
    return float(np.sum(w * y) / np.sum(w)) if np.sum(w) > 0 else np.nan
