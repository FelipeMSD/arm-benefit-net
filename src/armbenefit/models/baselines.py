"""
Classical baselines with the same interface as the PyTorch models:
fit(G, C, arm, y) and predict_arms(G, C) -> (n, K) probabilities.

    ClinicalLR  subtype-level model: arm × (HR, MammaPrint) interactions, no genes.
                It's the "best arm per clinical subtype" strategy a trialist would use.
    RidgeS      ridge (L2) logistic on genes + clinical + arm (S-learner, main effects).
    RidgeT      one ridge logistic per arm (T-learner; Künzel et al. 2019).
    LGBMS       gradient-boosted trees with arm as a feature (S-learner). Trees can
                learn gene × arm interactions on their own (Grinsztajn et al. 2022).

Penalty strengths are tuned by internal CV on the training fold only
(LogisticRegressionCV), so they're nested inside our outer folds.

Why ridge rather than elastic net? We started with elastic net (saga solver). With the
design columns rescaled (see RidgeS) saga converged so slowly that one outer fold took
over 40 minutes on a laptop. Ridge with L-BFGS solves the same problem class in
seconds, and for expression-based prediction it performs on a par with sparse
penalties (Bøvelstad et al., Bioinformatics 2007).
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _onehot(arm, k):
    return np.eye(k, dtype=np.float32)[np.asarray(arm)]


class ClinicalLR:
    def __init__(self, n_arms: int):
        self.k = n_arms

    def _x(self, C, arm):
        A = _onehot(arm, self.k)
        hr, mp = C[:, [0]], C[:, [1]]
        return np.column_stack([C, A[:, 1:], A[:, 1:] * hr, A[:, 1:] * mp])

    def fit(self, G, C, arm, y):
        self.m = LogisticRegression(C=1.0, max_iter=5000).fit(self._x(C, arm), y)
        return self

    def predict_arms(self, G, C):
        n = len(C)
        return np.column_stack([self.m.predict_proba(self._x(C, np.full(n, k)))[:, 1]
                                for k in range(self.k)])


class RidgeS:
    """Ridge S-learner.

    Pitfall fixed here: the penalty shrinks ALL coefficients equally, so with
    2,000 genes the 7 arm indicators were shrunk to ~0, and the model then
    predicted the same pCR for every arm. Multiplying the arm and clinical
    columns by ARM_SCALE makes their effective penalty ARM_SCALE² times smaller
    (a coefficient β on a column scaled by s needs only β/s), so these key
    covariates stay (almost) unpenalised. That's the usual practice of not
    penalising design variables.
    """

    ARM_SCALE = 30.0

    def __init__(self, n_arms: int, seed: int = 0):
        self.k, self.seed = n_arms, seed

    def _x(self, G, C, arm):
        return np.column_stack([G, self.ARM_SCALE * C, self.ARM_SCALE * _onehot(arm, self.k)[:, 1:]])

    def fit(self, G, C, arm, y):
        X = self._x(G, C, arm)
        self.m = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), penalty="l2", solver="lbfgs", cv=3,
                                      scoring="neg_log_loss", max_iter=5000).fit(X, y)
        return self

    def predict_arms(self, G, C):
        n = len(G)
        return np.column_stack([self.m.predict_proba(self._x(G, C, np.full(n, k)))[:, 1]
                                for k in range(self.k)])


class RidgeT:
    def __init__(self, n_arms: int, seed: int = 0):
        self.k, self.seed = n_arms, seed

    def fit(self, G, C, arm, y):
        arm, y = np.asarray(arm), np.asarray(y)
        self.ms = []
        for k in range(self.k):
            m = arm == k
            self.ms.append(LogisticRegressionCV(
                Cs=np.logspace(-4, 0, 9), penalty="l2", solver="lbfgs", cv=3,
                scoring="neg_log_loss", max_iter=5000).fit(np.column_stack([G[m], C[m]]), y[m]))
        return self

    def predict_arms(self, G, C):
        X = np.column_stack([G, C])
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.ms])


class LGBMS:
    def __init__(self, n_arms: int, seed: int = 0):
        self.k, self.seed = n_arms, seed
        self.params = dict(n_estimators=300, learning_rate=0.03, num_leaves=7,
                           min_child_samples=15, colsample_bytree=0.2, subsample=0.8,
                           subsample_freq=1, reg_lambda=1.0, random_state=seed, verbose=-1)

    def fit(self, G, C, arm, y):
        X = np.column_stack([G, C, _onehot(arm, self.k)])
        self.m = lgb.LGBMClassifier(**self.params).fit(X, y)
        return self

    def predict_arms(self, G, C):
        n = len(G)
        return np.column_stack([
            self.m.predict_proba(np.column_stack([G, C, _onehot(np.full(n, k), self.k)]))[:, 1]
            for k in range(self.k)])
