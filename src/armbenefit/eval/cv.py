"""
Leakage-safe cross-validation utilities.

Three rules, each backed by evidence of how ML results get inflated:
  1. Split at the PATIENT level, stratified by arm × outcome, so each test fold
     mirrors the trial and no patient is in both training and test data.
  2. Fit EVERY data-dependent preprocessing step (gene filtering, scaling) on the
     training fold only, then apply it to the test fold (Ambroise & McLachlan 2002;
     Kapoor & Narayanan 2023).
  3. Tune hyperparameters in an INNER loop that never sees the outer test fold
     (nested CV; Varma & Simon 2006, Cawley & Talbot 2010).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def outer_folds(clin: pd.DataFrame, n_splits: int = 5, seed: int = 0,
                strata_cols: tuple[str, ...] = ("arm_code", "subtype", "pcr")
                ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) positional indices, stratified by the joint strata.

    Rare strata (fewer patients than folds) are merged into one "other" level so
    that StratifiedKFold can still balance the common ones.
    """
    strata = clin[list(strata_cols)].astype(str).agg("|".join, axis=1)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= n_splits, "rare")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(clin)), strata.values)


class FoldPreprocessor:
    """Unsupervised gene filter + z-scoring, learned on the training fold only.

    * Keeps the `top_k` most variable genes. Variance is computed on TRAINING
      patients only and never uses labels, so test data can't influence the choice.
    * Standardises with training means and SDs. Test patients are transformed
      with those same numbers, exactly as a new patient would be in practice.
    """

    def __init__(self, top_k: int = 2000):
        self.top_k = top_k
        self.genes_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> FoldPreprocessor:
        var = X.var(axis=0)
        self.genes_ = var.sort_values(ascending=False).index[: self.top_k].to_numpy()
        sub = X[self.genes_]
        self.mean_ = sub.mean(axis=0).to_numpy()
        self.std_ = sub.std(axis=0).replace(0, 1).to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.genes_ is None:
            raise RuntimeError("call fit() first")
        return ((X[self.genes_].to_numpy() - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def clinical_matrix(clin: pd.DataFrame, arms: list[str], include_arm: bool = True,
                    include_platform: bool = True) -> np.ndarray:
    """Encode covariates known before treatment: HR, MammaPrint (MP), platform, and (optionally) arm one-hot."""
    cols = [clin["hr"].astype(float).to_numpy(), clin["mp"].astype(float).to_numpy()]
    if include_platform:
        cols.append((clin["platform"] == "GPL30493").astype(float).to_numpy())
    M = np.column_stack(cols)
    if include_arm:
        onehot = np.column_stack([(clin["arm_code"] == a).astype(float) for a in arms])
        M = np.column_stack([M, onehot])
    return M.astype(np.float32)
