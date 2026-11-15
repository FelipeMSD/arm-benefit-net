"""
Step 4–7: nested cross-validated benchmark on HER2-negative I-SPY2.

Cohort: HER2-negative patients (n = 741), 8 arms sharing ONE control arm.
For each outer fold (5-fold, stratified by arm × subtype × pCR, repeated over seeds):
  1. FoldPreprocessor: top-2000 most variable genes + z-scoring, fitted on training only.
  2. Torch models: (dropout, weight decay) chosen by inner 3-fold CV on the training fold.
  3. Every model predicts P(pCR) under ALL 8 arms for the held-out patients.

Out-of-fold predictions (patient-level) go to data_processed/ (git-ignored).
Aggregate results go to results/.

Usage:  python scripts/04_benchmark_ispy2.py [--repeats 3] [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from armbenefit.data.ispy2 import load_ispy2
from armbenefit.eval.cv import FoldPreprocessor, clinical_matrix, outer_folds
from armbenefit.models.baselines import ClinicalLR, RidgeS, RidgeT, LGBMS
from armbenefit.models.nets import TorchArmModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_processed"
ARMS = ["CTRL", "AMG386", "GANE", "GANI", "MK2206", "NERA", "PEMBRO", "VELI_CARBO"]
GRID = list(itertools.product([0.3, 0.6], [1e-3, 1e-2]))     # (dropout, weight_decay)


def load_cohort():
    clin, expr = load_ispy2()
    keep = clin.her2.eq(0) & clin.arm_code.isin(ARMS)
    clin, expr = clin[keep], expr[keep]
    expr = expr.loc[:, expr.notna().all()]
    clin = clin.assign(arm_idx=clin.arm_code.map({a: i for i, a in enumerate(ARMS)}),
                       stratum=clin.hr.astype(str) + "|" + clin.mp.astype(str))
    return clin, expr


def tune_torch(kind, G, C, arm, y, seed):
    """Inner 3-fold CV over GRID. Returns the (dropout, wd) with the lowest factual log-loss."""
    from sklearn.model_selection import StratifiedKFold
    strata = pd.Series(arm).astype(str) + "|" + pd.Series(y).astype(int).astype(str)
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    scores = {}
    for do, wd in GRID:
        losses = []
        for tr, va in skf.split(G, strata):
            m = TorchArmModel(len(ARMS), kind=kind, dropout=do, weight_decay=wd, seed=seed)
            P = m.fit(G[tr], C[tr], arm[tr], y[tr]).predict_arms(G[va], C[va])
            p = np.clip(P[np.arange(len(va)), arm[va]], 1e-6, 1 - 1e-6)
            losses.append(-np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)))
        scores[(do, wd)] = np.mean(losses)
    return min(scores, key=scores.get)


def main(repeats: int, quick: bool) -> None:
    clin, expr = load_cohort()
    print(f"HER2-negative cohort: {len(clin)} patients, {expr.shape[1]} genes, arms={ARMS}")
    C_all = clinical_matrix(clin, ARMS, include_arm=False)
    arm_all, y_all = clin.arm_idx.to_numpy(), clin.pcr.astype(int).to_numpy()
    records, chosen = [], []
    for rep in range(repeats):
        for fold, (tr, te) in enumerate(outer_folds(clin, 5, seed=rep)):
            t0 = time.time()
            pp = FoldPreprocessor(top_k=500 if quick else 2000).fit(expr.iloc[tr])
            Gtr, Gte = pp.transform(expr.iloc[tr]), pp.transform(expr.iloc[te])
            Ctr, Cte = C_all[tr], C_all[te]
            a, y = arm_all[tr], y_all[tr]
            seed = 100 * rep + fold
            models = {
                "Clinical-LR": ClinicalLR(len(ARMS)),
                "Ridge-S": RidgeS(len(ARMS), seed),
                "Ridge-T": RidgeT(len(ARMS), seed),
                "LGBM-S": LGBMS(len(ARMS), seed),
            }
            for kind, name in [("pooled", "MLP-S"), ("armnet", "ArmNet")]:
                do, wd = (0.6, 1e-2) if quick else tune_torch(kind, Gtr, Ctr, a, y, seed)
                chosen.append(dict(rep=rep, fold=fold, model=name, dropout=do, weight_decay=wd))
                models[name] = TorchArmModel(len(ARMS), kind, dropout=do, weight_decay=wd, seed=seed)
                if kind == "armnet":
                    models["ArmNet-Ens5"] = TorchArmModel(len(ARMS), kind, dropout=do,
                                                          weight_decay=wd, n_seeds=5, seed=seed)
            for name, m in models.items():
                P = m.fit(Gtr, Ctr, a, y).predict_arms(Gte, Cte)
                df = pd.DataFrame(P, columns=[f"p_{x}" for x in ARMS], index=clin.index[te])
                df["model"], df["rep"], df["fold"] = name, rep, fold
                records.append(df)
            print(f"rep {rep} fold {fold}: {time.time() - t0:.0f}s", flush=True)
    oof = pd.concat(records)
    oof = oof.join(clin[["arm_code", "arm_idx", "pcr", "subtype", "hr", "mp", "stratum", "platform"]])
    OUT.mkdir(exist_ok=True)
    oof.to_csv(OUT / "ispy2_oof_predictions.csv.gz")
    pd.DataFrame(chosen).to_csv(ROOT / "results" / "tuning_choices.csv", index=False)
    print("saved", OUT / "ispy2_oof_predictions.csv.gz", oof.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(args.repeats, args.quick)
