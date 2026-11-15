"""
Step 5–7 evaluation: prediction, calibration, benefit and policy value.

Input: data_processed/ispy2_oof_predictions.csv.gz (from 04_benchmark_ispy2.py).
Each patient has one out-of-fold prediction per repeat. We average them over repeats
(repeated-CV averaging reduces split-to-split noise) and report the SD across repeats.

Outputs (aggregate only):
  results/table2_prediction.csv      factual AUC / Brier / calibration per model
  results/table3_benefit.csv         per arm × model: interaction β, c-for-benefit
  results/table4_policy.csv          policy value (IPW) per strategy
  docs/figures/fig2_prediction.png, fig3_benefit.png, fig4_policy.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from armbenefit.eval.metrics import (benefit_interaction, bootstrap_ci,  # noqa: E402
                                     c_for_benefit, policy_value, summarise_predictions)

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "docs" / "figures"
ARMS = ["CTRL", "AMG386", "GANE", "GANI", "MK2206", "NERA", "PEMBRO", "VELI_CARBO"]
MODELS = ["Clinical-LR", "Ridge-S", "Ridge-T", "LGBM-S", "MLP-S", "ArmNet", "ArmNet-Ens5"]
PCOLS = [f"p_{a}" for a in ARMS]
COLORS = dict(zip(MODELS, ["#7f7f7f", "#1f77b4", "#aec7e8", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]))


def averaged(oof: pd.DataFrame) -> pd.DataFrame:
    meta = ["arm_code", "arm_idx", "pcr", "subtype", "hr", "mp", "stratum", "platform"]
    avg = oof.groupby(["model", oof.index])[PCOLS].mean()
    avg = avg.reset_index(level=0).join(oof.groupby(level=0)[meta].first())
    avg["p_factual"] = avg[PCOLS].to_numpy()[np.arange(len(avg)), avg.arm_idx.to_numpy()]
    return avg


def prediction_table(oof, avg) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        d = avg[avg.model == m]
        s = summarise_predictions(d.pcr.to_numpy(), d.p_factual.to_numpy())
        per_rep = []
        for _, r in oof[oof.model == m].groupby("rep"):
            per_rep.append(roc_auc_score(r.pcr, r[PCOLS].to_numpy()[np.arange(len(r)), r.arm_idx]))
        rows.append(dict(model=m, **s, auc_sd_repeats=np.std(per_rep), n=len(d)))
    return pd.DataFrame(rows)


def benefit_table(avg) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        d = avg[avg.model == m]
        for arm in ARMS[1:]:
            s = d[d.arm_code.isin(["CTRL", arm])]
            y, t = s.pcr.to_numpy(), (s.arm_code == arm).to_numpy().astype(int)
            tau = (s[f"p_{arm}"] - s["p_CTRL"]).to_numpy()
            strata = s.stratum.to_numpy()
            inter = benefit_interaction(y, t, tau, strata)
            cfb = c_for_benefit(y, t, tau, strata)
            lo, hi = bootstrap_ci(lambda yy, tt, ta, ss: c_for_benefit(yy, tt, ta, ss),
                                  y, t, tau, strata, n_boot=200)
            rows.append(dict(model=m, arm=arm, n_treated=int(t.sum()), n_ctrl=int((1 - t).sum()),
                             mean_pred_benefit=tau[t == 1].mean(), **inter,
                             c_benefit=cfb, c_benefit_lo=lo, c_benefit_hi=hi))
    return pd.DataFrame(rows)


def added_value_table(avg) -> pd.DataFrame:
    """Does a genomic model's predicted benefit add information BEYOND the clinical model's?

    Among patients randomised to arm a or control:
      logit P(pCR) = β0 + β1·t + β2·τ̂_clin + β3·t·τ̂_clin + β4·τ̂_gen + β5·t·τ̂_gen + strata
    β5 (per SD of τ̂_gen) tests whether genomic benefit modifies the treatment effect after
    allowing the clinical (subtype-level) benefit to do so. Both τ̂ are out-of-fold.
    """
    import statsmodels.api as sm
    clin = avg[avg.model == "Clinical-LR"]
    rows = []
    for m in MODELS[1:]:
        d = avg[avg.model == m]
        for arm in ARMS[1:]:
            s = d[d.arm_code.isin(["CTRL", arm])]
            c = clin.loc[s.index]
            t = (s.arm_code == arm).to_numpy().astype(float)
            tg = (s[f"p_{arm}"] - s["p_CTRL"]).to_numpy()
            tc = (c[f"p_{arm}"] - c["p_CTRL"]).to_numpy()
            tg, tc = (tg - tg.mean()) / tg.std(), (tc - tc.mean()) / (tc.std() + 1e-12)
            S = pd.get_dummies(s.stratum, drop_first=True, dtype=float).to_numpy()
            X = sm.add_constant(np.column_stack([t, tc, t * tc, tg, t * tg, S]))
            fit = sm.GLM(s.pcr.to_numpy(), X, family=sm.families.Binomial()).fit()
            b, se = fit.params[5], fit.bse[5]
            rows.append(dict(model=m, arm=arm, beta_added=b, ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se,
                             p=fit.pvalues[5]))
    out = pd.DataFrame(rows)
    # Benjamini–Hochberg FDR across all model × arm tests (multiple comparisons).
    from statsmodels.stats.multitest import multipletests
    out["q_bh"] = multipletests(out.p, method="fdr_bh")[1]
    return out


def policy_table(avg) -> pd.DataFrame:
    """Policy value by Hájek IPW, with propensity = P(arm | HR × MP stratum) estimated empirically."""
    base = avg[avg.model == MODELS[0]]
    prop_tab = pd.crosstab(base.stratum, base.arm_code, normalize="index")
    rng = np.random.default_rng(0)
    rows = []

    def evaluate(name, d, rec):
        y, arm = d.pcr.to_numpy(), d.arm_code.to_numpy()
        prop = np.array([prop_tab.loc[s, a] for s, a in zip(d.stratum, arm)])
        v = policy_value(y, arm, rec, prop)
        boots = []
        n = len(d)
        for _ in range(1000):
            i = rng.integers(0, n, n)
            boots.append(policy_value(y[i], arm[i], rec[i], prop[i]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        share = pd.Series(rec).value_counts(normalize=True).round(2).to_dict()
        rows.append(dict(strategy=name, value=v, lo=lo, hi=hi, n_concordant=int((arm == rec).sum()),
                         recommendation_mix=str(share)))

    evaluate("All control (paclitaxel)", base, np.full(len(base), "CTRL"))
    for arm in ["PEMBRO", "VELI_CARBO"]:
        evaluate(f"All {arm}", base, np.full(len(base), arm))
    for m in MODELS:
        d = avg[avg.model == m]
        rec = np.array(ARMS)[d[PCOLS].to_numpy().argmax(1)]
        evaluate(f"Model: {m}", d, rec)
    return pd.DataFrame(rows)


def figures(pred, ben, pol, avg) -> None:
    # Fig 2: AUC forest + calibration curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, r in pred.iterrows():
        ax[0].errorbar(r.auc, i, xerr=[[r.auc - r.auc_lo], [r.auc_hi - r.auc]], fmt="o",
                       color=COLORS[r.model], capsize=3)
    ax[0].set_yticks(range(len(pred)), pred.model)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("AUC for pCR under the received arm (95% CI)")
    ax[0].set_title("a  Discrimination (HER2−, n=741)")
    for m in ["Clinical-LR", "Ridge-S", "LGBM-S", "ArmNet-Ens5"]:
        d = avg[avg.model == m]
        bins = pd.qcut(d.p_factual, 8, duplicates="drop")
        g = d.groupby(bins, observed=True).agg(p=("p_factual", "mean"), o=("pcr", "mean"))
        ax[1].plot(g.p, g.o, "o-", color=COLORS[m], label=m, ms=4)
    ax[1].plot([0, 1], [0, 1], ":", c="grey")
    ax[1].set_xlabel("Predicted pCR probability")
    ax[1].set_ylabel("Observed pCR rate")
    ax[1].set_title("b  Calibration (octiles)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_prediction.png", dpi=200)

    # Fig 3: interaction β forest per arm, Clinical-LR vs genomic models
    sel = ["Clinical-LR", "Ridge-T", "LGBM-S", "ArmNet-Ens5"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, m in enumerate(sel):
        b = ben[ben.model == m].reset_index(drop=True)
        yy = np.arange(len(b)) + (j - 1.5) * 0.18
        ax[0].errorbar(b.beta_int, yy, xerr=[b.beta_int - b.ci_lo, b.ci_hi - b.beta_int], fmt="o",
                       ms=4, capsize=2, color=COLORS[m], label=m)
        ax[1].errorbar(b.c_benefit, yy, xerr=[b.c_benefit - b.c_benefit_lo, b.c_benefit_hi - b.c_benefit],
                       fmt="o", ms=4, capsize=2, color=COLORS[m])
    arms = ben[ben.model == sel[0]].arm.tolist()
    for a_, lab, ref in [(ax[0], "Interaction β per SD of predicted benefit (log-odds)", 0),
                         (ax[1], "c-for-benefit (95% bootstrap CI)", 0.5)]:
        a_.set_yticks(range(len(arms)), arms)
        a_.axvline(ref, c="grey", ls=":")
        a_.set_xlabel(lab)
        a_.invert_yaxis()
    ax[0].set_title("a  Does predicted benefit modify the observed effect?")
    ax[1].set_title("b  Discrimination of benefit")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_benefit.png", dpi=200)

    # Fig 4: policy value
    fig, ax = plt.subplots(figsize=(7, 4.2))
    p = pol.reset_index(drop=True)
    ax.errorbar(p.value, range(len(p)), xerr=[p.value - p.lo, p.hi - p.value], fmt="o", capsize=3, c="k")
    ax.set_yticks(range(len(p)), p.strategy)
    ax.invert_yaxis()
    ax.set_xlabel("Estimated pCR rate if the strategy were followed (IPW, 95% CI)")
    ax.set_title("Policy value, HER2-negative I-SPY2")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_policy.png", dpi=200)


def main(path: Path) -> None:
    oof = pd.read_csv(path, index_col=0)
    avg = averaged(oof)
    pred, ben, pol = prediction_table(oof, avg), benefit_table(avg), policy_table(avg)
    pred.to_csv(RES / "table2_prediction.csv", index=False, float_format="%.3f")
    ben.to_csv(RES / "table3_benefit.csv", index=False, float_format="%.3f")
    pol.to_csv(RES / "table4_policy.csv", index=False, float_format="%.3f")
    add = added_value_table(avg)
    add.to_csv(RES / "table3b_benefit_beyond_clinical.csv", index=False, float_format="%.4f")
    from statsmodels.stats.multitest import multipletests
    ben["q_bh"] = multipletests(ben.p, method="fdr_bh")[1]
    ben.to_csv(RES / "table3_benefit.csv", index=False, float_format="%.4f")
    print(add[add.arm.isin(["PEMBRO", "VELI_CARBO", "NERA"])].round(3).to_string(index=False))
    figures(pred, ben, pol, avg)
    pd.set_option("display.width", 200)
    print(pred[["model", "auc", "auc_lo", "auc_hi", "auc_sd_repeats", "brier", "cal_slope"]].round(3))
    print(ben.pivot(index="arm", columns="model", values="beta_int").round(2)[MODELS])
    print(ben.pivot(index="arm", columns="model", values="c_benefit").round(3)[MODELS])
    print(pol[["strategy", "value", "lo", "hi", "n_concordant"]].round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(ROOT / "data_processed" / "ispy2_oof_predictions.csv.gz"))
    main(Path(ap.parse_args().oof))
