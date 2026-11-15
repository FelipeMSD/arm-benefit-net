"""
Step 2b: exploratory analysis of I-SPY2-990.

Outputs (aggregate only, no patient-level data):
  results/table1_cohort.csv        patients and pCR rate by arm × receptor subtype
  results/table_platform_by_arm.csv  Agilent platform × arm (to expose confounding)
  docs/figures/fig1_cohort.png     (a) pCR by arm in HER2− disease; (b) PCA coloured by platform

Why PCA by platform? Batch effects usually dominate the first principal
components of expression data (Leek et al. 2010). If PC1/PC2 still separate
platforms after the authors' harmonisation, residual batch signal exists and
any model could exploit it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from armbenefit.data.ispy2 import load_ispy2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "docs" / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (better than Wald for small n)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def main() -> None:
    clin, expr = load_ispy2()
    expr = expr.loc[:, expr.notna().all()]      # drop genes with missing values (label-free filter)
    print(f"cohort: {len(clin)} patients, {expr.shape[1]} complete genes")

    # --- Table 1: arm × subtype ------------------------------------------------
    rows = []
    for (arm, sub), g in clin.groupby(["arm_code", "subtype"]):
        k, n = int(g.pcr.sum()), len(g)
        lo, hi = wilson_ci(k, n)
        rows.append(dict(arm=arm, subtype=sub, n=n, pcr_n=k, pcr_rate=k / n, ci_lo=lo, ci_hi=hi))
    t1 = pd.DataFrame(rows)
    t1.to_csv(RES / "table1_cohort.csv", index=False, float_format="%.3f")
    pd.crosstab(clin.arm_code, clin.platform).to_csv(RES / "table_platform_by_arm.csv")

    # --- Figure 1 ----------------------------------------------------------------
    her2neg = clin[clin.her2.eq(0)]
    order = her2neg.groupby("arm_code").pcr.mean().sort_values().index
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, sub in enumerate(["HR+HER2-", "TNBC"]):
        sub_df = her2neg[her2neg.subtype == sub]
        rates, los, his, ns = [], [], [], []
        for arm in order:
            g = sub_df[sub_df.arm_code == arm]
            k, n = int(g.pcr.sum()), len(g)
            lo, hi = wilson_ci(k, n)
            rates.append(k / n if n else np.nan)
            los.append(lo)
            his.append(hi)
            ns.append(n)
        y = np.arange(len(order)) + (i - 0.5) * 0.35
        rates, los, his = map(np.array, (rates, los, his))
        ax[0].errorbar(rates, y, xerr=[rates - los, his - rates], fmt="o", capsize=2,
                       label=sub, color=["#1f77b4", "#d62728"][i])
    ax[0].set_yticks(np.arange(len(order)), order)
    ax[0].set_xlabel("pCR rate (95% Wilson CI)")
    ax[0].set_title("a  pCR by arm, HER2-negative disease")
    ax[0].axvline(her2neg[her2neg.arm_code == "CTRL"].pcr.mean(), ls=":", c="grey")
    ax[0].legend(frameon=False)

    z = (expr - expr.mean()) / expr.std()
    pcs = PCA(n_components=2, random_state=0).fit(z.values)
    sc = pcs.transform(z.values)
    for plat, c in [("GPL20078", "#2ca02c"), ("GPL30493", "#9467bd")]:
        m = (clin.platform == plat).values
        ax[1].scatter(sc[m, 0], sc[m, 1], s=6, alpha=0.6, c=c, label=f"{plat} (n={m.sum()})")
    ev = pcs.explained_variance_ratio_ * 100
    ax[1].set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax[1].set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax[1].set_title("b  Expression PCA by Agilent platform")
    ax[1].legend(frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_cohort.png", dpi=200)

    # Quantify residual platform signal: how well do the top 10 PCs predict the platform?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    pcs10 = PCA(n_components=10, random_state=0).fit_transform(z.values)
    auc = cross_val_score(LogisticRegression(max_iter=2000), pcs10,
                          (clin.platform == "GPL30493").astype(int), cv=5, scoring="roc_auc")
    print(f"platform predictable from top-10 PCs: AUC = {auc.mean():.3f} ± {auc.std():.3f}")
    print(t1[t1.subtype.isin(["HR+HER2-", "TNBC"])].to_string(index=False))


if __name__ == "__main__":
    main()
