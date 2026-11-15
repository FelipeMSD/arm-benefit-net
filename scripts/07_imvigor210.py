"""
Step 9: IMvigor210 (atezolizumab, metastatic urothelial cancer, RNA-seq).

Part A: cross-trial, cross-cancer, cross-platform TRANSFER.
  The I-SPY2 ArmNet learned which tumours benefit from pembrolizumab (anti-PD-1)
  over paclitaxel. We turn its Integrated-Gradients attributions into a linear
  "PEMBRO-benefit score" (Σ_g w_g · z_g) and ask, in a different cancer, platform and
  anti-PD-(L)1 drug, whether the score is associated with response and survival.
  The weights are frozen: nothing is fitted on IMvigor210, so this is a true external test.
  Comparators: CD8 T-effector signature (Mariathasan 2018), tumour mutational burden (TMB).
  Negative control: the VELI_CARBO (PARP + platinum) benefit score.
  Caveat: IMvigor210 is single-arm, so it can show PROGNOSIS under atezolizumab,
  not benefit over another treatment.

Part B: survival prediction benchmark (5-fold × 3 repeats CV):
  ridge Cox (lifelines) vs linear Cox and DeepCox in PyTorch, on fold-selected top
  genes + log TMB, plus foundation-model (BulkFormer) embeddings if present.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import torch  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from armbenefit.data.imvigor210 import DIR, load_imvigor210, signature_genes  # noqa: E402
from armbenefit.eval.cv import FoldPreprocessor  # noqa: E402
from armbenefit.eval.metrics import bootstrap_ci  # noqa: E402
from armbenefit.models.survival import DeepCox, LinearCox, fit_cox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "docs" / "figures"


def zscore(df):
    return (df - df.mean()) / df.std().replace(0, 1)


def part_a(clin, expr):
    w = pd.read_csv(RES / "benefit_signature_weights.csv", index_col=0)
    z = zscore(expr)
    scores = pd.DataFrame(index=clin.index)
    cover = {}
    for arm in ["PEMBRO", "VELI_CARBO"]:
        shared = w.index.intersection(z.columns)
        cover[arm] = (len(shared), len(w))
        scores[f"{arm}_benefit_score"] = z[shared].to_numpy() @ w.loc[shared, arm].to_numpy()
    sig = signature_genes()
    cd8 = [g for g in sig["CD 8 T effector"] if g in z.columns]
    scores["CD8_Teff"] = z[cd8].mean(axis=1)
    scores["log_TMB"] = np.log1p(clin.tmb)
    scores = (scores - scores.mean()) / scores.std()            # per-SD effects

    rows = []
    resp = clin.response.notna()
    for s in scores.columns:
        ok = resp & scores[s].notna()
        y, x = clin.response[ok].to_numpy(), scores[s][ok].to_numpy()
        auc = roc_auc_score(y, x)
        lo, hi = bootstrap_ci(roc_auc_score, y, x, n_boot=2000)
        d = pd.DataFrame({"T": clin.os_months, "E": clin.os_event, "x": scores[s],
                          "ecog": pd.to_numeric(clin.ecog, errors="coerce")}).dropna()
        cph = CoxPHFitter().fit(d, "T", "E")
        hr = cph.summary.loc["x"]
        rows.append(dict(score=s, n_response=int(ok.sum()), auc_response=auc, auc_lo=lo, auc_hi=hi,
                         n_os=len(d), hr_os_per_sd=hr["exp(coef)"], hr_lo=hr["exp(coef) lower 95%"],
                         hr_hi=hr["exp(coef) upper 95%"], p_os=hr["p"]))
    ta = pd.DataFrame(rows)

    # Does the transferred score add information beyond CD8 T-effector + TMB? (likelihood-ratio test)
    d = pd.concat([clin.response, scores], axis=1).dropna()
    base = sm.Logit(d.response, sm.add_constant(d[["CD8_Teff", "log_TMB"]])).fit(disp=0)
    full = sm.Logit(d.response, sm.add_constant(d[["CD8_Teff", "log_TMB", "PEMBRO_benefit_score"]])).fit(disp=0)
    lr = 2 * (full.llf - base.llf)
    from scipy.stats import chi2
    added = dict(n=len(d), or_pembro_score=float(np.exp(full.params["PEMBRO_benefit_score"])),
                 lr_stat=lr, p_lr=float(chi2.sf(lr, 1)), gene_coverage=str(cover))
    return ta, added, scores


def part_b(clin, expr, n_rep=3, top_k=1000):
    d = clin[clin.os_months.notna()]
    X_all = expr.loc[d.index]
    T, E = d.os_months.to_numpy(), d.os_event.to_numpy()
    tmb = np.log1p(d.tmb.to_numpy())
    emb_path = DIR / "bulkformer_embeddings.csv.gz"
    emb = pd.read_csv(emb_path, index_col=0).loc[d.index] if emb_path.exists() else None
    rows = []
    for rep in range(n_rep):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        for fold, (tr, te) in enumerate(skf.split(X_all, E)):
            pp = FoldPreprocessor(top_k=top_k).fit(X_all.iloc[tr])
            med = np.nanmedian(tmb[tr])
            t_tr = np.nan_to_num(tmb[tr], nan=med)
            t_te = np.nan_to_num(tmb[te], nan=med)
            mu, sd = t_tr.mean(), t_tr.std()
            Xtr = np.column_stack([pp.transform(X_all.iloc[tr]), (t_tr - mu) / sd])
            Xte = np.column_stack([pp.transform(X_all.iloc[te]), (t_te - mu) / sd])
            feats = {"genes+TMB": (Xtr, Xte)}
            if emb is not None:
                e_tr, e_te = emb.iloc[tr].to_numpy(), emb.iloc[te].to_numpy()
                m, s = e_tr.mean(0), e_tr.std(0) + 1e-8
                feats["BulkFormer+TMB"] = (np.column_stack([(e_tr - m) / s, (t_tr - mu) / sd]),
                                           np.column_stack([(e_te - m) / s, (t_te - mu) / sd]))
            feats["TMB only"] = (Xtr[:, [-1]], Xte[:, [-1]])
            for fname, (a, b) in feats.items():
                preds = {}
                if a.shape[1] == 1:
                    cph = CoxPHFitter().fit(pd.DataFrame({"x": a[:, 0], "T": T[tr], "E": E[tr]}), "T", "E")
                    preds["Cox"] = cph.predict_partial_hazard(pd.DataFrame({"x": b[:, 0]})).to_numpy()
                else:
                    # Classical comparator: ridge Cox on the top 50 principal components (PC-Cox).
                    # lifelines' Newton–Raphson diverges with ~1,000 covariates and ~280 patients,
                    # so we reduce dimension first, inside the fold (Bøvelstad et al. 2007).
                    from sklearn.decomposition import PCA
                    k = min(50, a.shape[1] - 1)
                    pca = PCA(n_components=k, random_state=0).fit(a[:, :-1])
                    pa = np.column_stack([pca.transform(a[:, :-1]), a[:, -1]])
                    pb = np.column_stack([pca.transform(b[:, :-1]), b[:, -1]])
                    cols = [f"f{i}" for i in range(pa.shape[1])]
                    cph = CoxPHFitter(penalizer=0.5, l1_ratio=0.0).fit(
                        pd.DataFrame(pa, columns=cols).assign(T=T[tr], E=E[tr]), "T", "E",
                        fit_options=dict(step_size=0.3))
                    preds["PC-Cox, ridge (lifelines)"] = cph.predict_partial_hazard(
                        pd.DataFrame(pb, columns=cols)).to_numpy()
                    for name, net in [("Linear Cox (PyTorch)", LinearCox(a.shape[1])),
                                      ("DeepCox (PyTorch)", DeepCox(a.shape[1]))]:
                        net = fit_cox(net, a, T[tr], E[tr], seed=rep * 10 + fold)
                        with torch.no_grad():
                            preds[name] = net(torch.as_tensor(b, dtype=torch.float32)).numpy()
                for mname, r in preds.items():
                    rows.append(dict(rep=rep, fold=fold, features=fname, model=mname,
                                     cindex=concordance_index(T[te], -r, E[te])))
    res = pd.DataFrame(rows)
    summ = (res.groupby(["features", "model"]).cindex.agg(["mean", "std", "count"])
            .reset_index().sort_values("mean", ascending=False))
    return summ


def main():
    clin, expr, _, _ = load_imvigor210()
    ta, added, scores = part_a(clin, expr)
    ta.to_csv(RES / "table6_imvigor_transfer.csv", index=False, float_format="%.4g")
    pd.Series(added).to_csv(RES / "table6b_imvigor_added_value.csv", header=False)
    print(ta.round(3).to_string(index=False))
    print(added)
    tb = part_b(clin, expr)
    tb.to_csv(RES / "table7_imvigor_survival_cv.csv", index=False, float_format="%.4g")
    print(tb.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, r in ta.iterrows():
        ax[0].errorbar(r.auc_response, i, xerr=[[r.auc_response - r.auc_lo], [r.auc_hi - r.auc_response]],
                       fmt="o", capsize=3, c="k")
        ax[1].errorbar(r.hr_os_per_sd, i, xerr=[[r.hr_os_per_sd - r.hr_lo], [r.hr_hi - r.hr_os_per_sd]],
                       fmt="o", capsize=3, c="k")
    for a_, lab, ref in [(ax[0], "AUC for CR/PR (95% CI)", 0.5), (ax[1], "OS hazard ratio per SD (ECOG-adjusted)", 1)]:
        a_.set_yticks(range(len(ta)), ta.score)
        a_.axvline(ref, ls=":", c="grey")
        a_.set_xlabel(lab)
        a_.invert_yaxis()
    ax[0].set_title("a  Response to atezolizumab (IMvigor210)")
    ax[1].set_title("b  Overall survival")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_imvigor_transfer.png", dpi=200)


if __name__ == "__main__":
    main()
