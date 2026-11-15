"""
Step 10a: final ArmNet ensemble on ALL HER2-negative patients + Integrated Gradients.

What we explain: the predicted BENEFIT of arm a over control, on the logit scale,
    f_a(x) = logit_a(x) − logit_CTRL(x),
averaged over the 5 ensemble members. Integrated Gradients (Sundararajan et al.,
2017) attributes f_a(x) − f_a(baseline) to each input gene, integrating gradients
along the straight path from a baseline to x. Baseline = all zeros = the average
training patient (inputs are z-scored). IG satisfies "completeness": the attributions
sum to f(x) − f(baseline), and we check this numerically (the convergence delta).

Biological sanity checks (hypotheses fixed BEFORE looking at the results):
  PEMBRO     benefit should load on T-cell / IFN-γ / checkpoint genes (CD8 T effector,
             Immune Checkpoint, APM signatures from Mariathasan et al. 2018).
  VELI_CARBO benefit should load on DNA-damage-repair / proliferation genes (DDR,
             homologous recombination, Fanconi, cell cycle).
Test: mean |attribution| of the set vs 10,000 random gene sets of the same size.

The model is trained on all patients, so these attributions DESCRIBE the fitted model;
they are not an independent validation of it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from captum.attr import IntegratedGradients  # noqa: E402
from torch import nn  # noqa: E402

from armbenefit.data.imvigor210 import signature_genes  # noqa: E402
from armbenefit.eval.cv import FoldPreprocessor, clinical_matrix  # noqa: E402
from armbenefit.models.nets import TorchArmModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, ART = ROOT / "results", ROOT / "docs" / "figures", ROOT / "artifacts"
ART.mkdir(exist_ok=True)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "04_benchmark_ispy2.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)
ARMS = bench.ARMS

SETS = {
    "PEMBRO": ["CD 8 T effector", "Immune Checkpoint", "APM"],
    "VELI_CARBO": ["DDR", "Homologous recombination", "Fanconi", "Cell cycle"],
}


class Benefit(nn.Module):
    """Wraps the ensemble: input x (genes + clinical) -> ensemble-mean benefit logit of arm k vs control."""

    def __init__(self, nets: list[nn.Module], k: int):
        super().__init__()
        self.nets, self.k = nn.ModuleList(nets), k

    def forward(self, x):
        out = [m(x) for m in self.nets]
        return torch.stack([o[:, self.k] - o[:, 0] for o in out]).mean(0)


def set_test(attr_abs: pd.Series, genes: list[str], rng, n_perm: int = 10_000) -> dict:
    genes = [g for g in genes if g in attr_abs.index]
    if not genes:
        return dict(n_in_model=0, obs=np.nan, p_perm=np.nan)
    obs = attr_abs[genes].mean()
    vals = attr_abs.to_numpy()
    null = np.array([vals[rng.choice(len(vals), len(genes), replace=False)].mean() for _ in range(n_perm)])
    return dict(n_in_model=len(genes), genes=",".join(genes), obs=obs,
                null_mean=null.mean(), p_perm=(1 + (null >= obs).sum()) / (n_perm + 1))


def main() -> None:
    clin, expr = bench.load_cohort()
    tune = pd.read_csv(RES / "tuning_choices.csv")
    do, wd = tune[tune.model == "ArmNet"][["dropout", "weight_decay"]].mode().iloc[0]
    print(f"final hyperparameters (mode over outer folds): dropout={do}, weight_decay={wd}")
    pp = FoldPreprocessor(top_k=2000).fit(expr)
    G = pp.transform(expr)
    C = clinical_matrix(clin, ARMS, include_arm=False)
    model = TorchArmModel(len(ARMS), "armnet", dropout=do, weight_decay=wd, n_seeds=5, seed=2026)
    model.fit(G, C, clin.arm_idx.to_numpy(), clin.pcr.astype(int).to_numpy())
    torch.save(dict(states=[m.state_dict() for m in model.models], genes=list(pp.genes_),
                    mean=pp.mean_, std=pp.std_, arms=ARMS, dropout=do, weight_decay=wd),
               ART / "armnet_final.pt")

    X = torch.from_numpy(np.column_stack([G, C]).astype(np.float32))
    sigs = signature_genes()
    rng = np.random.default_rng(0)
    rows_top, rows_set, sig_weights = [], [], {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, arm in zip(axes, ["PEMBRO", "VELI_CARBO"]):
        k = ARMS.index(arm)
        ig = IntegratedGradients(Benefit(model.models, k))
        attrs, deltas = [], []
        for chunk in torch.split(X, 64):
            a, d = ig.attribute(chunk, baselines=torch.zeros_like(chunk), n_steps=64,
                                return_convergence_delta=True)
            attrs.append(a.detach().numpy())
            deltas.append(d.detach().numpy())
        A = np.concatenate(attrs)[:, : G.shape[1]]                       # gene columns only
        delta = np.abs(np.concatenate(deltas)).max()
        signed = pd.Series(A.mean(0), index=pp.genes_)
        absval = pd.Series(np.abs(A).mean(0), index=pp.genes_)
        sig_weights[arm] = signed
        print(f"{arm}: max |IG completeness error| = {delta:.2e}")
        top = absval.sort_values(ascending=False).head(25)
        for g in top.index:
            rows_top.append(dict(arm=arm, gene=g, mean_abs_attr=absval[g], mean_signed_attr=signed[g]))
        for s in SETS[arm]:
            rows_set.append(dict(arm=arm, signature=s, **set_test(absval, sigs[s], rng)))
        t15 = top.head(15)[::-1]
        ax.barh(t15.index, [signed[g] for g in t15.index],
                color=["#d62728" if signed[g] > 0 else "#1f77b4" for g in t15.index])
        ax.set_xlabel(f"Mean IG attribution to {arm} benefit (logit)")
    axes[0].set_title("a  Top genes driving predicted PEMBRO benefit")
    axes[1].set_title("b  Top genes driving predicted VELI_CARBO benefit")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_explainability.png", dpi=200)
    pd.DataFrame(rows_top).to_csv(RES / "table5_ig_top_genes.csv", index=False, float_format="%.4g")
    st = pd.DataFrame(rows_set)
    st.to_csv(RES / "table5b_geneset_tests.csv", index=False, float_format="%.4g")
    pd.DataFrame(sig_weights).to_csv(RES / "benefit_signature_weights.csv", float_format="%.5g")
    print(pd.DataFrame(rows_top).groupby("arm").head(12).to_string(index=False))
    print(st.drop(columns="genes", errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
