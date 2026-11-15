# Lesson 09: Custom losses (deep Cox survival) and external transfer

**Goal:** write a loss PyTorch doesn't ship (the Cox partial likelihood), verify it against a trusted library, and test whether something learned in one trial carries over to another.
Code: [`src/armbenefit/models/survival.py`](../src/armbenefit/models/survival.py), [`scripts/07_imvigor210.py`](../scripts/07_imvigor210.py). Tests: [`tests/test_survival.py`](../tests/test_survival.py).

## 1. Any differentiable function can be a loss
Autograd doesn't care whether your loss is in `torch.nn`. If it's built from tensor operations, `.backward()` works. Survival data needs this, because outcomes are **censored**: for patients still alive at last follow-up, we only know survival exceeded that time.

## 2. Cox partial likelihood, vectorised
The Cox model: hazard_i(t) = h₀(t)·exp(f(xᵢ)). The partial likelihood compares each death with everyone still at risk at that moment, and the unknown baseline h₀ cancels out (Cox, 1972):

    log PL = Σ_{i: event} [ f(xᵢ) − log Σ_{j: Tⱼ ≥ Tᵢ} exp(f(xⱼ)) ]

Sort patients by time descending. Then the risk set of patient *i* is everyone before them in the sorted order, and the inner sum becomes a running `torch.logcumsumexp`: one line, O(n), numerically stable.
```python
order = torch.argsort(time, descending=True)
risk, event = risk[order], event[order]
loss = -((risk - torch.logcumsumexp(risk, 0)) * event).sum() / event.sum()
```
**Verification:** `test_cox_loss_matches_lifelines` checks our value equals lifelines' log-likelihood at its fitted coefficients (rtol 1e-6). Never trust a hand-written loss without such a check.

**DeepSurv** (Katzman et al., 2018) replaces the linear f(x) = βᵀx with an MLP. **Full-batch** training matters: the risk set needs many patients, and small mini-batches give poor risk-set estimates.

**Metric:** Harrell's C-index, the probability that, of two comparable patients, the one with higher predicted risk dies first (Harrell et al., 1996).

## 3. External transfer: the strongest test of generalisation
Internal CV estimates performance on *new patients from the same trial*. Pharma decisions need more: does a biomarker hold **across studies** ("cross-study validation")? We freeze the I-SPY2 ArmNet's explanation of pembrolizumab benefit (Integrated Gradients weights per gene) into a linear score and apply it unchanged to IMvigor210:
- different cancer (urothelial vs breast), platform (RNA-seq vs microarray), drug (anti-PD-L1 vs anti-PD-1) and setting (metastatic vs neoadjuvant);
- no IMvigor210 data is used for fitting, so the result can't be overfit;
- **negative control:** the PARP + platinum benefit score, which should *not* track immunotherapy outcome;
- **caveat:** IMvigor210 is single-arm, so association with outcome on atezolizumab is *prognostic under treatment*, not proof of benefit vs another drug.

## ✅ Check yourself
1. Why does the baseline hazard h₀(t) disappear from the partial likelihood?
2. What breaks if you train a Cox model with mini-batches of 8 patients?
3. The PEMBRO score has AUC 0.65 for atezolizumab response. Name two alternative explanations besides "it captures immunotherapy benefit".

<details><summary>Answers</summary>

1. Each term is a ratio of hazards at the same time t: exp(fᵢ)h₀(t) / Σ exp(fⱼ)h₀(t). The h₀(t) factors cancel.
2. Each risk set contains only batch members, so the loss is a very noisy, biased estimate of the full partial likelihood.
3. (a) It's generally prognostic (e.g. immune infiltration is associated with better outcome regardless of therapy); (b) it correlates with known factors like TMB or tumour subtype; (c) chance, which is what the CI is for.
</details>

## References
- Cox DR. Regression models and life-tables. *J R Stat Soc B* 1972;34:187–220.
- Katzman JL et al. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med Res Methodol* 2018;18:24. https://doi.org/10.1186/s12874-018-0482-1
- Harrell FE, Lee KL, Mark DB. Multivariable prognostic models. *Stat Med* 1996;15:361–387.
- Mariathasan S et al. TGFβ attenuates tumour response to PD-L1 blockade by contributing to exclusion of T cells. *Nature* 2018;554:544–548. https://doi.org/10.1038/nature25501
- Davidson-Pilon C. lifelines: survival analysis in Python. *JOSS* 2019;4:1317. https://doi.org/10.21105/joss.01317
- Riley RD et al. External validation of clinical prediction models using big datasets from e-health records or IPD meta-analysis. *BMJ* 2016;353:i3140. https://doi.org/10.1136/bmj.i3140
