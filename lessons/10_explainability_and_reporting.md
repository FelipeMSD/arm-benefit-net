# Lesson 10: Explaining a network, and reporting a negative result well

**Goal:** use Integrated Gradients (Captum) correctly, read attributions critically, and write up results, including ones that don't confirm the hypothesis, the way a translational team needs them.
Code: [`scripts/06_final_model_explain.py`](../scripts/06_final_model_explain.py). Report: [`docs/report.md`](../docs/report.md).

## 1. Integrated Gradients in practice
IG attributes a model output F(x) to input features by integrating gradients along a straight path from a baseline x′ to x (Sundararajan et al., 2017):

    IGᵢ(x) = (xᵢ − x′ᵢ) · ∫₀¹ ∂F(x′ + α(x − x′))/∂xᵢ dα

- **What to explain matters.** We explain *benefit*: logit(head *a*) − logit(control head), averaged over the ensemble. Explaining the arm's own prediction instead would mostly show prognosis.
- **The baseline matters.** Our inputs are z-scored, so the all-zero baseline is "the average training patient". Attributions answer "why does this patient's predicted benefit differ from the average patient's?"
- **Check completeness:** Σᵢ IGᵢ ≈ F(x) − F(x′). Captum returns this error as `convergence_delta`. Ours is ≤ 3.6e-3, so 64 integration steps are enough.
- In Captum, wrap whatever scalar you want explained in an `nn.Module` (our `Benefit` class) and call `IntegratedGradients(wrapper).attribute(x, baselines=...)`.

## 2. Reading attributions critically
Attributions describe the **model**, not the biology. Our checks:
1. **Pre-registered hypotheses:** immune genes for pembrolizumab, DNA repair for PARP + platinum, written in the script *before* running it.
2. **Permutation test** against random gene sets of the same size.
3. **Specificity:** do two different drugs get different explanations? Ours had ρ = 0.62 and 14 of 25 top genes shared, so the heads mostly encode a common axis.
4. **External transfer:** a frozen attribution score applied to another cohort (lesson 09).

All four agreed with the benefit statistics: there was no drug-specific signal. Explanation methods are a *diagnostic* here. They exposed what the benefit tests suggested.

## 3. Reporting: what the report does, and why
- **Structure (Nature style):** Abstract → Introduction (gap) → Results (claims + numbers + CIs) → Discussion (meaning, limitations, next steps) → Methods (enough to reproduce).
- **Every number carries uncertainty** (95% CI or SD across repeats) and every test is corrected for multiplicity (Benjamini–Hochberg).
- **Positive controls** (TMB and CD8 T-effector in IMvigor210 reproduce the original paper). Without them, a null transfer result could just be a broken pipeline.
- **Negative results are results.** "AUC 0.77 but no benefit information beyond subtype" is exactly what a pharma team needs before spending money on a biomarker-guided trial. TRIPOD+AI and the PATH statement both ask for this kind of transparency (Collins et al., 2024; Kent et al., 2020).

## 4. What this whole project taught (the PyTorch side)
| Concept | Where you used it |
|---|---|
| Tensors, autograd, manual gradient descent | Lesson 01, `logreg_scratch.py` |
| `nn.Module`, `ModuleList`, `DataLoader`, AdamW, dropout, early stopping | `nets.py` |
| Custom indexing in a loss (`gather`) | ArmNet factual loss |
| Custom loss from scratch (`logcumsumexp`) | `survival.py` (Cox) |
| Deep ensembles | `TorchArmModel(n_seeds=5)` |
| Attention, transformers, `huggingface_hub`, safetensors | Lesson 08, BulkFormer |
| Explainability (Captum IG) | `06_final_model_explain.py` |
| Testing ML code (equivalence to reference libraries, leakage, determinism, synthetic ground truth) | `tests/` |

## ✅ Check yourself
1. Why explain the benefit logit rather than P(pCR | arm)?
2. IG attributions are large for *S100P*. Can you conclude *S100P* causes pembrolizumab benefit?
3. Why include TMB and CD8 T-effector in the IMvigor210 analysis if they aren't "your" model?

<details><summary>Answers</summary>

1. P(pCR | arm) mixes prognosis (which affects every arm) with benefit. The difference isolates what's specific to the drug.
2. No. Attributions explain the model's function, which may reflect correlation, confounding (subtype) or noise. Causality needs perturbation or randomised evidence.
3. They're positive controls. If known biomarkers reproduce the original findings, a null result for our score reflects the score, not a broken pipeline.
</details>

## References
- Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. ICML 2017. https://arxiv.org/abs/1703.01365
- Kokhlikyan N et al. Captum: a unified and generic model interpretability library for PyTorch. 2020. https://arxiv.org/abs/2009.07896
- Adebayo J et al. Sanity checks for saliency maps. NeurIPS 2018. https://arxiv.org/abs/1810.03292
- Collins GS et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
- Kent DM et al. The PATH statement. *Ann Intern Med* 2020;172:35–45. https://doi.org/10.7326/M18-3667
- Benjamini Y, Hochberg Y. Controlling the false discovery rate. *J R Stat Soc B* 1995;57:289–300.
