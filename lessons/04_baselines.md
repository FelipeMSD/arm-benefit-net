# Lesson 04: Baselines first, the models deep learning has to beat

**Goal:** understand the four classical baselines and *why* they're the bar.
Code: [`src/armbenefit/models/baselines.py`](../src/armbenefit/models/baselines.py). Run: `python scripts/04_benchmark_ispy2.py` (nested CV), then `python scripts/05_evaluate_ispy2.py`.

## 1. Why baselines come first
On tabular data such as an expression matrix, gradient-boosted trees and penalised regression often match or beat deep networks (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022). Christodoulou et al. (2019) reviewed 71 clinical prediction studies and found no average performance benefit of ML over logistic regression. A deep model is only interesting if it beats these **under the same leakage-safe protocol**.

## 2. The baselines, and what each one assumes
| Model | Inputs | What it can learn about *benefit* |
|---|---|---|
| **Clinical-LR** | HR, MammaPrint, platform, arm, arm×HR, arm×MP | Benefit differs only by **clinical subtype**. That's the trialist's "best arm per subtype" |
| **Ridge-S** (S-learner) | genes + clinical + arm | Genes shift the baseline risk, and the arm adds a constant log-odds. Benefit varies only through the logistic curve (no gene×arm terms) |
| **Ridge-T** (T-learner) | one penalised model **per arm** | Any gene-specific benefit, but each model sees only 49–179 patients, so its estimates are noisy (Künzel et al., 2019) |
| **LGBM-S** | genes + clinical + arm, trees | Interactions found automatically by tree splits |

**Ridge (L2) logistic regression** shrinks all gene coefficients towards zero without setting any exactly to zero. With thousands of correlated genes it predicts about as well as sparse penalties such as the elastic net (L1 + L2; Zou & Hastie, 2005), as Bøvelstad et al. (2007) found for expression-based prediction. The penalty strength is tuned by internal CV on the training fold.

### Two real problems we hit, and what they teach
**(a) Solver choice is part of the method.** We first used an elastic net with the `saga` solver. After the fix in (b), one outer fold took over 40 minutes, because rescaled columns make saga's step size tiny and every fit ran to its iteration cap. Switching to ridge with L-BFGS gave the same kind of model in about 1 minute per fold. Always time one fold before launching the full run.

**(b) The penalty can silently remove the treatment.**

In the first run, the S-learner recommended **control for every patient**. The penalty treats all columns equally, so with 2,000 genes it shrank the 7 arm indicators to exactly zero. With no arm effect, every arm got the same prediction, and `argmax` picked the first one (control). Fix: scale the design columns (arm, clinical) by 30 so their effective penalty is 900× smaller (`RidgeS.ARM_SCALE`). That's the usual practice of not penalising design variables, and a regression test (`test_ridge_s_keeps_arm_effects`) now guards it.
**Lesson:** always check that a model's counterfactual predictions actually vary across arms.

## ✅ Check yourself
1. Why can a T-learner estimate gene-specific benefit while an S-learner ridge model (main effects only) cannot?
2. Why does scaling a column by 30 reduce its effective L2 penalty by 900?
3. The Clinical-LR model has no genes. Why is it still essential in a genomics paper?

<details><summary>Answers</summary>

1. The T-learner fits separate gene coefficients per arm, so the difference between arms depends on genes. The S-learner has one set of gene coefficients plus an arm constant, so the arm effect is the same for every patient on the log-odds scale.
2. A coefficient β on x becomes β/30 on 30x for the same fit. The penalty (β/30)² is 900 times smaller.
3. It answers "does expression add anything beyond what clinicians already know (HR, MammaPrint)?" Without it, a high AUC could just be subtype.
</details>

## References
- Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on tabular data? NeurIPS 2022. https://arxiv.org/abs/2207.08815
- Shwartz-Ziv R, Armon A. Tabular data: deep learning is not all you need. *Inf Fusion* 2022;81:84–90. https://doi.org/10.1016/j.inffus.2021.11.011
- Christodoulou E et al. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *J Clin Epidemiol* 2019;110:12–22. https://doi.org/10.1016/j.jclinepi.2019.02.004
- Bøvelstad HM et al. Predicting survival from microarray data: a comparative study. *Bioinformatics* 2007;23:2080–2087. https://doi.org/10.1093/bioinformatics/btm305
- Zou H, Hastie T. Regularization and variable selection via the elastic net. *J R Stat Soc B* 2005;67:301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
- Künzel SR et al. Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS* 2019;116:4156–4165. https://doi.org/10.1073/pnas.1804597116
- Ke G et al. LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 2017.
