# Lesson 07: Uncertainty and calibration, so a probability means what it says

**Goal:** read a calibration plot, understand why deep ensembles help, and know what calibration slope and intercept tell a clinician.
Outputs: `results/table2_prediction.csv` (columns `cal_intercept`, `cal_slope`) and `docs/figures/fig2_prediction.png` (b).

## 1. Calibration
A model is calibrated if, among patients given P(pCR) = 0.4, about 40% achieve pCR. Discrimination (AUC) and calibration are different: a model can rank perfectly and still be badly miscalibrated. For a treatment decision, the absolute probability matters (Van Calster et al., 2019).

**Logistic recalibration:** fit logit P(y) = a + b·logit(p̂).
- slope *b* < 1: predictions too extreme (typical overfitting); *b* > 1: too timid.
- intercept *a* ≠ 0 (with *b* fixed at 1): systematic over- or under-prediction.

**Calibration plot:** group patients into bins of predicted risk and plot observed vs predicted. With n ≈ 740, use few bins (we use octiles) and treat the curve as noisy.

## 2. Why neural networks are often miscalibrated
Modern networks trained with cross-entropy can become over-confident as capacity grows (Guo et al., 2017). Weight decay, dropout and early stopping temper this. **Deep ensembles** (Lakshminarayanan et al., 2017) average M independently initialised networks:
- averaging cancels part of the seed-specific noise, which improves accuracy;
- disagreement between members reflects *epistemic* uncertainty (what the model doesn't know because data is scarce);
- averaged probabilities are usually better calibrated, though sometimes **under**-confident (slope > 1), because averaging pulls probabilities towards the middle.

In code, an ensemble is just a loop: `TorchArmModel(..., n_seeds=5)` trains 5 nets and `predict_arms` averages them.

## 3. Repeated cross-validation
We repeat 5-fold CV with different splits (3 repeats) and average each patient's out-of-fold predictions. The SD of AUC across repeats (`auc_sd_repeats`) shows how much a single split could mislead you.

## ✅ Check yourself
1. Model A has AUC 0.78 and slope 0.55. Model B has AUC 0.76 and slope 0.98. Which would you use to tell a patient "your chance of pCR is 35%"?
2. Why might averaging 5 networks produce slope > 1?
3. Which kind of uncertainty can an ensemble capture, and which can't it?

<details><summary>Answers</summary>

1. Model B. Its probabilities can be taken at face value. Model A would need recalibration first.
2. Averaging pulls individual extreme predictions towards the mean, so the ensemble's probabilities are less spread than the truth.
3. Epistemic uncertainty (model or parameter uncertainty from limited data) shows up as member disagreement. Aleatoric uncertainty (irreducible outcome randomness) is already in the probability itself. Neither covers distribution shift to a different population.
</details>

## References
- Van Calster B et al. Calibration: the Achilles heel of predictive analytics. *BMC Med* 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
- Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017. https://arxiv.org/abs/1706.04599
- Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS 2017. https://arxiv.org/abs/1612.01474
- Steyerberg EW. *Clinical Prediction Models*, 2nd ed. Springer 2019. https://doi.org/10.1007/978-3-030-16399-0
