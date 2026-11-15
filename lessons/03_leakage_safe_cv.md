# Lesson 03: Leakage-safe validation, so you can trust the numbers

**Goal:** build the evaluation machinery *before* any model, so no model gets a chance to cheat.
Code: [`src/armbenefit/eval/cv.py`](../src/armbenefit/eval/cv.py), [`src/armbenefit/eval/metrics.py`](../src/armbenefit/eval/metrics.py). Tests: [`tests/test_cv_metrics.py`](../tests/test_cv_metrics.py).

## 1. Leakage: the most common way ML papers overstate results
**Leakage** means information from the test patients reaches the model during training. Kapoor & Narayanan (2023) found leakage in 294 papers across 17 fields. In genomics the classic case is selecting genes by their association with the outcome on *all* samples, then cross-validating. Ambroise & McLachlan (2002) showed this can produce near-perfect accuracy on pure noise.

Our three rules:
| Rule | Implementation | Test that enforces it |
|---|---|---|
| Patient-level splits, stratified by arm × subtype × pCR | `outer_folds()` | `test_folds_partition_patients` |
| Preprocessing fitted on the training fold only | `FoldPreprocessor` (variance filter + z-score) | `test_preprocessor_never_sees_test_data` |
| Hyperparameters tuned in an inner loop | nested CV in `scripts/04_benchmark.py` | outer test folds are never passed to tuning |

**Why nested CV?** If you try 20 hyperparameter settings and report the best CV score, that score is optimistically biased, because you selected it on the same data you report it on (Varma & Simon, 2006; Cawley & Talbot, 2010). The outer loop estimates performance and the inner loop chooses settings.

## 2. Metrics that a clinical reviewer expects
- **AUC:** discrimination. It's rank-based and insensitive to calibration.
- **Brier score:** mean squared error of the probabilities.
- **Calibration intercept and slope:** do predicted risks match observed rates? A slope < 1 means overfitting (predictions too extreme). Calibration is "the Achilles heel of predictive analytics" (Van Calster et al., 2019).
- **95% bootstrap CIs:** resample *patients*. With ~700 patients, a single AUC without a CI says little.

TRIPOD+AI (Collins et al., 2024) asks for all of these.

## 3. Metrics for "which arm benefits whom"
Predicting pCR is not the same as predicting **benefit**. A gene that marks good prognosis in every arm (e.g. proliferation in TNBC) is *prognostic*. A gene that changes the effect of one drug is *predictive* (Ballman, 2015). For an arm *a* versus control, predicted benefit is

    τ̂_a(x) = P̂(pCR | x, arm a) − P̂(pCR | x, control)

We can never observe both outcomes for one patient (the fundamental problem of causal inference; Holland, 1986). So benefit is validated at the **group** level, using randomisation:
1. **Interaction test:** among patients randomised to *a* or control, does τ̂ modify the observed treatment effect? We fit `pCR ~ treated + τ̂ + treated×τ̂ + subtype`. A positive interaction term means the model ranks benefit correctly.
2. **c-for-benefit** (van Klaveren et al., 2018): match treated and control patients by predicted benefit and check whether pairs with larger predicted benefit show larger observed benefit. A value of 0.5 means the model tells us nothing about benefit.
3. **Policy value (IPW):** what pCR rate would we expect if each patient got the arm the model recommends? Patients whose randomised arm equals the recommendation are re-weighted by 1/P(arm | subtype) (Hájek estimator; see Kosorok & Laber, 2019, on precision-medicine policies).

## ✅ Check yourself
1. You z-score all 986 patients, then split into folds. Is that leakage? How serious is it compared with selecting genes by p-value on all patients?
2. Why do we stratify the folds by arm as well as by pCR?
3. A model has AUC 0.75 but calibration slope 0.4. What does that mean in practice?
4. Why can't we compute an AUC for benefit at the patient level?

<details><summary>Answers</summary>

1. Yes, technically: the test-fold means and SDs influenced the training features. It's usually mild because it's unsupervised and doesn't use labels. Supervised selection is far worse. It's free to avoid, so we avoid it.
2. pCR rates differ strongly by arm (17–66%). Unbalanced folds would mix arm-effect differences into the fold-to-fold variance, and some folds could miss small arms.
3. The ranking is good, but the probabilities are far too extreme. A predicted 90% might really be 60%, so it can't be used for decisions without recalibration.
4. Each patient has only one observed outcome (under the arm they received), so their true individual benefit is never known.
</details>

## References
- Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS* 2002;99:6562–6566. https://doi.org/10.1073/pnas.102102699
- Kapoor S, Narayanan A. Leakage and the reproducibility crisis in ML-based science. *Patterns* 2023;4:100804. https://doi.org/10.1016/j.patter.2023.100804
- Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
- Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR* 2010;11:2079–2107.
- Van Calster B et al. Calibration: the Achilles heel of predictive analytics. *BMC Med* 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
- Collins GS et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
- Ballman KV. Biomarker: predictive or prognostic? *J Clin Oncol* 2015;33:3968–3971. https://doi.org/10.1200/JCO.2015.63.3651
- Holland PW. Statistics and causal inference. *J Am Stat Assoc* 1986;81:945–960. https://doi.org/10.1080/01621459.1986.10478354
- van Klaveren D et al. The proposed "concordance-statistic for benefit" provided a useful metric when modeling heterogeneous treatment effects. *J Clin Epidemiol* 2018;94:59–68. https://doi.org/10.1016/j.jclinepi.2017.10.021
- Kosorok MR, Laber EB. Precision medicine. *Annu Rev Stat Appl* 2019;6:263–286. https://doi.org/10.1146/annurev-statistics-030718-105251
