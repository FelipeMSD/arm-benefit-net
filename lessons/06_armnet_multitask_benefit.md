# Lesson 06: ArmNet, a multi-task network that predicts response under every arm

**Goal:** understand the central model of this project, and how its "benefit" predictions are validated with randomisation.

## 1. From prediction to treatment-effect heterogeneity
We want τ_a(x) = P(pCR | x, arm a) − P(pCR | x, control): the benefit of arm *a* for a tumour with expression *x*. Under randomisation (within the subtype strata that I-SPY2 uses), each arm's patients are exchangeable with the control patients, so P(pCR | x, arm) is identified from the trial data (Rubin, 1974; Hernán & Robins, 2020).

## 2. ArmNet = shared trunk + one head per arm
```python
h = trunk(x)                        # tumour representation, learned from ALL arms
logits = [head_k(h) for k in arms]  # one logit per arm  -> (n, K)
loss = BCE(logits[i, arm_i], y_i)   # FACTUAL loss: only the arm the patient got
```
`gather(1, arm.view(-1, 1))` picks each patient's own column, which is the PyTorch idiom for "index a different column per row". This design is TARNet (Shalit, Johansson & Sontag, 2017) extended to K arms.
- **Why share?** 49–179 patients per arm can't support a separate deep net (Ridge-T shows the noise). The trunk is trained by every patient, which is "borrowing strength" across arms, like a hierarchical model.
- **Why separate heads?** So the effect of each drug can depend on the representation (heterogeneity), unlike a pooled model where the arm enters as just another input (`PooledMLP`, the S-learner).
- **Deep ensemble (ArmNet-Ens5):** 5 networks from different seeds, probabilities averaged. Ensembles improve accuracy and calibration under small data (Lakshminarayanan et al., 2017).

## 3. Validating benefit
Individual benefit is never observed (one outcome per patient). We therefore use the randomised comparison at group level (lesson 03): the **interaction test**, **c-for-benefit** and **IPW policy value**. Results are in `results/table3_benefit.csv`, `results/table4_policy.csv` and figures 3–4.

**Key comparison:** Clinical-LR (benefit by subtype only) vs ArmNet (benefit from expression). If ArmNet doesn't beat Clinical-LR on benefit metrics, the extra genomic complexity isn't justified, however good its AUC.

## 4. Limits worth stating
- Adaptive randomisation changes allocation over **time**. Our propensities condition on HR × MP strata but not calendar time, so the IPW values are approximate.
- Arm and platform are confounded for some arms (lesson 02).
- Policy value with 8 arms rests on few "concordant" patients per recommendation, so CIs are wide.

## ✅ Check yourself
1. Why is the loss computed only on the received arm's head?
2. ArmNet predicts PEMBRO benefit of +30 points for a patient. How could you check that this is right *for that patient*? (Trick question.)
3. What would you expect the c-for-benefit to be for a model with only main effects on the log-odds scale?

<details><summary>Answers</summary>

1. We only know the outcome under the received arm. Training other heads on that label would teach them the wrong arm's outcome.
2. You can't. Individual effects aren't observable. Only groups of similar patients randomised to different arms can validate predicted benefit.
3. Above 0.5 is possible but limited: on the probability scale, benefit varies with baseline risk even without interactions. Any discrimination comes from risk magnitude, not from biology specific to the drug.
</details>

## References
- Shalit U, Johansson FD, Sontag D. Estimating individual treatment effect: generalization bounds and algorithms. ICML 2017. https://arxiv.org/abs/1606.03976
- Rubin DB. Estimating causal effects of treatments in randomized and nonrandomized studies. *J Educ Psychol* 1974;66:688–701.
- Hernán MA, Robins JM. *Causal Inference: What If.* Chapman & Hall/CRC 2020. https://miguelhernan.org/whatifbook
- Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS 2017. https://arxiv.org/abs/1612.01474
- Kent DM et al. The Predictive Approaches to Treatment effect Heterogeneity (PATH) statement. *Ann Intern Med* 2020;172:35–45. https://doi.org/10.7326/M18-3667
- van Klaveren D et al. c-for-benefit. *J Clin Epidemiol* 2018;94:59–68. https://doi.org/10.1016/j.jclinepi.2017.10.021
