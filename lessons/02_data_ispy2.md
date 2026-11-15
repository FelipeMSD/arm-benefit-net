# Lesson 02: Getting to know the trial before modelling it

**Goal:** download I-SPY2-990 reproducibly, turn GEO files into tidy tables, and find the design features that will shape (and could fool) every model.

Run:
```bash
python scripts/01_download_ispy2.py   # ~50 MB, writes data_raw/ispy2/ + MANIFEST.json (SHA-256)
python scripts/02_eda_ispy2.py        # results/table1_cohort.csv, docs/figures/fig1_cohort.png
```
Code to read: [`src/armbenefit/data/ispy2.py`](../src/armbenefit/data/ispy2.py).

---

## 1. What I-SPY2 is
I-SPY2 is an **adaptive, multi-arm, neoadjuvant platform trial** in high-risk stage II/III breast cancer (Barker et al., 2009).
- Every patient gets weekly paclitaxel, followed by anthracycline chemotherapy and surgery.
- Experimental arms **add** a drug to paclitaxel. Arms "graduate" when their Bayesian predicted probability of success in a phase 3 trial is high enough within a subtype signature.
- **Endpoint:** pathologic complete response (pCR), meaning no invasive cancer left in the breast and nodes at surgery.
- **Randomisation is adaptive.** Allocation probabilities change over time and by subtype (HR, HER2, MammaPrint). So the arm assignment is random *given* subtype and calendar period, not unconditionally. Lesson 06 builds on this.

The 990-patient resource (Wolf et al., 2022) includes pre-treatment Agilent 44K expression for 10 drug arms plus controls.

## 2. What we found (all from `scripts/02_eda_ispy2.py`)
| Finding | Number | Consequence for modelling |
|---|---|---|
| Patients with expression + pCR | 986 | n is small for deep learning. Regularisation and honest CV matter more than architecture |
| Genes without missing values | 17,993 of 19,134 | We drop genes with missing values. This filter doesn't use labels, so it isn't leakage |
| HER2-negative patients | 741 | Seven experimental arms share **one control arm** (n=179). This is where "which arm benefits whom" can be asked |
| HER2-positive patients | 245 | Their own control arm (n=31) is too small. We exclude them from the benefit analysis |
| Platform signal after harmonisation | AUC 0.54 from the top 10 PCs | Batch largely removed. **But** the platform is confounded with arm: NERA and VELI_CARBO are only on GPL30493, PEMBRO and GANE only on GPL20078 |
| ⚠️ Durvalumab + olaparib arm | **not in this resource** | It was published separately (Pusztai 2021). The PARP-inhibitor class is represented by veliparib + carboplatin |

TNBC pCR: control 20% (95% CI 13–30), pembrolizumab 66% (47–80), veliparib + carboplatin 56% (41–71). These match the published direction of effects.

## 3. Concepts
- **Patient-level identifiers as the key.** Every split in later lessons uses `patient_id`, so the same patient can never be in both training and test data.
- **Checksums (SHA-256)** guarantee that the analysed file is the downloaded one.
- **Wilson intervals** for proportions. The usual p ± 1.96·SE (Wald) interval misbehaves for small n or rates near 0 or 1 (Brown, Cai & DasGupta, 2001).
- **PCA as a batch detector.** Batch effects often dominate the top principal components (Leek et al., 2010). Predicting the batch from the PCs gives a single number for "how much batch is left".
- **Pre-processing done by the data authors** (ComBat across platforms) used *all* samples. That's a mild form of leakage we can't undo, so we report it as a limitation.

## ✅ Check yourself
1. Why can't we compare the pCR rate of PEMBRO with that of CTRL pooled across all patients, without looking at subtype?
2. A model reaches AUC 0.80 for predicting pCR in the VELI_CARBO arm. Name one non-biological reason it could be inflated.
3. Why is dropping genes with missing values not leakage, while dropping genes "not associated with pCR" before CV is?

<details><summary>Answers</summary>

1. The arms enrolled different subtype mixes (randomisation is adaptive by subtype), and pCR depends strongly on subtype (TNBC > HR+). Pooled comparisons are confounded by subtype mix.
2. Platform or batch: all VELI_CARBO samples are on one platform, so platform-specific artefacts could correlate with that arm's labels. Calendar time is another (control patients from other periods).
3. The missing-value filter doesn't look at the outcome. Selecting genes by their association with pCR uses the test patients' labels, so the test set is no longer unseen (Ambroise & McLachlan, 2002).
</details>

## References
- Barker AD et al. I-SPY 2: an adaptive breast cancer trial design in the setting of neoadjuvant chemotherapy. *Clin Pharmacol Ther* 2009;86:97–100. https://doi.org/10.1038/clpt.2009.68
- Wolf DM et al. Redefining breast cancer subtypes to guide treatment prioritization and maximize response. *Cancer Cell* 2022;40:609–623. https://doi.org/10.1016/j.ccell.2022.05.005
- Pusztai L et al. Durvalumab with olaparib and paclitaxel for high-risk HER2-negative stage II/III breast cancer. *Cancer Cell* 2021;39:989–998. https://doi.org/10.1016/j.ccell.2021.05.009
- Leek JT et al. Tackling the widespread and critical impact of batch effects in high-throughput data. *Nat Rev Genet* 2010;11:733–739. https://doi.org/10.1038/nrg2825
- Brown LD, Cai TT, DasGupta A. Interval estimation for a binomial proportion. *Stat Sci* 2001;16:101–133. https://doi.org/10.1214/ss/1009213286
- Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS* 2002;99:6562–6566. https://doi.org/10.1073/pnas.102102699
