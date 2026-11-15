# Lesson 00: Setting up a reproducible PyTorch project

**Goal:** understand *why* the repository is organised this way before writing any model code. In translational science, a model you can't reproduce is a model nobody can trust.

---

## 1. What PyTorch is, in one paragraph
PyTorch is a Python library for **tensors** (n-dimensional arrays, like NumPy) that can (a) run on CPU or GPU and (b) **record the operations applied to them**, so derivatives can be computed automatically (*autograd*). Training a neural network is just: compute a prediction → compute a loss → let autograd compute ∂loss/∂parameters → nudge the parameters downhill (gradient descent). Everything else (`nn.Module`, optimizers, `DataLoader`) is convenience built on top of those two ideas. PyTorch uses an *eager*, define-by-run style: code runs line by line like normal Python, which makes debugging easy (Paszke et al., 2019).

## 2. Why conda + a pinned `environment.yml`
- Different versions of numpy, torch or scikit-learn can change numerical results or break code. Pinning exact versions lets anyone rebuild your environment (Wilson et al., 2017).
- conda isolates the project from other Python installations on the laptop (you already have Python 3.12 system-wide; the project uses 3.11 in its own env).
- **CPU-only torch:** our GPU (MX110, 2 GB) is too small. Our datasets (~1,000 patients × ~20,000 genes) are tiny by deep-learning standards and train fine on CPU.

## 3. Why the `src/` layout and an installable package
- Code that is reused (data loaders, models, CV) lives in `src/armbenefit/`. Lessons and scripts *import* it instead of copy-pasting it.
- `pip install -e .` installs it in "editable" mode: you edit the code, and the import sees the change immediately.
- Tests import the *installed* package, which catches packaging mistakes early.

## 4. Why tests from day one
Research code fails silently: a shuffled label, or a scaler fitted on the test set, still gives a number, just a wrong one. Small automated tests catch this:
- **Unit tests:** does a function do the one thing it should? (e.g. "our autograd logistic regression matches scikit-learn").
- **Leakage tests:** does preprocessing ever see test-fold samples? (Kapoor & Narayanan, 2023, documented leakage in 294 papers across 17 fields.)
- **End-to-end tests on synthetic data:** the full pipeline runs in seconds in GitHub Actions without downloading patient data.
(Taschuk & Wilson, 2017: "Ten simple rules for making research software more robust".)

## 5. Why data is never committed
- Patient-level data belongs to its sources. GEO and IMvigor210 are public, but redistributing copies is unnecessary and bloats git.
- **Scripts that download + checksum** give the same guarantee (identical input) with no copies.

## 6. Reproducibility in PyTorch: seeds
Neural network training is random: weight initialisation, data shuffling, dropout. Fixing seeds (`torch.manual_seed`, `numpy.random.seed`, `random.seed`) makes runs repeatable on the same machine and versions. PyTorch's documentation notes that full determinism across platforms is **not guaranteed** (https://pytorch.org/docs/stable/notes/randomness.html). That's another reason to report variability across seeds and folds instead of a single number.

## 7. How we'll work (the "small batch" rule)
Each step: one deliverable → saved → tested → committed. Then you review it before we continue. This is also good scientific practice: small, reviewable changes make errors visible.

---

## ✅ Check yourself
1. In one sentence: what are the two things a PyTorch tensor can do that a plain Python list can't?
2. Why would a scaler fitted on *all* samples before cross-validation inflate your AUC?
3. You get AUC 0.81 with seed 1 and 0.74 with seed 2. What should you report?

## Your setup commands (run in the Anaconda Prompt or the terminal)
```bash
git clone https://github.com/FelipeMSD/arm-benefit-net.git
cd arm-benefit-net
conda env create -f environment.yml
conda activate armbenefit
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected output: `2.5.1+cpu False`.

## References
- Paszke A et al. *PyTorch: An imperative style, high-performance deep learning library.* NeurIPS 2019. https://arxiv.org/abs/1912.01703
- Wilson G et al. *Good enough practices in scientific computing.* PLoS Comput Biol 2017;13:e1005510. https://doi.org/10.1371/journal.pcbi.1005510
- Taschuk M, Wilson G. *Ten simple rules for making research software more robust.* PLoS Comput Biol 2017;13:e1005412. https://doi.org/10.1371/journal.pcbi.1005412
- Kapoor S, Narayanan A. *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 2023;4:100804. https://doi.org/10.1016/j.patter.2023.100804
- PyTorch docs, *Reproducibility.* https://pytorch.org/docs/stable/notes/randomness.html
