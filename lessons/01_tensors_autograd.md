# Lesson 01: Tensors and autograd, the two ideas everything else is built on

**Goal:** by the end of this lesson you can:
1. create and manipulate PyTorch tensors, and predict their `shape`, `dtype` and `device`;
2. use broadcasting without bugs;
3. explain what `requires_grad`, `.backward()` and `.grad` do;
4. train a logistic regression **by hand** with gradient descent, and prove it matches scikit-learn.

Why start with logistic regression? It's the simplest "neural network": one linear layer plus a sigmoid. It's also the **baseline** every biomarker paper should beat. If you can build it from tensors and autograd, every later model (MLP, multi-task net, transformer) is the same recipe with more layers.

Hands-on script: [`01_tensors_autograd.py`](01_tensors_autograd.py). Run it with `python lessons/01_tensors_autograd.py`.

---

## 1. Tensors
A **tensor** is an n-dimensional array, like a NumPy `ndarray`, with two extra powers:
- it can live on a **device** (CPU or GPU);
- it can **record the operations** applied to it, so derivatives can be computed later (autograd).

In our project, an expression matrix is a 2-D tensor of shape `(n_patients, n_genes)`, e.g. `(990, 20000)`. The pCR labels form a 1-D tensor of shape `(990,)`.

| Attribute | What it is | Why it matters |
|---|---|---|
| `shape` | size of each dimension | most bugs are shape bugs |
| `dtype` | `float32`, `float64`, `int64`, `bool`… | deep learning defaults to `float32` (fast). We use `float64` when we compare against scikit-learn to many decimals |
| `device` | `cpu` or `cuda:0` | all tensors in one operation must be on the same device |
| `requires_grad` | whether to record operations for autograd | parameters: `True`; data: `False` |

**NumPy interop:** `torch.from_numpy(a)` and `t.numpy()` **share memory**: changing one changes the other. `torch.tensor(a)` makes a copy. This difference causes subtle bugs, and the script demonstrates it.

## 2. Broadcasting
When two tensors have different shapes, PyTorch (like NumPy) "stretches" dimensions of size 1 so they match. Shapes are compared **from the right**. Two dimensions are compatible if they're equal or if one of them is 1.

Example: standardising genes. `X` has shape `(n, p)`, and `X.mean(dim=0)` has shape `(p,)`. Then `X - mean` works: `(p,)` is treated as `(1, p)` and stretched to `(n, p)`.

⚠️ **The classic bug:** `y` with shape `(n,)` and predictions with shape `(n, 1)`. `y - pred` then broadcasts to `(n, n)`. That's a silent, wrong number, not an error. Always check shapes.

## 3. Autograd: automatic differentiation
Training means finding parameters θ that minimise a loss L(θ). Gradient descent repeats:

    θ ← θ − η · ∂L/∂θ        (η = learning rate)

Computing ∂L/∂θ by hand is error-prone for big models. **Autograd** does it automatically:
1. Mark parameters with `requires_grad=True`.
2. Compute the loss with ordinary tensor operations. PyTorch silently builds a **computational graph**: each result remembers which operation produced it (`.grad_fn`).
3. `loss.backward()` walks the graph backwards, applying the chain rule (*reverse-mode automatic differentiation*, also called *backpropagation*), and stores ∂L/∂θ in `θ.grad`.

Three rules beginners trip on:
- **Gradients accumulate.** `.backward()` *adds* to `.grad`. Reset it every step with `θ.grad = None` (or `optimizer.zero_grad()` in later lessons).
- **Update parameters inside `torch.no_grad()`**, otherwise the update itself would be recorded in the graph.
- **`.detach()`** gives a tensor that shares data but is cut from the graph. Use it for logging and plotting.

Reverse mode is efficient because a single backward pass gives the gradient with respect to *all* parameters at a cost of a few forward passes, whether there are 10 parameters or 10 million (Baydin et al., 2018).

## 4. Logistic regression as a tiny neural network
For patient *i* with expression vector **x**ᵢ:

    zᵢ = **x**ᵢ · **w** + b            (linear layer: "logit")
    pᵢ = σ(zᵢ) = 1 / (1 + e^(−zᵢ))     (sigmoid: probability of pCR)

Loss = mean binary cross-entropy (the negative log-likelihood of a Bernoulli model), plus an L2 penalty that shrinks **w** towards 0:

    L(**w**, b) = −(1/n) Σᵢ [ yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ) ]  +  λ/2 · ‖**w**‖²

The analytic gradient is ∂L/∂**w** = (1/n) Xᵀ(**p** − **y**) + λ**w**. In the script, autograd reproduces this to machine precision. That's how you learn to trust it.

**Numerical stability:** computing `log(sigmoid(z))` directly can give `log(0) = −inf` for large |z|. PyTorch's `binary_cross_entropy_with_logits` takes the **logit** z and uses the log-sum-exp trick. **Always pass logits, not probabilities**, to the loss.

### Matching scikit-learn exactly
scikit-learn's `LogisticRegression(penalty="l2", C=C)` minimises

    ½‖**w**‖² + C · Σᵢ logloss_i        (sum, not mean; intercept not penalised)

Divide by C·n and you get our loss with **λ = 1/(C·n)**. Same minimiser, so the coefficients must agree. Our test checks this: if they disagree, something in our code is wrong.

## 5. Why this matters for the project
- Elastic net or logistic regression on gene expression is the **baseline** that our deep models must beat (Grinsztajn et al., 2022). Now you know it's just one linear layer.
- With ~20,000 genes and ~1,000 patients (p ≫ n), the unpenalised model can separate the training data perfectly, so the weights blow up. The **penalty (λ) is essential**, and choosing it must happen inside cross-validation (lesson 03).

---

## ✅ Check yourself
1. `a = torch.ones(3, 1)` and `b = torch.arange(4.)`. What is `(a + b).shape`? Why?
2. You call `loss.backward()` twice without resetting `.grad`. What happens to `w.grad`?
3. Why do we give `binary_cross_entropy_with_logits` the logits `z` instead of `sigmoid(z)`?
4. scikit-learn uses `C = 0.5` and n = 200. What λ makes our loss equivalent?
5. Without the L2 penalty, with p = 20,000 genes and n = 990 patients, what happens to ‖**w**‖ during training, and why?

<details><summary>Answers</summary>

1. `(3, 4)`: `(3,1)` and `(4,)` → `(1,4)` stretch to `(3,4)`.
2. The two gradients are **added**, so `w.grad` is twice the true gradient and the next update step is wrong.
3. Numerical stability: the loss uses a stable log-sum-exp formula on logits, avoiding `log(0)`.
4. λ = 1/(C·n) = 1/(0.5·200) = 0.01.
5. When p ≫ n the classes are (almost surely) linearly separable. The loss keeps decreasing as ‖**w**‖ → ∞, so weights diverge and overfit. The penalty bounds them.
</details>

## References
- Paszke A et al. PyTorch: an imperative style, high-performance deep learning library. NeurIPS 2019. https://arxiv.org/abs/1912.01703
- Baydin AG, Pearlmutter BA, Radul AA, Siskind JM. Automatic differentiation in machine learning: a survey. *JMLR* 2018;18(153):1–43. https://jmlr.org/papers/v18/17-468.html
- Rumelhart DE, Hinton GE, Williams RJ. Learning representations by back-propagating errors. *Nature* 1986;323:533–536. https://doi.org/10.1038/323533a0
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*, 2nd ed., ch. 4.4 (logistic regression) and 18 (p ≫ n). Springer 2009. https://hastie.su.domains/ElemStatLearn/
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*, ch. 6.5 (back-propagation). MIT Press 2016. https://www.deeplearningbook.org/
- scikit-learn User Guide, Logistic regression (objective function). https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- PyTorch docs: Autograd mechanics. https://pytorch.org/docs/stable/notes/autograd.html · Broadcasting semantics. https://pytorch.org/docs/stable/notes/broadcasting.html
- Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on tabular data? NeurIPS 2022. https://arxiv.org/abs/2207.08815
