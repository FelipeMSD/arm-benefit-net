"""
Lesson 01: Tensors and autograd (hands-on script)
=================================================

Run from the repository root:

    conda activate armbenefit
    python lessons/01_tensors_autograd.py

Read it top to bottom alongside `lessons/01_tensors_autograd.md`.
Each PART prints what it demonstrates, so you can compare the code with
the output. Nothing here downloads data: we use a small SYNTHETIC
"expression matrix" so the lesson runs in seconds and anyone can repeat it.

PARTS
  1. Creating tensors: shape, dtype, device
  2. NumPy interop: shared memory vs copies
  3. Broadcasting, and the (n,) vs (n, 1) bug
  4. Autograd on a toy function: requires_grad, backward, grad accumulation
  5. Logistic regression BY HAND: gradient descent with autograd
  6. Check 1: autograd gradient == analytic gradient
  7. Check 2: our coefficients == scikit-learn's coefficients
"""

# --- Imports -----------------------------------------------------------------
# numpy : the array library most of scientific Python is built on.
# torch : PyTorch. Tensors + autograd + neural-network tools.
# sklearn.linear_model.LogisticRegression : the reference implementation we
#         will try to reproduce exactly. That's how we validate our understanding.
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def banner(title: str) -> None:
    """Print a visible section header so the console output is easy to follow."""
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --- Reproducibility -----------------------------------------------------------
# Fix every random number generator we use. Same seed + same library versions
# + same machine -> identical numbers on every run (see lesson 00, section 6).
SEED = 2026
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# =============================================================================
banner("PART 1: Creating tensors: shape, dtype, device")
# =============================================================================
# A 2-D tensor is a matrix. Think: rows = patients, columns = genes.
X_toy = torch.tensor([[2.1, 0.3, 5.0],
                      [1.8, 0.1, 4.2]])          # 2 patients x 3 genes
print("X_toy =\n", X_toy)
print("shape :", X_toy.shape)    # torch.Size([2, 3])
print("dtype :", X_toy.dtype)    # torch.float32 is the default for decimals
print("device:", X_toy.device)   # cpu (we have no usable GPU)

# Common constructors. Each one is used later in the course.
zeros = torch.zeros(2, 3)        # e.g. initialise an accumulator
ones = torch.ones(3)             # e.g. a vector of weights = 1
ar = torch.arange(5)             # 0,1,2,3,4: integer dtype (int64)
rnd = torch.randn(2, 3)          # standard-normal noise: weight initialisation
print("arange dtype:", ar.dtype, "| randn sample:\n", rnd)

# Changing dtype. float64 (double precision) is slower but more precise. We
# use it in this lesson so we can compare against scikit-learn to ~1e-6.
X64 = X_toy.to(torch.float64)
print("converted dtype:", X64.dtype)

# Indexing works like NumPy: [row, column]
print("gene 3 of patient 1:", X_toy[0, 2].item())   # .item() -> Python float
print("all patients, gene 1:", X_toy[:, 0])           # a column -> shape (2,)

# Reductions along a dimension: dim=0 collapses rows (one value per gene),
# dim=1 collapses columns (one value per patient).
print("mean per gene   (dim=0):", X_toy.mean(dim=0))
print("mean per patient(dim=1):", X_toy.mean(dim=1))

# =============================================================================
banner("PART 2: NumPy interop: shared memory vs copies")
# =============================================================================
a = np.array([1.0, 2.0, 3.0])
shared = torch.from_numpy(a)     # SHARES memory with `a` (no copy, fast)
copied = torch.tensor(a)         # COPIES the data (independent)
a[0] = 100.0                     # modify the NumPy array...
print("numpy a      :", a)
print("from_numpy   :", shared, " <- changed too (shared memory!)")
print("torch.tensor :", copied, "   <- unchanged (a copy)")
# Lesson: use torch.from_numpy for speed only if you won't modify the
# source array afterwards. Otherwise, copy.

# =============================================================================
banner("PART 3: Broadcasting, and the (n,) vs (n, 1) bug")
# =============================================================================
# Standardising genes (z-scores) is a textbook use of broadcasting:
#   X: (n, p);  X.mean(dim=0): (p,)  -> treated as (1, p) -> stretched to (n, p)
X = torch.randn(4, 3) * 10 + 5
Z = (X - X.mean(dim=0)) / X.std(dim=0)
print("Z-scored column means (≈0):", Z.mean(dim=0))
print("Z-scored column stds  (≈1):", Z.std(dim=0))

# THE BUG: labels y have shape (n,), predictions have shape (n, 1).
y = torch.tensor([1.0, 0.0, 1.0, 0.0])          # shape (4,)
pred = torch.tensor([[0.9], [0.2], [0.8], [0.1]])  # shape (4, 1)
wrong = (y - pred)                                # broadcasts to (4, 4)!
right = (y - pred.squeeze(1))                     # squeeze -> (4,), correct
print("shape of y - pred (WRONG):", tuple(wrong.shape), "<- silent bug, no error raised")
print("shape after squeeze(1)   :", tuple(right.shape))
# Defensive habit we'll use everywhere: assert shapes in your functions.
assert right.shape == y.shape

# =============================================================================
banner("PART 4: Autograd on a toy function")
# =============================================================================
# f(w) = (3w - 2)^2.  By hand: df/dw = 2*(3w - 2)*3 = 18w - 12.
# At w = 1: df/dw = 6.
w = torch.tensor(1.0, requires_grad=True)   # "track operations on w"
f = (3 * w - 2) ** 2                        # builds a graph: mul -> sub -> pow
print("f =", f.item(), "| f.grad_fn =", f.grad_fn)  # the last op that made f
f.backward()                                # chain rule, backwards through the graph
print("autograd df/dw at w=1:", w.grad.item(), "(by hand: 6.0)")

# Gradients ACCUMULATE: calling backward again ADDS to w.grad.
f = (3 * w - 2) ** 2
f.backward()
print("after a 2nd backward without reset:", w.grad.item(), "<- 12, not 6!")
w.grad = None                               # reset (optimizer.zero_grad() does this later)
f = (3 * w - 2) ** 2
f.backward()
print("after resetting w.grad = None     :", w.grad.item())

# =============================================================================
banner("PART 5: Logistic regression BY HAND with gradient descent")
# =============================================================================
# Synthetic stand-in for our real problem:
#   n = 200 "patients", p = 10 "genes", binary outcome ("pCR" yes/no).
# Only genes 0, 1 and 2 truly matter. This mimics a sparse biomarker signal.
n, p = 200, 10
X_np = rng.normal(size=(n, p))
true_w = np.array([1.5, -2.0, 1.0] + [0.0] * (p - 3))
true_b = -0.5
prob = 1 / (1 + np.exp(-(X_np @ true_w + true_b)))
y_np = rng.binomial(1, prob).astype(np.float64)
print(f"synthetic data: X {X_np.shape}, pCR rate = {y_np.mean():.2f}")

# Convert to float64 tensors (copies, so independent of the NumPy arrays).
Xt = torch.tensor(X_np, dtype=torch.float64)
yt = torch.tensor(y_np, dtype=torch.float64)

# Penalty strength. We pick C as scikit-learn would, then convert it to our
# λ so that both objectives have the SAME minimiser:  λ = 1 / (C · n).
C = 1.0
lam = 1.0 / (C * n)

# PARAMETERS: the things we learn. Start at zero; requires_grad=True so
# autograd computes ∂loss/∂w and ∂loss/∂b for us.
w = torch.zeros(p, dtype=torch.float64, requires_grad=True)
b = torch.zeros(1, dtype=torch.float64, requires_grad=True)


def loss_fn(w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy + L2 penalty on w (the intercept b is NOT penalised).

    We pass LOGITS (z) to binary_cross_entropy_with_logits, not probabilities:
    it uses a numerically stable log-sum-exp formula (no log(0) = -inf).
    """
    z = Xt @ w + b                     # (n, p) @ (p,) -> (n,), plus b broadcast
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, yt)  # mean over n
    return bce + 0.5 * lam * (w ** 2).sum()


lr = 0.5            # learning rate η: step size of gradient descent
n_steps = 3000
for step in range(n_steps):
    loss = loss_fn(w, b)       # 1) FORWARD: compute the loss (graph is built)
    w.grad, b.grad = None, None  # 2) reset old gradients (they accumulate!)
    loss.backward()            # 3) BACKWARD: autograd fills w.grad and b.grad
    with torch.no_grad():      # 4) UPDATE outside the graph
        w -= lr * w.grad
        b -= lr * b.grad
    if step in (0, 10, 100, 1000, n_steps - 1):
        print(f"step {step:5d} | loss = {loss.item():.6f} | ‖grad w‖ = {w.grad.norm():.2e}")

print("learned w (first 5):", np.round(w.detach().numpy()[:5], 3))
print("true    w (first 5):", true_w[:5])
# The learned weights are shrunk towards 0 by the penalty and by noise,
# but genes 0–2 clearly stand out. That's the biomarker signal.

# =============================================================================
banner("PART 6: Check 1: autograd gradient == analytic gradient")
# =============================================================================
# Analytic gradient of our loss:  ∂L/∂w = (1/n) Xᵀ(p − y) + λ w
# Evaluate both at a RANDOM point (not the optimum, where both are ~0).
w_test = torch.randn(p, dtype=torch.float64, requires_grad=True)
b_test = torch.randn(1, dtype=torch.float64, requires_grad=True)
loss_fn(w_test, b_test).backward()
with torch.no_grad():
    p_hat = torch.sigmoid(Xt @ w_test + b_test)
    analytic = Xt.T @ (p_hat - yt) / n + lam * w_test
max_diff = (w_test.grad - analytic).abs().max().item()
print(f"max |autograd − analytic| = {max_diff:.2e}  (machine precision ≈ 1e-16)")

# =============================================================================
banner("PART 7: Check 2: our coefficients == scikit-learn's")
# =============================================================================
# scikit-learn minimises  ½‖w‖² + C·Σ logloss   (equivalent to ours, see .md §4).
# Tight tolerance so the reference itself is well converged.
sk = LogisticRegression(penalty="l2", C=C, tol=1e-12, max_iter=10_000)
sk.fit(X_np, y_np)
diff_w = np.abs(sk.coef_.ravel() - w.detach().numpy()).max()
diff_b = abs(sk.intercept_[0] - b.item())
print(f"max |w_ours − w_sklearn| = {diff_w:.2e}")
print(f"    |b_ours − b_sklearn| = {diff_b:.2e}")
print("MATCH ✔" if max(diff_w, diff_b) < 1e-4 else "MISMATCH ✘: check λ, steps, lr")
print("\nTakeaway: logistic regression = one linear layer + sigmoid + BCE loss.")
print("Every later model in this course is the same loop with a bigger network.")
