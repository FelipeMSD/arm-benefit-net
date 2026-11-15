# Lesson 05: `nn.Module`, `DataLoader` and the training loop

**Goal:** go from lesson 01's hand-written gradient descent to the standard PyTorch toolkit, and understand each line of [`src/armbenefit/models/nets.py`](../src/armbenefit/models/nets.py).

## 1. `nn.Module`: a model is a class
```python
class PooledMLP(nn.Module):
    def __init__(self, n_in, hidden=64, dropout=0.5):
        super().__init__()                       # registers parameters, submodules, train/eval state
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_in, hidden), nn.ReLU(), ...)
    def forward(self, x):                        # called by model(x); never call forward directly
        return self.net(x).squeeze(-1)
```
- `nn.Linear(n_in, h)` holds a weight matrix (h × n_in) and a bias. They're created with `requires_grad=True`, which is lesson 01's `w` and `b`, now managed for you.
- `model.parameters()` yields every learnable tensor, and that's what the optimiser updates.
- **`nn.ModuleList`** (used for ArmNet's heads) registers a *list* of submodules. A plain Python list would hide them from `.parameters()`, and those heads would never train. That's a classic silent bug.

## 2. The loop (compare with lesson 01)
```python
for epoch in range(max_epochs):
    model.train()                               # dropout ON
    for x, y in loader:                         # mini-batches of 64 patients, reshuffled each epoch
        opt.zero_grad()                         # = w.grad = None
        loss = loss_fn(model, x, y)             # forward
        loss.backward()                         # autograd
        opt.step()                              # = w -= lr * w.grad (AdamW version)
    model.eval()                                # dropout OFF
    with torch.no_grad(): val = loss_fn(model, *val_tensors)
    # early stopping: keep the best weights; stop after `patience` epochs without improvement
```
- **`DataLoader`** batches and shuffles. Mini-batch gradients are noisy estimates of the full gradient, and that noise also regularises (stochastic gradient descent).
- **AdamW** adapts the step size per parameter using running gradient moments (Kingma & Ba, 2015). It applies weight decay *decoupled* from the gradient, which works better than L2-in-the-loss for Adam (Loshchilov & Hutter, 2019).
- **Dropout** randomly zeroes units during training, so the network can't rely on any single gene or unit (Srivastava et al., 2014). It must be OFF at evaluation, which is why `model.eval()` matters.
- **Early stopping** on a validation split is a regulariser in its own right (Prechelt, 1998).
- With n ≈ 600 training patients and 2,000 genes, the network has ~130k parameters for ~600 labels. **Regularisation is the whole game:** input dropout, dropout, weight decay, early stopping, and a small architecture.

## 3. Hyperparameters chosen by inner CV
Dropout ∈ {0.3, 0.6} × weight decay ∈ {1e-3, 1e-2}, chosen per outer fold by inner 3-fold CV on factual log-loss (`tune_torch` in `scripts/04_benchmark_ispy2.py`). The chosen values are saved in `results/tuning_choices.csv`.

## ✅ Check yourself
1. What breaks if you forget `model.eval()` before computing validation loss?
2. Why does `nn.ModuleList` matter for ArmNet?
3. Name four regularisers used here, and say which one also decides *when to stop*.

<details><summary>Answers</summary>

1. Dropout stays on, so validation predictions are noisy and pessimistic. Early stopping then reacts to noise.
2. Without it, the per-arm heads aren't registered: `.parameters()` misses them, so AdamW never updates them and `state_dict()` doesn't save them.
3. Input dropout, hidden dropout, weight decay (AdamW) and early stopping. Early stopping decides when to stop.
</details>

## References
- Paszke A et al. PyTorch. NeurIPS 2019. https://arxiv.org/abs/1912.01703
- Kingma DP, Ba J. Adam: a method for stochastic optimization. ICLR 2015. https://arxiv.org/abs/1412.6980
- Loshchilov I, Hutter F. Decoupled weight decay regularization. ICLR 2019. https://arxiv.org/abs/1711.05101
- Srivastava N et al. Dropout: a simple way to prevent neural networks from overfitting. *JMLR* 2014;15:1929–1958.
- Prechelt L. Early stopping, but when? In *Neural Networks: Tricks of the Trade*. Springer 1998. https://doi.org/10.1007/3-540-49430-8_3
- PyTorch tutorial: Learning PyTorch with examples. https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
