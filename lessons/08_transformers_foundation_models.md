# Lesson 08: Transformers, `huggingface_hub` and a transcriptomic foundation model

**Goal:** understand attention, load a pretrained biological foundation model from the Hugging Face Hub, and use it honestly as a feature extractor.
Script: [`scripts/08_bulkformer_embeddings.py`](../scripts/08_bulkformer_embeddings.py) (separate env `bulkformer`). Its use in models: [`scripts/07_imvigor210.py`](../scripts/07_imvigor210.py).

## 1. Attention in five lines
A transformer turns each input token into three vectors: **query** *q*, **key** *k* and **value** *v* (learned linear maps). Each token's output is a weighted average of all values, with weights from query–key similarity:

    Attention(Q, K, V) = softmax(Q Kᵀ / √d) · V          (Vaswani et al., 2017)

For gene expression, a "token" is a **gene** (gene identity embedding + its expression value). Attention lets every gene's representation depend on every other gene's, so it can learn co-regulation context. Full attention over ~20,000 genes costs O(n²). BulkFormer therefore combines a graph network over a gene–gene graph with **Performer** linear-attention layers (Choromanski et al., 2021).

```python
import torch, torch.nn.functional as F
q = k = v = torch.randn(1, 5, 8)                               # 1 sample, 5 "genes", 8 dims
w = F.softmax(q @ k.transpose(1, 2) / 8**0.5, dim=-1)           # (1, 5, 5) attention weights
out = w @ v                                                     # each gene = mix of all genes
torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v))   # PyTorch's built-in agrees
```

## 2. Foundation models for bulk transcriptomes
A **foundation model** is pretrained on a very large unlabelled corpus with a self-supervised task, then reused for many downstream tasks (Bommasani et al., 2021). Here:
- **BulkFormer:** pretrained on ~500k human bulk RNA-seq profiles to reconstruct **masked** expression values (like BERT's masked tokens).
- **scGPT** (Cui et al., 2024) does the same for single cells. **DNABERT** (Ji et al., 2021) does it for DNA sequence. These are the three best-known biological foundation models for expression and sequence data.
- **Usage modes:** (i) **frozen feature extractor** (embeddings into a simple model), which we use here, because n = 348 is too small to fine-tune 37M parameters without severe overfitting; (ii) fine-tuning, for larger cohorts.

## 3. `huggingface_hub` and `transformers` in practice
```python
from bulkformer import BulkFormerModel, extract_features
model = BulkFormerModel.from_pretrained("zhfanrui/BulkFormer-37M")   # downloads + caches weights (safetensors)
model.config.gene_ids        # the 20,010 Ensembl IDs, in the order the model expects
```
`from_pretrained` is the `transformers` convention. Under the hood, `huggingface_hub` downloads `config.json` and `model.safetensors` into a local cache (`~/.cache/huggingface`). **Safetensors** stores only tensors, unlike pickle-based `.pt` files, which can run arbitrary code when loaded.

**Input contract**, read from the package source rather than assumed:
1. raw counts with Ensembl IDs. IMvigor210 has Entrez IDs, so we map them with HGNC;
2. `normalize_data`: counts → TPM (gene-length corrected) → log1p;
3. `align_to_vocab`: reorder to the vocabulary, and fill missing genes with the mask value −10 (4.3% missing here);
4. `extract_features(..., "sample_level", aggregate_type="mean")`: mean of per-gene embeddings, giving a 131-dimensional vector per patient.

**Why not apply it to I-SPY2?** I-SPY2 is a microarray (log2 intensities, no counts, no gene-length meaning). Feeding it to a model pretrained on RNA-seq TPM is a domain shift with no principled correction. Knowing **when not to use** a foundation model is part of using it well.

## 4. Engineering lesson: isolate conflicting dependencies
The BulkFormer package pulled `transformers` 5.x, which needs `torch>=2.6`, while the project is pinned to torch 2.5.1. Separate conda environments keep both reproducible, and only a CSV of embeddings crosses between them.

## ✅ Check yourself
1. Why divide by √d inside the softmax?
2. Why are frozen embeddings computed on *all* patients before cross-validation not leakage?
3. Give one reason a foundation-model embedding might predict *worse* than 1,000 selected genes on a specific clinical task.

<details><summary>Answers</summary>

1. Dot products grow with dimension d. Large logits saturate the softmax into near one-hot weights with vanishing gradients, and scaling keeps their variance about 1.
2. The model is frozen, uses no labels, and processes each sample independently. It's a fixed function of one patient's own data, like log-transforming.
3. The pretraining objective (reconstruct expression) doesn't target the clinical signal. Mean pooling over 20k genes can dilute a few key immune genes, and the cohort (tumour type, library prep) may differ from the pretraining data.
</details>

## References
- Vaswani A et al. Attention is all you need. NeurIPS 2017. https://arxiv.org/abs/1706.03762
- Choromanski K et al. Rethinking attention with Performers. ICLR 2021. https://arxiv.org/abs/2009.14794
- Bommasani R et al. On the opportunities and risks of foundation models. 2021. https://arxiv.org/abs/2108.07258
- Kang B, Fan R, Yi M, Cui C, Cui Q. BulkFormer: a large-scale foundation model for bulk transcriptomes. *Cell Systems* 2026;17:101657. doi:10.1016/j.cels.2026.101657 (preprint: bioRxiv 2025, doi:10.1101/2025.06.11.659222). Weights: https://huggingface.co/zhfanrui/BulkFormer-37M (MIT)
- Cui H et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nat Methods* 2024;21:1470–1480. https://doi.org/10.1038/s41592-024-02201-0
- Ji Y et al. DNABERT: pre-trained bidirectional encoder representations from transformers model for DNA-language in genome. *Bioinformatics* 2021;37:2112–2120. https://doi.org/10.1093/bioinformatics/btab083
- Wolf T et al. Transformers: state-of-the-art natural language processing. EMNLP 2020 (Demos). https://doi.org/10.18653/v1/2020.emnlp-demos.6
- Hugging Face safetensors documentation. https://huggingface.co/docs/safetensors
