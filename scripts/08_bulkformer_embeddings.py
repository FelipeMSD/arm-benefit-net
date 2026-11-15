"""
Step 8: sample embeddings of IMvigor210 from the BulkFormer foundation model.

BulkFormer (Kang, Fan et al., Cell Systems 2026; bioRxiv 2025) is a transformer
pretrained on ~500k bulk RNA-seq profiles to reconstruct masked gene expression. Its
per-gene embeddings, averaged over genes, give a 131-dimensional sample vector
(dim 128 + 3). We freeze the model (no fine-tuning, no labels), so the embeddings
are a fixed, label-free transformation, safe to compute on all patients before CV.

Runs in the SEPARATE conda env `bulkformer` (needs torch>=2.6 and transformers 5.x,
which would conflict with the main env):
    conda create -n bulkformer python=3.11 pip
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.6.0
    pip install "git+https://github.com/zhfanrui/BulkFormer_hf.git"
    python scripts/08_bulkformer_embeddings.py [--n 4]   # --n to time a few samples first

Input requirements (from the package): raw counts, columns = Ensembl gene IDs.
IMvigor210 uses Entrez IDs, so we map Entrez -> Ensembl with the HGNC complete set.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bulkformer import BulkFormerModel, extract_features
from bulkformer.preprocessing import align_to_vocab, normalize_data

ROOT = Path(__file__).resolve().parents[1]
IMV = ROOT / "data_processed" / "imvigor210"
HGNC = ROOT / "data_raw" / "reference" / "hgnc_complete_set.txt"
MODEL_ID = "zhfanrui/BulkFormer-37M"


def entrez_to_ensembl_counts() -> pd.DataFrame:
    counts = pd.read_csv(IMV / "counts.csv.gz", index_col=0)            # Entrez × patients
    counts.index = counts.index.astype(str)
    hg = pd.read_csv(HGNC, sep="\t", usecols=["entrez_id", "ensembl_gene_id"], dtype=str).dropna()
    hg["entrez_id"] = hg["entrez_id"].str.replace(r"\.0$", "", regex=True)
    m = hg.drop_duplicates("entrez_id").set_index("entrez_id")["ensembl_gene_id"]
    ens = counts.index.map(m)
    counts = counts[ens.notna()]
    counts.index = ens.dropna()
    counts = counts.groupby(level=0).sum()                                  # several Entrez -> one ENSG
    return counts.T                                                         # patients × ENSG


def main(n: int | None) -> None:
    torch.set_num_threads(4)
    X = entrez_to_ensembl_counts()
    if n:
        X = X.iloc[:n]
    model = BulkFormerModel.from_pretrained(MODEL_ID)
    log_tpm = normalize_data(X)
    aligned, mask_prob = align_to_vocab(log_tpm, model.config.gene_ids)
    print(f"{len(X)} samples; vocabulary {len(model.config.gene_ids)} genes; "
          f"missing (masked) fraction = {mask_prob:.3f}")
    t0 = time.time()
    emb = extract_features(model, aligned.values.astype(np.float32), output_feature_type="sample_level",
                           aggregate_type="mean", mask_prob=mask_prob, device="cpu", batch_size=4)
    emb = emb.cpu().numpy() if hasattr(emb, "cpu") else np.asarray(emb)
    print(f"embedding shape {emb.shape}; {time.time() - t0:.1f}s ({(time.time() - t0) / len(X):.2f}s/sample)")
    out = pd.DataFrame(emb, index=X.index, columns=[f"bf{i}" for i in range(emb.shape[1])])
    name = "bulkformer_embeddings.csv.gz" if not n else "bulkformer_embeddings_test.csv.gz"
    out.to_csv(IMV / name, float_format="%.6g")
    (IMV / "bulkformer_run.txt").write_text(f"model={MODEL_ID}\nmask_prob={mask_prob:.4f}\nn={len(X)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    main(ap.parse_args().n)
