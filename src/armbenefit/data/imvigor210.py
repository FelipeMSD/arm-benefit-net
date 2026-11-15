"""
IMvigor210 (atezolizumab, metastatic urothelial carcinoma; Mariathasan et al., Nature 2018).

Exported once from the IMvigor210CoreBiologies R package by scripts/03_export_imvigor210.R.
Expression: raw RNA-seq counts -> log2(CPM + 1). CPM is a per-sample normalisation
(it doesn't use other samples), so it can be computed before any split without leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[3] / "data_processed" / "imvigor210"


def load_imvigor210(directory: Path = DIR, min_mean_cpm: float = 1.0):
    """Return (clinical, log2CPM expression with gene SYMBOL columns, raw counts with Entrez columns, features)."""
    counts = pd.read_csv(directory / "counts.csv.gz", index_col=0)          # genes × patients (Entrez IDs)
    pheno = pd.read_csv(directory / "pheno.csv.gz", index_col=0)
    feat = pd.read_csv(directory / "features.csv.gz", index_col=0)
    counts.index = counts.index.astype(str)
    feat.index = feat.index.astype(str)

    cpm = counts / counts.sum(axis=0) * 1e6
    keep = cpm.mean(axis=1) >= min_mean_cpm                                  # expressed genes only
    logcpm = np.log2(cpm[keep] + 1)
    sym = feat.loc[logcpm.index, "symbol"]
    logcpm = logcpm[sym.notna().to_numpy()]
    logcpm.index = sym.dropna().to_numpy()
    logcpm = logcpm.groupby(level=0).mean().T.astype(np.float32)             # patients × symbols

    clin = pd.DataFrame(index=pheno.index)
    clin["response"] = pheno["binaryResponse"].map({"CR/PR": 1, "SD/PD": 0})
    clin["os_months"] = pheno["os"]
    clin["os_event"] = pheno["censOS"]
    clin["tmb"] = pheno["FMOne mutation burden per MB"]
    clin["ecog"] = pheno["Baseline ECOG Score"]
    clin["immune_phenotype"] = pheno["Immune phenotype"]
    clin["ic_level"] = pheno["IC Level"]
    clin["tissue"] = pheno["Tissue"]
    return clin, logcpm.loc[clin.index], counts.T.loc[clin.index], feat


def signature_genes(directory: Path = DIR) -> dict[str, list[str]]:
    """Gene signatures shipped with IMvigor210CoreBiologies, mapped from Entrez to symbols."""
    sig = pd.read_csv(directory / "signatures.csv.gz")
    feat = pd.read_csv(directory / "features.csv.gz", index_col=0)
    feat.index = feat.index.astype(str)
    sig["gene"] = sig["gene"].astype(str)
    sym = sig["gene"].map(feat["symbol"]).fillna(sig["gene"])     # some entries may already be symbols
    sig = sig.assign(symbol=sym)
    return {k: sorted(set(v)) for k, v in sig.groupby("signature")["symbol"]}
