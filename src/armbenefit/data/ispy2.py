"""
I-SPY2-990 (GEO GSE194040) loaders.

Design choices (and why):
  * Patient ID is the key everywhere. Splits are made at the patient level so no
    patient can appear in both training and test data (patient-level validation).
  * The two Agilent platforms (GPL20078, GPL30493) are kept as a `platform`
    column. The authors already harmonised them (ComBat / linear factors; Wolf et
    al. 2022). That adjustment was fitted on ALL samples, a mild form of
    leakage we cannot undo. We report it as a limitation and test robustness by
    using the platform as a stratification and sensitivity variable.
  * Arms get short, stable codes, so models and plots never depend on long strings.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[3] / "data_raw" / "ispy2"
PROCESSED = Path(__file__).resolve().parents[3] / "data_processed"

# Short arm codes. "Paclitaxel" alone is the shared control arm.
ARM_CODES = {
    "Paclitaxel": "CTRL",
    "Paclitaxel + Neratinib": "NERA",
    "Paclitaxel + MK-2206": "MK2206",
    "Paclitaxel + MK-2206 + Trastuzumab": "MK2206_T",
    "Paclitaxel + Ganitumab": "GANI",
    "Paclitaxel + Ganetespib": "GANE",
    "Paclitaxel + Pembrolizumab": "PEMBRO",
    "Paclitaxel + Durvalumab + Olaparib": "DURVA_OLA",
    "Paclitaxel + Trastuzumab": "CTRL_HER2",
    "Paclitaxel + Pertuzumab + Trastuzumab": "PERTU_T",
    "Paclitaxel + Trastuzumab + Pertuzumab": "PERTU_T",
    "T-DM1 + Pertuzumab": "TDM1_P",
    "Paclitaxel + ABT 888 + Carboplatin": "VELI_CARBO",
    "Paclitaxel + ABT-888 + Carboplatin": "VELI_CARBO",
    "Paclitaxel + AMG 386": "AMG386",
    "Paclitaxel + AMG-386": "AMG386",       # one sample uses a hyphen in GEO
    "Paclitaxel + AMG 386 + Trastuzumab": "AMG386_T",
}


def _parse_series_matrix(path: Path, platform: str) -> pd.DataFrame:
    """Read the '!Sample_characteristics_ch1' lines ("key: value") into a table."""
    rows: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                rows["gsm"] = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                key = vals[0].split(":", 1)[0].strip()
                rows[key] = [v.split(":", 1)[1].strip() if ":" in v else np.nan for v in vals]
    df = pd.DataFrame(rows)
    df["platform"] = platform
    return df


def load_clinical(raw_dir: Path = RAW) -> pd.DataFrame:
    """One row per patient: patient_id, gsm, platform, hr, her2, mp, pcr, arm, arm_code."""
    parts = [
        _parse_series_matrix(raw_dir / "series_matrix_GPL20078.txt.gz", "GPL20078"),
        _parse_series_matrix(raw_dir / "series_matrix_GPL30493.txt.gz", "GPL30493"),
    ]
    df = pd.concat(parts, ignore_index=True).rename(
        columns={"patient id": "patient_id", "adjustment method": "adjustment"})
    for col in ("hr", "her2", "mp", "pcr"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["patient_id"] = df["patient_id"].astype(str)
    df["arm_code"] = df["arm"].map(ARM_CODES)
    unknown = df.loc[df["arm_code"].isna(), "arm"].unique()
    if len(unknown):
        raise ValueError(f"unmapped arm names: {unknown}")
    # Receptor subtype used throughout (I-SPY2 randomises within HR/HER2/MP strata).
    df["subtype"] = np.select(
        [df.her2.eq(1) & df.hr.eq(1), df.her2.eq(1) & df.hr.eq(0),
         df.her2.eq(0) & df.hr.eq(1), df.her2.eq(0) & df.hr.eq(0)],
        ["HR+HER2+", "HR-HER2+", "HR+HER2-", "TNBC"], default="NA")
    return df.drop(columns=[c for c in ("tissue",) if c in df]).set_index("patient_id")


def load_expression(raw_dir: Path = RAW) -> pd.DataFrame:
    """Gene-level log2 expression as a patients × genes DataFrame (float32)."""
    expr = pd.read_csv(raw_dir / "expression_gene_level.txt.gz", sep="\t", index_col=0)
    expr.columns = expr.columns.astype(str)
    expr = expr.T.astype(np.float32)            # rows = patients, columns = genes
    expr.index.name = "patient_id"
    return expr


def load_ispy2(raw_dir: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clinical, expression) restricted to patients present in both, same order."""
    clin, expr = load_clinical(raw_dir), load_expression(raw_dir)
    ids = clin.index.intersection(expr.index)
    clin = clin.loc[ids]
    clin = clin[clin["pcr"].notna()]
    return clin, expr.loc[clin.index]
