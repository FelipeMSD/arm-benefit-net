"""
Step 2a: download the I-SPY2-990 mRNA resource from GEO (GSE194040).

Files (public, NCBI GEO):
  * GSE194040_..._meanCol_geneLevel_n988.txt.gz   gene-level expression, 988 patients
    (two Agilent 44K platforms already combined by the authors; see Wolf et al. 2022, STAR Methods)
  * GSE194040-GPL20078_series_matrix.txt.gz        sample annotations (arm, HR, HER2, MP, pCR), platform 1
  * GSE194040-GPL30493_series_matrix.txt.gz        sample annotations, platform 2

Why record SHA-256 checksums? A checksum proves that the file analysed is
byte-for-byte the file downloaded. If GEO ever changes a file, the mismatch is
detected instead of silently changing the results (Wilson et al. 2017).

Usage (from the repo root):  python scripts/01_download_ispy2.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194040"
FILES = {
    "expression_gene_level.txt.gz":
        f"{BASE}/suppl/GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt.gz",
    "series_matrix_GPL20078.txt.gz": f"{BASE}/matrix/GSE194040-GPL20078_series_matrix.txt.gz",
    "series_matrix_GPL30493.txt.gz": f"{BASE}/matrix/GSE194040-GPL30493_series_matrix.txt.gz",
}
OUT = Path(__file__).resolve().parents[1] / "data_raw" / "ispy2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in FILES.items():
        dest = OUT / name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
        manifest[name] = {"url": url, "sha256": sha256(dest), "bytes": dest.stat().st_size}
        print(f"  {name}: {manifest[name]['bytes'] / 1e6:.1f} MB  sha256={manifest[name]['sha256'][:16]}…")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest written to {OUT / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
