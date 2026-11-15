"""Data integrity tests for I-SPY2. Skipped automatically when the raw data isn't downloaded (e.g. in CI)."""

import pytest

from armbenefit.data.ispy2 import RAW, load_ispy2

pytestmark = pytest.mark.skipif(not (RAW / "expression_gene_level.txt.gz").exists(),
                                reason="I-SPY2 raw data not downloaded")


def test_shapes_and_alignment():
    clin, expr = load_ispy2()
    assert len(clin) == 986
    assert (clin.index == expr.index).all()          # same patients, same order
    assert clin.index.is_unique                       # one row per patient


def test_labels_and_arms():
    clin, _ = load_ispy2()
    assert set(clin.pcr.unique()) <= {0, 1}
    assert clin.arm_code.notna().all()
    assert (clin[clin.her2 == 0].arm_code == "CTRL").sum() == 179
