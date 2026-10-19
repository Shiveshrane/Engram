"""Test 3 (design §6): parameter table for every arm, and parameter-matching invariants (design §5.3).

Run with -s to see the tables.
"""
import pytest

from engramdelta.scripts.param_table import format_table, param_rows

MATCHED = {"A1", "A5", "T1", "T2", "LA", "A1_noGconv"}


@pytest.mark.parametrize("phase,vocab", [("phase0", None), ("phase1", None), ("phase2", 16384)])
def test_param_table(phase, vocab):
    rows = {r["arm"]: r for r in param_rows(phase, vocab=vocab)}
    print(f"\n{phase}\n" + format_table(list(rows.values())))
    a2 = rows["A2"]
    for arm, r in rows.items():
        if arm in MATCHED:
            rel = abs(r["non_embedding"] / a2["non_embedding"] - 1)
            assert rel <= 0.02, f"{phase} {arm}: non-embedding params {r['non_embedding']} vs A2 {a2['non_embedding']}"
    if "A3" in rows:
        assert rows["A3"]["total"] == a2["total"] and rows["A3"]["trainable"] == a2["total"] - a2["engram_table"]
    if "A4" in rows:
        assert rows["A4"]["engram_table"] == a2["engram_table"]
        assert rows["A4"]["non_embedding"] == a2["non_embedding"]
    ratio = a2["engram_table"] / rows["A1"]["non_embedding"]
    assert 0.5 <= ratio <= 1.5, f"{phase}: Engram table is {ratio:.2f}x the backbone (design: about 1x)"
