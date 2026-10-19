"""Test 7 (design §6) and the scaled-up test 6: checks on the Phase 0 Kaggle runs (E0.4 known result, E0.5 leak).

These read finished result JSONs. The training happens in NB02. They are skipped until those results exist:
    ENGRAMDELTA_RESULTS=results python -m pytest tests/test_repro_known.py -s
Pre-registered criteria (PREREG.md): A1 (GDN) beats vanilla linear attention by > 2 pooled seed std at >= 2
values of N; A1 recall is non-increasing in N beyond the knee; A6 (no mixer) recall < 1% at every N.
"""
import os

import pytest

from engramdelta import analysis

RESULTS = os.environ.get("ENGRAMDELTA_RESULTS", "results")


@pytest.fixture(scope="module")
def df():
    d = analysis.load(RESULTS)
    if d.empty or "E0.4" not in set(d["group"]):
        pytest.skip("no E0.4 results yet (run notebooks/02_phase0_known_result)")
    return d


def test_gdn_beats_linear_attention_and_degrades_with_N(df):
    chk = analysis.known_result_check(df)
    print(chk["table"].to_string())
    print("N_star =", chk["N_star"])
    assert chk["criterion_1_pass"], "GDN does not clearly beat vanilla linear attention: the pipeline is suspect"
    assert chk["criterion_2_pass"], "GDN recall does not degrade with N beyond the knee"


def test_engram_only_probe_at_chance(df):
    t = analysis.leak_table(df)
    t = t[t["group"].isin(["E0.5", "E1.2l"])]
    if t.empty:
        pytest.skip("no E0.5 results yet")
    print(t.to_string())
    assert t["pass_lt_1pct"].all(), "A6 above 1% recall: possible answer leakage"
