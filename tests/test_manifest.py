"""The experiment manifests produce exactly the run counts of the one-week plan (§6): Phase 1 = 167 runs."""
from engramdelta.scripts.make_manifest import build

PREREG = {"N_star": 64, "phase1_steps": 2000, "seeds": [0, 1, 2], "lr_candidates": [3e-4, 1e-3, 3e-3],
          "lr": {"A1": 1e-3, "A2": 1e-3, "T1": 3e-4},
          "primary_cell": {"F": 512, "P": 64, "placement": "a", "filler_weight": 1.0},
          "phase2_train_tokens": 120_000_000, "phase2_hours_per_run": 1.7, "phase2_max_tokens": 150_000_000}

EXPECTED = {"E0.3": 2, "E0.4": 35, "E0.5": 5, "E0.6": 2, "E1.0a": 9, "E1.0c": 6, "E1.1": 54, "E1.2": 78,
            "E1.2r": 6, "E1.2l": 2, "E1.3": 12, "E2.0": 2, "E2.1": 3, "E2.2": 2}


def test_run_counts(tmp_path):
    counts = {g: len(build(g, PREREG, str(tmp_path))) for g in EXPECTED}
    assert counts == EXPECTED
    assert sum(counts[g] for g in EXPECTED if g.startswith("E1")) == 167


def test_run_ids_unique_and_cells_correct(tmp_path):
    runs = [r for g in EXPECTED for r in build(g, PREREG, str(tmp_path))]
    ids = [r["run_id"] for r in runs]
    assert len(ids) == len(set(ids))
    e11 = build("E1.1", PREREG, str(tmp_path))
    assert all("data.filler_weight=1.0" in r["set"] for r in e11)
    assert {s for r in e11 for s in r["set"] if s.startswith("data.N=")} == {"data.N=64", "data.N=128"}
    e10c = build("E1.0c", PREREG, str(tmp_path))
    assert all("data.filler_weight=0.0" in r["set"] for r in e10c)
    a3 = [r for r in build("E1.2", PREREG, str(tmp_path)) if "/A3.yaml" in r["config"]]
    assert all("train.lr=0.001" in r["set"] for r in a3), "A3 must use A2's learning rate"


def test_phase0_steps_and_e06_cell(tmp_path):
    for g in ("E0.4", "E0.5"):
        assert all("train.steps=4000" in r["set"] for r in build(g, PREREG, str(tmp_path)))
    e06 = build("E0.6", dict(PREREG, N_star=32), str(tmp_path))
    assert {s for r in e06 for s in r["set"] if s.startswith(("data.N=", "data.F="))} == {"data.N=32", "data.F=512"}


def test_e06_needs_n_star(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        build("E0.6", dict(PREREG, N_star=None), str(tmp_path))
