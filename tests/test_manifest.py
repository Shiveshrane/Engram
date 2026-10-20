"""The experiment manifests produce exactly the run counts of the one-week plan (option A, PREREG.md 2026-10-20):
Phase 1 = 67 runs at N = N*, placement a, primary cell F = 512 with the answer-only loss (lambda = 0)."""
import pytest

from engramdelta.scripts.make_manifest import build

PREREG = {"N_star": 32, "phase1_steps": 6000, "seeds": [0, 1, 2], "lr_candidates": [3e-4, 1e-3, 3e-3],
          "lr": {"A1": 1e-3, "A2": 3e-4},
          "primary_cell": {"F": 512, "P": 64, "placement": "a", "filler_weight": 0.0},
          "phase2_train_tokens": 120_000_000, "phase2_hours_per_run": 1.7, "phase2_max_tokens": 150_000_000}

EXPECTED = {"E0.3": 2, "E0.4": 35, "E0.5": 5, "E0.6": 2, "E0.7": 4, "E1.0a": 6, "E1.0c": 6, "E1.1": 12, "E1.2": 30,
            "E1.2l": 1, "E1.3": 12, "E2.0": 2, "E2.1": 3, "E2.2": 2}


def sets(runs, prefix):
    return {s for r in runs for s in r["set"] if s.startswith(prefix)}


def test_run_counts(tmp_path):
    counts = {g: len(build(g, PREREG, str(tmp_path))) for g in EXPECTED}
    assert counts == EXPECTED
    assert sum(counts[g] for g in EXPECTED if g.startswith("E1")) == 67


def test_run_ids_unique_and_cells_correct(tmp_path):
    runs = [r for g in EXPECTED for r in build(g, PREREG, str(tmp_path))]
    ids = [r["run_id"] for r in runs]
    assert len(ids) == len(set(ids))
    p1 = [r for g in EXPECTED if g.startswith("E1") for r in build(g, PREREG, str(tmp_path))]
    assert sets(p1, "data.N=") == {"data.N=32"} and sets(p1, "data.placement=") == {"data.placement=a"}
    assert not any("/T1.yaml" in r["config"] for r in p1)
    main = [r for g in ("E1.0a", "E1.1", "E1.2", "E1.2l", "E1.3") for r in build(g, PREREG, str(tmp_path))]
    assert sets(main, "data.filler_weight=") == {"data.filler_weight=0.0"}
    assert sets(build("E1.0c", PREREG, str(tmp_path)), "data.filler_weight=") == {"data.filler_weight=1.0"}
    assert sets(build("E1.1", PREREG, str(tmp_path)), "data.F=") == {"data.F=0", "data.F=128", "data.F=256", "data.F=512"}
    e12 = build("E1.2", PREREG, str(tmp_path))
    assert sets([r for r in e12 if "/A2.yaml" not in r["config"]], "data.F=") == {"data.F=0", "data.F=512"}
    for g in ("E1.0a", "E1.0c", "E1.2l", "E1.3"):
        assert sets(build(g, PREREG, str(tmp_path)), "data.F=") == {"data.F=512"}
    a3 = [r for r in e12 if "/A3.yaml" in r["config"]]
    assert all("train.lr=0.0003" in r["set"] for r in a3), "A3 must use A2's learning rate"
    assert all("train.steps=6000" in r["set"] for r in p1)


def test_phase0_steps_and_e06_cell(tmp_path):
    for g in ("E0.4", "E0.5"):
        assert all("train.steps=4000" in r["set"] for r in build(g, PREREG, str(tmp_path)))
    assert sets(build("E0.6", PREREG, str(tmp_path)), "data.N=") == {"data.N=32"}


def test_e06_needs_n_star(tmp_path):
    with pytest.raises(SystemExit):
        build("E0.6", dict(PREREG, N_star=None), str(tmp_path))


def test_e07_clutter_pilot_cells(tmp_path):
    e07 = build("E0.7", PREREG, str(tmp_path))
    cells = {(r["set"][3], r["set"][6]) for r in e07}          # data.F, data.filler_weight
    assert cells == {("data.F=64", "data.filler_weight=1.0"), ("data.F=128", "data.filler_weight=1.0"),
                     ("data.F=256", "data.filler_weight=1.0"), ("data.F=512", "data.filler_weight=0.0")}
    assert all("data.N=32" in r["set"] and "train.eval_every=250" in r["set"] for r in e07)


def test_phase1_needs_primary_f(tmp_path):
    with pytest.raises(SystemExit):
        build("E1.1", dict(PREREG, primary_cell=dict(PREREG["primary_cell"], F=None)), str(tmp_path))
