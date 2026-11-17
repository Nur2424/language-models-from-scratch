"""MS8: lifecycle drivers on synthetic data and temporary study roots (O2, O8, O9 and dry runs; plan §8.5).

Synthetic research records and study conditions are written per test; the frozen conditions are
computed from this repository. ``PROTOCOL_COUNTS`` is replaced by the synthetic data's counts (the
production check stays strict). Nothing here reads ``data/``, ``results/`` or the real record.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

import models.recurrent.rnn_training as rt
from training.rnn_study import (build_probe, diagnostic_timing, init_gate, lr_pilot, posthoc_spectral,
                                research_runs, summarize_gate, summarize_pilot, thread_benchmark)
from training.rnn_study.common import REPO_ROOT, Blocked, StudyPaths, study_session
from training.rnn_study.conditions import CONDITIONS_SCHEMA, IntegrityBlocked, compute_frozen_conditions
from training.rnn_study.verify_runs import verify_study

HEADER = "| ID | Date | Decision | Status | Reason | Consequences | Reference |\n|---|---|---|---|---|---|---|\n"


def make_study(tmp: Path, decisions: dict[str, str], values: dict, *, status="ACCEPTED (user approval, 2026-12-01)"):
    paths = StudyPaths(tmp / "rnn_study")
    paths.root.mkdir(parents=True)
    ids = sorted(set(decisions.values()), key=lambda d: int(d[2:]))
    rows = {d: f"| {d} | 2026-12-01 | synthetic {d} | {status} | r | c | ref |" for d in ids}
    record = tmp / "record.md"
    record.write_text("# R\n\n## 11. Decisions and Rationale\n\n" + HEADER + "\n".join(rows.values())
                      + "\n\n## 12. Limitations\n", encoding="utf-8")
    refs = {k: {"id": d, "row_sha256": hashlib.sha256(rows[d].encode()).hexdigest()} for k, d in decisions.items()}
    conds = {"schema": CONDITIONS_SCHEMA, "frozen": compute_frozen_conditions(REPO_ROOT), "decisions": refs,
             "values": {k: {"value": v, "decision": "D-90"} for k, v in values.items()},
             "record_reviewed_through": ids[-1]}
    paths.conditions.write_text(json.dumps(conds, indent=1))
    return paths, record


def threads():
    return [torch.get_num_threads(), torch.get_num_interop_threads()]


BASE = {"implementation_acceptance": "D-90"}


def session(paths, record, driver, **kw):
    return study_session(driver.COMMAND, paths=paths, required_decisions=driver.REQUIRED_DECISIONS,
                         required_values=driver.REQUIRED_VALUES, record=record, **kw)


@pytest.fixture
def synthetic_counts(monkeypatch, data):
    N = int(data.train.lengths.numel())
    S = -(-N // 32)
    monkeypatch.setattr(rt, "PROTOCOL_COUNTS", {"N": N, "M": int(data.train.lengths.sum()), "steps_per_epoch": S,
                                                "final_batch_names": N - (S - 1) * 32, "N_dev": int(data.dev.lengths.numel()),
                                                "M_dev": int(data.dev.lengths.sum())})


# --------------------------------------------------------------------------- O2 / O9 prerequisites
@pytest.mark.parametrize("driver", [build_probe, thread_benchmark, diagnostic_timing, init_gate, lr_pilot,
                                    research_runs, posthoc_spectral])
def test_O2_each_driver_refuses_without_its_decisions(tmp_path, driver) -> None:
    """O2: a driver whose decisions are not all referenced (implementation acceptance included) is refused
    before anything runs; nothing is written to run directories."""
    paths, record = make_study(tmp_path, BASE, {})
    with pytest.raises(IntegrityBlocked, match="decision"):
        with session(paths, record, driver):
            pass
    assert not paths.all_run_dirs()


def test_O2_values_and_staleness_and_status(tmp_path) -> None:
    decisions = {**BASE, "authorisation_L4": "D-91"}
    paths, record = make_study(tmp_path, decisions, {"probe_sha256": "x"})
    with pytest.raises(Blocked, match="threads"):
        with session(paths, record, diagnostic_timing):
            pass
    paths2, record2 = make_study(tmp_path / "b", decisions, {"probe_sha256": "x", "threads": threads()},
                                 status="ACCEPTED (direction); details PENDING")
    with pytest.raises(IntegrityBlocked, match="status"):
        with session(paths2, record2, diagnostic_timing):
            pass
    paths3, record3 = make_study(tmp_path / "c", decisions, {"probe_sha256": "x", "threads": threads()})
    with open(record3, "a") as fh:  # a new decision recorded after the conditions were reviewed
        fh.write("")
    text = record3.read_text().replace("\n\n## 12.", "\n| D-99 | 2026-12-02 | new | ACCEPTED (user approval, 2026-12-02) | r | c | ref |\n\n## 12.")
    record3.write_text(text)
    with pytest.raises(IntegrityBlocked, match="after D-91"):
        with session(paths3, record3, diagnostic_timing):
            pass


def test_O9_pilot_requires_the_gate_decision_and_gate_seeds(tmp_path, data) -> None:
    """O9: ``lr_pilot`` refuses without the recorded gate decision; ``init_gate`` refuses any seed set other
    than 1000 – 1009."""
    paths, record = make_study(tmp_path, {**BASE, "authorisation_L6": "D-92"},
                               {"probe_sha256": "x", "threads": threads(), "g": 1.0, "s2": 0.5, "lr_grid": [0.1]})
    with pytest.raises(IntegrityBlocked, match="gate_decision"):
        with session(paths, record, lr_pilot):
            pass
    paths2, record2 = make_study(tmp_path / "g", {**BASE, "authorisation_L5": "D-93"},
                                 {"probe_sha256": "x", "threads": threads()})
    with session(paths2, record2, init_gate) as s:
        with pytest.raises(Blocked, match="1000 – 1009"):
            init_gate.run(s, data, seeds=(100, 101, 102))


def test_frozen_condition_change_blocks_every_driver(tmp_path) -> None:
    """O3 at session level: a changed frozen hash blocks before decisions are even read."""
    paths, record = make_study(tmp_path, {**BASE, "authorisation_L2": "D-91"}, {})
    conds = json.loads(paths.conditions.read_text())
    conds["frozen"]["files"]["models/recurrent/rnn.py"] = "0" * 64
    paths.conditions.write_text(json.dumps(conds))
    with pytest.raises(IntegrityBlocked, match="frozen"):
        with session(paths, record, build_probe):
            pass


# --------------------------------------------------------------------------- O8 static scan
def test_O8_study_code_never_builds_or_reads_the_test_split() -> None:
    """O8: no ``.test`` attribute, no ``["test"]`` subscript and no ``get("test")`` anywhere in
    ``training/rnn_study``; names are loaded only by ``common.load_study_data``."""
    problems = []
    for path in sorted((REPO_ROOT / "training" / "rnn_study").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "test":
                problems.append(f"{path.name}:{node.lineno} .test")
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "test":
                problems.append(f"{path.name}:{node.lineno} ['test']")
            if isinstance(node, ast.Call) and any(isinstance(a, ast.Constant) and a.value == "test" for a in node.args):
                problems.append(f"{path.name}:{node.lineno} call with 'test'")
            if isinstance(node, ast.Name) and node.id == "load_names" and path.name != "common.py":
                problems.append(f"{path.name}:{node.lineno} load_names")
    assert not problems, problems


# --------------------------------------------------------------------------- dry runs on synthetic data
def test_dry_run_probe_gate_timing(tmp_path, data) -> None:
    """L2, L4 and L5 drivers end to end on synthetic data: the probe is built once; the timing and the
    gate report are written once; no decision is made by code."""
    paths, record = make_study(tmp_path, {**BASE, "authorisation_L2": "D-91"}, {})
    with session(paths, record, build_probe) as s:
        out = build_probe.run(s, data)
    assert paths.probe.exists() and out["f8_dev_targets"] == build_probe.f8_count(data.dev)
    with session(paths, record, build_probe) as s, pytest.raises(Blocked, match="built once"):
        build_probe.run(s, data)
    probe_sha = out["probe_sha256"]
    for driver, key in ((diagnostic_timing, "authorisation_L4"), (init_gate, "authorisation_L5")):
        conds = json.loads(paths.conditions.read_text())
        conds["values"].update({"probe_sha256": {"value": probe_sha, "decision": "D-90"},
                                "threads": {"value": threads(), "decision": "D-90"}})
        conds["decisions"][key] = conds["decisions"]["authorisation_L2"]
        paths.conditions.write_text(json.dumps(conds))
        with session(paths, record, driver) as s:
            result = driver.run(s, data)
        assert Path(result.get("path") or result["report"]).exists()
    report = json.loads((paths.step_dir("L5_gate") / "initial_statistics.json").read_text())
    summary = summarize_gate.summarize(report)
    assert summary["seeds"] == list(range(1000, 1010)) and summary["positions"] and summary["pairs"]


def test_dry_run_pilot_research_posthoc_verify(tmp_path, data, synthetic_counts) -> None:
    """L6, L7 and L8 drivers end to end (tiny epochs): attempts recorded and counted once per key;
    research runs carry the full plan and ``o3.json``; the verifier finds no inconsistency; the pilot
    summary applies the predeclared rules."""
    vals = {"probe_sha256": None, "threads": threads(), "g": 1.0, "s2": 0.5, "lr_grid": [0.1, 0.3]}
    paths, record = make_study(tmp_path, {**BASE, "authorisation_L2": "D-91", "gate_decision": "D-92",
                                          "authorisation_L6": "D-93"}, vals)
    with session(paths, record, build_probe) as s:
        sha = build_probe.run(s, data)["probe_sha256"]
    conds = json.loads(paths.conditions.read_text())
    conds["values"]["probe_sha256"]["value"] = sha
    paths.conditions.write_text(json.dumps(conds))
    with session(paths, record, lr_pilot) as s:
        out = lr_pilot.run(s, data, epochs=2)
    assert len(out["runs"]) == 6 and {r["status"] for r in out["runs"]} <= {"completed", "failed_diverged", "failed_mechanical"}
    with session(paths, record, lr_pilot) as s:
        assert lr_pilot.run(s, data, epochs=2) == {"runs": []}  # every key already counted
    outcomes = summarize_pilot.load_pilot_outcomes(paths.root)
    assert sorted(outcomes) == [0.1, 0.3] and all(len(v) == 3 for v in outcomes.values())
    summary = summarize_pilot.select_learning_rate(outcomes, horizon=2)
    assert summary["evaluated"] == [0.1, 0.3]
    conds = json.loads(paths.conditions.read_text())
    for key, did in (("horizon_decision", "authorisation_L2"), ("final_seeds", "authorisation_L2"),
                     ("failed_seed_policy", "authorisation_L2"), ("authorisation_L7", "authorisation_L2"),
                     ("authorisation_L8", "authorisation_L2")):
        conds["decisions"][key] = conds["decisions"][did]
    conds["values"].update({"lr": {"value": 0.1, "decision": "D-90"}, "max_epochs": {"value": 3, "decision": "D-90"},
                            "final_seeds": {"value": [0], "decision": "D-90"}})
    paths.conditions.write_text(json.dumps(conds))
    with session(paths, record, research_runs) as s:
        res = research_runs.run(s, data)
    assert res["runs"][0]["diagnostics_status"] == "complete"
    run_dir = next(p for p in paths.all_run_dirs() if "L7_research" in str(p))
    assert (run_dir / "research_diagnostics" / "o3.json").exists()
    with session(paths, record, posthoc_spectral) as s:
        posthoc_spectral.run(s, data)
    report = verify_study(paths.root)
    assert all(not f for f in report.values()), report


def test_research_runs_refuse_overlapping_seeds(tmp_path, data) -> None:
    decisions = {**BASE, **{k: "D-91" for k in research_runs.REQUIRED_DECISIONS}}
    paths, record = make_study(tmp_path, decisions, {"probe_sha256": "x", "threads": threads(), "g": 1.0, "s2": 0.5,
                                                     "lr": 0.1, "max_epochs": 2, "final_seeds": [101]})
    with session(paths, record, research_runs) as s, pytest.raises(Blocked, match="overlaps"):
        research_runs.run(s, data)


# --------------------------------------------------------------------------- summary rules (hand fixtures)
def _hist(values):
    return [{"completed_epochs": e, "dev_nll": v, "running_train_nll": None if e == 0 else v} for e, v in enumerate(values)]


def test_pilot_rules_on_hand_fixtures() -> None:
    """Training spec §7.3 / §7.4 on hand-made histories: eligibility, boundary extension, near-tie
    (smallest η within max(1e‑3, SE)), plateau epoch and the hybrid horizon rule."""
    ok = lambda vals: {"status": "completed", "history": _hist(vals)}  # noqa: E731
    outcomes = {
        0.03: {s: ok([3.3, 2.5, 2.20 + d]) for s, d in ((100, 0.0), (101, 0.001), (102, -0.001))},
        0.1: {s: ok([3.3, 2.4, 2.1995 + d]) for s, d in ((100, 0.0), (101, 0.0002), (102, -0.0002))},
        0.3: {100: ok([3.3, 2.4, 2.3]), 101: {"status": "failed_diverged", "history": _hist([3.3, 3.4])},
              102: ok([3.3, 2.4, 2.3])},
        1.0: {s: ok([3.3, 2.6, 2.5]) for s in (100, 101, 102)},
    }
    r = summarize_pilot.select_learning_rate(outcomes, horizon=2)
    assert r["eligible"] == [0.03, 0.1, 1.0] and r["primary"] == 0.1 and r["at_boundary"] is None
    assert r["selected_if_grid_final"] == 0.03  # within max(1e-3, SE) of the best mean
    edge = {0.03: outcomes[0.03], 0.1: {s: ok([3.3, 2.5, 2.4]) for s in (100, 101, 102)}}
    r2 = summarize_pilot.select_learning_rate(edge, horizon=2)
    assert r2["primary"] == 0.03 and r2["at_boundary"] == "lower" and r2["extension_indicated"] == 0.01
    flat = [_hist([3.3, 2.5, 2.3, 2.2005, 2.2, 2.2001])] * 3
    p = summarize_pilot.plateau_epoch(flat, last=5)
    assert p["e_star"] == 3 and p["requires_review"] is False
    assert summarize_pilot.horizon_rule(25) == {"E": 30, "extension_needed": False}
    assert summarize_pilot.horizon_rule(30)["extension_needed"] is True
    assert summarize_pilot.horizon_rule(30, 41) == {"E": 41, "extension_needed": False}
    assert summarize_pilot.horizon_rule(30, 60)["E"] is None
    assert summarize_pilot.horizon_rule(30, 41, x_failed=True)["E"] is None
    flags = summarize_pilot.pilot_flags(_hist([3.3, 2.5, 2.6, 2.4, 2.7]), horizon=4)
    assert flags["dev_nll_rose"] == [2, 4] and flags["dev_nll_rose_repeated"] and flags["final_above_best"]


@pytest.mark.parametrize("driver", [build_probe, thread_benchmark, diagnostic_timing, init_gate, lr_pilot,
                                    research_runs, posthoc_spectral])
def test_cli_exits_2_when_blocked(tmp_path, driver, capsys) -> None:
    """Every driver's command line fails closed (exit 2, ``BLOCKED`` on stderr) on a study root without
    recorded conditions; no data is loaded and no run directory is created."""
    root = tmp_path / "rnn_study"
    assert driver.main(["--root", str(root)]) == 2
    assert "BLOCKED" in capsys.readouterr().err
    assert not StudyPaths(root).all_run_dirs()


def test_pilot_flags_ignore_the_epoch_one_change() -> None:
    """A rise from epoch 0 to epoch 1 is not a flag (the epoch-0 row is the untrained reference)."""
    flags = summarize_pilot.pilot_flags(_hist([3.0, 3.2, 3.1]), horizon=2)
    assert flags["dev_nll_rose"] == [] and flags["final_above_best"] is False
