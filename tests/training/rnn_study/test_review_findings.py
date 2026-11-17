"""Stage 9 regression tests for the independent review (plan §7.3): orchestration findings.

MJ-7 (pilot rule needs every seed), mn-10 (X failure takes precedence in the horizon rule), mn-2
(an exception before the first commit is ``aborted_before_commit``), mn-9 (an integrity mismatch
stops X), and the verifier's new invalid-point consistency check (C6). Synthetic data only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import models.recurrent.rnn_training as rt
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from training.rnn_study import extension_x, summarize_pilot
from training.rnn_study.common import REPO_ROOT, Blocked, StudyPaths, execute_training_attempt, study_session
from training.rnn_study.conditions import CONDITIONS_SCHEMA, compute_frozen_conditions
from training.rnn_study.verify_runs import verify_run
from utils.reproducibility import make_generator

HEADER = "| ID | Date | Decision | Status | Reason | Consequences | Reference |\n|---|---|---|---|---|---|---|\n"


def _hist(values):
    return [{"completed_epochs": e, "dev_nll": v, "running_train_nll": None if e == 0 else v} for e, v in enumerate(values)]


def _ok(vals):
    return {"status": "completed", "history": _hist(vals)}


# --------------------------------------------------------------------------- MJ-7, mn-10
def test_MJ7_eta_rule_refuses_a_grid_with_missing_seeds() -> None:
    """training §7.3: the mean is over seeds 100 – 102. A grid point with fewer (or other) seeds makes
    the rule inconclusive; nothing is selected, however good the single-seed mean looks."""
    full = {s: _ok([3.3, 2.5, 2.30]) for s in (100, 101, 102)}
    outcomes = {0.03: full, 0.1: {100: _ok([3.3, 2.0, 1.0])}, 0.3: full}
    r = summarize_pilot.select_learning_rate(outcomes, horizon=2)
    assert r["selected"] is None and r["primary"] is None and "missing" in r["inconclusive"]
    assert "0.1" in r["inconclusive"] and "101" in r["inconclusive"]
    extra = {0.03: full, 0.1: {**full, 7: _ok([3.3, 2.0, 1.0])}}
    assert summarize_pilot.select_learning_rate(extra, horizon=2)["selected"] is None
    complete = {0.03: full, 0.1: {s: _ok([3.3, 2.5, 2.2]) for s in (100, 101, 102)}}
    assert summarize_pilot.select_learning_rate(complete, horizon=2)["primary"] == 0.1


def test_mn10_failed_extension_is_inconclusive_whatever_the_argument_order() -> None:
    assert "inconclusive" in summarize_pilot.horizon_rule(30, None, x_failed=True)
    assert summarize_pilot.horizon_rule(30, None, x_failed=True)["E"] is None
    assert summarize_pilot.horizon_rule(30, 41, x_failed=True)["E"] is None
    assert summarize_pilot.horizon_rule(29, None, x_failed=True) == {"E": 30, "extension_needed": False}


# --------------------------------------------------------------------------- mn-2
def _session(tmp: Path, data):
    paths = StudyPaths(tmp / "rnn_study")
    paths.root.mkdir(parents=True)
    row = "| D-90 | 2026-12-01 | synthetic | ACCEPTED (user approval, 2026-12-01) | r | c | ref |"
    record = tmp / "record.md"
    record.write_text("# R\n\n## 11. Decisions and Rationale\n\n" + HEADER + row + "\n\n## 12. Limitations\n",
                      encoding="utf-8")
    ref = {"id": "D-90", "row_sha256": hashlib.sha256(row.encode()).hexdigest()}
    paths.conditions.write_text(json.dumps({
        "schema": CONDITIONS_SCHEMA, "frozen": compute_frozen_conditions(REPO_ROOT),
        "decisions": {"implementation_acceptance": ref, "authorisation_L6": ref}, "values": {},
        "record_reviewed_through": "D-90"}))
    return study_session("test", paths=paths, required_decisions=("authorisation_L6",), required_values=(),
                         record=record), paths


@pytest.mark.parametrize("exc", [RuntimeError("boom before any commit"), KeyboardInterrupt()])
def test_mn2_exception_before_first_commit_is_aborted_before_commit(tmp_path, data, exc) -> None:
    """plan §8.3, §9.1: an attempt that ends before its first committed checkpoint is final
    ``aborted_before_commit`` (never resumable), for an ordinary exception and for an interrupt."""
    ctx, paths = _session(tmp_path, data)

    def build(run_dir):
        raise exc

    with ctx as session:
        with pytest.raises(type(exc)):
            execute_training_attempt(session, step="L6_pilot", arm="0.1", seed=100, authorisation="D-90",
                                     build=build, data=data)
        rec = next(iter(session.store.attempts().values()))
    assert rec["status"] == "aborted_before_commit" and "first commit" in rec["reason"]


# --------------------------------------------------------------------------- mn-9
def test_mn9_integrity_mismatch_stops_extension_x(monkeypatch) -> None:
    """training §7.4: a mismatch of epochs 0 – 30 between X and its pilot run stops X for review;
    later seeds are not run."""
    calls = []

    class Store:
        def counted(self, key):
            return {"status": "completed", "run_dir": f"pilot_{key[2]}"} if key[0] == "L6_pilot" else None

        def attempts(self):
            return {}

    class S:
        values = {"threads": [1, 1], "lr": 0.1}
        conditions = {"decisions": {"authorisation_X": {"id": "D-91"}}}
        store = Store()
        paths = type("P", (), {"root": Path("/nonexistent")})()

    monkeypatch.setattr("training.rnn_study.common.set_threads", lambda *a: a)
    monkeypatch.setattr(extension_x, "_builder", lambda *a, **k: (lambda seed: None))

    def fake_attempt(session, **kw):
        calls.append(kw["seed"])
        return {"status": "completed", "run_dir": f"x_{kw['seed']}"}

    monkeypatch.setattr("training.rnn_study.common.execute_training_attempt", fake_attempt)
    monkeypatch.setattr(extension_x, "_history_hashes",
                        lambda d: ["a"] * 31 if "pilot" in str(d) else ["a"] * 30 + ["b"])
    with pytest.raises(Blocked, match="stopped for review"):
        extension_x.run(S(), data=None)
    assert calls == [100]


# --------------------------------------------------------------------------- verifier: C6 consistency
def test_verifier_flags_metadata_without_the_invalid_point_list(tmp_path, data) -> None:
    from models.recurrent import rnn_diagnostics as dg
    from utils.run_manager import RunManager

    probe = dg.build_probe(data.train, data.train_names, data.split_info, tmp_path / "probe.json")
    run = RunManager.create("rnn", config={"t": 1}, results_root=tmp_path / "runs")
    plan = dg.RNNDiagnosticsPlan(probe=probe, train=data.train, train_names=data.train_names, dev=data.dev,
                                 steps_per_epoch=5, max_epochs=2, recurrent_gain=1.0,
                                 out_dir=run.run_dir / dg.RESEARCH_DIAGNOSTICS_DIR)
    model = RNNLanguageModel(data.vocab, RNNConfig(hidden_dim=4), generator=make_generator(2))
    cfg = rt.RNNTrainingConfig(lr=0.3, max_epochs=2, batch_size=4, eval_batch_size=7)
    rt.train_rnn(model, data.train, data.dev, training=cfg, seed=3, run=run, diagnostics=plan)
    assert verify_run(run.run_dir) == []
    pts = [json.loads(x) for x in (plan.out_dir / "points.jsonl").read_text().splitlines()]
    pts[1]["valid"] = False
    (plan.out_dir / "points.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pts))
    meta = json.loads(run.metadata_path.read_text())
    meta["diagnostics_status"] = "incomplete"
    run.metadata_path.write_text(json.dumps(meta))
    findings = verify_run(run.run_dir)
    assert any("invalid_diagnostic_points" in f for f in findings)
