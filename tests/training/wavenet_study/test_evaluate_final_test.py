"""F4: protected one-time test evaluation, on synthetic data only.

One module-level synthetic experiment (four arms, seeds 0 and 1, three epochs) is trained with the
F3 runner and analysed on dev (F5). Every test then starts from that state without a freeze: the
``fresh`` fixture removes ``test_freeze/`` and restores ``invocations.jsonl`` afterwards (the
manifest and the runs are never changed by the evaluator). Crashes are simulated in-process with
a ``BaseException`` the evaluator does not catch (the process "dies"), and once with a real child
process killed by ``os._exit``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.training.wavenet_study.test_final_experiment import approve, make_spec, make_world, quiet, tampered
from training.wavenet_study import analyze_final_experiment as an
from training.wavenet_study import context_buckets as cb
from training.wavenet_study import evaluate_final_test as ev
from training.wavenet_study import final_experiment as fe
from training.wavenet_study import study_common as sc
from training.wavenet_study import verify_final_runs as vf
from utils.metrics import token_nll

REPO = fe.REPO
KEYS = [f"{a}/{s}" for s in (0, 1) for a in "ABCD"]


class SimulatedCrash(BaseException):
    """Stands for the death of the evaluating process (never caught by the evaluator)."""


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("f4_world")
    w = make_world(base)
    spec = make_spec(w, base / "final", seeds=(0, 1))
    assert fe.run_stage(spec, "validation", out=quiet([])) == 0
    approve(spec)
    assert fe.run_stage(spec, "completion", out=quiet([])) == 0
    with pytest.raises(sc.Blocked, match="dev analysis"):
        ev.create_freeze(spec.root, authorized_by="Nur", note="too early")
    an.dev_analysis(spec.root, out=quiet([]))
    return {"spec": spec, "root": Path(spec.root), "world": w}


@pytest.fixture()
def fresh(world):
    root = world["root"]
    inv = root / "invocations.jsonl"
    saved = inv.read_bytes()
    shutil.rmtree(root / fe.FREEZE_DIR, ignore_errors=True)
    yield world
    shutil.rmtree(root / fe.FREEZE_DIR, ignore_errors=True)
    inv.write_bytes(saved)


def freeze(world):
    return ev.create_freeze(world["root"], authorized_by="Nur", note="all decisions frozen")


def states(world):
    return ev.ledger_status(world["root"])["states"]


# ============================================================ freeze
def test_no_evaluation_before_freeze(fresh, monkeypatch):
    called = []
    monkeypatch.setattr(ev, "_test_windows", lambda *a, **k: called.append(1))
    with pytest.raises(sc.Blocked, match="no test freeze"):
        ev.evaluate(fresh["root"], out=quiet([]))
    assert called == []
    with pytest.raises(sc.Blocked):
        ev.ledger_status(fresh["root"])


def test_freeze_contents_and_training_closure(fresh):
    doc = freeze(fresh)
    report = vf.verify(fresh["root"])
    assert doc["checkpoints"] == vf.selected_checkpoints(report) and doc["order"] == KEYS
    assert set(ev.CODE_FILES) <= set(doc["code"]) and "models/hierarchical/wavenet.py" in doc["code"]
    defs = cb.from_json((fresh["root"] / ev.DEV_ANALYSIS / ev.BUCKETS_FILE).read_text())
    assert doc["bucket_definitions"]["definition_sha256"] == defs["definition_sha256"]
    assert doc["evaluation"]["batch_size"] == 4096
    assert doc["freeze_sha256"] == sc.sha256_json({k: v for k, v in doc.items() if k != "freeze_sha256"})
    assert fe.status(fresh["spec"])["state"] == "CLOSED"
    with pytest.raises(sc.Blocked, match="training is closed"):
        fe.run_stage(fresh["spec"], "completion", out=quiet([]))
    with pytest.raises(sc.Blocked, match="exactly once"):
        freeze(fresh)
    with pytest.raises(sc.Blocked, match="frozen"):
        an.dev_analysis(fresh["root"], out=quiet([]))
    with tampered(ev._paths(fresh["root"])["freeze"], lambda b: b.replace(b"all decisions", b"some decisions")):
        with pytest.raises(sc.Blocked, match="freeze_sha256"):
            ev.load_freeze(fresh["root"])
    assert states(fresh) == {k: "PENDING" for k in KEYS}


def test_freeze_refused_when_verification_fails(fresh, monkeypatch):
    monkeypatch.setattr(vf, "verify", lambda root, **k: {"status": "fail", "problems": ["synthetic failure"]})
    with pytest.raises(sc.Blocked, match="verification does not pass"):
        freeze(fresh)
    assert not ev._paths(fresh["root"])["freeze"].exists()


def test_changed_inputs_block_evaluation(fresh):
    freeze(fresh)
    dev_summary = fresh["root"] / ev.DEV_ANALYSIS / ev.DEV_SUMMARY_FILE
    with tampered(dev_summary, lambda b: b + b" "):
        with pytest.raises(sc.Blocked, match="dev analysis changed"):
            ev.evaluate(fresh["root"], out=quiet([]))
    defs = fresh["root"] / ev.DEV_ANALYSIS / ev.BUCKETS_FILE
    with tampered(defs, lambda b: b.replace(b'"min_support": 500', b'"min_support": 400')):
        with pytest.raises(sc.Blocked, match="bucket definitions changed"):
            ev.evaluate(fresh["root"], out=quiet([]))
    ckpt = vf._run_path(ev.load_freeze(fresh["root"])["checkpoints"]["C/1"]["run_dir"]) / "model_best.pt"
    with tampered(ckpt, lambda b: b + b"\0"):
        with pytest.raises(sc.Blocked, match="checkpoint C/1 changed"):
            ev.evaluate(fresh["root"], out=quiet([]))
    assert states(fresh) == {k: "PENDING" for k in KEYS}


# ============================================================ one complete, once-only evaluation
@pytest.fixture()
def evaluated(fresh):
    freeze(fresh)
    lines = []
    assert ev.evaluate(fresh["root"], out=quiet(lines)) == 0
    return {**fresh, "lines": lines}


def test_every_checkpoint_evaluated_exactly_once(evaluated):
    root = evaluated["root"]
    assert states(evaluated) == {k: "COMPLETED" for k in KEYS}
    assert ev.verify_outputs(root)["status"] == "pass"
    out_dir = ev._paths(root)["evaluations"]
    before = {p.name: sc.sha256_file(p / ev.RESULT_FILE) for p in out_dir.iterdir()}
    lines = []
    assert ev.evaluate(root, out=quiet(lines)) == 0
    assert "0 checkpoint(s) to evaluate" in lines[0]
    assert {p.name: sc.sha256_file(p / ev.RESULT_FILE) for p in out_dir.iterdir()} == before
    started = [e for e in ev.read_ledger(root) if e["event"] == "started"]
    assert sorted(e["key"] for e in started) == sorted(KEYS)
    for fn in (ev.adopt, ev.authorize_reevaluation):
        with pytest.raises(sc.Blocked, match="never repeated"):
            fn(root, "A/0", deviation_id="DEV-1", reason="try", authorized_by="Nur")


def test_results_match_independent_recomputation(evaluated):
    root, spec = evaluated["root"], evaluated["spec"]
    fz = ev.load_freeze(root)
    defs = cb.from_json((root / ev.DEV_ANALYSIS / ev.BUCKETS_FILE).read_text())
    for key in ("A/0", "D/1"):
        arm, seed = key.split("/")[0], int(key.split("/")[1])
        res = json.loads((ev._paths(root)["evaluations"] / ev._dir_name(key) / ev.RESULT_FILE).read_text())
        per_target = np.load(ev._paths(root)["evaluations"] / ev._dir_name(key) / ev.PER_TARGET_FILE)
        loaded = ev.load_selected_model(spec, arm, seed, fz["checkpoints"][key])
        test = ev._test_windows(spec, loaded["built"])
        acc = loaded["built"]["module"].evaluate_nll(loaded["model"], test, batch_size=4096)
        assert res["test_nll"] == acc.mean_nll and res["num_tokens"] == acc.num_tokens == test.num_examples
        with torch.no_grad():
            ref = token_nll(loaded["model"](test.contexts), test.targets, reduction="none").double().numpy()
        assert np.allclose(per_target, ref, rtol=0, atol=1e-6)
        assignment = cb.assign(defs, test, cb.count_train_contexts(loaded["built"]["train"]))
        assert cb.aggregate(assignment, per_target) == res["buckets"]
        for block in res["buckets"]["stratifications"].values():
            assert sum(c["n"] for c in block["categories"].values()) == res["num_tokens"]
        if arm == "D":
            assert res["checks"]["batchnorm"]["recalibration_reproduced"] is True
        assert res["checks"]["dev_reproduced"] is True
    assert cb.from_json((root / ev.DEV_ANALYSIS / ev.BUCKETS_FILE).read_text()) == defs   # never changed


def test_ledger_tampering_is_detected(evaluated):
    root = evaluated["root"]
    ledger = ev._paths(root)["ledger"]
    with tampered(ledger, lambda b: b.replace(b'"completed"', b'"completeD"', 1)):
        with pytest.raises(sc.Blocked):
            ev.ledger_status(root)
    lines = ledger.read_bytes().split(b"\n")
    with tampered(ledger, lambda b: b"\n".join(lines[:-2]) + b"\n"):        # last record removed
        with pytest.raises(sc.Blocked, match="recorded"):
            ev.ledger_status(root)
    out = ev._paths(root)["evaluations"] / "B-s1" / ev.PER_TARGET_FILE
    with tampered(out, lambda b: b[:-8] + np.float64(9.0).tobytes()):
        assert ev.verify_outputs(root)["status"] == "fail"
    result = ev._paths(root)["evaluations"] / "C-s1" / ev.RESULT_FILE   # self-consistent edit: only the ledger knows
    with tampered(result, lambda b: json.dumps({**json.loads(b), "seconds": 0.0}, indent=2).encode()):
        check = ev.verify_outputs(root)
        assert check["status"] == "fail" and "differ from the ledger" in check["problems"][0]


def test_completed_evaluation_cannot_be_restarted_even_by_ledger_entry(evaluated):
    spec, root = evaluated["spec"], evaluated["root"]
    with sc.PilotLock(spec.paths, "test") as lock:
        session = ev._session(spec, lock, "test")
        ev._append(spec, session, root, {"event": "started", "key": "A/0", "evaluation_id": "A/0-e2"})
        session.end()
    with pytest.raises(sc.Blocked, match="INTEGRITY_FAILURE"):
        ev.ledger_status(root)


def test_invalid_transition_is_an_integrity_failure(fresh):
    freeze(fresh)
    spec, root = fresh["spec"], fresh["root"]
    with sc.PilotLock(spec.paths, "test") as lock:
        session = ev._session(spec, lock, "test")
        ev._append(spec, session, root, {"event": "completed", "key": "A/0", "evaluation_id": "A/0-e1"})
        session.end()
    with pytest.raises(sc.Blocked, match="INTEGRITY_FAILURE"):
        ev.ledger_status(root)
    with pytest.raises(sc.Blocked, match="INTEGRITY_FAILURE"):
        ev.evaluate(root, out=quiet([]))


# ============================================================ BatchNorm inference state
@pytest.mark.parametrize("arm", ["B", "D"])
def test_batchnorm_calibration_and_inference_checks(fresh, arm):
    spec = fresh["spec"]
    sel = vf.selected_checkpoints(vf.verify(fresh["root"]))[f"{arm}/1"]
    loaded = ev.load_selected_model(spec, arm, 1, sel)
    assert ev.selection_checks(loaded, sel, 4096)["batchnorm"]["recalibration_reproduced"]
    bn = next(m for _, m in loaded["model"].named_modules() if hasattr(m, "inference_mean"))
    with torch.no_grad():
        bn.inference_mean.add_(1e-3)
    with pytest.raises(ev.EvaluationIntegrityError, match="recalibration"):
        ev.selection_checks(loaded, sel, 4096)
    loaded = ev.load_selected_model(spec, arm, 1, sel)
    bn = next(m for _, m in loaded["model"].named_modules() if hasattr(m, "inference_mean"))
    bn.num_calibrations.add_(1)
    with pytest.raises(ev.EvaluationIntegrityError, match="num_calibrations"):
        ev.selection_checks(loaded, sel, 4096)
    loaded = ev.load_selected_model(spec, arm, 1, sel)
    loaded["model"].train()
    with pytest.raises(ev.EvaluationIntegrityError, match="evaluation mode"):
        ev.selection_checks(loaded, sel, 4096)


def test_selection_and_checkpoint_identity_checks(fresh):
    spec = fresh["spec"]
    sel = vf.selected_checkpoints(vf.verify(fresh["root"]))["A/0"]
    loaded = ev.load_selected_model(spec, "A", 0, sel)
    with torch.no_grad():
        next(loaded["model"].parameters()).view(-1)[0] += 1e-2
    with pytest.raises(ev.EvaluationIntegrityError, match="dev NLL"):
        ev.selection_checks(loaded, sel, 4096)
    with pytest.raises(ev.EvaluationIntegrityError, match="SHA-256"):
        ev.load_selected_model(spec, "A", 0, {**sel, "model_best_sha256": "0" * 64})
    with pytest.raises(ev.EvaluationIntegrityError, match="state hash"):
        ev.load_selected_model(spec, "A", 0, {**sel, "state_sha256": "0" * 64})
    with pytest.raises(ev.EvaluationIntegrityError, match="earliest"):
        ev.selection_checks(ev.load_selected_model(spec, "A", 0, sel), {**sel, "best_epoch": sel["best_epoch"] + 1},
                            4096)


# ============================================================ crashes and interruptions
def _crash_on(monkeypatch, key, *, after_commit):
    original = ev._commit

    def commit(tmp, final):
        if final.name == ev._dir_name(key):
            if after_commit:
                original(tmp, final)
            raise SimulatedCrash(f"process died {'after' if after_commit else 'before'} commit")
        return original(tmp, final)

    monkeypatch.setattr(ev, "_commit", commit)


def test_crash_before_commit_blocks_then_authorised_reevaluation(fresh, monkeypatch):
    root = fresh["root"]
    freeze(fresh)
    _crash_on(monkeypatch, "B/0", after_commit=False)
    with pytest.raises(SimulatedCrash):
        ev.evaluate(root, out=quiet([]))
    monkeypatch.undo()
    st = states(fresh)
    assert st["A/0"] == "COMPLETED" and st["B/0"] == "STARTED" and st["C/0"] == "PENDING"
    assert list(ev._paths(root)["evaluations"].glob(".tmp-B-s0-*"))       # uncommitted output left behind
    lines = []
    assert ev.evaluate(root, out=quiet(lines)) == ev.EXIT["blocked"]
    assert "human review" in lines[-1] and states(fresh)["B/0"] == "BLOCKED"
    assert ev.evaluate(root, out=quiet([])) == ev.EXIT["blocked"]         # stays blocked; nothing evaluated
    assert states(fresh)["C/0"] == "PENDING"
    with pytest.raises(sc.Blocked, match="no committed output"):
        ev.adopt(root, "B/0", deviation_id="DEV-1", reason="adopt", authorized_by="Nur")
    for bad in ({"deviation_id": "", "reason": "r", "authorized_by": "Nur"},
                {"deviation_id": "DEV-1", "reason": " ", "authorized_by": "Nur"},
                {"deviation_id": "DEV-1", "reason": "r", "authorized_by": ""}):
        with pytest.raises(sc.Blocked):
            ev.authorize_reevaluation(root, "B/0", **bad)
    ev.authorize_reevaluation(root, "B/0", deviation_id="DEV-1", reason="crash before commit, inspected",
                              authorized_by="Nur")
    assert not list(ev._paths(root)["evaluations"].glob(".tmp-B-s0-*"))
    assert list((ev._paths(root)["quarantine"] / "B-s0-DEV-1").iterdir())
    assert ev.evaluate(root, out=quiet([])) == 0
    status = ev.ledger_status(root)
    assert status["states"] == {k: "COMPLETED" for k in KEYS}
    assert status["deviations"]["B/0"][0]["deviation_id"] == "DEV-1"
    res = json.loads((ev._paths(root)["evaluations"] / "B-s0" / ev.RESULT_FILE).read_text())
    assert res["evaluation_id"] == "B/0-e2"
    assert ev.verify_outputs(root)["status"] == "pass"


def test_crash_after_commit_is_never_adopted_silently(fresh, monkeypatch):
    root = fresh["root"]
    freeze(fresh)
    _crash_on(monkeypatch, "C/0", after_commit=True)
    with pytest.raises(SimulatedCrash):
        ev.evaluate(root, out=quiet([]))
    monkeypatch.undo()
    assert (ev._paths(root)["evaluations"] / "C-s0").is_dir() and states(fresh)["C/0"] == "STARTED"
    assert ev.evaluate(root, out=quiet([])) == ev.EXIT["blocked"]
    last = [e for e in ev.read_ledger(root) if e.get("key") == "C/0"][-1]
    assert last["event"] == "interrupted" and last["output_present"] is True
    target = ev._paths(root)["evaluations"] / "C-s0" / ev.PER_TARGET_FILE
    with tampered(target, lambda b: b[:-8] + np.float64(1.0).tobytes()):
        with pytest.raises(ev.EvaluationIntegrityError):
            ev.adopt(root, "C/0", deviation_id="DEV-2", reason="committed output inspected", authorized_by="Nur")
    assert states(fresh)["C/0"] == "BLOCKED"
    ev.adopt(root, "C/0", deviation_id="DEV-2", reason="committed output inspected", authorized_by="Nur")
    assert states(fresh)["C/0"] == "ADOPTED"
    assert ev.evaluate(root, out=quiet([])) == 0
    assert set(states(fresh).values()) == {"COMPLETED", "ADOPTED"}
    assert ev.verify_outputs(root)["status"] == "pass"


def test_failure_and_keyboard_interrupt_block(fresh, monkeypatch):
    root = fresh["root"]
    freeze(fresh)
    original = ev.evaluate_one
    calls = []

    def flaky(spec, fz, key, eid, defs, r):
        calls.append(key)
        if key == "B/0" and calls.count(key) == 1:
            raise RuntimeError("synthetic evaluation failure")
        if key == "C/0":
            raise KeyboardInterrupt
        return original(spec, fz, key, eid, defs, r)

    monkeypatch.setattr(ev, "evaluate_one", flaky)
    assert ev.evaluate(root, out=quiet([])) == ev.EXIT["failed"]
    assert states(fresh)["B/0"] == "BLOCKED"
    ev.authorize_reevaluation(root, "B/0", deviation_id="DEV-3", reason="software failure fixed", authorized_by="Nur")
    assert ev.evaluate(root, out=quiet([])) == ev.EXIT["interrupted"]
    st = states(fresh)
    assert st["B/0"] == "COMPLETED" and st["C/0"] == "BLOCKED"
    assert [e for e in ev.read_ledger(root) if e.get("key") == "B/0"][1]["event"] == "failed"


def test_real_process_crash_after_commit(fresh, tmp_path, monkeypatch):
    root = fresh["root"]
    freeze(fresh)
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(REPO)!r})
        from training.wavenet_study import evaluate_final_test as ev
        original = ev._commit
        def commit(tmp, final):
            original(tmp, final)
            if final.name == "A-s1":
                os._exit(9)
        ev._commit = commit
    """))
    env = {**__import__("os").environ, "PYTHONPATH": str(site)}
    proc = subprocess.run([sys.executable, str(REPO / "training/wavenet_study/evaluate_final_test.py"), "--root",
                           str(root), "evaluate"], cwd=REPO, env=env, capture_output=True, text=True)
    assert proc.returncode == 9, proc.stderr[-2000:]
    st = states(fresh)
    assert st["A/1"] == "STARTED" and all(st[k] == "COMPLETED" for k in KEYS[:4])
    assert ev.evaluate(root, out=quiet([])) == ev.EXIT["blocked"]
    assert [e for e in ev.read_ledger(root) if e.get("key") == "A/1"][-1]["output_present"] is True


def test_only_the_evaluator_builds_test_windows():
    pattern = "splits.test"
    for name in ("final_experiment", "verify_final_runs", "analyze_final_experiment", "context_buckets",
                 "final_statistics", "study_common"):
        assert pattern not in (REPO / f"training/wavenet_study/{name}.py").read_text(), name
    src = (REPO / "training/wavenet_study/evaluate_final_test.py").read_text()
    assert src.count(pattern) == 1 and "def _test_windows" in src
