"""MS7: run-directory classification and the attempt store (O4 – O7) (plan §8.5)."""


from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.rnn_study import attempts as at
from training.rnn_study import conditions as cd
from training.rnn_study.common import REPO_ROOT, Blocked, StudyLock, StudyPaths, lock_held_elsewhere



# --------------------------------------------------------------------------- run-directory classification (O4 – O6)
def _run_dir(tmp_path, *, status, outcome=None, checkpoint=True) -> Path:
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"status": status, "outcome": outcome}))
    if checkpoint:
        (d / "checkpoint_last.pt").write_bytes(b"x")
    return d


@pytest.mark.parametrize(("status", "outcome", "checkpoint", "expected"), [
    ("running", None, False, "aborted_before_commit"),
    ("failed", None, False, "aborted_before_commit"),
    ("interrupted", None, False, "aborted_before_commit"),
    ("failed", "failed_mechanical", False, "failed_mechanical"),
    ("running", None, True, "crashed"),
    ("interrupted", None, True, "interrupted"),
    ("failed", None, True, "error"),
    ("failed", "failed_mechanical", True, "failed_mechanical"),
    ("completed", "completed", True, "completed"),
    ("completed", "failed_diverged", True, "failed_diverged"),
])
def test_classify_run_dir(tmp_path, status, outcome, checkpoint, expected) -> None:
    assert at.classify_run_dir(_run_dir(tmp_path, status=status, outcome=outcome, checkpoint=checkpoint))[0] == expected


# --------------------------------------------------------------------------- attempt store (O6, O7)
def _mk_run(study: StudyPaths, n: int, **kw) -> Path:
    d = study.step_dir("L6_pilot") / "runs" / "rnn" / f"run_{n:03d}"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"status": kw.get("status", "running"), "outcome": kw.get("outcome")}))
    if kw.get("checkpoint", False):
        (d / "checkpoint_last.pt").write_bytes(b"x")
    return d


def test_attempt_lifecycle_counting_and_retries(study) -> None:
    with StudyLock(study, "test") as lock:
        store = at.AttemptStore(study, lock)
        d1 = _mk_run(study, 1, status="running")  # killed before the first commit
        a1 = store.launch(step="L6_pilot", arm="0.1", seed=100, run_dir=d1, authorisation="D-70")
        assert store.open_attempts() == [a1]
        assert store.reconcile() == [(a1, "aborted_before_commit")]
        d2 = _mk_run(study, 2, status="running", checkpoint=True)
        with pytest.raises(Blocked, match="retry_of"):
            store.launch(step="L6_pilot", arm="0.1", seed=100, run_dir=d2, authorisation="D-70")
        with pytest.raises(Blocked, match="reason"):
            store.launch(step="L6_pilot", arm="0.1", seed=100, run_dir=d2, authorisation="D-71", retry_of=a1)
        a2 = store.launch(step="L6_pilot", arm="0.1", seed=100, run_dir=d2, authorisation="D-71", retry_of=a1,
                          reason="killed by power loss before the first commit")
        assert store.reconcile() == [(a2, "crashed")]
        store.resumed(a2, recovery_head="abc")
        store.finish(a2, "completed")
        assert store.counted(("L6_pilot", "0.1", 100))["attempt_id"] == a2
        with pytest.raises(Blocked, match="counted outcome"):
            store.launch(step="L6_pilot", arm="0.1", seed=100, run_dir=d2, authorisation="D-72", retry_of=a2, reason="x")
        # O6: a Category A / D outcome is final and never retried
        d3 = _mk_run(study, 3)
        a3 = store.launch(step="L6_pilot", arm="1.0", seed=101, run_dir=d3, authorisation="D-70")
        store.finish(a3, "failed_diverged")
        assert store.counted(("L6_pilot", "1.0", 101))["status"] == "failed_diverged"
        with pytest.raises(Blocked):
            store.resumed(a3, recovery_head=None)
    with pytest.raises(RuntimeError, match="lock"):  # every write needs the study lock
        at.AttemptStore(study).launch(step="L6_pilot", arm="0.03", seed=100, run_dir=_mk_run(study, 9),
                                      authorisation="D-70")


def test_orphans_and_tampered_manifest(study) -> None:
    with StudyLock(study, "test") as lock:
        store = at.AttemptStore(study, lock)
        d1 = _mk_run(study, 1)
        store.launch(step="L6_pilot", arm="0.3", seed=102, run_dir=d1, authorisation="D-70")
        _mk_run(study, 2)
        assert store.orphans() == ["L6_pilot/runs/rnn/run_002"]
        store.launch(step="L6_pilot", arm="0.1", seed=102, run_dir=study.step_dir("L6_pilot") / "runs" / "rnn" / "run_002",
                     authorisation="D-70")
    text = study.manifest.read_text()
    first, rest = text.split("\n", 1)
    # An edit of any line but the last breaks the chain (an edit of the last line alone is not detectable
    # by a hash chain: tamper evidence, not immutability, as for the WaveNet log).
    study.manifest.write_text(first.replace("0.3", "0.4") + "\n" + rest)
    with pytest.raises(Blocked, match="verification"):
        at.AttemptStore(study).attempts()
    study.manifest.write_text(text + '{"event": "launched"')  # incomplete last line: never repaired
    with pytest.raises(Blocked, match="verification"):
        at.AttemptStore(study).attempts()


