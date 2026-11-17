"""MS7: kills, reconciliation, partial-record recovery and resume (O4, O5, O10; plan §9; D-49 P7-3, P7-4).

Each run, kill and resume happens in a child process (SIGKILL is real: no Python cleanup runs) with
fixed thread counts, so the comparisons are between identically configured processes. Children build
their synthetic data inline and never import the ``tests`` package by name.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from training.rnn_study import attempts as at
from training.rnn_study import recovery as rc
from training.rnn_study.common import REPO_ROOT

CHILD = r'''
import os, signal, string, sys
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from pathlib import Path
from training.wavenet_study.study_common import disable_git
disable_git()
import models.recurrent.rnn_training as rt
from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from utils.data import make_next_token_sequences
from utils.reproducibility import derive_seed, make_generator
from utils.run_manager import RunManager
from utils.vocabulary import Vocabulary

def names(count, seed):
    g = torch.Generator().manual_seed(seed)
    return ["".join(chr(97 + int(c)) for c in torch.randint(0, 26, (int(torch.randint(1, 9, (1,), generator=g)),), generator=g)) for _ in range(count)]

mode, root, kill, with_plan = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4] == "plan"
V = Vocabulary(chars=tuple(string.ascii_lowercase))
tr, dv = names(18, 40), names(6, 41)
train, dev = make_next_token_sequences(tr, V), make_next_token_sequences(dv, V)
cfg = rt.RNNTrainingConfig(lr=0.4, max_epochs=3, batch_size=4, eval_batch_size=4)
if mode == "run":
    run = RunManager.create("rnn", config={"f": 1}, results_root=root)
    print(run.run_dir, flush=True)
else:
    run = RunManager.open(root)
plan = None
if with_plan:
    probe_path = run.run_dir.parent.parent.parent / "probe.json"
    info = {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1], "N": 18, "M": int(train.lengths.sum())}
    probe = dg.load_probe(probe_path) if probe_path.exists() else dg.build_probe(train, tr, info, probe_path)
    plan = dg.RNNDiagnosticsPlan(probe=probe, train=train, train_names=tr, dev=dev, steps_per_epoch=5, max_epochs=3,
                                 recurrent_gain=1.0, out_dir=run.run_dir / "research_diagnostics")
if kill.startswith("o2:"):
    target, real, calls = int(kill[3:]), rt._o2, {"n": 0}
    def o2(model):
        calls["n"] += 1
        if calls["n"] == target:
            os.kill(os.getpid(), signal.SIGKILL)
        return real(model)
    rt._o2 = o2
elif kill.startswith("append:"):
    target, real = int(kill[7:]), rt._append_line_fsync
    def append(path, record):
        if path.name == "steps.jsonl" and record["step"] == target:
            import json as _j
            line = _j.dumps(record, sort_keys=True, separators=(",", ":"))
            with open(path, "a") as fh:
                fh.write(line[: len(line) // 2]); fh.flush(); os.fsync(fh.fileno())
            os.kill(os.getpid(), signal.SIGKILL)
        return real(path, record)
    rt._append_line_fsync = append
model = RNNLanguageModel(V, RNNConfig(hidden_dim=4), generator=make_generator(derive_seed(3, "init")))
res = rt.train_rnn(model, train, dev, training=cfg, seed=3, run=run, resume=(mode == "resume"), diagnostics=plan)
print("STATUS", res.status, flush=True)
'''


def child(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", CHILD, *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          timeout=600)


def start(root: Path, kill: str, plan: str) -> tuple[subprocess.CompletedProcess, Path]:
    out = child("run", str(root), kill, plan)
    return out, Path(out.stdout.splitlines()[0])


def lines(path: Path, strip=rc.TIMING_FIELDS) -> list[dict]:
    return [{k: v for k, v in json.loads(x).items() if k not in strip} for x in path.read_text().splitlines()]


def assert_same_records(a: Path, b: Path, with_plan: bool) -> None:
    assert lines(a / "history.jsonl") == lines(b / "history.jsonl")
    assert (a / "diagnostics" / "steps.jsonl").read_bytes() == (b / "diagnostics" / "steps.jsonl").read_bytes()
    import torch
    for name in ("model_best.pt", "model_final.pt", "checkpoint_last.pt"):
        x, y = torch.load(a / name, weights_only=True), torch.load(b / name, weights_only=True)
        assert all(torch.equal(x["model_state"][k], y["model_state"][k]) for k in x["model_state"]), name
        assert x["extra"] == y["extra"], name
    if with_plan:
        assert lines(a / "research_diagnostics" / "points.jsonl") == lines(b / "research_diagnostics" / "points.jsonl")
        assert (a / "research_diagnostics" / "primary.jsonl").read_bytes() == (b / "research_diagnostics" / "primary.jsonl").read_bytes()


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    out = {}
    for plan in ("none", "plan"):
        root = tmp_path_factory.mktemp(f"ref_{plan}")
        res, run_dir = start(root, "none", plan)
        assert res.returncode == 0 and "STATUS completed" in res.stdout, res.stderr
        out[plan] = run_dir
    return out


def test_O4_kill_before_first_commit(tmp_path) -> None:
    """O4: SIGKILL before the first commit → ``aborted_before_commit``; recovery and resume both refuse;
    the directory is unchanged; a retry needs a new directory (attempt store, ``test_attempts.py``)."""
    res, run_dir = start(tmp_path, "o2:2", "none")
    assert res.returncode == -signal.SIGKILL
    assert at.classify_run_dir(run_dir)[0] == "aborted_before_commit"
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    with pytest.raises(rc.RecoveryError, match="no committed checkpoint"):
        rc.preflight(run_dir, "s1")
    assert child("resume", str(run_dir), "none", "none").returncode != 0
    after = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file() and "recovery" not in p.parts
             and p.name != "recovery.jsonl"}
    assert after == before


@pytest.mark.parametrize("plan", ["none", "plan"])
@pytest.mark.parametrize("kill", ["o2:3", "append:8"], ids=["after-commit", "mid-append"])
def test_O5_O10_kill_recover_resume_equals_uninterrupted(tmp_path, reference, plan, kill) -> None:
    """O5 / O10: SIGKILL after a commit (``crashed``), or in the middle of a step-record append (a
    partial last line): the preflight preserves every file, validates all records, removes only the
    partial tail beyond the commit point, and records it; the resumed run (P7-3: same identity, no new
    authorisation) equals the uninterrupted run; the post-check finds the records consistent."""
    res, run_dir = start(tmp_path, kill, plan)
    assert res.returncode == -signal.SIGKILL, res.stderr
    assert at.classify_run_dir(run_dir)[0] == "crashed"
    steps = run_dir / "diagnostics" / "steps.jsonl"
    partial = not steps.read_bytes().endswith(b"\n")
    assert partial == kill.startswith("append")
    original = steps.read_bytes()
    entry = rc.preflight(run_dir, "s1")
    assert entry["result"] == "recovered" and entry["commit"] == {"epoch": 1, "step": 5}
    assert (run_dir / "recovery" / "s1" / "original" / "diagnostics" / "steps.jsonl").read_bytes() == original
    assert (entry["files"]["diagnostics/steps.jsonl"]["partial_tail_bytes"] > 0) == partial
    assert original.startswith(steps.read_bytes()) and steps.read_bytes().endswith(b"\n")
    at.mark_crashed_interrupted(run_dir)
    resumed = child("resume", str(run_dir), "none", plan)
    assert resumed.returncode == 0 and "STATUS completed" in resumed.stdout, resumed.stderr
    assert_same_records(run_dir, reference[plan], plan == "plan")
    check = rc.postcheck(run_dir, "s1")
    assert check["result"] == "consistent", check
    log = [json.loads(x) for x in (run_dir / rc.RECOVERY_LOG).read_text().splitlines()]
    assert [e["phase"] for e in log] == ["preflight", "postcheck"]


def test_preflight_aborts_on_damaged_committed_record(tmp_path) -> None:
    """P7-4: a malformed record before the commit point cannot be recovered safely: the preflight
    aborts, records the abort, and changes no active file (the run becomes ``unrecoverable``)."""
    res, run_dir = start(tmp_path, "o2:3", "none")
    assert res.returncode == -signal.SIGKILL
    hist = run_dir / "history.jsonl"
    rows = hist.read_text().splitlines(keepends=True)
    hist.write_text(rows[0] + "{not json}\n" + "".join(rows[2:]))
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    with pytest.raises(rc.RecoveryError, match="malformed"):
        rc.preflight(run_dir, "s1")
    after = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file() and "recovery" not in p.parts
             and p.name != rc.RECOVERY_LOG}
    assert after == before
    log = [json.loads(x) for x in (run_dir / rc.RECOVERY_LOG).read_text().splitlines()]
    assert log[-1]["result"] == "aborted"


def test_postcheck_flags_a_mismatch_for_review(tmp_path, reference) -> None:
    """A resumed record that differs from its preserved original is flagged, not accepted silently."""
    res, run_dir = start(tmp_path, "o2:3", "none")
    rc.preflight(run_dir, "s1")
    at.mark_crashed_interrupted(run_dir)
    assert child("resume", str(run_dir), "none", "none").returncode == 0
    steps = run_dir / "diagnostics" / "steps.jsonl"
    recs = steps.read_text().splitlines(keepends=True)
    recs[0] = recs[0].replace('"finite":true', '"finite":false')
    steps.write_text("".join(recs))
    assert rc.postcheck(run_dir, "s1")["result"] == "review_required"
