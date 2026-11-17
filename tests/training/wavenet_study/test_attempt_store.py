"""Stage 3: exclusive lock and attempt store (protocol §9.4, §10.5, §12.3; approvals A4).

Synthetic records only. Real concurrency and crashes use short-lived child Python processes
that import only ``study_common`` (no model, no data, no Git: every child disables all
subprocess creation before importing anything else).
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from training.wavenet_study import study_common as sc

REPO = Path(sc.__file__).resolve().parents[2]
KEY = {"arm": "A", "cap": 30, "seed": 100}


def aid_of(key, n):
    return f"{key['arm']}-c{key['cap']}-s{key['seed']}-a{n}"


@pytest.fixture()
def paths(tmp_path):
    return sc.PilotPaths(tmp_path / "pilot")


def store_for(paths, lock=None):
    return sc.AttemptStore(paths, key_fields=("arm", "cap", "seed"), attempt_id=aid_of, lock=lock)


def result_for(aid, key, status, **extra):
    return {"attempt_id": aid, **key, "status": status, "phase": "training", "run_dir": None,
            "failure": None if status == "completed" else {"type": "X", "message": "m", "traceback": "tb"},
            "conditions_sha256": "f" * 64, **extra}


def run_attempt(paths, key, status, *, retry=None, write_result=True, **extra):
    with sc.PilotLock(paths, "test") as lock:
        store = store_for(paths, lock)
        aid = store.launch(key, retry=retry)
        if write_result:
            sc.write_attempt_result(paths, aid, result_for(aid, key, status, **extra))
        return store.finish(aid, returncode=0)


# ============================================================ lock
def test_lock_is_exclusive_and_released_on_exit_and_on_exception(paths):
    with sc.PilotLock(paths, "first") as lock:
        assert lock.held and sc.lock_held_elsewhere(paths)
        assert "command first" in paths.lock_file.read_text()
        with pytest.raises(sc.Blocked, match="another pilot command"), sc.PilotLock(paths, "second"):
            pass
    assert not sc.lock_held_elsewhere(paths)
    with pytest.raises(ValueError), sc.PilotLock(paths, "third"):
        raise ValueError("boom")
    with sc.PilotLock(paths, "fourth"):
        pass


def test_writing_without_the_lock_is_refused(paths):
    store = store_for(paths)
    with pytest.raises(RuntimeError, match="lock"):
        store.launch(KEY)
    with sc.PilotLock(paths, "x") as lock:
        aid = store_for(paths, lock).launch(KEY)
    with pytest.raises(RuntimeError, match="lock"):
        store_for(paths, lock).finish(aid)  # lock object no longer held
    with pytest.raises(RuntimeError, match="lock"):
        sc.log_invocation(paths, lock, "run")


CHILD = textwrap.dedent("""
    import os, subprocess, sys, time
    def _refuse(*a, **k):
        raise PermissionError("subprocess creation disabled in this test child")
    subprocess.Popen._execute_child = _refuse
    os.system = _refuse
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from training.wavenet_study import study_common as sc
    paths, mode = sc.PilotPaths(Path(sys.argv[2])), sys.argv[3]
    key = {"arm": "A", "cap": 30, "seed": int(sys.argv[4])}
    aid_of = lambda k, n: f"{k['arm']}-c{k['cap']}-s{k['seed']}-a{n}"
    try:
        with sc.PilotLock(paths, "child-" + mode) as lock:
            store = sc.AttemptStore(paths, key_fields=("arm", "cap", "seed"), attempt_id=aid_of, lock=lock)
            if mode == "hold":
                print("ready", flush=True)
                time.sleep(60)
            aid = store.launch(key)
            print("ready " + aid, flush=True)
            if mode == "launch_and_hang":
                time.sleep(60)
            time.sleep(0.3)  # keep the lock while the other child tries
            sc.write_attempt_result(paths, aid, {"attempt_id": aid, **key, "status": "completed"})
            store.finish(aid, returncode=0)
            print("done " + aid, flush=True)
    except sc.Blocked as exc:
        print("BLOCKED " + exc.message, flush=True)
        sys.exit(2)
""")


def spawn(paths, mode, seed=100):
    return subprocess.Popen([sys.executable, "-c", CHILD, str(REPO), str(paths.root), mode, str(seed)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def stop(proc, sig=signal.SIGKILL):
    if proc.poll() is None:
        proc.send_signal(sig)
    proc.wait()
    proc.stdout.close()
    proc.stderr.close()


def wait_ready(proc):
    line = proc.stdout.readline()
    assert line.startswith("ready"), (line, proc.stderr.read() if proc.poll() is not None else "")
    return line.split()[1] if len(line.split()) > 1 else None


def test_lock_held_by_another_process_blocks_every_command(paths):
    holder = spawn(paths, "hold")
    try:
        wait_ready(holder)
        assert sc.lock_held_elsewhere(paths)
        with pytest.raises(sc.Blocked, match="another pilot command"), sc.PilotLock(paths, "parent"):
            pass
    finally:
        stop(holder)
    with sc.PilotLock(paths, "parent"):  # the dead process's lock is gone
        pass


def test_two_concurrent_invocations_never_both_launch(paths):
    procs = [spawn(paths, "race"), spawn(paths, "race")]
    outs = [p.communicate(timeout=60)[0] for p in procs]  # communicate() closes the pipes
    codes = sorted(p.returncode for p in procs)
    assert codes == [0, 2], outs
    attempts = store_for(paths).attempts()
    assert list(attempts) == ["A-c30-s100-a1"] and attempts["A-c30-s100-a1"]["status"] == "completed"
    assert any("BLOCKED" in o for o in outs)


def test_killed_runner_is_detected_and_registered_as_crashed(paths):
    proc = spawn(paths, "launch_and_hang")
    try:
        aid = wait_ready(proc)
        rec = store_for(paths).verify_launched(aid, KEY)  # child-side check while the runner lives
        assert rec["status"] == "launched"
        with pytest.raises(sc.Blocked):
            store_for(paths).verify_launched(aid, {**KEY, "seed": 101})
    finally:
        stop(proc)  # SIGKILL: the runner dies without closing its attempt
    with pytest.raises(sc.Blocked, match="not registered as launched"):
        store_for(paths).verify_launched(aid, KEY)  # no active invocation any more
    with sc.PilotLock(paths, "recovery") as lock:
        store = store_for(paths, lock)
        with pytest.raises(sc.Blocked, match="unresolved"):
            store.require_runnable()
        assert store.check_state() == [f"{aid} -> crashed"]
        rec = store.attempts()[aid]
        assert rec["status"] == "crashed" and rec["events"] == ["launched", "registered"]
        with pytest.raises(sc.Blocked, match="unresolved"):
            store.require_runnable()
        assert store.check_state() == []  # registration happens once


# ============================================================ lifecycle
def test_launch_finish_lifecycle_and_hash_chained_manifest(paths):
    rec = run_attempt(paths, KEY, "completed")
    assert rec["attempt_id"] == "A-c30-s100-a1" and rec["status"] == "completed"
    assert rec["events"] == ["launched", "finished"] and rec["key"] == ("A", 30, 100)
    assert rec["result_sha256"] == sc.sha256_file(paths.attempt_result("A-c30-s100-a1"))
    assert "traceback" not in json.dumps(rec)
    entries, head = sc.HashChainedLog(paths.manifest).read()
    assert len(entries) == 2 and head != sc.GENESIS


def test_attempts_without_a_result_are_crashed_or_interrupted(paths):
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        a = store.launch(KEY)
        assert store.finish(a, returncode=-9)["status"] == "crashed"
        b = store.launch({**KEY, "seed": 101})
        assert store.finish(b, interrupted=True)["status"] == "interrupted"
        with pytest.raises(sc.Blocked, match="not an unfinished"):
            store.finish(a)


def test_a_runner_that_died_after_the_child_wrote_its_result_is_registered_from_it(paths):
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        aid = store.launch(KEY)
        sc.write_attempt_result(paths, aid, result_for(aid, KEY, "mechanically_unstable"))
    with sc.PilotLock(paths, "y") as lock:
        store = store_for(paths, lock)
        assert store.check_state() == [f"{aid} -> mechanically_unstable"]


def test_duplicate_and_unauthorised_launches_are_refused(paths):
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        aid = store.launch(KEY)
        with pytest.raises(sc.Blocked, match="unfinished"):
            store.launch(KEY)
        sc.write_attempt_result(paths, aid, result_for(aid, KEY, "error"))
        store.finish(aid)
        with pytest.raises(sc.Blocked, match="authorised retry"):
            store.launch(KEY)
        with pytest.raises(ValueError, match="may not set"):
            store.launch({**KEY, "seed": 102}, retry_of=aid)
        with pytest.raises(sc.Blocked, match="no attempt to retry"):
            store.launch({**KEY, "seed": 102}, retry={"retry_of": aid})


def test_completed_attempts_are_final(paths):
    run_attempt(paths, KEY, "completed")
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        with pytest.raises(sc.Blocked, match="final"):
            store.launch(KEY)
        with pytest.raises(sc.Blocked, match="final"):
            store.authorize_retry(KEY, reason="a sufficiently long reason", authorized_by="Nur",
                                  conditions_sha256="f" * 64)


def test_result_files_are_written_once_and_must_match_the_attempt(paths):
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        aid = store.launch(KEY)
        sc.write_attempt_result(paths, aid, result_for(aid, {**KEY, "seed": 101}, "completed"))
        with pytest.raises(sc.Blocked, match="already exists"):
            sc.write_attempt_result(paths, aid, result_for(aid, KEY, "completed"))
        with pytest.raises(sc.Blocked, match="does not belong"):
            store.finish(aid)
    with pytest.raises(ValueError):
        sc.write_attempt_result(paths, "other", result_for("x", KEY, "completed"))


@pytest.mark.parametrize("bad", [{"arm": "A", "cap": 30.0, "seed": 100}, {"arm": "A", "cap": True, "seed": 100},
                                 {"arm": "A", "seed": 100}])
def test_key_values_must_be_complete_strings_or_ints(paths, bad):
    with sc.PilotLock(paths, "x") as lock, pytest.raises(sc.Blocked):
        store_for(paths, lock).launch(bad)


def test_pending_removes_duplicates_and_existing_keys(paths):
    run_attempt(paths, KEY, "completed")
    keys = [KEY, ("A", 30, 101), {"arm": "A", "cap": 30, "seed": 101}, ("B", 30, 100)]
    assert store_for(paths).pending(keys) == [{"arm": "A", "cap": 30, "seed": 101},
                                              {"arm": "B", "cap": 30, "seed": 100}]


# ============================================================ manifest integrity
def _raw_append(paths, row):
    sc.HashChainedLog(paths.manifest).append(row)


@pytest.mark.parametrize("rows,match", [
    ([{"event": "finished", "attempt_id": "x", **KEY, "status": "completed"}], "unknown attempt"),
    ([{"event": "launched", "attempt_id": "x", **KEY}, {"event": "launched", "attempt_id": "x", **KEY}],
     "launched twice"),
    ([{"event": "launched", "attempt_id": "x", **KEY}, {"event": "finished", "attempt_id": "x", **KEY,
                                                         "status": "great"}], "unknown status"),
    ([{"event": "launched", "attempt_id": "x", **KEY},
      {"event": "finished", "attempt_id": "x", **KEY, "status": "error"},
      {"event": "registered", "attempt_id": "x", **KEY, "status": "crashed"}], "already closed"),
    ([{"event": "launched", "attempt_id": "x", **KEY},
      {"event": "finished", "attempt_id": "x", **KEY, "seed": 1, "status": "error"}], "key changed"),
    ([{"event": "deleted", "attempt_id": "x"}], "invalid event"),
])
def test_invalid_manifest_sequences_block(paths, rows, match):
    paths.root.mkdir(parents=True)
    for row in rows:
        _raw_append(paths, row)
    with pytest.raises(sc.Blocked, match=match):
        store_for(paths).attempts()


def test_edited_manifest_breaks_the_chain(paths):
    run_attempt(paths, KEY, "error")
    run_attempt(paths, {**KEY, "seed": 101}, "completed")
    original = paths.manifest.read_text()
    lines = original.splitlines()
    obj = json.loads(lines[1])
    obj["status"] = "completed"  # rewrite an error into a success
    paths.manifest.write_text("\n".join([lines[0], sc.canonical_json(obj), *lines[2:]]) + "\n")
    with pytest.raises(sc.Blocked, match="failed verification"):
        store_for(paths).attempts()
    paths.manifest.write_text("\n".join(lines[:1] + lines[2:]) + "\n")  # a deleted line
    with pytest.raises(sc.Blocked, match="failed verification"):
        store_for(paths).attempts()


def test_edit_of_the_last_manifest_line_is_visible_only_through_the_recorded_head(paths):
    run_attempt(paths, KEY, "error")
    head = sc.HashChainedLog(paths.manifest).read()[1]
    lines = paths.manifest.read_text().splitlines()
    obj = json.loads(lines[-1])
    obj["status"] = "completed"
    paths.manifest.write_text("\n".join([*lines[:-1], sc.canonical_json(obj)]) + "\n")
    assert store_for(paths).attempts()["A-c30-s100-a1"]["status"] == "completed"  # chain alone: undetected
    assert sc.HashChainedLog(paths.manifest).read()[1] != head  # the recorded head exposes it (A5 limitation)


def test_invocations_are_hash_chained(paths):
    with sc.PilotLock(paths, "x") as lock:
        h1 = sc.log_invocation(paths, lock, "run", arms=["A"], argv=["x"])
        h2 = sc.log_invocation(paths, lock, "retry", reason="r")
    entries, head = sc.HashChainedLog(paths.invocations).read()
    assert [e["command"] for e in entries] == ["run", "retry"] and head == h2 != h1


# ============================================================ orphans
def _run_dir(paths, name, attempt_id=None, metadata=True):
    d = paths.runs_dir / "mlp" / name
    d.mkdir(parents=True)
    if metadata:
        (d / "metadata.json").write_text(json.dumps({"pilot": {"attempt_id": attempt_id}}))
    return d


@pytest.mark.parametrize("case", ["unknown_attempt", "no_metadata", "two_dirs", "orphan_result"])
def test_orphan_records_block_the_pilot(paths, case):
    rec = run_attempt(paths, KEY, "completed")
    if case == "unknown_attempt":
        _run_dir(paths, "run_001", "Z-c30-s1-a1")
    elif case == "no_metadata":
        _run_dir(paths, "run_001", metadata=False)
    elif case == "two_dirs":
        _run_dir(paths, "run_001", rec["attempt_id"])
        _run_dir(paths, "run_002", rec["attempt_id"])
    else:
        (paths.attempts_dir / "ghost.json").write_text("{}")
    with sc.PilotLock(paths, "x") as lock, pytest.raises(sc.Blocked, match="without an attempt"):
        store_for(paths, lock).check_state()


def test_crashed_attempt_records_its_single_run_directory(paths):
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        aid = store.launch(KEY)
        _run_dir(paths, "run_001", aid)
        rec = store.finish(aid, returncode=-9)
    assert rec["status"] == "crashed" and rec["run_dir"].endswith("runs/mlp/run_001")
    with sc.PilotLock(paths, "y") as lock:
        assert store_for(paths, lock).check_state() == []


# ============================================================ halts, integrity blocks and retries
def test_mechanical_instability_halts_runs_and_retries(paths):
    run_attempt(paths, KEY, "mechanically_unstable")
    run_attempt(paths, {**KEY, "seed": 101}, "error")
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        assert store.sweep_state()["halted"] == ["A-c30-s100-a1"]
        with pytest.raises(sc.Blocked, match="halted"):
            store.require_runnable()
        for key in (KEY, {**KEY, "seed": 101}):
            with pytest.raises(sc.Blocked, match="halted"):
                store.authorize_retry(key, reason="a sufficiently long reason", authorized_by="Nur",
                                      conditions_sha256="f" * 64)


def test_integrity_blocked_requires_authorised_retry(paths):
    run_attempt(paths, KEY, "integrity_blocked")
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        with pytest.raises(sc.Blocked, match="integrity_blocked"):
            store.require_runnable()
        for who in (None, "", "   "):
            with pytest.raises(sc.Blocked, match="--authorized-by"):
                store.authorize_retry(KEY, reason="G5 mismatch reviewed: rerun", authorized_by=who,
                                      conditions_sha256="f" * 64)
        token = store.authorize_retry(KEY, reason="G5 mismatch reviewed: rerun", authorized_by=" Nur ",
                                      conditions_sha256="f" * 64)
        assert token == {"retry_of": "A-c30-s100-a1", "retry_reason": "G5 mismatch reviewed: rerun",
                         "retry_authorized_by": "Nur", "conditions_sha256": "f" * 64}
        forged = {**token, "retry_of": "A-c30-s100-a0"}  # not the attempt being retried
        with pytest.raises(sc.Blocked, match="does not match"):
            store.launch(KEY, retry=forged)
        aid = store.launch(KEY, retry=token)
        rec = store.attempts()[aid]
        assert aid == "A-c30-s100-a2" and rec["retry_of"] == "A-c30-s100-a1"
        assert rec["retry_authorized_by"] == "Nur" and "conditions_sha256" not in rec
        with pytest.raises(sc.Blocked, match="unfinished"):
            store.launch(KEY, retry=token)  # a token cannot be reused
        sc.write_attempt_result(paths, aid, result_for(aid, KEY, "completed"))
        store.finish(aid)
        store.require_runnable()  # the sweep may continue after the authorised retry
        assert [r["status"] for r in store.by_key()[("A", 30, 100)]] == ["integrity_blocked", "completed"]


@pytest.mark.parametrize("status", ["error", "crashed"])
def test_unresolved_attempts_need_a_reason_and_matching_conditions(paths, status):
    run_attempt(paths, KEY, status, write_result=status == "error")  # no result file -> crashed
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        assert store.by_key()[("A", 30, 100)][-1]["status"] == status
        with pytest.raises(sc.Blocked, match="unresolved"):
            store.require_runnable()
        for empty in ("", "   \n\t", None):
            with pytest.raises(sc.Blocked, match="non-empty written reason"):
                store.authorize_retry(KEY, reason=empty, authorized_by=None, conditions_sha256="f" * 64)
        assert store.authorize_retry(KEY, reason="ok", authorized_by=None,  # no minimum length
                                     conditions_sha256="f" * 64)["retry_reason"] == "ok"
        if status == "error":
            with pytest.raises(sc.Blocked, match="different conditions"):
                store.authorize_retry(KEY, reason="fixed infrastructure problem", authorized_by=None,
                                      conditions_sha256="0" * 64)
        token = store.authorize_retry(KEY, reason="fixed infrastructure problem", authorized_by=None,
                                      conditions_sha256="f" * 64)
        assert token["retry_authorized_by"] is None
        assert store.launch(KEY, retry=token) == "A-c30-s100-a2"


def test_interrupted_attempt_retry(paths):
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        aid = store.launch(KEY)
        store.finish(aid, interrupted=True)
        with pytest.raises(sc.Blocked, match="unresolved"):
            store.require_runnable()
        token = store.authorize_retry(KEY, reason="Ctrl-C by the operator", authorized_by=None,
                                      conditions_sha256="f" * 64)
        assert store.launch(KEY, retry=token) == "A-c30-s100-a2"


def test_every_attempt_is_kept_in_launch_order(paths):
    run_attempt(paths, KEY, "error")
    with sc.PilotLock(paths, "x") as lock:
        store = store_for(paths, lock)
        token = store.authorize_retry(KEY, reason="fixed infrastructure problem", authorized_by=None,
                                      conditions_sha256="f" * 64)
        aid = store.launch(KEY, retry=token)
        time.sleep(0.01)
        store.finish(aid, returncode=-9)
    recs = store_for(paths).by_key()[("A", 30, 100)]
    assert [(r["attempt_id"], r["status"]) for r in recs] == [("A-c30-s100-a1", "error"), ("A-c30-s100-a2", "crashed")]
