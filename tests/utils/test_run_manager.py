import json
import threading

import pytest

from utils.io import read_jsonl
from utils.run_manager import RunError, RunManager, RunStatus, git_info


@pytest.fixture
def make_run(tmp_path):
    def _make(model="toy", **kw):
        return RunManager.create(model, config={"model": {"name": model}}, results_root=tmp_path, **kw)
    return _make


def test_layout_and_initial_metadata(make_run, tmp_path):
    run = make_run(dataset_fingerprint={"names_sha256": "abc"})
    assert run.run_dir == tmp_path / "runs" / "toy" / "run_001"
    assert run.config_path.is_file() and run.logs_dir.is_dir() and run.diagnostics_dir.is_dir()
    assert json.loads(run.config_path.read_text()) == {"model": {"name": "toy"}}
    meta = run.read_metadata()
    assert meta["run_id"] == "toy/run_001" and meta["status"] == "created"
    assert meta["dataset"] == {"names_sha256": "abc"}
    assert "torch" in meta["environment"] and "available" in meta["git"]
    assert run.checkpoint_path("best").name == "model_best.pt"


def test_unique_directories_never_overwrite(make_run):
    runs = [make_run() for _ in range(3)]
    assert [r.run_dir.name for r in runs] == ["run_001", "run_002", "run_003"]
    assert make_run("other").run_dir.name == "run_001"


def test_concurrent_creation_is_collision_free(make_run):
    results, errors = [], []

    def worker():
        try:
            results.append(make_run().run_dir.name)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(set(results)) == 8


def test_lifecycle_completed(make_run):
    run = make_run()
    with run.track():
        run.append_history({"step": 0, "train_nll": 3.3})
        run.append_history({"step": 1, "train_nll": 3.1})
    meta = run.read_metadata()
    assert meta["status"] == "completed" and meta["started_at"] and meta["ended_at"]
    assert meta["active_duration_seconds"] >= 0 and len(meta["sessions"]) == 1
    assert meta["sessions"][0]["outcome"] == "completed"
    assert [r["step"] for r in read_jsonl(run.history_path)] == [0, 1]
    with pytest.raises(RunError, match="invalid status transition"):
        run.mark_running()


def test_failure_records_exception_and_reraises(make_run):
    run = make_run()
    with pytest.raises(ZeroDivisionError):
        with run.track():
            1 / 0
    meta = run.read_metadata()
    assert meta["status"] == "failed" and meta["error"]["type"] == "ZeroDivisionError"
    assert "1 / 0" in meta["error"]["traceback"]


def test_interrupt_then_resume(make_run):
    run = make_run()
    with pytest.raises(KeyboardInterrupt):
        with run.track():
            raise KeyboardInterrupt
    assert run.status is RunStatus.INTERRUPTED
    reopened = RunManager.open(run.run_dir)
    with reopened.track():
        pass
    meta = reopened.read_metadata()
    assert meta["status"] == "completed" and meta["error"] is None
    assert [s["outcome"] for s in meta["sessions"]] == ["interrupted", "completed"]


def test_summary_and_metadata_updates(make_run):
    run = make_run()
    run.mark_running()
    run.update_metadata(results={"dev_nll": 2.1})
    run.update_metadata(results={"test_nll": 2.2}, notes="x")
    run.mark_completed(summary={"best_step": 10})
    meta = run.read_metadata()
    assert meta["results"] == {"dev_nll": 2.1, "test_nll": 2.2} and meta["notes"] == "x"
    assert meta["summary"] == {"best_step": 10}
    with pytest.raises(RunError, match="mark_"):
        run.update_metadata(status="running")


def test_validation(tmp_path):
    with pytest.raises(ValueError):
        RunManager.create("Bad Name", config={}, results_root=tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        RunManager.create("toy", config={}, results_root=tmp_path, extra_metadata={"status": "x"})
    with pytest.raises(RunError, match="not a run directory"):
        RunManager.open(tmp_path)


def test_git_info_never_raises(tmp_path):
    info = git_info(tmp_path)  # not a repository
    assert info["available"] is False and info["reason"]
