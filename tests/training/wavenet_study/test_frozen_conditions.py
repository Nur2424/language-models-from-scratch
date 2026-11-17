"""Stage 3: frozen conditions and checksums (protocol §10.1; DEC-1; AM-7). Synthetic files only."""

from __future__ import annotations

import json
import math

import pytest
import torch

from training.wavenet_study import study_common as sc


def _current(**over):
    c = {
        "protocol": {"seeds": (100, 101, 102), "caps": [30, 60], "epsilon_floor": 0.001},
        "arms": {"A": {"config": {"training": {"lr": 0.03, "batch_size": 32}}}},
        "threads": {"intra_op": 1, "inter_op": 8},
        "environment": {"python": "3.12.6", "packages": {"torch": "2.10.0"}},
        "checksums": {"blocking": {"utils/io.py": "a" * 64}, "recorded": {"wavenet_spec.md": "b" * 64}},
    }
    for path, value in over.items():
        node = c
        *head, last = path.split("/")
        for part in head:
            node = node[part]
        node[last] = value
    return c


@pytest.fixture()
def paths(tmp_path):
    return sc.PilotPaths(tmp_path / "pilot")


def _freeze(paths, current=None):
    with sc.PilotLock(paths, "test") as lock:
        return sc.verify_or_freeze(paths, current or _current(), may_create=True, lock=lock)


def test_first_run_freezes_and_later_identical_conditions_verify(paths):
    frozen, nonblocking = _freeze(paths)
    assert nonblocking == [] and paths.conditions.is_file()
    assert frozen["conditions"]["protocol"]["seeds"] == [100, 101, 102]  # JSON-normalised (tuples → lists)
    assert frozen["conditions_sha256"] == sc.sha256_json(frozen["conditions"])
    again, nb = sc.verify_or_freeze(paths, _current(), may_create=False, lock=None)
    assert again == frozen and nb == []


def test_freezing_requires_the_lock_and_may_create(paths):
    with pytest.raises(RuntimeError, match="lock"):
        sc.verify_or_freeze(paths, _current(), may_create=True, lock=None)
    with pytest.raises(sc.Blocked, match="not been started"):
        sc.verify_or_freeze(paths, _current(), may_create=False, lock=None)
    assert not paths.conditions.exists()


@pytest.mark.parametrize("change,path", [
    ({"arms/A/config/training/lr": math.nextafter(0.03, 1.0)}, '["arms", "A", "config", "training", "lr"]'),
    ({"threads/intra_op": 2}, '["threads", "intra_op"]'),
    ({"environment/packages": {"torch": "2.11.0"}}, '["environment", "packages", "torch"]'),
    ({"checksums/blocking": {"utils/io.py": "c" * 64}}, '["checksums", "blocking", "utils/io.py"]'),
    ({"checksums/blocking": {"utils/io.py": "a" * 64, "utils/new.py": "d" * 64}},
     '["checksums", "blocking", "utils/new.py"]'),
    ({"protocol/caps": [30]}, '["protocol", "caps"]'),
])
def test_any_blocking_difference_is_refused_with_its_path(paths, change, path):
    _freeze(paths)
    with pytest.raises(sc.Blocked) as info:
        sc.verify_or_freeze(paths, _current(**change), may_create=False, lock=None)
    assert path in info.value.message and "amendment" in info.value.message


def test_recorded_document_checksums_are_reported_but_never_block(paths):
    _freeze(paths)
    frozen, nonblocking = sc.verify_or_freeze(
        paths, _current(**{"checksums/recorded": {"wavenet_spec.md": "e" * 64}}), may_create=False, lock=None)
    assert nonblocking == [("checksums", "recorded", "wavenet_spec.md")]
    assert frozen["conditions"]["checksums"]["recorded"]["wavenet_spec.md"] == "b" * 64  # never rewritten


def test_edited_or_corrupt_conditions_file_is_refused(paths):
    _freeze(paths)
    data = json.loads(paths.conditions.read_text())
    data["conditions"]["threads"]["intra_op"] = 4  # edited after freezing, hash left unchanged
    paths.conditions.write_text(json.dumps(data))
    with pytest.raises(sc.Blocked, match="conditions_sha256"):
        sc.verify_or_freeze(paths, _current(**{"threads/intra_op": 4}), may_create=False, lock=None)
    paths.conditions.write_text("{not json")
    with pytest.raises(sc.Blocked, match="not valid JSON"):
        sc.load_frozen(paths)
    paths.conditions.write_text(json.dumps({"conditions": {}}))
    with pytest.raises(sc.Blocked, match="lacks"):
        sc.load_frozen(paths)


def test_manifest_without_conditions_is_inconsistent(paths):
    paths.root.mkdir(parents=True)
    sc.HashChainedLog(paths.manifest).append({"event": "launched", "attempt_id": "x"})
    with sc.PilotLock(paths, "test") as lock, pytest.raises(sc.Blocked, match="inconsistent"):
        sc.verify_or_freeze(paths, _current(), may_create=True, lock=lock)


def test_non_finite_values_cannot_be_frozen(paths):
    with sc.PilotLock(paths, "test") as lock, pytest.raises(ValueError):
        sc.verify_or_freeze(paths, _current(**{"protocol/epsilon_floor": float("nan")}), may_create=True, lock=lock)


def test_sha256_json_is_canonical():
    assert sc.sha256_json({"a": 1, "b": [1, 2]}) == sc.sha256_json({"b": [1, 2], "a": 1})
    assert sc.sha256_json({"a": 1.0}) != sc.sha256_json({"a": 1})


# ----------------------------------------------------------------- checksum records
@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "utils" / "sub" / "__pycache__").mkdir(parents=True)
    (r / "utils" / "a.py").write_text("A = 1\n")
    (r / "utils" / "sub" / "b.py").write_text("B = 2\n")
    (r / "utils" / "sub" / "__pycache__" / "b.py").write_text("ignored\n")
    (r / "spec.md").write_text("spec\n")
    return r


def test_checksum_record_classes_and_change_detection(repo):
    files = sc.python_files_under(repo, "utils")
    assert files == ["utils/a.py", "utils/sub/b.py"]
    rec = sc.checksum_record(repo, blocking=files, recorded=["spec.md"])
    assert set(rec) == {"blocking", "recorded"} and rec["blocking"]["utils/a.py"] == sc.sha256_file(repo / "utils/a.py")
    (repo / "utils" / "a.py").write_text("A = 2\n")
    assert sc.checksum_record(repo, blocking=files, recorded=["spec.md"])["blocking"]["utils/a.py"] != \
        rec["blocking"]["utils/a.py"]


def test_missing_or_doubly_classified_checksum_files_are_refused(repo):
    with pytest.raises(sc.Blocked, match="missing"):
        sc.checksum_record(repo, blocking=["utils/nope.py"], recorded=[])
    with pytest.raises(ValueError, match="both"):
        sc.checksum_record(repo, blocking=["spec.md"], recorded=["spec.md"])


def test_environment_record_contains_the_thread_configuration():
    env = sc.environment_record()  # runs under the test package's Git guard
    assert env["intra_op_threads"] == torch.get_num_threads()
    assert env["inter_op_threads"] == torch.get_num_interop_threads()
    assert env["packages"]["torch"] is not None and torch.__version__.startswith(env["packages"]["torch"])
    assert {"python", "python_executable", "platform", "machine", "cpu"} <= set(env)
