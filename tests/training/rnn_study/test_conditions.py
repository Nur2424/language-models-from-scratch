"""MS7: frozen conditions and study conditions (O3) (plan §8.5)."""


from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.rnn_study import attempts as at
from training.rnn_study import conditions as cd
from training.rnn_study.common import REPO_ROOT, Blocked, StudyLock, StudyPaths, lock_held_elsewhere



# --------------------------------------------------------------------------- frozen conditions (O3)
def _fake_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    for rel in cd.FROZEN_FILES + ("training/rnn_study/common.py",):
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content of {rel}\n")
    return repo


def test_O3_frozen_conditions_detect_any_change(tmp_path) -> None:
    repo = _fake_repo(tmp_path)
    frozen = cd.compute_frozen_conditions(repo)
    conditions = {"frozen": frozen}
    cd.verify_frozen(conditions, repo)
    (repo / "models/recurrent/rnn.py").write_text("changed\n")
    with pytest.raises(cd.IntegrityBlocked, match="rnn.py"):
        cd.verify_frozen(conditions, repo)
    (repo / "models/recurrent/rnn.py").write_text("content of models/recurrent/rnn.py\n")
    (repo / "training/rnn_study/new_driver.py").write_text("x = 1\n")
    with pytest.raises(cd.IntegrityBlocked, match="new_driver"):
        cd.verify_frozen(conditions, repo)
    (repo / "training/rnn_study/new_driver.py").unlink()
    env = {**frozen, "environment": {**frozen["environment"], "torch": "0.0.0"}}
    with pytest.raises(cd.IntegrityBlocked, match="environment"):
        cd.verify_frozen({"frozen": env}, repo)


def test_conditions_file_and_values(study) -> None:
    with pytest.raises(Blocked, match="does not exist"):
        cd.load_conditions(study)
    study.root.mkdir(parents=True)
    study.conditions.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(cd.IntegrityBlocked, match="schema"):
        cd.load_conditions(study)
    study.conditions.write_text(json.dumps({"schema": cd.CONDITIONS_SCHEMA, "frozen": {}, "decisions": {},
                                            "values": {"threads": {"value": [1, 8], "decision": "D-60"},
                                                       "bad": 3}, "record_reviewed_through": "D-60"}))
    c = cd.load_conditions(study)
    assert cd.require_values(c, ["threads"]) == {"threads": [1, 8]}
    with pytest.raises(Blocked, match="not yet fixed"):
        cd.require_values(c, ["eta"])
    with pytest.raises(cd.IntegrityBlocked, match="malformed"):
        cd.require_values(c, ["bad"])


def test_frozen_files_exist_in_this_repository() -> None:
    """Every listed frozen file exists in the repository the study runs from."""
    assert not [f for f in cd.FROZEN_FILES if not (REPO_ROOT / f).is_file()]


