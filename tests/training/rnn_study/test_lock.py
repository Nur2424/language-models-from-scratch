"""MS7: exclusive study lock across processes; Git guard in study code (O1) (plan §8.5)."""


from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.rnn_study import attempts as at
from training.rnn_study import conditions as cd
from training.rnn_study.common import REPO_ROOT, Blocked, StudyLock, StudyPaths, lock_held_elsewhere



# --------------------------------------------------------------------------- lock
def test_lock_is_exclusive_across_processes(study) -> None:
    with StudyLock(study, "test") as lock:
        assert lock.held and lock_held_elsewhere(study)  # flock: a second open file description is another holder
        code = ("import sys; from pathlib import Path; from training.rnn_study.common import StudyLock, StudyPaths, Blocked\n"
                f"p = StudyPaths(Path({str(study.root)!r}))\n"
                "try:\n    StudyLock(p, 'child').__enter__(); print('acquired')\nexcept Blocked: print('blocked')\n")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                             env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"})
        assert out.stdout.strip() == "blocked", out.stderr
    assert not lock_held_elsewhere(study)
    with StudyLock(study, "again"):
        pass


# --------------------------------------------------------------------------- O1 Git guard in study code
def test_O1_git_is_blocked_in_the_study_package() -> None:
    from training.rnn_study.common import install_git_guard

    install_git_guard()
    assert getattr(subprocess.Popen, "_study_git_guard", False) is True
    # Matched by type family and message: another frozen test guard may fire first in a whole-tree run.
    with pytest.raises(PermissionError, match=r"^git disabled by project policy"):
        subprocess.run(["git", "--version"], check=False)
