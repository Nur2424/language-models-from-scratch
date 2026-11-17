"""The test-package Git guard (conftest.py): no Git process can start during these tests.

Every deliberate attempt below targets a *fake* ``git`` executable placed first on ``PATH``
(or addressed by absolute path). The fake writes a marker file if it is ever executed, so a
missing marker proves that the guard refused the call before any process started. Even a
failure of the guard could therefore never run the real Git. A positive control shows that
the same marker mechanism does detect an executed program.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys

import pytest

from utils import run_manager

GIT_DISABLED = {"available": False, "reason": "git disabled by project policy"}


@pytest.fixture()
def fake_bin(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "executed.txt"
    for name in ("git", "notgit"):
        exe = bin_dir / name
        exe.write_text(f'#!/bin/sh\necho "{name} $@" >> "{marker}"\n')
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return bin_dir, marker


def test_positive_control_marker_detects_an_executed_program(fake_bin):
    bin_dir, marker = fake_bin
    subprocess.run([str(bin_dir / "notgit"), "x"], check=True)
    assert marker.read_text() == "notgit x\n"


ATTEMPTS = {
    "run_list": lambda b: subprocess.run(["git", "--version"], check=False),
    "run_absolute_path": lambda b: subprocess.run([str(b / "git"), "status"], check=False),
    "run_executable_kw": lambda b: subprocess.run(["anything"], executable=str(b / "git"), check=False),
    "run_shell_string": lambda b: subprocess.run("git --version", shell=True, check=False),
    "check_output_sh_c": lambda b: subprocess.check_output(["sh", "-c", "cd /tmp && git status"]),
    "env_wrapper": lambda b: subprocess.call(["env", "git", "log"]),
    "popen": lambda b: subprocess.Popen(["git", "rev-parse", "HEAD"]),
    "getoutput": lambda b: subprocess.getoutput("git status"),
    "os_system": lambda b: os.system("git --version"),
    "os_popen": lambda b: os.popen("git --version").read(),
    "posix_spawnp": lambda b: os.posix_spawnp("git", ["git", "status"], dict(os.environ)),
    "posix_spawn": lambda b: os.posix_spawn(str(b / "git"), ["git"], dict(os.environ)),
    "spawnvp": lambda b: os.spawnvp(os.P_WAIT, "git", ["git", "status"]),
    "spawnlp": lambda b: os.spawnlp(os.P_WAIT, "git", "git", "status"),
    "execvp": lambda b: os.execvp("git", ["git", "status"]),
    "execv": lambda b: os.execv(str(b / "git"), ["git"]),
}


@pytest.mark.parametrize("name", sorted(ATTEMPTS))
def test_deliberate_git_invocation_is_intercepted_before_execution(fake_bin, name):
    bin_dir, marker = fake_bin
    result = None
    with pytest.raises(PermissionError, match="git disabled by project policy"):
        result = ATTEMPTS[name](bin_dir)
    assert result is None
    assert not marker.exists(), f"{name}: a git process was started"


def test_original_git_info_code_path_is_refused_without_running_git(fake_bin, tmp_path):
    _, marker = fake_bin
    # The shared utility's own implementation (still importable as code) must not reach Git:
    # it catches the guard's PermissionError (an OSError) and reports Git as unavailable.
    import inspect
    import types

    source = inspect.getsource(sys.modules["utils.run_manager"])
    module = types.ModuleType("run_manager_unpatched_copy")
    exec(compile(source, "run_manager_unpatched_copy", "exec"), module.__dict__)  # noqa: S102
    info = module.git_info(tmp_path)  # unpatched implementation: subprocess.run(["git", ...])
    assert info["available"] is False
    assert "git disabled by project policy" in info["reason"]
    assert not marker.exists()


def test_run_manager_records_the_disabled_git_stub():
    assert run_manager.git_info() == GIT_DISABLED


def test_synthetic_fixture_runs_record_git_disabled(synthetic_runs):
    import json

    for run in synthetic_runs.values():
        meta = json.loads((run.run_dir / "metadata.json").read_text())
        assert meta["git"] == GIT_DISABLED


def test_non_git_programs_and_git_like_words_are_not_blocked(fake_bin):
    _, marker = fake_bin
    subprocess.run(["notgit", "digit", ".git/index", "legit"], check=True)
    assert marker.read_text() == "notgit digit .git/index legit\n"
