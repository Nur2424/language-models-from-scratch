"""MS0 tests: Git guard, documented guard dependency, synthetic fixtures, no data access.

Milestone MS0 of ``models/recurrent/rnn_implementation_plan.md`` v0.2 (approved, D-49):

* the guard is active: ``git`` is refused before any process starts, in every form;
* ``utils.run_manager.git_info()`` and ``RunManager.create`` record ``GIT_DISABLED``;
* ordinary subprocesses still work (the guard is not over-broad);
* the guard dependency on ``training/wavenet_study/study_common.py`` is narrow: importing it
  and installing the guard loads no other WaveNet study module and no model module;
* the synthetic fixtures are self-contained (27 classes, PAD = 27, names inside the vocabulary);
* no Model 6 test file names the data or results directories, the names file, or the
  data-loading and splitting functions (static AST scan).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tests.models.recurrent.conftest as m6conf
from training.wavenet_study.study_common import GIT_DISABLED, GitBlockedError, disable_git
from utils import run_manager
from utils.run_manager import RunManager

REPO_ROOT = Path(__file__).resolve().parents[3]
M6_TEST_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- Git guard
#: A refusal is recognised by its type family and message, not by one exception class: when the whole
#: test tree is collected, ``tests/training/wavenet_study/conftest.py`` installs its own (frozen) guard
#: on top of ``study_common.disable_git`` with its own ``GitBlockedError``; whichever guard fires first,
#: Git is refused before a process exists. ``test_guard_installation_is_idempotent`` checks that the
#: ``study_common`` guard itself is installed.
GIT_REFUSED = r"^git disabled by project policy"


def test_git_by_absolute_path_is_refused_before_a_process_exists() -> None:
    # The path does not exist: if a process were attempted, the error would be
    # FileNotFoundError. A PermissionError with the policy message shows the refusal happens before exec.
    with pytest.raises(PermissionError, match=GIT_REFUSED):
        subprocess.run(["/nonexistent-m6-dir/git", "--version"], check=False)


@pytest.mark.parametrize(
    "call",
    [
        lambda: subprocess.run(["git", "--version"], check=False),
        lambda: subprocess.run("git status", shell=True, check=False),
        lambda: subprocess.run(["sh", "-c", "git log"], check=False),
        lambda: subprocess.run(["env", "git", "status"], check=False),
        lambda: subprocess.check_output(["git", "rev-parse", "HEAD"]),
        lambda: os.system("git --version"),
    ],
    ids=["argv", "shell-string", "sh-c-wrapper", "env-wrapper", "check_output", "os.system"],
)
def test_every_form_of_git_is_refused(call) -> None:
    with pytest.raises(PermissionError, match=GIT_REFUSED):
        call()


def test_non_git_subprocess_still_runs() -> None:
    out = subprocess.run([sys.executable, "-c", "print('m6')"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "m6"


def test_git_info_returns_the_disabled_record() -> None:
    assert run_manager.git_info() == dict(GIT_DISABLED)
    assert GIT_DISABLED == {"available": False, "reason": "git disabled by project policy"}


def test_run_manager_create_records_git_disabled(tmp_path: Path) -> None:
    run = RunManager.create("rnn", config={"fixture": True}, results_root=tmp_path)
    meta = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    assert meta["git"] == dict(GIT_DISABLED)
    assert run.run_dir.is_relative_to(tmp_path)  # nothing written under results/


def test_guard_installation_is_idempotent() -> None:
    before = subprocess.Popen._execute_child
    disable_git()
    assert subprocess.Popen._execute_child is before
    assert getattr(subprocess.Popen, "_study_git_guard", False) is True


def test_conftest_reexports_the_same_guard_objects() -> None:
    assert m6conf.GitBlockedError is GitBlockedError
    assert m6conf.GIT_DISABLED is GIT_DISABLED


# --------------------------------------------------------------------------- dependency (O-2)
_COUPLING_PROBE = """
import json, sys
before = set(sys.modules)
from training.wavenet_study.study_common import GIT_DISABLED, GitBlockedError, disable_git
disable_git()
print(json.dumps(sorted(set(sys.modules) - before)))
"""


def test_guard_import_loads_no_other_study_or_model_module() -> None:
    """Fresh interpreter (isolated mode, explicit path): importing the guard and installing it
    must not load any other ``training.*`` module or any ``models.*`` module."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    out = subprocess.run(
        [sys.executable, "-c", _COUPLING_PROBE],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    study = [m for m in loaded if m.startswith("training")]
    assert set(study) <= {"training", "training.wavenet_study", "training.wavenet_study.study_common"}, study
    assert not [m for m in loaded if m == "models" or m.startswith("models.")]
    assert not [m for m in loaded if m.startswith("tests")]


# --------------------------------------------------------------------------- synthetic fixtures
def test_full_vocab_has_the_project_conventions(full_vocab) -> None:
    assert full_vocab.num_classes == 27
    assert full_vocab.pad_id == 27
    assert full_vocab.num_input_ids == 28
    assert full_vocab.boundary == "." and full_vocab.boundary_id == 0
    assert full_vocab.stoi["a"] == 1 and full_vocab.stoi["z"] == 26


def test_synthetic_names_are_inside_the_vocabulary(full_vocab, synthetic_names) -> None:
    assert synthetic_names, "fixture must not be empty"
    assert not full_vocab.unknown_chars(synthetic_names)
    lengths = {len(n) for n in synthetic_names}
    assert min(lengths) == 1 and max(lengths) == 16  # covers every Stage 5 length stratum
    assert all(n == n.lower() and n.isalpha() for n in synthetic_names)


def test_generator_factory_is_deterministic_and_leaves_the_global_rng(generator_factory) -> None:
    import torch

    state = torch.random.get_rng_state()
    a = torch.rand(5, generator=generator_factory(0), dtype=torch.float64)
    b = torch.rand(5, generator=generator_factory(0), dtype=torch.float64)
    c = torch.rand(5, generator=generator_factory(1), dtype=torch.float64)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_float64_fixture_does_not_change_the_default_dtype(float64) -> None:
    import torch

    assert float64 is torch.float64
    assert torch.get_default_dtype() is torch.float32


def test_tiny_hidden_dims(tiny_hidden_dim) -> None:
    assert tiny_hidden_dim in m6conf.TINY_HIDDEN_DIMS and 0 < tiny_hidden_dim < 128


# --------------------------------------------------------------------------- no data access
_FORBIDDEN_NAMES = frozenset({"load_names", "split_names"})
_FORBIDDEN_TEXT = ("data/", "results/", "names.txt")


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
        and isinstance(n.body[0].value.value, str)
    }  # documentation may name the directories it promises not to read
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) in docstrings:
            continue
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            found.append(f"{path.name}:{node.lineno} name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            found.append(f"{path.name}:{node.lineno} attribute {node.attr}")
        elif isinstance(node, ast.alias) and node.name.split(".")[-1] in _FORBIDDEN_NAMES:
            found.append(f"{path.name} import {node.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            hits = [t for t in _FORBIDDEN_TEXT if t in node.value]
            if hits:
                found.append(f"{path.name}:{node.lineno} string mentions {hits}")
        elif isinstance(node, ast.Call):
            # Path("data") / ... or Path(..., "results", ...): a bare directory component.
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value in ("data", "results"):
                    found.append(f"{path.name}:{node.lineno} path component {arg.value!r}")
    return found


def test_no_model6_test_names_project_data_or_results() -> None:
    """Scans every ``.py`` file under ``tests/models/recurrent/`` (recursively, so later
    milestones are covered automatically). This file is excluded because it must spell
    the forbidden tokens to search for them; it is reviewed by hand instead."""
    files = sorted(p for p in M6_TEST_DIR.rglob("*.py") if p.resolve() != Path(__file__).resolve())
    assert any(p.name == "conftest.py" for p in files)
    violations = [v for p in files for v in _violations(p)]
    assert not violations, "\n".join(violations)


def test_scanner_detects_each_forbidden_form(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from utils.data import load_names\n"
        "import utils.data as d\n"
        "x = d.split_names\n"
        "p = 'data/names.txt'\n"
        "q = 'results/runs'\n"
        "from pathlib import Path\n"
        "r = Path('data')\n"
        "def f():\n"
        "    '''Docstrings may mention data/ and results/.'''\n",
        encoding="utf-8",
    )
    joined = "\n".join(_violations(sample))
    for needle in ("import load_names", "attribute split_names", ":4 string mentions ['data/', 'names.txt']",
                   ":5 string mentions ['results/']", "path component 'data'"):
        assert needle in joined, (needle, joined)
    assert ":9 " not in joined  # the docstring on line 9 is not a violation
