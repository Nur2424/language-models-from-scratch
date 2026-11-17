"""Synthetic fixtures for the study_common tests.

Everything here is synthetic: a handful of made-up names over a 6-letter alphabet, tiny
model widths, a few epochs. No project data, no LR-pilot result and no pilot configuration
is used. The accepted trainers are called unchanged; the only addition is a test-side
wrapper of ``run.append_history`` that records the state hash at the history write, which
is where the Stage-2 runner will record it (protocol §10.2).

Git guard (project policy: no Git command may ever run). ``RunManager.create`` records
``utils.run_manager.git_info()``, which starts ``git`` subprocesses. Before anything else in
this test package runs, this module therefore

1. replaces ``utils.run_manager.git_info`` with a stub returning ``GIT_DISABLED`` (the same
   record the LR pilot writes), and
2. installs a process-wide guard that refuses, *before* any process is started, every attempt
   to execute a program named ``git``: ``subprocess.Popen`` (and so ``run``, ``call``,
   ``check_output``, ``getoutput`` and ``os.popen``) at ``Popen._execute_child``, plus
   ``os.system``, ``os.posix_spawn``/``posix_spawnp``, ``os.spawn*`` and ``os.exec*``.
   Shell command strings and wrappers such as ``sh -c 'git …'`` or ``env git …`` are refused
   too. The guard is test-only; it does not change any project file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from training.wavenet_study import study_common as sc
from utils import run_manager as _run_manager
from utils.data import make_context_windows
from utils.diagnostics.schedule import DiagnosticsSchedule
from utils.reproducibility import derive_seed, make_generator
from utils.run_manager import RunManager
from utils.vocabulary import Vocabulary

# ----------------------------------------------------------------- Git guard (installed at import)
GIT_DISABLED = {"available": False, "reason": "git disabled by project policy"}
_GIT_WORD = re.compile(r"(?:^|[\s;&|()`'\"/=])git(?:\.exe)?(?=$|[\s;&|()`'\"])")


class GitBlockedError(PermissionError):
    """Raised instead of starting a ``git`` process."""


def _mentions_git(*parts) -> bool:
    for part in parts:
        if part is None:
            continue
        items = [part] if isinstance(part, (str, bytes, os.PathLike)) else list(part)
        for item in items:
            text = os.fsdecode(item) if isinstance(item, (bytes, os.PathLike)) else str(item)
            if os.path.basename(text.strip()) in ("git", "git.exe") or _GIT_WORD.search(text):
                return True
    return False


def _refuse_git(what: str, *parts) -> None:
    if _mentions_git(*parts):
        raise GitBlockedError(f"git disabled by project policy (refused {what}: {parts!r})")


def _install_git_guard() -> None:
    if getattr(subprocess.Popen, "_git_guard_installed", False):
        return
    original_execute_child = subprocess.Popen._execute_child

    def _execute_child(self, args, executable, *rest, **kwargs):
        _refuse_git("subprocess", args, executable)
        return original_execute_child(self, args, executable, *rest, **kwargs)

    subprocess.Popen._execute_child = _execute_child
    subprocess.Popen._git_guard_installed = True

    def wrap(name: str, picks) -> None:
        original = getattr(os, name, None)
        if original is None:
            return

        def guarded(*args, **kwargs):
            _refuse_git(f"os.{name}", *picks(args))
            return original(*args, **kwargs)

        guarded.__name__ = name
        setattr(os, name, guarded)

    wrap("system", lambda a: a[:1])
    for name in ("posix_spawn", "posix_spawnp"):
        wrap(name, lambda a: a[:2])
    for name in ("execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe"):
        wrap(name, lambda a: a)
    for name in ("spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe"):
        wrap(name, lambda a: a[1:])
    _run_manager.git_info = lambda *args, **kwargs: dict(GIT_DISABLED)


_install_git_guard()

TRAIN_NAMES = ["abc", "bca", "cab", "abd", "bad", "dab", "ace", "cea", "eca", "bed", "deb", "fab",
               "abf", "caf", "fec", "edf", "fed", "def", "bef", "cbd", "adc", "ebf", "fda", "dcf"]
DEV_NAMES = ["abe", "fac", "dbc", "ecd"]


@dataclass
class SyntheticRun:
    arm: str
    run_dir: Path
    normalization: str
    epochs: int
    model: torch.nn.Module


def _arm(arm: str):
    from models.feedforward import batchnorm_mlp, mlp
    from models.hierarchical import wavenet

    if arm == "A":
        return mlp, mlp.MLPLanguageModel, mlp.MLPConfig(context_length=8, embedding_dim=3, hidden_dim=5), \
            mlp.MLPTrainingConfig, mlp.train_mlp, "none"
    if arm == "B":
        return batchnorm_mlp, batchnorm_mlp.BatchNormMLPLanguageModel, \
            batchnorm_mlp.BatchNormMLPConfig(context_length=8, embedding_dim=3, hidden_dim=5), \
            batchnorm_mlp.BatchNormMLPTrainingConfig, batchnorm_mlp.train_batchnorm_mlp, "batchnorm"
    norm = "none" if arm == "C" else "batchnorm"
    return wavenet, wavenet.WaveNetLanguageModel, \
        wavenet.WaveNetConfig(context_length=8, embedding_dim=3, hidden_dim=4, normalization=norm), \
        wavenet.WaveNetTrainingConfig, wavenet.train_wavenet, norm


def make_synthetic_run(root: Path, arm: str, *, epochs: int = 3, seed: int = 7, lr: float = 0.1) -> SyntheticRun:
    module, model_cls, model_cfg, train_cls, train_fn, norm = _arm(arm)
    vocab = Vocabulary.from_names(TRAIN_NAMES)
    train = make_context_windows(TRAIN_NAMES, vocab, context_length=8)
    dev = make_context_windows(DEV_NAMES, vocab, context_length=8)
    batch = next(b for b in (5, 6, 7, 9) if train.num_examples % b != 1)
    training = train_cls(lr=lr, max_epochs=epochs, batch_size=batch, eval_batch_size=16)
    model = model_cls(vocab, model_cfg, generator=make_generator(derive_seed(seed, "init")))
    model_name = getattr(model, "model_name", None) or module.MODEL_NAME
    run = RunManager.create(model_name, config=module.run_config(None, model.config, training), results_root=root)
    original = run.append_history

    def append_history(record):  # state hash at exactly the history write (protocol §10.2)
        original(record)
        sc_record = {"epoch": record["epoch"], "sha256": sc.model_state_hash(model)}
        with (run.run_dir / sc.STATE_HASH_FILE).open("a") as fh:
            fh.write(json.dumps(sc_record) + "\n")

    run.append_history = append_history
    schedule = DiagnosticsSchedule(enabled=True, every_n_steps=2, gradients=True, parameters=True, activations=False)
    with run.track():
        train_fn(model, train, dev, training=training, seed=seed, run=run, diagnostics=schedule, config=None)
    return SyntheticRun(arm, run.run_dir, norm, epochs, model)


@pytest.fixture(scope="session")
def synthetic_runs(tmp_path_factory) -> dict[str, SyntheticRun]:
    """One completed 3-epoch synthetic run per arm type (A, B, C, D); treat as read-only."""
    root = tmp_path_factory.mktemp("synthetic_runs")
    torch.set_num_threads(1)
    return {arm: make_synthetic_run(root, arm) for arm in "ABCD"}


@pytest.fixture()
def run_copy(tmp_path, synthetic_runs):
    """A writable copy of a synthetic run directory, for fault injection."""
    import shutil

    def copy(arm: str) -> Path:
        dst = tmp_path / f"copy_{arm}"
        shutil.copytree(synthetic_runs[arm].run_dir, dst)
        return dst

    return copy
