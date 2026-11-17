"""Model 6 (vanilla RNN) test scaffolding: Git guard and synthetic fixtures.

Milestone MS0 of ``models/recurrent/rnn_implementation_plan.md`` v0.2 (approved, D-49).

Git guard (project policy D-15; ``rnn_training_spec.md`` §10.8; plan §4.3, decision O-2)
-------------------------------------------------------------------------------------------
``utils.run_manager.RunManager.create`` records ``git_info()``, which starts ``git``
subprocesses. This module therefore installs the project's Git guard **at import time**,
before any Model 6 test runs:

* ``utils.run_manager.git_info`` returns ``GIT_DISABLED`` instead of running ``git``;
* starting any program named ``git`` is refused before a process exists (``subprocess``,
  ``os.system``, ``os.posix_spawn*``, ``os.spawn*``, ``os.exec*``, shell strings and wrappers).

**Documented dependency (D-49, O-2).** The guard is imported, unchanged, from the frozen
WaveNet study module ``training/wavenet_study/study_common.py`` (D-34). Only the three names
below are imported. That module imports the standard library at module level only, so the
import executes no study code; ``test_rnn_scaffolding.py`` verifies this in a fresh
interpreter. Nothing in the WaveNet study package is modified.

Synthetic data only (plan §7.4)
-------------------------------
No fixture reads ``data/``, ``results/``, the training split, the probe or any research
record. Names are invented; the vocabulary is built from the boundary ``.`` and ``a`` – ``z``
directly, which gives the project's 27 output classes without reading the names file.
"""

from __future__ import annotations

import string

import pytest
import torch

from training.wavenet_study.study_common import GIT_DISABLED, GitBlockedError, disable_git

disable_git()  # before anything else in this test package runs; idempotent

from utils.reproducibility import make_generator  # noqa: E402  (after the guard, by design)
from utils.vocabulary import Vocabulary  # noqa: E402

__all__ = ["GIT_DISABLED", "GitBlockedError", "SYNTHETIC_NAMES", "TINY_HIDDEN_DIMS"]

# --------------------------------------------------------------------------- constants
#: Invented names over a – z: lengths 1 – 16 characters, varied first and last letters,
#: repeated letters and one duplicate (duplicates are legal in a list of examples).
SYNTHETIC_NAMES: tuple[str, ...] = (
    "a", "zo", "bex", "qiun", "mavra", "tolkie", "ysabelo", "fennwick",
    "hhhhhhhhh", "corvinalt", "ludmeriaxo", "pqrstuvwxyz", "obelandrinas",
    "kestravelunio", "wyndamorphelus", "abcdefghijklmnop", "zo",
)

#: Hidden sizes for tiny models in fast tests (the approved model uses H = 128).
TINY_HIDDEN_DIMS: tuple[int, ...] = (4, 8)

#: Base seed for synthetic fixtures; unrelated to every research seed (100 – 102, 999,
#: 1000 – 1009 and the final seeds) and to the probe stream.
FIXTURE_SEED: int = 20261114


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def full_vocab() -> Vocabulary:
    """The 27-class vocabulary (``.`` = id 0, ``a`` – ``z`` = ids 1 – 26, PAD = 27)."""
    return Vocabulary(chars=tuple(string.ascii_lowercase))


@pytest.fixture(scope="session")
def synthetic_names() -> tuple[str, ...]:
    return SYNTHETIC_NAMES


@pytest.fixture(params=TINY_HIDDEN_DIMS, ids=lambda h: f"H{h}")
def tiny_hidden_dim(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture
def generator_factory():
    """``factory(offset=0)`` returns a fresh CPU ``torch.Generator`` seeded with
    ``FIXTURE_SEED + offset``; it never touches the global RNG."""

    def factory(offset: int = 0) -> torch.Generator:
        return make_generator(FIXTURE_SEED + int(offset))

    return factory


@pytest.fixture
def float64() -> torch.dtype:
    """dtype for criterion-A comparisons (``rnn_spec.md`` §13.1). The global default dtype
    is never changed by this package."""
    return torch.float64
