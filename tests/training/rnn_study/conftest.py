"""Model 6 orchestration tests: Git guard and synthetic fixtures (MS7, MS8; plan §8.5).

The Git guard is installed at import, before anything creates a ``RunManager`` (D-15; D-49 O-2: the
WaveNet guard is reused unchanged). Everything is synthetic: invented names, temporary study roots,
synthetic research records. No test reads ``data/``, ``results/`` or the project's research record,
except the one read-only parse test of the real record in ``test_decisions.py``.
"""

from __future__ import annotations

import string

import pytest
import torch

from training.wavenet_study.study_common import disable_git

disable_git()

from training.rnn_study.common import StudyData, StudyPaths  # noqa: E402
from utils.data import make_next_token_sequences  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

VOCAB = Vocabulary(chars=tuple(string.ascii_lowercase))


def synthetic_names(count: int, seed: int, max_len: int = 8) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        L = int(torch.randint(1, max_len + 1, (1,), generator=g))
        out.append("".join(chr(97 + int(c)) for c in torch.randint(0, 26, (L,), generator=g)))
    return out


def synthetic_study_data(n_train: int = 18, n_dev: int = 6, seed: int = 40) -> StudyData:
    tr, dv = synthetic_names(n_train, seed), synthetic_names(n_dev, seed + 1)
    train, dev = make_next_token_sequences(tr, VOCAB), make_next_token_sequences(dv, VOCAB)
    info = {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1], "N": n_train,
            "M": int(train.lengths.sum())}
    return StudyData(VOCAB, train, dev, tuple(tr), tuple(dv), info)


@pytest.fixture
def study(tmp_path) -> StudyPaths:
    return StudyPaths(tmp_path / "rnn_study")


@pytest.fixture
def data() -> StudyData:
    return synthetic_study_data()
