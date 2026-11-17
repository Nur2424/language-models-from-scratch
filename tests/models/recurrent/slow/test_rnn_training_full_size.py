"""MS5 full-width check of the training procedure at the approved H = 128 (slow suite).

Milestone MS5 of ``rnn_implementation_plan.md`` v0.2 (D-49). One epoch on synthetic names at
H = 128, B = 32, eval chunks of 1,024 (the research batch sizes), float32 CPU: finite records,
⌈N/B⌉ steps, ln 27 at epoch 0, and the epoch-0 hash equal to the untrained model's hash.
Run with ``python3 -m pytest tests/models/recurrent/slow``.
"""

from __future__ import annotations

import math
import string

import torch

import models.recurrent.rnn_training as rt
from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import RNNLanguageModel
from utils.data import make_next_token_sequences
from utils.reproducibility import derive_seed, make_generator
from utils.vocabulary import Vocabulary


def test_full_size_one_epoch():
    g = torch.Generator().manual_seed(3)
    names = ["".join(chr(97 + int(c)) for c in torch.randint(0, 26, (int(torch.randint(1, 16, (1,), generator=g)),),
                                                             generator=g)) for _ in range(200)]
    vocab = Vocabulary(chars=tuple(string.ascii_lowercase))
    train, dev = make_next_token_sequences(names[:160], vocab), make_next_token_sequences(names[160:], vocab)
    model = RNNLanguageModel(vocab, generator=make_generator(derive_seed(5, "init")))
    h0 = dg.param_sha256(model)
    res = rt.train_rnn(model, train, dev, training=rt.RNNTrainingConfig(lr=0.1, max_epochs=1), seed=5)
    assert res.status == "completed" and len(res.history) == 2
    assert abs(res.history[0]["train_nll"] - math.log(27)) <= 1e-6 and res.history[0]["param_sha256"] == h0
    assert len(res.history[1]["W_singular_values"]) == 128 and all(math.isfinite(v) for v in res.history[1]["W_singular_values"])
