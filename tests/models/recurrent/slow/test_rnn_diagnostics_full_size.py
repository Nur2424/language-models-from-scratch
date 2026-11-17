"""MS3 full-width checks of the diagnostic functions at the approved H = 128 (slow suite).

Milestone MS3 of ``rnn_implementation_plan.md`` v0.2 (D-49). Synthetic names only; tolerance A
(1e‑9, float64) against the accepted MS1 reference; g = 1.2, the largest gain of the A fixtures.
Run with ``python3 -m pytest tests/models/recurrent/slow``.
"""

from __future__ import annotations

import math
import string

import torch

import tests.models.recurrent.rnn_reference as ref
from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from utils.data import make_next_token_sequences
from utils.reproducibility import make_generator
from utils.vocabulary import Vocabulary

H = 128
NAMES = ["kestravelunio", "mavra", "zo", "obelandrinas", "bex", "ludmeriaxo", "fennwick", "qiun"]


def _model(vocab, g, seed):
    gen = make_generator(seed)
    Q, R = torch.linalg.qr(torch.randn(H, H, generator=gen, dtype=torch.float64))
    vals = (math.sqrt(0.5) * torch.randn(27, H, generator=gen, dtype=torch.float64),
            g * (Q * torch.sign(torch.diagonal(R)).unsqueeze(0)),
            0.1 * torch.randn(H, generator=gen, dtype=torch.float64),
            torch.randn(27, H, generator=gen, dtype=torch.float64) / math.sqrt(H),
            0.1 * torch.randn(27, generator=gen, dtype=torch.float64))
    m = RNNLanguageModel(vocab, RNNConfig(), generator=make_generator(1)).to(torch.float64)
    with torch.no_grad():
        for n, v in zip(dg.PARAM_ORDER, vals):
            getattr(m, n).copy_(v)
    return m


def test_full_size_g1_and_g3(tmp_path) -> None:
    vocab = Vocabulary(chars=tuple(string.ascii_lowercase))
    seqs = make_next_token_sequences(NAMES, vocab)
    info = {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1],
            "N": len(NAMES), "M": int(seqs.lengths.sum())}
    probe = dg.build_probe(seqs, NAMES, info, tmp_path / "probe.json")
    model = _model(vocab, 1.2, 128)
    cm = dg.float64_copy(model)
    g1 = dg.g1_products(cm, probe, seqs, spectral=True)
    p = ref.Params(*(getattr(model, n).detach().contiguous() for n in dg.PARAM_ORDER))
    fw = {r: ref.forward_name(p, seqs.inputs[r, : int(seqs.lengths[r])].tolist()) for r in probe.rows}
    raw = g1.raw
    for i in range(0, len(raw["g1_t"]), 7):
        P = ref.jacobian_product(p, fw[int(raw["g1_row"][i])], int(raw["g1_t"][i]), int(raw["g1_s"][i]))
        e = math.log(float(torch.linalg.matrix_norm(P)) / math.sqrt(H))
        assert abs(raw["g1_log_phi"][i] - e) <= 1e-9 + 1e-9 * abs(e)
    g3 = dg.g3_decomposition(cm, probe, seqs)
    assert all(g3.checks[k]["pass"] for k in g3.checks), g3.checks
