"""Tests for ``training/transformer_study/dev_diagnostics.py`` on small synthetic models and a small
subset of the dev split; no trained checkpoint is needed and the test split is never built."""
from __future__ import annotations

import math

import pytest
import torch

import training.transformer_study.dev_diagnostics as dd
import training.transformer_study.transformer_runs as tr
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from models.recurrent.rnn_diagnostics import c1_restart as rnn_c1, float64_copy
from utils.reproducibility import make_generator


@pytest.fixture(scope="module")
def small() -> tr.SequenceData:
    full = tr.load_study_data()
    splits = tr.split_names(tr.load_names(), fractions=tr.SPLIT_FRACTIONS, seed=tr.SPLIT_SEED,
                            duplicate_policy=tr.DUPLICATE_POLICY)
    dev = [n for n in splits.dev if len(n) >= 9][:25] + list(splits.dev[:25])   # include F₈ targets
    return tr.build_sequence_data(full.train_names[:50], dev, full.vocab)


def _rnn(vocab) -> RNNLanguageModel:
    m = RNNLanguageModel(vocab, RNNConfig(), generator=make_generator(3))
    with torch.no_grad():   # V and c start at zero: give the output layer weights so ΔNLL is not trivially 0
        m.output_weight.normal_(generator=torch.Generator().manual_seed(1))
    return m


def test_c1_reproduces_the_rnn_study(small) -> None:
    copy = float64_copy(_rnn(small.vocab))
    ref = rnn_c1(copy, small.dev).primary
    mine = dd.c1_restart(copy, small.dev)
    assert mine["noop_reproduces_full"]
    for label in ("F8", "F7"):
        for a, b in zip(mine[f"{label}_all"], ref[label]):
            assert a["targets"] == b["targets"] and a["names"] == b["names"]
            assert a["mean"] == pytest.approx(b["mean"], abs=1e-12) and a["se"] == pytest.approx(b["se"], abs=1e-12)
    for i, c in enumerate(mine["F8_all"]):   # the EOS/letter split partitions the targets
        assert c["targets"] == mine["F8_eos"][i]["targets"] + mine["F8_letter"][i]["targets"]


@pytest.mark.parametrize("variant", ["transformer_2017", "gpt2_style"])
def test_transformer_c1_both_variants(small, variant) -> None:
    m = tr.make_model(small.vocab, variant, 1, 42, 0).double()
    a = dd.c1_restart(m, small.dev)
    b = dd.c1_restart(m, small.dev, preserve_positions=True)
    assert a["noop_reproduces_full"] and b["noop_reproduces_full"]
    assert [c["targets"] for c in a["F8_all"]] == [c["targets"] for c in b["F8_all"]] and a["F8_all"][0]["targets"] > 0
    assert a["F8_all"][0]["mean"] != b["F8_all"][0]["mean"]       # positions do enter the model


def test_position_preserving_positions() -> None:
    pos = dd._positions_for([(0, 2), (0, 5)], 6)
    assert pos[0].tolist() == [0, 1, 2, 3, 4, 5]                    # j = 2: the original positions
    assert pos[1].tolist() == [0, 4, 5, 6, 7, 8]                    # BOS at 0, kept characters at their own positions


def test_attention_summary(small) -> None:
    m = tr.make_model(small.vocab, "gpt2_style", 2, 30, 0)
    s = dd.attention_summary(m, small.dev)
    w = torch.tensor(s["weight_by_distance"])
    assert w.shape == (2, 3, 16)
    assert torch.allclose(w.sum(-1).double(), torch.ones(2, 3, dtype=torch.float64), atol=1e-6)   # weights over distances sum to 1
    assert torch.allclose(torch.tensor(s["weight_on_bos"]), w.new_tensor(s["weight_on_bos"]))
    ent = torch.tensor(s["entropy"])
    assert (ent > 0).all() and (ent <= math.log(16) + 1e-9).all()


def test_by_position_sums(small) -> None:
    m = tr.make_model(small.vocab, "transformer_2017", 1, 42, 0)
    r = dd.nll_by_position_sequences(m, small.dev)
    assert sum(r["by_position"]["count"]) == r["count"] == small.dev.num_targets
    assert r["nll"] == pytest.approx(tr.evaluate(m, small.dev)["nll"], rel=1e-9)
