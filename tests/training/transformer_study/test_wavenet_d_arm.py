"""Tests D1–D10 of ``training/transformer_study/wavenet_d_arm_spec.md``, plus the selection rules.

Training tests use a small subset of the real split (fast, CPU); nothing reads the test split.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import models.recurrent.rnn_training as rnn_training
import training.transformer_study.wavenet_d_arm as d
from models.hierarchical.wavenet import WaveNetLanguageModel, calibrate_batchnorm, evaluate_nll
from utils.batching import EpochSampler
from utils.data import make_context_windows, make_next_token_sequences
from utils.reproducibility import derive_seed, make_generator


@pytest.fixture(scope="module")
def full() -> d.StudyData:
    return d.load_study_data()


@pytest.fixture(scope="module")
def small(full: d.StudyData) -> d.StudyData:
    train = full.train_names[:120]
    dev_names = [n for n in d.split_names(d.load_names(), fractions=d.SPLIT_FRACTIONS, seed=d.SPLIT_SEED,
                                          duplicate_policy=d.DUPLICATE_POLICY).dev[:40]]
    return d.StudyData(vocab=full.vocab, train_names=tuple(train),
                       train=d.build_name_windows(train, full.vocab), dev=d.build_name_windows(dev_names, full.vocab))


def _strip(history):
    return [{k: v for k, v in r.items() if k != "seconds"} for r in history]


# ---------------------------------------------------------------- D1: pairing with the RNN
def test_d1_name_batches_equal_the_rnn_sampler(full: d.StudyData) -> None:
    src = inspect.getsource(rnn_training.train_rnn)
    assert 'EpochSampler(N, B, seed=derive_seed(seed, "data"))' in src
    assert rnn_training.RNNTrainingConfig(lr=0.1, max_epochs=1).batch_size == d.BATCH_SIZE
    N = full.train.num_names
    assert N == make_next_token_sequences(full.train_names, full.vocab).lengths.numel() == 25_640
    for seed in (0, 7, 100):
        mine, rnn = d.make_sampler(N, seed), EpochSampler(N, 32, seed=derive_seed(seed, "data"))
        for epoch in (0, 1, 37):
            a, b = list(mine.batches(epoch)), list(rnn.batches(epoch))
            assert len(a) == len(b) == 802 and all(torch.equal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------- D2: windows of a name batch
def test_d2_rows_are_each_names_windows(full: d.StudyData) -> None:
    nw = full.train
    assert nw.num_targets == 182_666
    for n in (0, 1, 500, 25_639):
        rows = torch.arange(int(nw.offsets[n]), int(nw.offsets[n + 1]))
        assert torch.all(nw.windows.name_index[rows] == n)
        ref = make_context_windows([full.train_names[n]], full.vocab, context_length=8)
        assert torch.equal(nw.windows.contexts[rows], ref.contexts)
        assert torch.equal(nw.windows.targets[rows], ref.targets)
        assert int(nw.windows.targets[rows][-1]) == full.vocab.eos_id
    idx = next(iter(d.make_sampler(nw.num_names, 3).batches(0)))
    rows = nw.rows(idx)
    assert rows.numel() == sum(len(full.train_names[i]) + 1 for i in idx.tolist())
    expected = torch.cat([torch.arange(int(nw.offsets[i]), int(nw.offsets[i + 1])) for i in idx.tolist()])
    assert torch.equal(rows, expected)


def test_d2_positions(full: d.StudyData) -> None:
    pos = full.train.positions()
    n = 4
    rows = torch.arange(int(full.train.offsets[n]), int(full.train.offsets[n + 1]))
    assert pos[rows].tolist() == list(range(len(full.train_names[n]) + 1))


# ---------------------------------------------------------------- D3: loss constant
def test_d3_loss_constant(full: d.StudyData) -> None:
    seqs = make_next_token_sequences(full.train_names, full.vocab)
    C_rnn = rnn_training.loss_normalizer(rnn_training.RNNTrainingConfig(lr=0.1, max_epochs=1), seqs)
    assert d.loss_constant(full.train_names) == C_rnn == pytest.approx(227.976, abs=1e-3)
    W = 228
    zero = F.cross_entropy(torch.zeros(W, 27), torch.zeros(W, dtype=torch.long), reduction="sum") / C_rnn
    assert float(zero) == pytest.approx(W * math.log(27) / C_rnn, rel=1e-6)


# ---------------------------------------------------------------- D4: initialisation stream
def test_d4_initial_parameters(full: d.StudyData) -> None:
    a = d.make_model(full.vocab, 5)
    b = WaveNetLanguageModel(full.vocab, d.D_CONFIG, generator=make_generator(derive_seed(5, "init")))
    assert d.param_sha256(a) == d.param_sha256(b)
    assert sum(p.numel() for p in a.parameters()) == 22_397


# ---------------------------------------------------------------- D5: BatchNorm modes
def test_d5_eval_does_not_depend_on_batch(small: d.StudyData) -> None:
    model = d.make_model(small.vocab, 1)
    with torch.no_grad():  # the output layer starts at zero; give it weights so the logits differ
        model.out.weight.normal_(generator=torch.Generator().manual_seed(0))
    calibrate_batchnorm(model, small.train.windows, batch_size=4096)
    model.eval()
    x = small.dev.windows.contexts
    with torch.no_grad():
        full_out, part = model(x), model(x[:7])
    # Equal up to float32 rounding: the BLAS kernel (and so the last bits) can depend on the batch size,
    # but eval mode must not use the batch's statistics. Train mode below differs by far more.
    assert torch.allclose(full_out[:7], part, rtol=1e-5, atol=1e-5)
    model.train()
    with torch.no_grad():
        assert not torch.allclose(model(x)[:7], model(x[:7]))


# ---------------------------------------------------------------- D6, D10: evaluation
def test_d6_d10_evaluation(small: d.StudyData) -> None:
    model = d.make_model(small.vocab, 2)
    calibrate_batchnorm(model, small.train.windows, batch_size=4096)
    ev = d.evaluate_by_position(model, small.dev)
    ref = evaluate_nll(model, small.dev.windows, batch_size=4096)
    assert ev["nll"] == pytest.approx(ref.mean_nll, rel=1e-6)
    assert sum(ev["by_position"]["count"]) == ev["count"] == small.dev.num_targets
    assert sum(ev["by_position"]["sum"]) == pytest.approx(ev["sum"], rel=1e-12)


# ---------------------------------------------------------------- D7, D9: determinism and selection
def test_d7_d9_determinism_and_best_checkpoint(small: d.StudyData, tmp_path: Path) -> None:
    a = d.train_run(small, seed=3, lr=0.1, max_epochs=3, run_dir=tmp_path / "a")
    b = d.train_run(small, seed=3, lr=0.1, max_epochs=3, run_dir=tmp_path / "b")
    assert a["status"] == b["status"] == "completed"
    ha, hb = d._read_history(tmp_path / "a"), d._read_history(tmp_path / "b")
    assert _strip(ha) == _strip(hb) and [r["epoch"] for r in ha] == [0, 1, 2, 3]
    assert ha[0]["dev_nll"] == pytest.approx(math.log(27), abs=0.05)
    devs = [r["dev_nll"] for r in ha if r["epoch"] >= 1]
    assert a["best_epoch"] == 1 + devs.index(min(devs))  # strict <: earliest on ties
    best = torch.load(tmp_path / "a" / "model_best.pt", weights_only=False)
    assert best["epoch"] == a["best_epoch"] and best["param_sha256"] == ha[a["best_epoch"]]["param_sha256"]


# ---------------------------------------------------------------- D8: resume
class _Stop(Exception):
    pass


def test_d8_resume_equals_uninterrupted(small: d.StudyData, tmp_path: Path, monkeypatch) -> None:
    d.train_run(small, seed=4, lr=0.1, max_epochs=3, run_dir=tmp_path / "ref")
    real, calls = d.evaluate_by_position, {"n": 0}

    def flaky(model, data):
        calls["n"] += 1
        if calls["n"] == 3:  # the end of epoch 2: interrupt before it is committed
            raise _Stop
        return real(model, data)

    monkeypatch.setattr(d, "evaluate_by_position", flaky)
    with pytest.raises(_Stop):
        d.train_run(small, seed=4, lr=0.1, max_epochs=3, run_dir=tmp_path / "res")
    monkeypatch.setattr(d, "evaluate_by_position", real)
    assert [r["epoch"] for r in d._read_history(tmp_path / "res")] == [0, 1]
    meta = d.train_run(small, seed=4, lr=0.1, max_epochs=3, run_dir=tmp_path / "res")
    assert meta["status"] == "completed"
    assert _strip(d._read_history(tmp_path / "res")) == _strip(d._read_history(tmp_path / "ref"))


def test_finished_run_is_not_rerun(small: d.StudyData, tmp_path: Path) -> None:
    d.train_run(small, seed=1, lr=0.1, max_epochs=1, run_dir=tmp_path / "x")
    with pytest.raises(ValueError, match="already finished"):
        d.train_run(small, seed=1, lr=0.1, max_epochs=1, run_dir=tmp_path / "x")


def test_huge_learning_rate_fails_cleanly(small: d.StudyData, tmp_path: Path) -> None:
    meta = d.train_run(small, seed=1, lr=1e4, max_epochs=2, run_dir=tmp_path / "boom")
    assert meta["status"] in ("failed_mechanical", "failed_diverged")
    assert json.loads((tmp_path / "boom" / "metadata.json").read_text())["status"] == meta["status"]


# ---------------------------------------------------------------- rules
def test_plateau_rule() -> None:
    flat = [[2.0, 1.5, 1.2, 1.1999, 1.1998]] * 3
    out = d.plateau_epoch([list(c) for c in flat])
    assert out["e_star"] == 3 and out["epsilon"] == 1e-3 and not out["epsilon_from_se"]
    falling = [[3.0 - 0.1 * e for e in range(30)] for _ in range(3)]
    assert d.plateau_epoch(falling)["e_star"] == 30


def _res(mean, se=0.0, eligible=True):
    return {"eligible": eligible, "mean_final_dev": mean, "se_final_dev": se}


def test_learning_rate_rule() -> None:
    out = d.select_learning_rate({0.03: _res(2.10), 0.1: _res(2.05), 0.3: _res(2.0495), 1.0: _res(2.2, eligible=False)})
    assert out["primary"] == 0.3 and out["selected"] == 0.1  # near-tie → smaller η
    out = d.select_learning_rate({0.03: _res(2.00), 0.1: _res(2.05), 0.3: _res(2.10), 1.0: _res(2.2)})
    assert out["selected"] == 0.03 and out["at_lower_boundary"]
    out = d.select_learning_rate({0.03: _res(2.1, 0.01), 0.1: _res(2.105), 0.3: _res(2.09, 0.02)})
    assert out["primary"] == 0.3 and out["near_tie_margin"] == 0.02 and out["selected"] == 0.03
