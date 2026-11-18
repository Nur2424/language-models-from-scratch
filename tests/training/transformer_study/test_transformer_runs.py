"""Tests for ``training/transformer_study/transformer_runs.py``. Training tests use a small subset
of the real split (fast, CPU); nothing reads the test split."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import models.recurrent.rnn_training as rnn_training
import training.transformer_study.transformer_runs as tr
import training.transformer_study.wavenet_d_arm as d
from models.attention.transformer import TransformerLanguageModel


@pytest.fixture(scope="module")
def full() -> tr.SequenceData:
    return tr.load_study_data()


@pytest.fixture(scope="module")
def small(full: tr.SequenceData) -> tr.SequenceData:
    splits = tr.split_names(tr.load_names(), fractions=tr.SPLIT_FRACTIONS, seed=tr.SPLIT_SEED,
                            duplicate_policy=tr.DUPLICATE_POLICY)
    return tr.build_sequence_data(full.train_names[:120], splits.dev[:40], full.vocab)


def _strip(history):
    return [{k: v for k, v in r.items() if k != "seconds"} for r in history]


def test_protocol_is_shared_with_the_rnn_and_arm_d(full: tr.SequenceData) -> None:
    assert tr.make_sampler is d.make_sampler and tr.loss_constant is d.loss_constant
    assert tr.select_learning_rate is d.select_learning_rate and tr.plateau_epoch is d.plateau_epoch
    N = full.train.inputs.shape[0]
    assert N == 25_640 and full.train.num_targets == 182_666
    C_rnn = rnn_training.loss_normalizer(rnn_training.RNNTrainingConfig(lr=0.1, max_epochs=1), full.train)
    assert tr.loss_constant(full.train_names) == C_rnn
    assert tr.SHAPES == ((1, 42), (2, 30), (3, 24)) and tr.NUM_HEADS == 3


def test_evaluate_matches_direct_computation(small: tr.SequenceData) -> None:
    m = tr.make_model(small.vocab, "gpt2_style", 2, 30, 0)
    ev = tr.evaluate(m, small.dev)
    with torch.no_grad():
        logits = m(small.dev.inputs)
        ref = F.cross_entropy(logits[small.dev.mask], small.dev.targets[small.dev.mask], reduction="sum").item()
    assert ev["count"] == small.dev.num_targets == sum(ev["by_position"]["count"])
    assert ev["sum"] == pytest.approx(ref, rel=1e-6)
    assert sum(ev["by_position"]["sum"]) == pytest.approx(ev["sum"], rel=1e-12)
    assert ev["by_position"]["count"][0] == small.dev.inputs.shape[0]   # every name has a target after BOS


@pytest.mark.parametrize("variant", ["transformer_2017", "gpt2_style"])
def test_determinism_and_best_checkpoint(small: tr.SequenceData, tmp_path: Path, variant: str) -> None:
    kw = dict(variant=variant, num_layers=2, d_model=30, seed=3, lr=0.1, max_epochs=3)
    a = tr.train_run(small, run_dir=tmp_path / "a", **kw)
    b = tr.train_run(small, run_dir=tmp_path / "b", **kw)
    assert a["status"] == b["status"] == "completed"
    ha = tr._read_history(tmp_path / "a")
    assert _strip(ha) == _strip(tr._read_history(tmp_path / "b")) and [r["epoch"] for r in ha] == [0, 1, 2, 3]
    devs = [r["dev_nll"] for r in ha if r["epoch"] >= 1]
    assert a["best_epoch"] == 1 + devs.index(min(devs))
    best = TransformerLanguageModel.from_checkpoint(tmp_path / "a" / "model_best.pt", expected_vocab=small.vocab,
                                                    expected_variant=variant)
    assert tr.param_sha256(best) == ha[a["best_epoch"]]["param_sha256"]
    assert tr.evaluate(best, small.dev)["nll"] == pytest.approx(min(devs), rel=1e-12)
    row = ha[1]
    assert len(row["mean_grad_norm"]) == 3 and len(row["internals"]["ln_1_input_std"]) == 2
    assert ("final_ln_input_std" in row["internals"]) == (variant == "gpt2_style")


class _Stop(Exception):
    pass


def test_resume_equals_uninterrupted(small: tr.SequenceData, tmp_path: Path, monkeypatch) -> None:
    kw = dict(variant="gpt2_style", num_layers=1, d_model=42, seed=4, lr=0.1, max_epochs=3)
    tr.train_run(small, run_dir=tmp_path / "ref", **kw)
    real, calls = tr.evaluate, {"n": 0}

    def flaky(model, seqs):
        calls["n"] += 1
        if calls["n"] == 5:   # epoch 0 and 1 use 2 calls each: interrupt during epoch 2's evaluation
            raise _Stop
        return real(model, seqs)

    monkeypatch.setattr(tr, "evaluate", flaky)
    with pytest.raises(_Stop):
        tr.train_run(small, run_dir=tmp_path / "res", **kw)
    monkeypatch.setattr(tr, "evaluate", real)
    assert [r["epoch"] for r in tr._read_history(tmp_path / "res")] == [0, 1]
    assert tr.train_run(small, run_dir=tmp_path / "res", **kw)["status"] == "completed"
    assert _strip(tr._read_history(tmp_path / "res")) == _strip(tr._read_history(tmp_path / "ref"))


def test_adamw_and_failures(small: tr.SequenceData, tmp_path: Path) -> None:
    opt = tr.make_optimizer("adamw", tr.make_model(small.vocab, "gpt2_style", 2, 30, 0), 1e-3)
    assert isinstance(opt, torch.optim.AdamW) and opt.defaults["weight_decay"] == 0.0
    meta = tr.train_run(small, variant="gpt2_style", num_layers=1, d_model=42, seed=1, lr=1e3, max_epochs=2,
                        run_dir=tmp_path / "boom")
    assert meta["status"] in ("failed_mechanical", "failed_diverged")
    with pytest.raises(ValueError, match="already finished"):
        tr.train_run(small, variant="gpt2_style", num_layers=1, d_model=42, seed=1, lr=1e3, max_epochs=2,
                     run_dir=tmp_path / "boom")


def _sel(lr, lo=False, hi=False):
    return {"selected": lr, "at_lower_boundary": lo, "at_upper_boundary": hi}


def test_shape_choice_rule() -> None:
    def shape(finals, lr=0.1):
        return {"results": {lr: {"final_dev": finals}}, "selection": _sel(lr)}

    per = {(1, 42): shape([2.050, 2.051, 2.052]), (2, 30): shape([2.0500, 2.0505, 2.0510]),
           (3, 24): shape([2.040, 2.041, 2.042])}
    out = tr.choose_shape(per)
    assert out["best"] == "L3_d24" and out["chosen"]["num_layers"] == 3   # others are ≥ 0.009 worse
    per[(2, 30)] = shape([2.0405, 2.0412, 2.0419])                        # within 0.001 of the best
    out = tr.choose_shape(per)
    assert out["tied"] == ["L2_d30", "L3_d24"] and out["chosen"]["num_layers"] == 2
    per[(2, 30)]["selection"] = _sel(0.1, lo=True)                        # a boundary η is not usable yet
    assert tr.choose_shape(per)["chosen"]["num_layers"] == 3


def test_study_lock(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    from training.transformer_study.run_lock import LOCK_NAME, StudyLocked, study_lock

    with study_lock(tmp_path):
        assert (tmp_path / LOCK_NAME).read_text() == str(os.getpid())
        with pytest.raises(StudyLocked, match="another run"):
            with study_lock(tmp_path):
                pass
    assert not (tmp_path / LOCK_NAME).exists()                 # released on exit
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    (tmp_path / LOCK_NAME).write_text(dead.stdout.strip())     # a lock left by a process that has ended
    with study_lock(tmp_path):                                 # is replaced, not obeyed
        pass


def test_drivers_take_the_lock() -> None:
    import inspect

    assert "study_lock(root)" in inspect.getsource(tr.main)
    assert "study_lock(root)" in inspect.getsource(d.main)
