"""Tests for ``training/transformer_study/test_evaluation.py``.

The freeze/evaluate guards are exercised on a throwaway mini-study (1-epoch runs on 400 training names)
whose "test" split is replaced by dev names, so no test reads the real test split."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
from scipy import stats

import training.transformer_study.test_evaluation as te
import training.transformer_study.transformer_runs as tr
import training.transformer_study.wavenet_d_arm as wd

SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- statistics
def test_paired_matches_scipy() -> None:
    a = [2.05, 2.06, 2.04, 2.055, 2.052]
    b = [2.06, 2.065, 2.05, 2.058, 2.06]
    out = te.paired(a, b)
    ref = stats.ttest_rel(a, b)
    assert out["p_two_sided"] == pytest.approx(ref.pvalue, rel=1e-10)
    assert out["mean_diff"] == pytest.approx(sum(x - y for x, y in zip(a, b)) / 5)
    assert out["a_lower_in"] == 5


def test_tost() -> None:
    b = [2.050 + 0.001 * i for i in range(10)]
    near = [x + 0.002 + 0.0005 * (-1) ** i for i, x in enumerate(b)]   # tiny, consistent difference
    far = [x + 0.03 + 0.0005 * (-1) ** i for i, x in enumerate(b)]
    assert te.tost(near, b)["equivalent"] and te.tost(near, b)["p_tost"] < 0.05
    out = te.tost(far, b)
    assert not out["equivalent"] and out["p_tost"] > 0.5
    lo, hi = out["ci90"]
    assert lo < out["mean_diff"] < hi


def test_holm() -> None:
    out = te.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"]["reject"] and not out["b"]["reject"] and not out["c"]["reject"]   # 0.03 > 0.05 / 2
    assert out["a"]["p_holm"] == pytest.approx(0.03)
    assert out["c"]["p_holm"] == pytest.approx(0.06) and out["b"]["p_holm"] == pytest.approx(0.06)


# ---------------------------------------------------------------- freeze and one-time evaluation
@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    splits = te._splits()
    vocab = wd.Vocabulary.from_names(splits.train)
    train = list(splits.train[:400])
    seq = tr.build_sequence_data(train, splits.dev, vocab)
    win = wd.StudyData(vocab=vocab, train_names=tuple(train), train=wd.build_name_windows(train, vocab),
                       dev=wd.build_name_windows(splits.dev, vocab))
    (root).mkdir(exist_ok=True)
    json.dump({"shape_choice": {"chosen": {"num_layers": 1, "d_model": 42, "lr": 0.1}}},
              open(root / "shape_pilot_summary.json", "w"))
    for s in SEEDS:
        for arm in ("gpt2_style", "transformer_2017"):
            tr.train_run(seq, variant=arm, num_layers=1, d_model=42, seed=s, lr=0.1, max_epochs=1,
                         run_dir=te.run_dir(root, arm, s))
        wd.train_run(win, seed=s, lr=0.1, max_epochs=1, run_dir=te.run_dir(root, "wavenet_d", s))
    rnn = root / "rnn_result.json"
    json.dump({"per_seed": {str(s): {"test_nll": 2.06 + 0.001 * s, "dev_nll": 2.05, "best_epoch": 38} for s in SEEDS}},
              open(rnn, "w"))
    return root, rnn, splits


@pytest.fixture
def patched(mini, monkeypatch):
    root, rnn, splits = mini
    monkeypatch.setattr(te, "RNN_RESULT", rnn)
    monkeypatch.setattr(te, "_splits", lambda: dataclasses.replace(splits, test=splits.dev[:300]))
    return root


def test_freeze_evaluate_analyze_once(patched) -> None:
    root = patched
    fz = te.freeze(root, seeds=SEEDS)
    assert set(fz["checkpoints"]) == {"gpt2_style", "transformer_2017", "wavenet_d"}
    assert all(len(v) == 3 for v in fz["checkpoints"].values())
    with pytest.raises(SystemExit, match="already frozen"):
        te.freeze(root, seeds=SEEDS)
    res = te.evaluate(root)
    assert set(res["per_seed"]) == {"gpt2_style", "transformer_2017", "wavenet_d", "rnn"}
    assert all(math.isfinite(v["test_nll"]) for per in res["per_seed"].values() for v in per.values())
    with pytest.raises(SystemExit, match="one time only"):
        te.evaluate(root)
    out = te.analyze(root)
    assert set(out["holm"]) == {"T-RQ1a", "T-RQ1b_equivalence", "T-RQ2"}
    assert len(out["secondary"]) == 6


def test_evaluation_refuses_changed_checkpoint(mini, monkeypatch, tmp_path) -> None:
    import shutil
    root0, rnn, splits = mini
    root = tmp_path / "copy"
    shutil.copytree(root0, root, ignore=shutil.ignore_patterns("test_evaluation"))
    monkeypatch.setattr(te, "RNN_RESULT", rnn)
    monkeypatch.setattr(te, "_splits", lambda: dataclasses.replace(splits, test=splits.dev[:300]))
    te.freeze(root, seeds=SEEDS)
    best = te.run_dir(root, "gpt2_style", 1) / "model_best.pt"
    best.write_bytes(best.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="changed since the freeze"):
        te.evaluate(root)
    assert not (root / "test_evaluation" / "result.json").exists()
