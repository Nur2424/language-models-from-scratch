"""MS8: the read-only run verifier (``training/rnn_study/verify_runs.py``; plan §5.2, §8.5).

One tiny research-configured run (synthetic names, H = 4, full diagnostic plan) is trained once per
module; each test copies it and alters one record. The verifier must report nothing on the clean copy
and a finding on every altered copy, and must never change a file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import string
from pathlib import Path

import pytest
import torch

import models.recurrent.rnn_training as rt
from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from training.rnn_study.verify_runs import verify_run
from utils.reproducibility import derive_seed, make_generator
from utils.data import make_next_token_sequences
from utils.run_manager import RunManager
from utils.vocabulary import Vocabulary

VOCAB = Vocabulary(chars=tuple(string.ascii_lowercase))

CFG = rt.RNNTrainingConfig(lr=0.5, max_epochs=3, batch_size=4, eval_batch_size=5)


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("verify")
    g = torch.Generator().manual_seed(11)
    names = ["".join(chr(97 + int(c)) for c in torch.randint(0, 26, (int(torch.randint(1, 9, (1,), generator=g)),),
                                                                  generator=g)) for _ in range(25)]
    train_names, dev_names = tuple(names[:18]), names[18:]
    train, dev = make_next_token_sequences(train_names, VOCAB), make_next_token_sequences(dev_names, VOCAB)
    info = {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1], "N": len(train_names),
            "M": int(train.lengths.sum())}
    run = RunManager.create("rnn", config={"fixture": True}, results_root=root)
    probe = dg.build_probe(train, train_names, info, root / "probe.json")
    plan = dg.RNNDiagnosticsPlan(probe=probe, train=train, train_names=train_names, dev=dev,
                                 steps_per_epoch=-(-len(train_names) // CFG.batch_size),
                                 max_epochs=CFG.max_epochs, recurrent_gain=1.0,
                                 out_dir=run.run_dir / dg.RESEARCH_DIAGNOSTICS_DIR)
    model = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=4), generator=make_generator(derive_seed(3, "init")))
    rt.train_rnn(model, train, dev, training=CFG, seed=3, run=run, diagnostics=plan)
    return run.run_dir


def _tree_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode() + p.read_bytes())
    return h.hexdigest()


@pytest.fixture
def copy(clean_run, tmp_path) -> Path:
    dst = tmp_path / "run"
    shutil.copytree(clean_run, dst)
    return dst


def _rewrite_jsonl(path: Path, edit) -> None:
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    edit(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_clean_run_has_no_findings_and_is_not_modified(copy) -> None:
    before = _tree_hash(copy)
    assert verify_run(copy) == []
    assert _tree_hash(copy) == before


def test_history_hash_tamper_detected(copy) -> None:
    def edit(rows):
        rows[-1]["param_sha256"] = "0" * 64
    _rewrite_jsonl(copy / "history.jsonl", edit)
    findings = verify_run(copy)
    assert any("checkpoint_last.pt" in f for f in findings)


def test_selection_tamper_detected(copy) -> None:
    """A history edited so that another epoch would be selected no longer matches model_best.pt."""
    rows = [json.loads(x) for x in (copy / "history.jsonl").read_text().splitlines()]
    best = min((r for r in rows if r["completed_epochs"] >= 1), key=lambda r: (r["dev_nll"], r["completed_epochs"]))
    other = next(r for r in rows if r["completed_epochs"] >= 1 and r["completed_epochs"] != best["completed_epochs"])

    def edit(rs):
        rs[other["completed_epochs"]]["dev_nll"] = best["dev_nll"] - 1.0
    _rewrite_jsonl(copy / "history.jsonl", edit)
    assert any("best checkpoint" in f for f in verify_run(copy))


def test_step_record_removed_detected(copy) -> None:
    _rewrite_jsonl(copy / "diagnostics" / "steps.jsonl", lambda rows: rows.pop(2))
    assert any("step records" in f or "unequal steps" in f for f in verify_run(copy))


def test_raw_file_removed_detected(copy) -> None:
    raw = sorted((copy / dg.RESEARCH_DIAGNOSTICS_DIR / "raw").glob("point_*.npz"))[0]
    raw.unlink()
    assert any("raw file" in f for f in verify_run(copy))


def test_diagnostic_point_hash_tamper_detected(copy) -> None:
    def edit(rows):
        p = next(r for r in rows if r["kind"] == "epoch_end")
        p["param_sha256_after"] = "f" * 64
    _rewrite_jsonl(copy / dg.RESEARCH_DIAGNOSTICS_DIR / "points.jsonl", edit)
    assert any("changed the parameters" in f for f in verify_run(copy))


def test_invalid_point_requires_incomplete_status(copy) -> None:
    def edit(rows):
        rows[0]["valid"] = False
    _rewrite_jsonl(copy / dg.RESEARCH_DIAGNOSTICS_DIR / "points.jsonl", edit)
    assert any("diagnostics_status" in f for f in verify_run(copy))


def test_metadata_outcome_tamper_detected(copy) -> None:
    meta = json.loads((copy / "metadata.json").read_text())
    meta["outcome"] = "failed_diverged"
    (copy / "metadata.json").write_text(json.dumps(meta))
    assert any("failed_diverged" in f for f in verify_run(copy))


def test_selection_tie_resolves_to_the_earliest_epoch(copy) -> None:
    """Strict <: a later epoch that only ties the selected one must not be selected; a history edited
    so that a later epoch ties an earlier one moves selection to the earlier epoch → finding."""
    rows = [json.loads(x) for x in (copy / "history.jsonl").read_text().splitlines()]
    best = min((r for r in rows if r["completed_epochs"] >= 1), key=lambda r: (r["dev_nll"], r["completed_epochs"]))
    earlier = [r for r in rows if 1 <= r["completed_epochs"] < best["completed_epochs"]]
    later = [r for r in rows if r["completed_epochs"] > best["completed_epochs"]]
    if earlier:  # an earlier tie takes the selection away from the checkpointed epoch
        target = earlier[0]["completed_epochs"]
    else:  # a later tie must leave the selection where it is
        target = later[0]["completed_epochs"]

    def edit(rs):
        rs[target]["dev_nll"] = best["dev_nll"]
    _rewrite_jsonl(copy / "history.jsonl", edit)
    findings = verify_run(copy)
    assert bool(findings) is bool(earlier), findings
