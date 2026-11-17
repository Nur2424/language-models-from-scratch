"""MS5 tests of the training procedure (``models/recurrent/rnn_training.py``).

Milestone MS5 of ``rnn_implementation_plan.md`` v0.2 (D-49). Labels (D-48, C2): ``A:`` arch spec
§13.2 T1 – T8; ``T:`` training spec §14 T9 – T24; ``X2`` research configuration, ``X4`` abort before
the first commit. Isolation tests D:I1 – I8 are in ``test_rnn_isolation.py``.

Synthetic names only; tiny models (H = 4); the Git guard is installed by ``conftest.py``. Fields that
measure wall time (``seconds``) are excluded from identity comparisons; everything else is compared
bitwise. Tolerances: A = 1e‑9 (float64), B = 1e‑5 (float32).
"""

from __future__ import annotations

import json
import math
import string
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import models.recurrent.rnn_training as rt
from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import IncompatibleCheckpointError, RNNConfig, RNNLanguageModel
from utils.checkpoint import load_checkpoint
from utils.config import default_config_paths, load_config
from utils.data import make_next_token_sequences
from utils.metrics import NLLAccumulator, token_nll
from utils.reproducibility import derive_seed, make_generator
from utils.run_manager import RunManager
from utils.vocabulary import Vocabulary

VOCAB = Vocabulary(chars=tuple(string.ascii_lowercase))
SEED = 7
#: RNG stream name of the EpochSampler (training spec §10.1); a name, not a path.
SAMPLER_STREAM = "data"


def _names(count, seed, max_len=8):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        L = int(torch.randint(1, max_len + 1, (1,), generator=g))
        out.append("".join(chr(97 + int(c)) for c in torch.randint(0, 26, (L,), generator=g)))
    return out


TRAIN_NAMES, DEV_NAMES = _names(19, 1), _names(7, 2)
TRAIN = make_next_token_sequences(TRAIN_NAMES, VOCAB)
DEV = make_next_token_sequences(DEV_NAMES, VOCAB)
CFG = rt.RNNTrainingConfig(lr=0.5, max_epochs=3, batch_size=4, eval_batch_size=5)  # N = 19 → S = 5, r = 3


def new_model(dtype=torch.float32, H=4):
    return RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=H), generator=make_generator(derive_seed(SEED, "init"))).to(dtype)


def new_run(root: Path) -> RunManager:
    return RunManager.create("rnn", config={"fixture": True}, results_root=root)


def new_plan(run: RunManager, cfg=CFG):
    info = {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1], "N": len(TRAIN_NAMES),
            "M": int(TRAIN.lengths.sum())}
    probe_path = run.run_dir.parent / "probe.json"
    probe = dg.load_probe(probe_path) if probe_path.exists() else dg.build_probe(TRAIN, TRAIN_NAMES, info, probe_path)
    return dg.RNNDiagnosticsPlan(probe=probe, train=TRAIN, train_names=TRAIN_NAMES, dev=DEV,
                                 steps_per_epoch=-(-len(TRAIN_NAMES) // cfg.batch_size), max_epochs=cfg.max_epochs,
                                 recurrent_gain=1.0, out_dir=run.run_dir / dg.RESEARCH_DIAGNOSTICS_DIR)


def strip_timing(rows):
    return [{k: v for k, v in r.items() if k not in dg.TIMING_FIELDS} for r in rows]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def ckpt_core(path):
    c = load_checkpoint(path)
    return c["model_state"], c["extra"], c["epoch"], c["step"], c["best_metric"], c["config"]


def same_state(a, b):
    return all(torch.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()


# =========================================================================== configuration
def test_training_config_validation() -> None:
    for bad in (dict(lr=0.0, max_epochs=1), dict(lr=float("nan"), max_epochs=1), dict(lr=True, max_epochs=1),
                dict(lr=0.1, max_epochs=0), dict(lr=0.1, max_epochs=1, batch_size=0),
                dict(lr=0.1, max_epochs=1, eval_batch_size=1.5)):
        with pytest.raises(ValueError):
            rt.RNNTrainingConfig(**bad)
    assert rt.RNNTrainingConfig(lr=1, max_epochs=2) == rt.RNNTrainingConfig(lr=1.0, max_epochs=2, batch_size=32,
                                                                           eval_batch_size=1024)


# =========================================================================== T9 – T11 loss normalisation
def test_T9_loss_normalizer() -> None:
    """T:T9: C = B·M/N, recorded in the result and in ``derived`` (the protocol value is 730,664/3,205)."""
    M = int(TRAIN.lengths.sum())
    assert rt.loss_normalizer(CFG, TRAIN) == 4 * M / 19
    assert math.isclose(32 * 182_666 / 25_640, 730_664 / 3_205, rel_tol=1e-15)
    res = rt.train_rnn(new_model(), TRAIN, DEV, training=replace(CFG, max_epochs=1), seed=SEED)
    assert res.loss_normalizer == 4 * M / 19


def test_T10_static_weighting_at_fixed_parameters() -> None:
    """T:T10 (A, float64): at fixed θ, Σ_b L_b·C equals the total token NLL and Σ_b ∇L_b equals
    (N/B)·∇L_data over one epoch's batches (any permutation)."""
    model = new_model(torch.float64)
    with torch.no_grad():
        model.output_weight.normal_(0, 0.5, generator=make_generator(3))
    C = rt.loss_normalizer(CFG, TRAIN)
    from utils.batching import EpochSampler
    sampler = EpochSampler(19, 4, seed=derive_seed(SEED, SAMPLER_STREAM))
    total, grads = 0.0, [torch.zeros_like(p) for p in model.parameters()]
    for idx in sampler.batches(0):
        T = int(TRAIN.lengths[idx].max())
        model.zero_grad()
        loss = token_nll(model(TRAIN.inputs[idx, :T]), TRAIN.targets[idx, :T], reduction="sum") / C
        loss.backward()
        total += float(loss.detach()) * C
        grads = [g + p.grad for g, p in zip(grads, model.parameters())]
    model.zero_grad()
    full = token_nll(model(TRAIN.inputs), TRAIN.targets, reduction="sum")
    assert abs(total - float(full.detach())) <= 1e-9 * abs(float(full.detach()))
    (full / int(TRAIN.lengths.sum())).backward()
    for g, p in zip(grads, model.parameters()):
        expected = (19 / 4) * p.grad
        assert float(((g - expected).abs() - (1e-9 + 1e-9 * expected.abs())).max()) <= 0


def test_T11_T20_final_batch_and_step_records(tmp_path) -> None:
    """T:T11, T:T20: per epoch ⌈N/B⌉ = 5 step records; Σ names = N; Σ m_b = M; the final batch has
    r = 3 names and loss = loss_sum/C (same C as every step)."""
    run = new_run(tmp_path)
    rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run)
    steps = read_jsonl(run.diagnostics_dir / "steps.jsonl")
    C = rt.loss_normalizer(CFG, TRAIN)
    assert [s["step"] for s in steps] == list(range(1, 16))
    for e in range(3):
        ep = [s for s in steps if s["epoch"] == e]
        assert len(ep) == 5 and sum(s["names"] for s in ep) == 19 and sum(s["m_b"] for s in ep) == int(TRAIN.lengths.sum())
        assert ep[-1]["names"] == 3
        for s in ep:
            assert math.isclose(s["loss"], s["nll_mean"] * s["m_b"] / C, rel_tol=1e-12)
            assert set(s["grad_norms"]) == {"U", "W", "b", "V", "c"} and s["finite"] is True
            assert math.isclose(s["grad_norm"], math.sqrt(sum(v * v for v in s["grad_norms"].values())), rel_tol=1e-12)


# =========================================================================== T12 – T14 failures, selection
def _nan_once(monkeypatch, target, name, at_call, replace_with):
    real = getattr(target, name)
    calls = {"n": 0}

    def wrapper(*a, **k):
        out = real(*a, **k)
        if not out.requires_grad:  # evaluation calls run under no_grad: not training steps
            return out
        calls["n"] += 1
        return replace_with(out) if calls["n"] == at_call else out

    monkeypatch.setattr(target, name, wrapper)


def test_T12_A_T4_category_A_loss(tmp_path, monkeypatch) -> None:
    """T:T12 / A:T4: an injected NaN loss raises at that step, before backward; parameters bitwise
    unchanged from before the failing step; status ``failed_mechanical``; no commit of that epoch."""
    run = new_run(tmp_path)
    model = new_model()
    _nan_once(monkeypatch, rt, "token_nll", 7, lambda out: out * float("nan"))
    after = []
    real_step = torch.optim.SGD.step

    def spy_step(self, *a, **k):
        out = real_step(self, *a, **k)
        after.append([p.detach().clone() for p in model.parameters()])
        return out

    monkeypatch.setattr(torch.optim.SGD, "step", spy_step)
    with pytest.raises(rt.CategoryAFailure) as info:
        rt.train_rnn(model, TRAIN, DEV, training=CFG, seed=SEED, run=run)
    assert info.value.where == "loss" and info.value.result.status == "failed_mechanical"
    assert len(after) == 6 and info.value.details["step"] == 7  # the failing step made no update
    assert all(torch.equal(a, b.detach()) for a, b in zip(after[-1], model.parameters()))
    meta = json.loads(run.metadata_path.read_text())
    assert meta["outcome"] == "failed_mechanical" and meta["status"] == "failed" and meta["failure"]["where"] == "loss"
    assert load_checkpoint(run.checkpoint_path("resume"))["epoch"] == 1  # epoch 2 not committed


def test_T12_category_A_gradient_leaves_parameters_unchanged(tmp_path) -> None:
    """T:T12: a non-finite gradient raises before ``optimizer.step()``: parameters bitwise unchanged."""
    model = new_model()
    calls = {"n": 0}

    def poison(grad):
        calls["n"] += 1
        return grad * float("inf") if calls["n"] == 4 else grad

    model.recurrent_weight.register_hook(poison)
    snapshots = []
    real_step = torch.optim.SGD.step

    def spy(self, *a, **k):
        out = real_step(self, *a, **k)
        snapshots.append([p.detach().clone() for p in model.parameters()])
        return out

    torch.optim.SGD.step = spy
    try:
        with pytest.raises(rt.CategoryAFailure) as info:
            rt.train_rnn(model, TRAIN, DEV, training=CFG, seed=SEED)
    finally:
        torch.optim.SGD.step = real_step
    assert info.value.where == "gradient" and len(snapshots) == 3
    assert all(torch.equal(a, b.detach()) for a, b in zip(snapshots[-1], model.parameters()))


def test_T12_category_A_post_step_parameter_and_evaluation(tmp_path, monkeypatch) -> None:
    real_step = torch.optim.SGD.step

    def bad_step(self, *a, **k):
        out = real_step(self, *a, **k)
        with torch.no_grad():
            self.param_groups[0]["params"][2][0] = float("nan")
        return out

    monkeypatch.setattr(torch.optim.SGD, "step", bad_step)
    with pytest.raises(rt.CategoryAFailure) as info:
        rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED)
    assert info.value.where == "parameter"
    monkeypatch.setattr(torch.optim.SGD, "step", real_step)

    bad = new_model()
    with torch.no_grad():
        bad.output_bias[0] = float("nan")  # the epoch-0 evaluation is the first computation
    with pytest.raises(rt.CategoryAFailure) as info:
        rt.train_rnn(bad, TRAIN, DEV, training=CFG, seed=SEED)
    assert info.value.where == "evaluation"


def _scripted_eval(monkeypatch, values):
    """Replace evaluate_nll: call i returns mean NLL values[i] (train, dev alternate per epoch)."""
    calls = {"n": 0}

    def fake(model, sequences, *, batch_size):
        v = values[calls["n"]]
        calls["n"] += 1
        acc = NLLAccumulator()
        acc.update_from_sum(v * 10, 10)
        return acc

    monkeypatch.setattr(rt, "evaluate_nll", fake)


def test_T13_divergence_ends_run_after_commit(tmp_path, monkeypatch) -> None:
    """T:T13: the flag is set exactly when train NLL > ln 27 at k ≥ 1; that epoch is committed;
    ``model_final.pt`` is written; the run ends with ``failed_diverged``."""
    _scripted_eval(monkeypatch, [3.4, 3.4, 3.0, 3.1, math.log(27) + 1e-6, 3.0, 2.0, 2.0])
    run = new_run(tmp_path)
    res = rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run)
    assert res.status == "failed_diverged"
    assert [r["diverged"] for r in res.history] == [False, False, True]  # epoch 0 at 3.4 > ln 27: never flagged
    assert load_checkpoint(run.checkpoint_path("resume"))["epoch"] == 2
    assert load_checkpoint(run.checkpoint_path("final"))["epoch"] == 2
    assert json.loads(run.metadata_path.read_text())["outcome"] == "failed_diverged"
    assert len(read_jsonl(run.history_path)) == 3


def test_T14_A_T3_selection_rules(tmp_path, monkeypatch) -> None:
    """T:T14 / A:T3: rows k = 0 … E; k = 0 never selected (even with the lowest dev NLL); strict <
    keeps the earliest of an exact tie; ``model_best.pt`` holds that epoch."""
    _scripted_eval(monkeypatch, [3.0, 1.0, 2.9, 2.5, 2.8, 2.5, 2.7, 2.6])
    run = new_run(tmp_path)
    res = rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run)
    assert [r["completed_epochs"] for r in res.history] == [0, 1, 2, 3]
    assert res.best_epoch == 1 and res.best_dev_nll == 2.5 and res.history[0]["selectable"] is False
    assert load_checkpoint(run.checkpoint_path("best"))["epoch"] == 1
    assert [r["is_best"] for r in res.history] == [False, True, False, False]


# =========================================================================== T15 – T17 resume
class _Interrupt(Exception):
    pass


def _interrupt_at_epoch_end(monkeypatch, epoch_index, where="o2"):
    """Raise during the epoch-end of 0-based epoch ``epoch_index`` (k = epoch_index + 1), before its
    commit: at O2 (``where='o2'``) or at the commit write (``where='commit'``)."""
    if where == "o2":
        real, calls = rt._o2, {"n": 0}

        def o2(model):
            calls["n"] += 1
            if calls["n"] == epoch_index + 2:  # call 1 is epoch 0's row
                raise _Interrupt("injected interruption")
            return real(model)

        monkeypatch.setattr(rt, "_o2", o2)
    else:
        real = rt.save_checkpoint

        def save(path, *, kind, **kw):
            if kind == "resume" and kw["epoch"] == epoch_index + 1:
                raise _Interrupt("injected crash at the commit")
            return real(path, kind=kind, **kw)

        monkeypatch.setattr(rt, "save_checkpoint", save)


@pytest.mark.parametrize("with_plan", [False, True], ids=["no-plan", "plan"])
@pytest.mark.parametrize("where", ["o2", "commit"])
def test_T15_T16_resume_equals_uninterrupted(tmp_path, monkeypatch, with_plan, where) -> None:
    """T:T15 (and D:I7 with a plan): a run interrupted in epoch 2 and resumed equals the uninterrupted
    run: parameters, history, step records, checkpoints and diagnostic records. T:T16 (``commit``):
    the crash comes after the epoch-end diagnostic point; that point is rewritten once."""
    ref_run = new_run(tmp_path / "ref")
    ref_model = new_model()
    rt.train_rnn(ref_model, TRAIN, DEV, training=CFG, seed=SEED, run=ref_run,
                 diagnostics=new_plan(ref_run) if with_plan else None)
    run = new_run(tmp_path / "int")
    with monkeypatch.context() as m:
        _interrupt_at_epoch_end(m, 1, where)
        with pytest.raises(_Interrupt):
            rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run,
                         diagnostics=new_plan(run) if with_plan else None)
    if with_plan and where == "commit":
        kinds = [(r["kind"], r["completed_epochs"]) for r in dg.read_jsonl_complete(run.run_dir / "research_diagnostics" / "points.jsonl")]
        assert ("epoch_end", 2) in kinds  # written before the crash, not committed
    model = new_model()
    res = rt.train_rnn(model, TRAIN, DEV, training=CFG, seed=SEED, run=run, resume=True,
                       diagnostics=new_plan(run) if with_plan else None)
    assert res.status == "completed"
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), ref_model.parameters()))
    assert strip_timing(read_jsonl(run.history_path)) == strip_timing(read_jsonl(ref_run.history_path))
    assert (run.diagnostics_dir / "steps.jsonl").read_bytes() == (ref_run.diagnostics_dir / "steps.jsonl").read_bytes()
    for kind in ("best", "final", "resume"):
        a, b = ckpt_core(run.checkpoint_path(kind)), ckpt_core(ref_run.checkpoint_path(kind))
        assert same_state(a[0], b[0]) and a[1:] == b[1:], kind
    if with_plan:
        d1, d2 = run.run_dir / "research_diagnostics", ref_run.run_dir / "research_diagnostics"
        assert strip_timing(read_jsonl(d1 / "points.jsonl")) == strip_timing(read_jsonl(d2 / "points.jsonl"))
        assert (d1 / "primary.jsonl").read_bytes() == (d2 / "primary.jsonl").read_bytes()
        assert sorted(p.name for p in (d1 / "raw").iterdir()) == sorted(p.name for p in (d2 / "raw").iterdir())
        for p in (d1 / "raw").iterdir():
            assert p.read_bytes() == (d2 / "raw" / p.name).read_bytes()


@pytest.mark.parametrize("change", ["lr", "max_epochs", "batch_size", "eval_batch_size", "seed", "vocab",
                                    "hidden_dim", "threads", "diagnostics", "dtype"])
def test_T17_resume_rejects_each_changed_field(tmp_path, monkeypatch, change) -> None:
    """T:T17: resume refuses any change of the training-spec §10.6 fields."""
    run = new_run(tmp_path)
    with monkeypatch.context() as m:
        _interrupt_at_epoch_end(m, 1)
        with pytest.raises(_Interrupt):
            rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run)
    cfg, seed, model, plan = CFG, SEED, new_model(), None
    threads = torch.get_num_threads()
    if change in ("lr", "max_epochs", "batch_size", "eval_batch_size"):
        cfg = replace(CFG, **{change: {"lr": 0.25, "max_epochs": 4, "batch_size": 5, "eval_batch_size": 6}[change]})
    elif change == "seed":
        seed = SEED + 1
    elif change == "vocab":
        model = RNNLanguageModel(Vocabulary(chars=tuple("abcdefghijklmnopqrstuvwxy~")), RNNConfig(hidden_dim=4),
                                 generator=make_generator(derive_seed(SEED, "init")))
    elif change == "hidden_dim":
        model = new_model(H=5)
    elif change == "threads":
        torch.set_num_threads(threads + 1)
    elif change == "diagnostics":
        plan = new_plan(run)
    elif change == "dtype":
        model = new_model(torch.float64)
    try:
        with pytest.raises((IncompatibleCheckpointError, ValueError, RuntimeError)):
            rt.train_rnn(model, TRAIN, DEV, training=cfg, seed=seed, run=run, resume=True, diagnostics=plan)
    finally:
        torch.set_num_threads(threads)


# =========================================================================== T18 hook order
class SpyPlan(dg.RNNDiagnosticsPlan):
    def __init__(self, *a, run=None, **k):
        super().__init__(*a, **k)
        self.calls, self.run = [], run

    def _ckpt_epoch(self):
        p = self.run.checkpoint_path("resume")
        return load_checkpoint(p)["epoch"] if p.exists() else None

    def at_start(self, model):
        self.calls.append(("start", 0, dg.param_sha256(model), self._ckpt_epoch(), len(read_jsonl(self.run.history_path))))
        super().at_start(model)

    def after_step(self, model, *, sigma, epoch):
        self.calls.append(("step", sigma, dg.param_sha256(model), self._ckpt_epoch(), epoch))
        super().after_step(model, sigma=sigma, epoch=epoch)

    def at_epoch_end(self, model, *, sigma, completed_epochs):
        self.calls.append(("end", sigma, dg.param_sha256(model), self._ckpt_epoch(), completed_epochs))
        super().at_epoch_end(model, sigma=sigma, completed_epochs=completed_epochs)


def test_T18_hook_order_and_hash_alignment(tmp_path) -> None:
    """T:T18 / D:D4: P₀ after the epoch-0 row and before step 1; ``after_step`` at every step with
    σ = completed steps; the epoch-end call precedes that epoch's checkpoint write; hook hashes equal
    the history, checkpoint and diagnostic-point hashes."""
    run = new_run(tmp_path)
    base = new_plan(run)
    plan = SpyPlan(probe=base.probe, train=TRAIN, train_names=TRAIN_NAMES, dev=DEV, steps_per_epoch=5, max_epochs=3,
                   recurrent_gain=1.0, out_dir=run.run_dir / "research_diagnostics", run=run)
    rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run, diagnostics=plan)
    assert plan.calls[0][0] == "start" and plan.calls[0][4] == 1  # the epoch-0 row was already written
    steps = [c for c in plan.calls if c[0] == "step"]
    assert [c[1] for c in steps] == list(range(1, 16))
    ends = [c for c in plan.calls if c[0] == "end"]
    assert [(c[1], c[4]) for c in ends] == [(5, 1), (10, 2), (15, 3)]
    assert [c[3] for c in ends] == [None, 1, 2]  # checkpoint_last not yet written for this epoch
    history = read_jsonl(run.history_path)
    assert [c[2] for c in ends] == [r["param_sha256"] for r in history[1:]]
    assert plan.calls[0][2] == history[0]["param_sha256"]
    assert load_checkpoint(run.checkpoint_path("final"))["extra"]["param_sha256"] == history[-1]["param_sha256"]
    best = load_checkpoint(run.checkpoint_path("best"))
    assert best["extra"]["param_sha256"] == history[best["epoch"]]["param_sha256"]
    points = dg.read_jsonl_complete(run.run_dir / "research_diagnostics" / "points.jsonl")
    for p in points:
        if p["kind"] == "epoch_end":
            assert p["param_sha256_before"] == history[p["completed_epochs"]]["param_sha256"]


# =========================================================================== T19 – T21 evaluation, trimming
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-9), (torch.float32, 1e-5)])
def test_T19_A_T6_evaluation(dtype, tol) -> None:
    """T:T19 / A:T6: length-sorted chunked evaluation equals unsorted single-chunk evaluation and the
    manual token-weighted NLL, for any chunk size; position totals equal the full NLL."""
    model = new_model(dtype)
    with torch.no_grad():
        model.output_weight.normal_(0, 0.5, generator=make_generator(4))
    manual = float(token_nll(model(DEV.inputs), DEV.targets, reduction="sum").detach()) / int(DEV.lengths.sum())
    for bs in (1, 2, 5, 1024):
        got = rt.evaluate_nll(model, DEV, batch_size=bs).mean_nll
        assert abs(got - manual) <= tol + tol * abs(manual), bs
        total, count = rt.evaluate_position_nll(model, DEV, batch_size=bs).total()
        assert count == int(DEV.lengths.sum()) and abs(total / count - manual) <= tol + tol * abs(manual)


def test_A_T7_loss_unaffected_by_trimming() -> None:
    """A:T7: the Option-B loss equals ``token_nll(..., reduction="sum") / C`` and is unaffected by
    per-batch trimming (float64, A)."""
    model = new_model(torch.float64)
    with torch.no_grad():
        model.output_weight.normal_(0, 0.5, generator=make_generator(5))
    idx = torch.tensor([0, 3, 7])
    T = int(TRAIN.lengths[idx].max())
    C = rt.loss_normalizer(CFG, TRAIN)
    trimmed = float(token_nll(model(TRAIN.inputs[idx, :T]), TRAIN.targets[idx, :T], reduction="sum").detach()) / C
    full = float(token_nll(model(TRAIN.inputs[idx]), TRAIN.targets[idx], reduction="sum").detach()) / C
    assert abs(trimmed - full) <= 1e-9 * abs(full)


def test_T21_trimmed_and_untrimmed_trajectories_agree() -> None:
    """T:T21 (A, float64): the trimmed training trajectory equals an untrimmed re-implementation of
    the same SGD steps (same sampler, same C) in this test."""
    model = new_model(torch.float64)
    rt.train_rnn(model, TRAIN, DEV, training=replace(CFG, max_epochs=2), seed=SEED)
    other = new_model(torch.float64)
    opt = torch.optim.SGD(other.parameters(), lr=CFG.lr)
    from utils.batching import EpochSampler
    sampler = EpochSampler(19, 4, seed=derive_seed(SEED, SAMPLER_STREAM))
    C = rt.loss_normalizer(CFG, TRAIN)
    for e in range(2):
        for idx in sampler.batches(e):
            opt.zero_grad()
            (token_nll(other(TRAIN.inputs[idx]), TRAIN.targets[idx], reduction="sum") / C).backward()
            opt.step()
    for a, b in zip(model.parameters(), other.parameters()):
        a, b = a.detach(), b.detach()
        assert float(((a - b).abs() - (1e-9 + 1e-9 * b.abs())).max()) <= 0


# =========================================================================== T22, A:T2, A:T8 determinism and RNG
def test_T22_A_T2_A_T8_determinism_rng_and_git(tmp_path) -> None:
    """T:T22 / A:T2 / A:T8: two runs with the same seed are bitwise identical; ``train_rnn`` leaves the
    global RNG and the model mode as found; the Git guard refuses any git process."""
    rng = torch.random.get_rng_state()
    m1, m2 = new_model(), new_model()
    m1.train()
    r1 = rt.train_rnn(m1, TRAIN, DEV, training=CFG, seed=SEED, run=new_run(tmp_path / "a"))
    r2 = rt.train_rnn(m2, TRAIN, DEV, training=CFG, seed=SEED, run=new_run(tmp_path / "b"))
    assert torch.equal(torch.random.get_rng_state(), rng) and m1.training
    assert all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))
    assert strip_timing(r1.history) == strip_timing(r2.history)
    import subprocess
    assert getattr(subprocess.Popen, "_study_git_guard", False) is True
    # Matched by type family and message: another frozen test guard may fire first in a whole-tree run.
    with pytest.raises(PermissionError, match=r"^git disabled by project policy"):
        subprocess.run(["git", "status"], check=False)
    meta = json.loads((tmp_path / "a" / "runs" / "rnn" / "run_001" / "metadata.json").read_text())
    assert meta["git"]["available"] is False


# =========================================================================== T23 post-hoc spectral
def test_T23_posthoc_best_spectral(tmp_path, monkeypatch) -> None:
    """T:T23: the post-hoc point is written only after the three hashes agree; checkpoint and training
    records are byte-identical afterwards; a tampered checkpoint is refused."""
    cfg = replace(CFG, max_epochs=4)
    _scripted_eval(monkeypatch, [3.3, 3.3, 3.0, 2.9, 2.9, 2.5, 2.8, 2.7, 2.7, 2.6])
    run = new_run(tmp_path)
    plan = new_plan(run, cfg)
    res = rt.train_rnn(new_model(), TRAIN, DEV, training=cfg, seed=SEED, run=run, diagnostics=plan)
    assert res.best_epoch == 2
    protected = [run.checkpoint_path(k) for k in ("best", "final", "resume")] + [run.history_path,
                                                                              run.diagnostics_dir / "steps.jsonl"]
    before = {p: p.read_bytes() for p in protected}
    pid = dg.posthoc_best_spectral(run.run_dir, plan.probe, TRAIN)
    recs = dg.read_jsonl_complete(run.run_dir / "research_diagnostics" / "points.jsonl")
    assert recs[-1]["point_id"] == pid and recs[-1]["kind"] == "posthoc_best" and recs[-1]["completed_epochs"] == 2
    assert recs[-1]["valid"] and recs[-1]["spectral"]
    assert all(p.read_bytes() == b for p, b in before.items())
    with pytest.raises(ValueError, match="already"):
        dg.posthoc_best_spectral(run.run_dir, plan.probe, TRAIN)
    ck = torch.load(run.checkpoint_path("best"), weights_only=True)
    ck["model_state"]["hidden_bias"] = ck["model_state"]["hidden_bias"] + 1.0
    torch.save(ck, run.checkpoint_path("best"))
    lines = (run.run_dir / "research_diagnostics" / "points.jsonl").read_text().splitlines(keepends=True)[:-1]
    (run.run_dir / "research_diagnostics" / "points.jsonl").write_text("".join(lines))
    with pytest.raises(ValueError, match="disagree"):
        dg.posthoc_best_spectral(run.run_dir, plan.probe, TRAIN)


def test_T23_posthoc_not_needed_for_best_epoch_1_or_E(tmp_path, monkeypatch) -> None:
    _scripted_eval(monkeypatch, [3.3, 3.3, 3.0, 2.5, 2.9, 2.6, 2.8, 2.7])
    run = new_run(tmp_path)
    plan = new_plan(run)
    rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run, diagnostics=plan)
    assert dg.posthoc_best_spectral(run.run_dir, plan.probe, TRAIN) is None


# =========================================================================== A:T1 overfitting fixture
def test_A_T1_tiny_fixture_overfits() -> None:
    """A:T1 (debugging check, not a quality claim): ["ab", "ac", "ab"] approaches the
    prefix-conditional entropy bound 3·H(2/3, 1/3)/9 ≈ 0.2122 nats per target.
    Test-local settings [O]: H = 8, lr = 30, 400 epochs (B = 32 → C = 96, one step per epoch);
    margin 0.01 nats."""
    names = ["ab", "ac", "ab"]
    seqs = make_next_token_sequences(names, VOCAB)
    bound = 3 * -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)) / 9
    assert abs(bound - 0.2122) < 1e-4
    model = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=8), generator=make_generator(1))
    res = rt.train_rnn(model, seqs, seqs, training=rt.RNNTrainingConfig(lr=30.0, max_epochs=400), seed=1)
    final = res.history[-1]["train_nll"]
    assert bound - 1e-9 <= final <= bound + 0.01, final


# =========================================================================== X2 research configuration (C8)
def _experiment(**changes):
    cfg = load_config(default_config_paths("rnn"), model_config_cls=RNNConfig)
    dev = replace(cfg.device, device="cpu", dtype="float32")
    cfg = replace(cfg, device=dev)
    for path, value in changes.items():
        section, field_ = path.split(".")
        cfg = replace(cfg, **{section: replace(getattr(cfg, section), **{field_: value})})
    return cfg


def test_X2_validate_research_run() -> None:
    """X2: ``device: auto``, another device, dtype ≠ float32, a float64 model, shared diagnostics on,
    step-based resume, wrong batch sizes and wrong thread counts are each refused; the approved
    configuration passes."""
    threads = (torch.get_num_threads(), torch.get_num_interop_threads())
    research = rt.RNNTrainingConfig(lr=0.1, max_epochs=30)
    model = new_model()
    rt.validate_research_run(_experiment(), model, research, expected_threads=threads)
    cases = [(_experiment(**{"device.device": "auto"}), model, research, threads, "auto"),
             (_experiment(**{"device.device": "mps"}), model, research, threads, "cpu"),
             (_experiment(**{"device.dtype": "float64"}), model, research, threads, "float32"),
             (_experiment(), new_model(torch.float64), research, threads, "float32 on cpu"),
             (_experiment(**{"diagnostics.enabled": True}), model, research, threads, "disabled"),
             (_experiment(**{"checkpoint.resume_every_n_steps": 100}), model, research, threads, "resume_every"),
             (_experiment(), model, replace(research, batch_size=16), threads, "batch sizes"),
             (_experiment(), model, replace(research, eval_batch_size=512), threads, "batch sizes"),
             (_experiment(), model, research, (threads[0] + 1, threads[1]), "thread counts")]
    for exp, mdl, tr, th, msg in cases:
        with pytest.raises(rt.ResearchConfigError, match=msg):
            rt.validate_research_run(exp, mdl, tr, expected_threads=th)


def test_check_protocol_counts() -> None:
    """Training spec §1: data whose N, M, S, r or dev counts differ from the protocol are refused."""
    with pytest.raises(rt.ResearchConfigError, match="protocol"):
        rt.check_protocol_counts(TRAIN, DEV, rt.RNNTrainingConfig(lr=0.1, max_epochs=1))


# =========================================================================== X4 abort before the first commit (C7)
def test_X4_abort_before_first_commit(tmp_path, monkeypatch) -> None:
    """X4: a crash before the first commit leaves a directory that neither ``resume=True`` nor
    ``resume=False`` changes (both refuse); a fresh directory runs normally."""
    run = new_run(tmp_path)
    with monkeypatch.context() as m:
        _interrupt_at_epoch_end(m, 0)
        with pytest.raises(_Interrupt):
            rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run)
    assert not run.checkpoint_path("resume").exists()
    snapshot = {p: p.read_bytes() for p in run.run_dir.rglob("*") if p.is_file()}
    with pytest.raises(rt.ResumeError, match="no committed checkpoint"):
        rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run, resume=True)
    with pytest.raises(rt.ResumeError, match="already holds"):
        rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run)
    assert {p: p.read_bytes() for p in run.run_dir.rglob("*") if p.is_file()} == snapshot
    assert rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=new_run(tmp_path)).status == "completed"


def test_resume_refuses_partial_record(tmp_path, monkeypatch) -> None:
    """P7-4: ``train_rnn(resume=True)`` refuses a step-record file whose last line is incomplete and
    leaves the file unchanged (recovery is the orchestration preflight's job)."""
    run = new_run(tmp_path)
    with monkeypatch.context() as m:
        _interrupt_at_epoch_end(m, 1)
        with pytest.raises(_Interrupt):
            rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run)
    steps = run.diagnostics_dir / "steps.jsonl"
    with open(steps, "a") as fh:
        fh.write('{"epoch": 1, "step"')
    before = steps.read_bytes()
    with pytest.raises(rt.ResumeError, match="incomplete"):
        rt.train_rnn(new_model(), TRAIN, DEV, training=CFG, seed=SEED, run=run, resume=True)
    assert steps.read_bytes() == before
