"""MS5 isolation tests D:I1 – I8 (``rnn_diagnostics_spec.md`` §11.2; integration spec §4.8).

D:I1 = A:T5 = T:T24 (training identical with and without a full plan); D:I7 is the plan variant of
T:T15 in ``test_rnn_training.py`` (``test_T15_T16_resume_equals_uninterrupted[*-plan]``) and is not
duplicated here. Synthetic names only; Git guard from ``conftest.py``.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import pytest
import torch

import models.recurrent.rnn_training as rt
from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from utils.checkpoint import load_checkpoint
from utils.data import make_next_token_sequences
from utils.reproducibility import derive_seed, make_generator
from utils.run_manager import RunManager
from utils.vocabulary import Vocabulary

VOCAB = Vocabulary(chars=tuple(string.ascii_lowercase))
SEED = 11


def _names(count, seed):
    g = torch.Generator().manual_seed(seed)
    return ["".join(chr(97 + int(c)) for c in torch.randint(0, 26, (int(torch.randint(1, 9, (1,), generator=g)),),
                                                            generator=g)) for _ in range(count)]


NAMES, DEV_NAMES = _names(17, 21), _names(6, 22)
TRAIN, DEV = make_next_token_sequences(NAMES, VOCAB), make_next_token_sequences(DEV_NAMES, VOCAB)
CFG = rt.RNNTrainingConfig(lr=0.4, max_epochs=3, batch_size=4, eval_batch_size=4)  # S = 5


def model():
    return RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=4), generator=make_generator(derive_seed(SEED, "init")))


def plan_for(out: Path, probe_path: Path):
    info = {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1], "N": len(NAMES),
            "M": int(TRAIN.lengths.sum())}
    probe = dg.load_probe(probe_path) if probe_path.exists() else dg.build_probe(TRAIN, NAMES, info, probe_path)
    return dg.RNNDiagnosticsPlan(probe=probe, train=TRAIN, train_names=NAMES, dev=DEV, steps_per_epoch=5,
                                 max_epochs=3, recurrent_gain=1.0, out_dir=out)


def lines(path):
    return [{k: v for k, v in json.loads(x).items() if k not in dg.TIMING_FIELDS} for x in Path(path).read_text().splitlines()]


def _train(tmp: Path, *, with_plan: bool, name: str):
    run = RunManager.create("rnn", config={"f": 1}, results_root=tmp / name)
    m = model()
    plan = plan_for(run.run_dir / dg.RESEARCH_DIAGNOSTICS_DIR, tmp / "probe.json") if with_plan else None
    res = rt.train_rnn(m, TRAIN, DEV, training=CFG, seed=SEED, run=run, diagnostics=plan)
    return run, m, res


def test_I1_training_identical_with_and_without_plan(tmp_path) -> None:
    """D:I1 = A:T5 = T:T24 (bitwise): parameters, step records, history, checkpoints and the final
    global RNG state are identical with a full plan (P₀, P_early, P_epoch, E_sel) and without one."""
    rng = torch.random.get_rng_state()
    r0, m0, _ = _train(tmp_path, with_plan=False, name="none")
    rng0 = torch.random.get_rng_state()
    r1, m1, res1 = _train(tmp_path, with_plan=True, name="plan")
    assert torch.equal(rng, rng0) and torch.equal(rng0, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(m0.parameters(), m1.parameters()))
    assert (r0.diagnostics_dir / "steps.jsonl").read_bytes() == (r1.diagnostics_dir / "steps.jsonl").read_bytes()
    assert lines(r0.history_path) == lines(r1.history_path)
    for kind in ("best", "final", "resume"):
        a, b = load_checkpoint(r0.checkpoint_path(kind)), load_checkpoint(r1.checkpoint_path(kind))
        assert all(torch.equal(a["model_state"][k], b["model_state"][k]) for k in a["model_state"])
        assert a["extra"] == b["extra"] and a["epoch"] == b["epoch"] and a["step"] == b["step"]
    kinds = [r["kind"] for r in dg.read_jsonl_complete(r1.run_dir / "research_diagnostics" / "points.jsonl")]
    assert kinds == ["start", "early", "early", "early", "epoch_end", "epoch_end", "epoch_end"]
    assert res1.diagnostics_status == "complete"


def _state(m):
    hooks = {name: tuple(len(getattr(mod, name)) for mod in m.modules())
             for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_state_dict_hooks")}
    return {"params": [p.detach().clone() for p in m.parameters()],
            "grads": [None if p.grad is None else p.grad.clone() for p in m.parameters()],
            "requires_grad": [p.requires_grad for p in m.parameters()],
            "dtype_device": [(p.dtype, p.device) for p in m.parameters()], "training": m.training, "hooks": hooks,
            "grad_enabled": torch.is_grad_enabled(), "rng": torch.get_rng_state().clone(),
            "threads": (torch.get_num_threads(), torch.get_num_interop_threads()),
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def _equal_state(a, b):
    same = lambda x, y: (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))  # noqa: E731
    return (all(same(x, y) for x, y in zip(a["params"], b["params"])) and all(same(x, y) for x, y in zip(a["grads"], b["grads"]))
            and all(a[k] == b[k] for k in ("requires_grad", "dtype_device", "training", "hooks", "grad_enabled",
                                           "threads", "deterministic")) and torch.equal(a["rng"], b["rng"]))


def _populated_model():
    m = model()
    with torch.no_grad():
        m.output_weight.normal_(0, 0.5, generator=make_generator(2))
    from utils.metrics import token_nll
    token_nll(m(TRAIN.inputs), TRAIN.targets, reduction="sum").backward()  # .grad populated
    return m


def test_I2_per_call_state_unchanged(tmp_path) -> None:
    """D:I2: around every hook call — parameter and ``.grad`` bytes, ``requires_grad``, dtype/device,
    ``model.training``, hook dictionaries, grad mode, global RNG, thread counts and the deterministic
    flag are unchanged."""
    m = _populated_model()
    plan = plan_for(tmp_path / "d", tmp_path / "probe.json")
    for call in (lambda: plan.at_start(m), lambda: plan.after_step(m, sigma=1, epoch=0),
                 lambda: plan.at_epoch_end(m, sigma=5, completed_epochs=1)):
        before = _state(m)
        call()
        assert _equal_state(before, _state(m))
    assert plan.status == "complete"


def test_I3_copy_independence() -> None:
    """D:I3: the float64 copy is independent (changing it leaves the original), has no hooks and no
    ``.grad``, and is float64 on CPU."""
    m = _populated_model()
    m.register_forward_hook(lambda *a: None)
    m.register_forward_pre_hook(lambda *a: None)
    m.recurrent_weight.register_hook(lambda g: g)
    original = [p.detach().clone() for p in m.parameters()]
    c = dg.float64_copy(m)
    assert all(p.dtype == torch.float64 and p.device.type == "cpu" and p.grad is None for p in c.parameters())
    assert all(len(getattr(mod, h)) == 0 for mod in c.modules() for h in ("_forward_hooks", "_forward_pre_hooks"))
    assert not getattr(c.recurrent_weight, "_backward_hooks", None)
    with torch.no_grad():
        for p in c.parameters():
            p.add_(1.0)
    assert all(torch.equal(a, b.detach()) for a, b in zip(original, m.parameters()))
    assert len(m._forward_hooks) == 1  # the original keeps its own hooks


def test_I4_training_hooks_not_triggered(tmp_path) -> None:
    """D:I4: a counting forward hook on the training model is not triggered by a diagnostic call."""
    m = _populated_model()
    count = {"n": 0}
    m.register_forward_hook(lambda *a: count.__setitem__("n", count["n"] + 1))
    plan = plan_for(tmp_path / "d", tmp_path / "probe.json")
    plan.at_start(m)
    plan.after_step(m, sigma=1, epoch=0)
    assert count["n"] == 0


def test_I5_autograd_check_leaves_training_gradients(tmp_path) -> None:
    """D:I5: after the G3 autograd computation the training ``.grad`` is bitwise unchanged and no
    training parameter acquired a ``grad_fn``."""
    m = _populated_model()
    grads = [p.grad.clone() for p in m.parameters()]
    plan = plan_for(tmp_path / "d", tmp_path / "probe.json")
    plan.after_step(m, sigma=1, epoch=0)  # an early point runs G3
    assert all(torch.equal(g, p.grad) and p.grad_fn is None and p.is_leaf for g, p in zip(grads, m.parameters()))


def test_I6_failure_isolation(tmp_path, monkeypatch) -> None:
    """D:I6 (= D:D10 with training): with a mandatory diagnostic failing at every point, training is
    bitwise identical to a run without diagnostics; status ``incomplete`` in the result and metadata."""
    r0, m0, _ = _train(tmp_path, with_plan=False, name="none")

    def boom(*a, **k):
        raise RuntimeError("injected")

    monkeypatch.setattr(dg, "c1_restart", boom)
    with pytest.warns(UserWarning):
        r1, m1, res = _train(tmp_path, with_plan=True, name="failing")
    assert all(torch.equal(a, b) for a, b in zip(m0.parameters(), m1.parameters()))
    assert (r0.diagnostics_dir / "steps.jsonl").read_bytes() == (r1.diagnostics_dir / "steps.jsonl").read_bytes()
    assert res.diagnostics_status == "incomplete"
    assert json.loads(r1.metadata_path.read_text())["diagnostics_status"] == "incomplete"
    invalid = [r for r in dg.read_jsonl_complete(r1.run_dir / "research_diagnostics" / "points.jsonl") if not r["valid"]]
    assert {r["kind"] for r in invalid} == {"early", "epoch_end"}  # P₀ does not run C1


def test_I8_no_file_outside_the_diagnostics_directory(tmp_path) -> None:
    """D:I8: hook calls write only under the plan's ``out_dir``."""
    m = _populated_model()
    plan = plan_for(tmp_path / "run" / "research_diagnostics", tmp_path / "probe.json")
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    plan.at_start(m)
    plan.after_step(m, sigma=2, epoch=0)
    new = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    assert new and all((tmp_path / "run" / "research_diagnostics") in p.parents for p in new)
