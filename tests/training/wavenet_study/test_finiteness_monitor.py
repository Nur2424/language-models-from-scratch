"""Stage 2: the read-only finiteness monitor (protocol §9.2; I-B; clarification B).

Synthetic data and tiny models only. Faults are injected with test-side hooks: a forward hook
registered *before* the monitor modifies a module's output before the monitor sees it (hooks
run in registration order and a returned tensor replaces the output); a post-step hook
registered before the monitor corrupts a parameter right after ``optimizer.step()``.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch
from torch.optim.optimizer import _global_optimizer_post_hooks

from training.wavenet_study import study_common as sc
from utils.data import make_context_windows
from utils.metrics import token_nll
from utils.reproducibility import derive_seed, make_generator
from utils.vocabulary import Vocabulary

TRAIN_NAMES = ["abc", "bca", "cab", "abd", "bad", "dab", "ace", "cea", "eca", "bed", "deb", "fab",
               "abf", "caf", "fec", "edf", "fed", "def", "bef", "cbd", "adc", "ebf", "fda", "dcf"]
DEV_NAMES = ["abe", "fac", "dbc", "ecd"]
EXPECTED_LEAVES = {
    "A": ["emb", "hidden", "act", "out"],
    "B": ["emb", "hidden", "bn", "act", "out"],
    "C": ["emb", "hidden1", "act1", "hidden2", "act2", "hidden3", "act3", "out"],
    "D": ["emb", "hidden1", "bn1", "act1", "hidden2", "bn2", "act2", "hidden3", "bn3", "act3", "out"],
}


def _arm(arm):
    from models.feedforward import batchnorm_mlp, mlp
    from models.hierarchical import wavenet

    if arm == "A":
        return mlp, mlp.MLPLanguageModel(VOCAB, mlp.MLPConfig(context_length=8, embedding_dim=3, hidden_dim=5),
                                         generator=make_generator(derive_seed(7, "init")))
    if arm == "B":
        cfg = batchnorm_mlp.BatchNormMLPConfig(context_length=8, embedding_dim=3, hidden_dim=5)
        return batchnorm_mlp, batchnorm_mlp.BatchNormMLPLanguageModel(VOCAB, cfg,
                                                                      generator=make_generator(derive_seed(7, "init")))
    cfg = wavenet.WaveNetConfig(context_length=8, embedding_dim=3, hidden_dim=4,
                                normalization="none" if arm == "C" else "batchnorm")
    return wavenet, wavenet.WaveNetLanguageModel(VOCAB, cfg, generator=make_generator(derive_seed(7, "init")))


VOCAB = Vocabulary.from_names(TRAIN_NAMES)
TRAIN = make_context_windows(TRAIN_NAMES, VOCAB, context_length=8)
DEV = make_context_windows(DEV_NAMES, VOCAB, context_length=8)


def _trained(arm, steps=6):
    """A model after a few plain SGD steps in train mode (so grads, EMA and counters are non-trivial)."""
    module, model = _arm(arm)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
    model.train()
    for i in range(steps):
        idx = torch.arange(i * 5, i * 5 + 5) % TRAIN.num_examples
        opt.zero_grad(set_to_none=True)
        token_nll(model(TRAIN.contexts[idx]), TRAIN.targets[idx], reduction="mean").backward()
        opt.step()
    if model_uses_bn(model):
        module.calibrate_batchnorm(model, TRAIN, batch_size=16)
    return module, model, opt


def model_uses_bn(model):
    return any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def _snapshot(model, opt=None):
    return {
        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "grads": {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()},
        "modes": {n: m.training for n, m in model.named_modules()},
        "requires_grad": {n: p.requires_grad for n, p in model.named_parameters()},
        "opt": copy.deepcopy(opt.state_dict()) if opt is not None else None,
        "torch_rng": torch.get_rng_state().clone(),
        "py_rng": random.getstate(),
        "np_rng": np.random.get_state()[1].copy(),
        "grad_enabled": torch.is_grad_enabled(),
    }


def _same_snapshot(a, b):
    assert a["state"].keys() == b["state"].keys()
    for k in a["state"]:
        assert a["state"][k].dtype == b["state"][k].dtype and torch.equal(a["state"][k], b["state"][k]), k
    for k in a["grads"]:
        assert (a["grads"][k] is None) == (b["grads"][k] is None), k
        if a["grads"][k] is not None:
            assert torch.equal(a["grads"][k], b["grads"][k]), k
    assert a["modes"] == b["modes"] and a["requires_grad"] == b["requires_grad"]
    assert torch.equal(a["torch_rng"], b["torch_rng"]) and a["py_rng"] == b["py_rng"]
    assert np.array_equal(a["np_rng"], b["np_rng"]) and a["grad_enabled"] == b["grad_enabled"]
    if a["opt"] is not None:
        assert repr(a["opt"]) == repr(b["opt"])


def _no_hooks_left(model, global_before):
    assert all(not m._forward_hooks for m in model.modules())
    assert len(_global_optimizer_post_hooks) == global_before


# ----------------------------------------------------------------- monitored points
@pytest.mark.parametrize("arm", list("ABCD"))
def test_default_points_are_exactly_the_protocol_leaf_modules(arm):
    _, model = _arm(arm)
    assert [n for n, _ in sc.leaf_modules(model)] == EXPECTED_LEAVES[arm]
    assert [n for n, _ in sc.FinitenessMonitor(model).points] == EXPECTED_LEAVES[arm]


# ----------------------------------------------------------------- read-only: evaluation
@pytest.mark.parametrize("arm", list("ABCD"))
def test_evaluation_with_monitor_is_bit_identical_and_preserves_all_state(arm):
    module, model, opt = _trained(arm)
    plain = copy.deepcopy(model)
    before = _snapshot(model, opt)
    global_before = len(_global_optimizer_post_hooks)
    monitor = sc.FinitenessMonitor(model)
    with monitor:
        nll = module.evaluate_nll(model, DEV, batch_size=16)
        sat = module.per_unit_saturation(model, DEV, batch_size=16)
        model.train()  # evaluation helpers must also restore train mode
        nll_train_mode = module.evaluate_nll(model, TRAIN, batch_size=7)
    _no_hooks_left(model, global_before)
    _same_snapshot(before, _snapshot(model, opt))
    ref_nll = module.evaluate_nll(plain, DEV, batch_size=16)
    plain.train()
    ref_train = module.evaluate_nll(plain, TRAIN, batch_size=7)
    assert sc.float_bits(nll.mean_nll) == sc.float_bits(ref_nll.mean_nll)
    assert sc.float_bits(nll_train_mode.mean_nll) == sc.float_bits(ref_train.mean_nll)
    ref_sat = module.per_unit_saturation(plain, DEV, batch_size=16)  # tensor (MLPs) or per-level dict (WaveNet)
    sat, ref_sat = (sat, ref_sat) if isinstance(sat, dict) else ({"all": sat}, {"all": ref_sat})
    assert sat.keys() == ref_sat.keys() and all(torch.equal(sat[k], ref_sat[k]) for k in sat)
    cost = monitor.open_bucket()
    assert cost["module_checks_evaluation"] > 0 and cost["module_checks_training"] == 0
    assert cost["parameter_checks"] == 0


@pytest.mark.parametrize("arm", ["B", "D"])
def test_recalibration_with_monitor_installs_identical_statistics(arm):
    module, model, _ = _trained(arm)
    plain = copy.deepcopy(model)
    with sc.FinitenessMonitor(model):
        module.calibrate_batchnorm(model, TRAIN, batch_size=16)
    module.calibrate_batchnorm(plain, TRAIN, batch_size=16)
    assert sc.model_state_hash(model) == sc.model_state_hash(plain)


# ----------------------------------------------------------------- read-only: training steps
@pytest.mark.parametrize("arm", list("ABCD"))
def test_training_steps_with_monitor_are_bit_identical(arm):
    _, model, opt = _trained(arm, steps=2)
    twin = copy.deepcopy(model)
    twin_opt = torch.optim.SGD(twin.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
    twin_opt.load_state_dict(opt.state_dict())
    monitor = sc.FinitenessMonitor(model).attach()
    try:
        for m, o in ((model, opt), (twin, twin_opt)):
            m.train()
            for i in range(3):
                idx = torch.arange(i * 7, i * 7 + 7) % TRAIN.num_examples
                o.zero_grad(set_to_none=True)
                token_nll(m(TRAIN.contexts[idx]), TRAIN.targets[idx], reduction="mean").backward()
                o.step()
    finally:
        monitor.detach()
    _same_snapshot(_snapshot(model, opt), _snapshot(twin, twin_opt))
    cost = monitor.open_bucket()
    assert monitor.steps_completed == 3 and cost["parameter_checks"] == 3
    assert cost["module_checks_training"] == 3 * len(EXPECTED_LEAVES[arm])


def test_isfinite_checks_add_no_autograd_nodes():
    _, model = _arm("A")
    seen = []
    with sc.FinitenessMonitor(model):
        model.train()
        out = model(TRAIN.contexts[:4])
        seen.append(out.grad_fn.name())
    plain = model(TRAIN.contexts[:4])
    assert seen[0] == plain.grad_fn.name()


def test_steps_of_another_optimizer_are_ignored():
    _, model = _arm("A")
    other = torch.nn.Linear(2, 2)
    other_opt = torch.optim.SGD(other.parameters(), lr=0.1)
    partial_opt = torch.optim.SGD([next(model.parameters())], lr=0.1)
    with sc.FinitenessMonitor(model) as monitor:
        for opt in (other_opt, partial_opt):
            opt.step()
    assert monitor.steps_completed == 0 and monitor.open_bucket()["parameter_checks"] == 0


# ----------------------------------------------------------------- lifecycle
def test_attach_twice_refused_and_detach_removes_every_hook():
    _, model = _arm("D")
    before = len(_global_optimizer_post_hooks)
    monitor = sc.FinitenessMonitor(model).attach()
    assert len(_global_optimizer_post_hooks) == before + 1
    assert sum(len(m._forward_hooks) for m in model.modules()) == len(EXPECTED_LEAVES["D"])
    with pytest.raises(RuntimeError):
        monitor.attach()
    monitor.detach()
    monitor.detach()  # idempotent
    _no_hooks_left(model, before)
    model.eval()
    with torch.no_grad():
        model(DEV.contexts[:3])
    assert monitor.open_bucket()["module_checks_evaluation"] == 0


def test_end_epoch_cross_checks_the_trainer_step_count():
    _, model = _arm("A")
    monitor = sc.FinitenessMonitor(model)
    with pytest.raises(RuntimeError, match="step count mismatch"):
        monitor.end_epoch(steps_reported=1)
    assert monitor.end_epoch(steps_reported=0)["epoch"] == 1 and monitor.epochs_completed == 1


# ----------------------------------------------------------------- detection and attribution
def _inject_output(module, *, when, value=float("inf")):
    """Forward hook (register BEFORE the monitor): replace one element when ``when()`` is true."""
    def hook(_m, _inputs, output):
        if when():
            out = output.clone()
            out.view(-1)[0] = value
            return out
        return None
    return module.register_forward_hook(hook)


def test_infinite_preactivation_is_caught_before_tanh_masks_it():
    _, model = _arm("A")
    handle = _inject_output(model.hidden, when=lambda: torch.is_grad_enabled())
    seen = {}
    def observe(_m, _i, o):
        seen["act_finite"] = bool(torch.isfinite(o).all())

    act_hook = model.act.register_forward_hook(observe)
    model.train()
    # Without the monitor the infinity is masked: tanh(inf) = 1 and the loss is finite.
    loss = token_nll(model(TRAIN.contexts[:5]), TRAIN.targets[:5], reduction="mean")
    assert seen["act_finite"] and bool(torch.isfinite(loss))
    act_hook.remove()
    monitor = sc.FinitenessMonitor(model).attach()
    try:
        with pytest.raises(sc.NonFiniteDetected) as info:
            model(TRAIN.contexts[:5])
    finally:
        monitor.detach()
        handle.remove()
    f = info.value.failure
    assert isinstance(info.value, FloatingPointError)
    assert (f.kind, f.name, f.forward, f.epoch_1based, f.global_step, f.nonfinite_elements) == \
        ("module_output", "hidden", "training", 1, 0, 1)
    assert monitor.failure == f


def test_nonfinite_preactivation_in_a_hierarchical_level_is_attributed_to_that_level():
    _, model = _arm("D")
    handle = _inject_output(model.hidden2, when=lambda: True, value=float("nan"))
    monitor = sc.FinitenessMonitor(model).attach()
    try:
        model.train()
        with pytest.raises(sc.NonFiniteDetected) as info:
            model(TRAIN.contexts[:5])
    finally:
        monitor.detach()
        handle.remove()
    assert info.value.failure.name == "hidden2"  # caught before bn2 / act2 transform it


def test_evaluation_forward_failure_is_labelled_with_epoch_and_last_step():
    _, model = _arm("C")
    handle = _inject_output(model.out, when=lambda: not torch.is_grad_enabled())
    monitor = sc.FinitenessMonitor(model).attach()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        model.train()
        for _ in range(4):
            opt.zero_grad()
            token_nll(model(TRAIN.contexts[:5]), TRAIN.targets[:5], reduction="mean").backward()
            opt.step()
        monitor.end_epoch(steps_reported=4)
        model.eval()
        with torch.no_grad(), pytest.raises(sc.NonFiniteDetected) as info:
            model(DEV.contexts[:3])
    finally:
        monitor.detach()
        handle.remove()
    f = info.value.failure
    assert (f.kind, f.name, f.forward, f.module_training, f.epoch_1based, f.global_step, f.steps_completed) == \
        ("module_output", "out", "evaluation", False, 2, 3, 4)


def test_nonfinite_parameter_after_step_is_detected_immediately_with_its_name():
    from torch.optim.optimizer import register_optimizer_step_post_hook

    _, model = _arm("B")
    calls = {"n": 0}

    def corrupt(_opt, _a, _k):
        calls["n"] += 1
        if calls["n"] == 3:
            with torch.no_grad():
                model.bn.weight[1] = float("-inf")

    injector = register_optimizer_step_post_hook(corrupt)  # registered first: runs before the check
    monitor = sc.FinitenessMonitor(model).attach()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        model.train()
        with pytest.raises(sc.NonFiniteDetected) as info:
            for _ in range(5):
                opt.zero_grad()
                token_nll(model(TRAIN.contexts[:5]), TRAIN.targets[:5], reduction="mean").backward()
                opt.step()
    finally:
        monitor.detach()
        injector.remove()
    f = info.value.failure
    assert (f.kind, f.name, f.global_step, f.steps_completed, f.forward, f.nonfinite_elements) == \
        ("parameter_after_step", "bn.weight", 2, 2, None, 1)
    assert calls["n"] == 3  # stopped at once: no further step was taken
