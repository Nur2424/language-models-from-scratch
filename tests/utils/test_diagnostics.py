import gc
import weakref

import pytest
import torch
from torch import nn

from utils.diagnostics.activations import ActivationMonitor, sigmoid_saturation, tanh_saturation
from utils.diagnostics.gradients import (
    global_grad_norm,
    gradient_summaries,
    nonfinite_gradients,
    parameters_without_grad,
)
from utils.diagnostics.parameters import ParameterSnapshot, parameter_summaries
from utils.diagnostics.schedule import DiagnosticsSchedule


def _backward(model):
    model.zero_grad(set_to_none=True)
    model(torch.tensor([0, 1, 2, 3])).pow(2).mean().backward()


# ----------------------------------------------------------------- schedule
def test_schedule():
    s = DiagnosticsSchedule(enabled=True, every_n_steps=10, activations=False)
    assert s.should_collect(0, "gradients") and s.should_collect(20, "parameters")
    assert not s.should_collect(5, "gradients") and not s.should_collect(10, "activations")
    assert not DiagnosticsSchedule().should_collect(0, "gradients")  # disabled by default
    with pytest.raises(ValueError):
        DiagnosticsSchedule(every_n_steps=0)
    with pytest.raises(ValueError):
        s.should_collect(0, "attention")  # type: ignore[arg-type]


# ---------------------------------------------------------------- gradients
def test_global_grad_norm_matches_clip_grad_norm(tiny_net):
    _backward(tiny_net)
    ours = global_grad_norm(tiny_net)
    before = [p.grad.clone() for p in tiny_net.parameters() if p.grad is not None]
    ref = torch.nn.utils.clip_grad_norm_(tiny_net.parameters(), max_norm=1e9).item()
    assert ours == pytest.approx(ref, rel=1e-6)
    after = [p.grad for p in tiny_net.parameters() if p.grad is not None]
    assert all(torch.equal(a, b) for a, b in zip(before, after))   # we never modify grads
    assert global_grad_norm(tiny_net, norm_type=float("inf")) == pytest.approx(max(g.abs().max().item() for g in before))


def test_gradient_summaries_per_parameter_and_module(tiny_net):
    _backward(tiny_net)
    per = gradient_summaries(tiny_net)
    g = tiny_net.head.weight.grad
    assert per["head.weight"]["mean"] == pytest.approx(g.mean().item(), abs=1e-7)
    assert per["head.weight"]["std"] == pytest.approx(g.std(unbiased=False).item(), rel=1e-5)
    assert per["head.weight"]["min"] == pytest.approx(g.min().item())
    assert "unused.weight" not in per
    mod = gradient_summaries(tiny_net, group_by="module")
    body = torch.cat([p.grad.reshape(-1) for n, p in tiny_net.named_parameters() if n.startswith("body")])
    assert mod["body"]["numel"] == body.numel()
    assert mod["body"]["std"] == pytest.approx(body.std(unbiased=False).item(), rel=1e-5)
    assert mod["body"]["norm"] == pytest.approx(body.norm().item(), rel=1e-5)
    assert parameters_without_grad(tiny_net) == ["unused.weight", "unused.bias"]


def test_zero_fraction_detects_dead_relu_paths(tiny_net):
    with torch.no_grad():
        tiny_net.body[2].bias.fill_(-100.0)   # ReLU always off → upstream grads are exactly zero
    _backward(tiny_net)
    s = gradient_summaries(tiny_net)
    assert s["body.0.weight"]["zero_fraction"] == 1.0


def test_nonfinite_gradient_detection(tiny_net):
    _backward(tiny_net)
    tiny_net.head.bias.grad[0] = float("nan")
    tiny_net.head.weight.grad[0, 0] = float("inf")
    bad = nonfinite_gradients(tiny_net)
    assert bad == {"head.weight": {"nan": 0, "inf": 1}, "head.bias": {"nan": 1, "inf": 0}}
    assert gradient_summaries(tiny_net)["head.bias"]["nan_count"] == 1


# -------------------------------------------------------------- activations
def test_activation_monitor_summaries_and_cleanup(tiny_net):
    x = torch.tensor([0, 1, 2, 3])
    with ActivationMonitor(tiny_net, module_types=(nn.Tanh, nn.ReLU),
                           saturation_by_type={nn.Tanh: tanh_saturation(0.5)}) as mon:
        out_tanh = tiny_net.body[1](tiny_net.body[0](tiny_net.emb(x)))
        mon.reset()
        tiny_net(x)
        tiny_net(x)
        stats = mon.summaries()
    assert set(stats) == {"body.1", "body.3"}
    assert stats["body.1"]["calls"] == 2 and stats["body.1"]["numel"] == 2 * out_tanh.numel()
    assert stats["body.1"]["mean"] == pytest.approx(out_tanh.mean().item(), abs=1e-6)
    assert stats["body.1"]["saturation_fraction"] == pytest.approx((out_tanh.abs() > 0.5).float().mean().item())
    assert "saturation_fraction" not in stats["body.3"]
    assert 0 <= stats["body.3"]["zero_fraction"] <= 1
    assert not mon.attached
    assert all(len(m._forward_hooks) == 0 for m in tiny_net.modules())


def test_activation_monitor_removes_hooks_on_error(tiny_net):
    with pytest.raises(RuntimeError, match="boom"):
        with ActivationMonitor(tiny_net, module_names=["body.1"]):
            raise RuntimeError("boom")
    assert all(len(m._forward_hooks) == 0 for m in tiny_net.modules())


def test_activation_monitor_does_not_retain_tensors(tiny_net):
    x = torch.tensor([0, 1, 2, 3])
    mon = ActivationMonitor(tiny_net, module_names=["body.1"]).attach()
    out = tiny_net(x)
    ref = weakref.ref(out)
    out.sum().backward()
    del out
    gc.collect()
    assert ref() is None   # neither the output nor its graph is kept alive by the monitor
    mon.remove()


def test_activation_monitor_tuple_outputs_and_guards():
    lstm = nn.LSTM(3, 4, batch_first=True)
    model = nn.Sequential()
    model.add_module("rnn", lstm)
    with ActivationMonitor(model, module_names=["rnn"]) as mon:
        out, _ = lstm(torch.randn(2, 5, 3))
    assert mon.summaries()["rnn"]["numel"] == out.numel()
    with pytest.raises(KeyError):
        ActivationMonitor(model, module_names=["nope"])
    mon = ActivationMonitor(model, module_names=["rnn"]).attach()
    with pytest.raises(RuntimeError, match="already attached"):
        mon.attach()
    mon.remove()
    mon.remove()  # idempotent


def test_saturation_predicates():
    y = torch.tensor([0.01, 0.5, 0.99])
    assert sigmoid_saturation(0.03)(y).tolist() == [True, False, True]
    assert tanh_saturation(0.97)(torch.tensor([-0.98, 0.0])).tolist() == [True, False]
    with pytest.raises(ValueError):
        tanh_saturation(1.5)


# --------------------------------------------------------------- parameters
def test_parameter_summaries(tiny_net):
    s = parameter_summaries(tiny_net)
    w = tiny_net.head.weight
    assert s["head.weight"]["norm"] == pytest.approx(w.norm().item(), rel=1e-6)
    assert "unused.weight" in s


def test_measured_update_matches_actual_delta_for_adamw(tiny_net):
    lr = 1e-2
    opt = torch.optim.AdamW(tiny_net.parameters(), lr=lr, weight_decay=0.1)
    _backward(tiny_net)
    before = {n: p.detach().clone() for n, p in tiny_net.named_parameters()}
    grad_proxy = lr * tiny_net.head.weight.grad.std().item()
    snap = ParameterSnapshot.capture(tiny_net)
    opt.step()
    stats = snap.update_stats(tiny_net)
    delta = tiny_net.head.weight.detach() - before["head.weight"]
    per = stats["per_parameter"]["head.weight"]
    assert per["update_norm"] == pytest.approx(delta.norm().item(), rel=1e-5)
    assert per["relative_update"] == pytest.approx(delta.norm().item() / before["head.weight"].norm().item(), rel=1e-5)
    # Adam's first step is ≈ lr·sign(g) per element: very different from the lr·std(g) proxy.
    assert per["update_std"] != pytest.approx(grad_proxy, rel=0.5)
    assert per["update_max_abs"] == pytest.approx(delta.abs().max().item(), rel=1e-5)
    # unused params got no grad → AdamW skips them entirely
    assert stats["per_parameter"]["unused.weight"]["update_norm"] == 0.0
    assert stats["per_parameter"]["unused.weight"]["log10_update_to_param_std"] is None
    total = sum(((tiny_net.get_parameter(n).detach() - b) ** 2).sum().item() for n, b in before.items()) ** 0.5
    assert stats["global"]["update_norm"] == pytest.approx(total, rel=1e-5)


def test_snapshot_validation(tiny_net):
    with pytest.raises(KeyError):
        ParameterSnapshot.capture(tiny_net, names=["nope"])
    snap = ParameterSnapshot.capture(tiny_net, names=["head.weight"])
    assert list(snap.before) == ["head.weight"]
    tiny_net.head = nn.Linear(4, 6)
    with pytest.raises(ValueError, match="changed shape"):
        snap.update_stats(tiny_net)
