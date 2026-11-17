"""MS4 tests: diagnostics plan, records, truncation, failure handling and O3.

Milestone MS4 of ``rnn_implementation_plan.md`` v0.2 (D-49). Labels: D:D3 / D:D11 (plan
resolution), D:D9 (stored values reload bit-identically), D:D10 (injected faults and failed
checks give ``valid: false``), C6 failure path, P7-6 optional refusal, P7-4 strict readers.
Hook-order, resume, isolation (D:I1 – I8) and post-hoc tests need ``train_rnn`` and are in MS5 / MS6.
Synthetic data only.
"""

from __future__ import annotations

import json
import string

import numpy as np
import pytest
import torch

from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from utils.data import make_next_token_sequences
from utils.reproducibility import make_generator
from utils.vocabulary import Vocabulary

VOCAB = Vocabulary(chars=tuple(string.ascii_lowercase))


def _names(count, seed):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, 16, (count,), generator=g).tolist()
    return ["".join(chr(97 + int(c)) for c in torch.randint(0, 26, (L,), generator=g)) for L in lengths]


@pytest.fixture
def setup(tmp_path):
    names = _names(24, 3)
    train = make_next_token_sequences(names, VOCAB)
    dev = make_next_token_sequences(_names(8, 4), VOCAB)
    info = {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1],
            "N": len(names), "M": int(train.lengths.sum())}
    probe = dg.build_probe(train, names, info, tmp_path / "probe.json")
    model = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=5), generator=make_generator(3))
    with torch.no_grad():  # non-zero output layer, so G2 and G3 are defined
        model.output_weight.copy_(0.3 * torch.randn(27, 5, generator=torch.Generator().manual_seed(1)))
    return names, train, dev, info, probe, model


def _plan(setup, out, S=5, E=3, **kw):
    names, train, dev, info, probe, _ = setup
    return dg.RNNDiagnosticsPlan(probe=probe, train=train, train_names=names, dev=dev, steps_per_epoch=S,
                                 max_epochs=E, recurrent_gain=1.0, out_dir=out, **kw)


# --------------------------------------------------------------------------- D3 / D11 plan resolution
@pytest.mark.parametrize(("S", "E", "early", "spectral_epochs"), [
    (802, 30, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], [1, 30]),
    (8, 2, [1, 2, 4], [1, 2]),          # S a power of two: σ = 8 is the epoch end, not an early point
    (1, 1, [], [1]),                    # E = 1: best = final = epoch 1
    (5, 4, [1, 2, 4], [1, 4]),
])
def test_D3_point_resolution(S, E, early, spectral_epochs) -> None:
    """D:D3 (= D:D11 plan part): P₀; σ = 2^j < S; every epoch end k·S; spectral at {0, 1, E}; ids by σ."""
    pts = dg.resolve_points(S, E)
    assert pts[0] == dg.EvaluationPoint(0, "start", 0, 0, True)
    assert [p.sigma for p in pts if p.kind == "early"] == early
    ends = [p for p in pts if p.kind == "epoch_end"]
    assert [p.sigma for p in ends] == [k * S for k in range(1, E + 1)]
    assert [p.completed_epochs for p in ends if p.spectral] == spectral_epochs
    assert [p.point_id for p in pts] == list(range(len(pts)))
    assert [p.sigma for p in pts] == sorted(p.sigma for p in pts)


def test_optional_diagnostics_are_refused() -> None:
    """P7-6: optional diagnostics stay disabled; enabling either is refused."""
    assert dg.OptionalDiagnostics() == dg.OptionalDiagnostics(False, False)
    for kw in ({"h1_on_dev": True}, {"spectral_at_powers_of_two": True}):
        with pytest.raises(ValueError, match="not authorised"):
            dg.OptionalDiagnostics(**kw)
    assert not hasattr(dg, "positional_data_bound")  # deferred, not created (D-49)


def test_plan_validates_probe_and_plan_file(setup, tmp_path) -> None:
    """C5 at construction; ``plan.json`` written once and must match on reconstruction."""
    names, train, dev, info, probe, model = setup
    _plan(setup, tmp_path / "d")
    _plan(setup, tmp_path / "d")  # identical: accepted (resume)
    with pytest.raises(ValueError, match="differs"):
        _plan(setup, tmp_path / "d", E=4)
    with pytest.raises(dg.ProbeMismatchError):
        dg.RNNDiagnosticsPlan(probe=probe, train=train, train_names=list(reversed(names)), dev=dev,
                              steps_per_epoch=5, max_epochs=3, recurrent_gain=1.0, out_dir=tmp_path / "e")


# --------------------------------------------------------------------------- hooks, records, D9
def test_hooks_write_the_right_points_and_records_reload(setup, tmp_path) -> None:
    """Hooks act exactly at the resolved points (σ = 2^j < S after a step; epoch ends); records carry
    the before/after hashes; raw files reload bit-identically (D:D9); status complete."""
    model = setup[-1]
    plan = _plan(setup, tmp_path / "d")
    plan.at_start(model)
    for sigma in range(1, 6):
        plan.after_step(model, sigma=sigma, epoch=0)
    plan.at_epoch_end(model, sigma=5, completed_epochs=1)
    recs = dg.read_jsonl_complete(tmp_path / "d" / "points.jsonl")
    assert [(r["kind"], r["sigma"]) for r in recs] == [("start", 0), ("early", 1), ("early", 2), ("early", 4),
                                                       ("epoch_end", 5)]
    assert all(r["valid"] and r["param_sha256_before"] == r["param_sha256_after"] == dg.param_sha256(model)
               and r["rng_sha256_before"] == r["rng_sha256_after"] for r in recs)
    assert recs[0]["diagnostics"] == ["H1", "G1"] and recs[1]["diagnostics"] == ["H1", "G1", "G2", "G3", "C1"]
    prim = dg.read_jsonl_complete(tmp_path / "d" / "primary.jsonl")
    assert [p["point_id"] for p in prim] == [r["point_id"] for r in recs]
    assert prim[0]["results"]["G1"]["spectral"] is True and prim[1]["results"]["G1"]["spectral"] is False
    for r in recs:
        path = tmp_path / "d" / "raw" / f"point_{r['point_id']:04d}.npz"
        with np.load(path, allow_pickle=False) as a, np.load(path, allow_pickle=False) as b:
            assert sorted(a.files) == sorted(b.files) and all(a[k].tobytes() == b[k].tobytes() for k in a.files)
    with np.load(tmp_path / "d" / "raw" / "point_0001.npz") as z:
        assert {"g1_log_phi", "g2_log_R", "g3_inner_full", "c1_delta_nll"} <= set(z.files)
    assert plan.status == "complete"
    with pytest.raises(RuntimeError, match="already recorded"):
        plan.at_start(model)
    with pytest.raises(ValueError):
        plan.at_epoch_end(model, sigma=6, completed_epochs=1)


def test_diagnostics_do_not_change_the_model(setup, tmp_path) -> None:
    """The plan reads the model only: parameters, grads, mode and the global RNG unchanged."""
    model = setup[-1]
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rng = torch.random.get_rng_state()
    plan = _plan(setup, tmp_path / "d")
    plan.at_start(model)
    plan.after_step(model, sigma=1, epoch=0)
    assert all(torch.equal(p.detach(), before[n]) and p.grad is None for n, p in model.named_parameters())
    assert model.training and torch.equal(torch.random.get_rng_state(), rng)


# --------------------------------------------------------------------------- D10 / C6 failure path
def test_D10_injected_exception_gives_invalid_point(setup, tmp_path, monkeypatch) -> None:
    """D:D10 / C6: an exception inside a mandatory diagnostic gives ``valid: false`` with the type,
    message and traceback; the remaining diagnostics still run; status becomes ``incomplete``."""
    model = setup[-1]
    plan = _plan(setup, tmp_path / "d")

    def boom(*a, **k):
        raise ArithmeticError("injected G2 failure")

    monkeypatch.setattr(dg, "g2_attenuation", boom)
    with pytest.warns(UserWarning, match="invalid"):
        plan.after_step(model, sigma=1, epoch=0)
    rec = dg.read_jsonl_complete(tmp_path / "d" / "points.jsonl")[0]
    assert rec["valid"] is False and plan.status == "incomplete" and plan.invalid_points == (1,)
    err = rec["errors"][0]
    assert err["diagnostic"] == "G2" and err["type"] == "ArithmeticError" and "injected" in err["traceback"]
    assert set(rec["checks"]) == {"H1", "G1", "G3", "C1"}


def test_D10_failed_runtime_check_gives_invalid_point(setup, tmp_path, monkeypatch) -> None:
    model = setup[-1]
    plan = _plan(setup, tmp_path / "d")
    real = dg.g3_decomposition

    def broken(*a, **k):
        res = real(*a, **k)
        res.checks["identity_full"] = {"rel_error": 1.0, "pass": False}
        return res

    monkeypatch.setattr(dg, "g3_decomposition", broken)
    with pytest.warns(UserWarning):
        plan.after_step(model, sigma=1, epoch=0)
    rec = dg.read_jsonl_complete(tmp_path / "d" / "points.jsonl")[0]
    assert rec["valid"] is False and rec["errors"][0]["type"] == "FailedCheck"


def test_isolation_breach_is_recorded(setup, tmp_path, monkeypatch) -> None:
    """A diagnostic that consumes the global RNG is caught by the before/after state comparison."""
    model = setup[-1]
    plan = _plan(setup, tmp_path / "d")
    real = dg.h1_statistics

    def rng_user(*a, **k):
        torch.rand(1)
        return real(*a, **k)

    monkeypatch.setattr(dg, "h1_statistics", rng_user)
    with pytest.warns(UserWarning):
        plan.at_start(model)
    rec = dg.read_jsonl_complete(tmp_path / "d" / "points.jsonl")[0]
    assert rec["valid"] is False and any(e["type"] == "IsolationBreach" for e in rec["errors"])


def test_failure_to_write_the_record_propagates(setup, tmp_path, monkeypatch) -> None:
    """C6: if the record itself cannot be written, the exception propagates (the run fails)."""
    model = setup[-1]
    plan = _plan(setup, tmp_path / "d")

    def no_disk(*a, **k):
        raise OSError("injected: disk full")

    monkeypatch.setattr(dg, "_append_line_fsync", no_disk)
    with pytest.raises(OSError, match="disk full"):
        plan.at_start(model)


# --------------------------------------------------------------------------- truncate / strict readers
def test_truncate_keeps_committed_points_only(setup, tmp_path) -> None:
    """Resume truncation: points with σ ≤ σ_K kept, an uncommitted epoch end removed with its raw file."""
    model = setup[-1]
    plan = _plan(setup, tmp_path / "d", S=4, E=2)
    plan.at_start(model)
    for s in (1, 2):
        plan.after_step(model, sigma=s, epoch=0)
    plan.at_epoch_end(model, sigma=4, completed_epochs=1)
    plan.at_epoch_end(model, sigma=8, completed_epochs=2)  # pretend the crash came before its commit
    plan.truncate(sigma=4, completed_epochs=1)
    recs = dg.read_jsonl_complete(tmp_path / "d" / "points.jsonl")
    assert [r["point_id"] for r in recs] == [0, 1, 2, 3]
    assert not (tmp_path / "d" / "raw" / "point_0004.npz").exists()
    assert [p["point_id"] for p in dg.read_jsonl_complete(tmp_path / "d" / "primary.jsonl")] == [0, 1, 2, 3]
    plan.at_epoch_end(model, sigma=8, completed_epochs=2)  # rewritten once, not duplicated
    assert [r["point_id"] for r in dg.read_jsonl_complete(tmp_path / "d" / "points.jsonl")] == [0, 1, 2, 3, 4]


def test_partial_last_line_is_refused_not_repaired(setup, tmp_path) -> None:
    """P7-4: an incomplete last line raises; nothing is discarded or rewritten by the plan."""
    model = setup[-1]
    plan = _plan(setup, tmp_path / "d")
    plan.at_start(model)
    path = tmp_path / "d" / "points.jsonl"
    with open(path, "a") as fh:
        fh.write('{"point_id": 1, "kind": "ea')
    before = path.read_bytes()
    with pytest.raises(dg.DiagnosticsRecordError, match="incomplete"):
        plan.truncate(sigma=0, completed_epochs=0)
    assert path.read_bytes() == before


# --------------------------------------------------------------------------- O3
def test_o3_record_is_descriptive() -> None:
    history = [{"completed_epochs": 0, "dev_nll": 3.3, "running_train_nll": None},
               {"completed_epochs": 1, "dev_nll": 2.5, "running_train_nll": 2.9},
               {"completed_epochs": 2, "dev_nll": 2.6, "running_train_nll": 3.0}]
    steps = [{"epoch": 0, "grad_norm": 1.0}, {"epoch": 0, "grad_norm": 3.0}, {"epoch": 0, "grad_norm": 2.0},
             {"epoch": 1, "grad_norm": 4.0}]
    rec = dg.o3_record(history, steps, status="completed", diagnostics_status="complete")
    assert rec["epoch_events"] == [{"epoch": 2, "event": "dev_nll_rose"}, {"epoch": 2, "event": "running_train_nll_rose"}]
    assert rec["gradient_norm_profile"][0] == {"epoch": 1, "max": 3.0, "median": 2.0, "max_over_median": 1.5}
    assert rec["research_use"] == "eligible" and rec["failed_run"] is False
    assert dg.o3_record(history, steps, status="failed_diverged", diagnostics_status="complete")["failed_run"]
    inc = dg.o3_record(history, steps, status="completed", diagnostics_status="incomplete", invalid_points=[4, 1])
    assert "human review" in inc["research_use"] and inc["invalid_points"] == [1, 4]
    assert rec["invalid_points"] == []
    with pytest.raises(ValueError, match="list of invalid points"):  # C6: incomplete always names its points
        dg.o3_record(history, steps, status="completed", diagnostics_status="incomplete")
    assert "spike" not in json.dumps(rec)
