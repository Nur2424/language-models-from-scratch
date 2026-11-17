"""Stage 9 regression tests for the independent review (plan §7.3) of Model 6.

Each test names the review finding it closes (MJ-n major, mn-n minor, N-n note; review of 2026-11-15).
Independent baselines: the accepted MS1 reference (``rnn_reference.py``), explicit matrices, hand
computation. Synthetic data only (plan §7.4). Tolerance A = 1e‑9 (float64).
"""

from __future__ import annotations

import dataclasses
import json
import math
import string
from collections import defaultdict

import numpy as np
import pytest
import torch

import models.recurrent.rnn_training as rt
import tests.models.recurrent.rnn_reference as ref
from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from utils.data import make_next_token_sequences
from utils.reproducibility import make_generator
from utils.run_manager import RunManager
from utils.vocabulary import Vocabulary

VOCAB = Vocabulary(chars=tuple(string.ascii_lowercase))
TOL = 1e-9


def close(a: float, e: float, what: str) -> None:
    assert abs(a - e) <= TOL + TOL * abs(e), f"{what}: {a} vs {e}"


def names_fixture(count: int, seed: int) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(1, 16, (count,), generator=g).tolist()
    return ["".join(chr(97 + int(c)) for c in torch.randint(0, 26, (L,), generator=g)) for L in lens]


def info_for(seqs) -> dict:
    return {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1],
            "N": int(seqs.lengths.numel()), "M": int(seqs.lengths.sum())}


def perturbed(H: int, g: float, seed: int) -> RNNLanguageModel:
    gen = make_generator(seed)
    Q, R = torch.linalg.qr(torch.randn(H, H, generator=gen, dtype=torch.float64))
    W = g * (Q * torch.sign(torch.diagonal(R)).unsqueeze(0))
    vals = (math.sqrt(.5) * torch.randn(27, H, generator=gen, dtype=torch.float64), W,
            .1 * torch.randn(H, generator=gen, dtype=torch.float64),
            torch.randn(27, H, generator=gen, dtype=torch.float64) / math.sqrt(H),
            .1 * torch.randn(27, generator=gen, dtype=torch.float64))
    m = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=H), generator=make_generator(1)).to(torch.float64)
    with torch.no_grad():
        for n, v in zip(dg.PARAM_ORDER, vals):
            getattr(m, n).copy_(v)
    return m


def as_ref(m) -> ref.Params:
    return ref.Params(*(getattr(m, n).detach().to(torch.float64).contiguous() for n in dg.PARAM_ORDER))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    names = names_fixture(40, 9)
    seqs = make_next_token_sequences(names, VOCAB)
    census = dg.build_probe(seqs, names, info_for(seqs), tmp_path_factory.mktemp("probe") / "probe.json")
    # Non-unit design weights (re-review of MJ-4): the 40-name probe is a census (every w = 1), so
    # declare larger stratum populations; weights N_ℓ/n_ℓ then differ across strata.
    pop = tuple(n * f for n, f in zip(census.sample_by_stratum, (3, 7, 2, 5)))
    probe = dataclasses.replace(census, population_by_stratum=pop,
                                weights_by_stratum=tuple(p / n if n else 0.0 for p, n in zip(pop, census.sample_by_stratum)))
    assert len(set(probe.weights().tolist())) > 1
    m = perturbed(6, 1.2, 61)
    return names, seqs, probe, m, as_ref(m)


def wq(values, weights, keys, qs=(0.25, 0.5, 0.75)):
    """Oracle: smallest x_(j), ascending with ties by key, with cumulative weight ≥ q·Σw (diag §4)."""
    order = sorted(range(len(values)), key=lambda i: (values[i], keys[i]))
    total = math.fsum(weights)
    out = []
    for q in qs:
        for j in range(len(order)):
            if math.fsum(weights[i] for i in order[: j + 1]) >= q * total:
                out.append(values[order[j]])
                break
    return out


# --------------------------------------------------------------------------- MJ-3: G3 summaries
def test_MJ3_g3_summaries_match_explicit_matrices(setup) -> None:
    """Q̂(Δ), π̂(Δ), π̂±(Δ), κ̂_c, Π̂(Δ ≥ 3) and ‖Ĝ‖ for the full probe and each half, from every term
    T = w D_k P_{t,k}ᵀ r_t h_{k−1}ᵀ built explicitly with the MS1 reference (diag §4.4); and the
    split-half spread |a − b|/2 of Π̂(Δ ≥ 3) and κ̂_c."""
    _, seqs, probe, m, p = setup
    res = dg.g3_decomposition(dg.float64_copy(m), probe, seqs)
    summary = {}
    for key, wts in (("full", probe.weights()), ("A", probe.half_weights("A")), ("B", probe.half_weights("B"))):
        terms = []
        for b, r in enumerate(probe.rows):
            w = float(wts[b])
            if w == 0:
                continue
            x = seqs.inputs[r, : int(seqs.lengths[r])].tolist()
            f = ref.forward_name(p, x)
            for t in range(2, len(x) + 1):
                rt_ = ref.hidden_residual(p, f, t)
                for k in range(2, t + 1):
                    P = ref.jacobian_product(p, f, t, k)
                    Dk = 1 - f.states[k - 1] ** 2
                    terms.append((t - k, w * torch.outer(Dk * (P.T @ rt_), f.states[k - 2])))
        G = sum(T for _, T in terms)
        G2 = float((G * G).sum())
        share, pos, neg, mass = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        for d, T in terms:
            ip = float((T * G).sum())
            share[d] += ip / G2
            (pos if ip > 0 else neg)[d] += ip / G2
            mass[d] += float(T.norm())
        kappa = math.sqrt(G2) / math.fsum(float(T.norm()) for _, T in terms)
        out = res.primary[key]
        for d in share:
            close(out["share"][str(d)], share[d], f"{key} π̂({d})")
            close(out["share_pos"][str(d)], pos[d], f"{key} π̂+({d})")
            close(out["share_neg"][str(d)], neg[d], f"{key} π̂−({d})")
            close(out["mass"][str(d)], mass[d], f"{key} Q̂({d})")
        close(out["kappa"], kappa, f"{key} κ̂")
        long3 = math.fsum(v for d, v in share.items() if d >= 3)
        close(out["long_share_ge3"], long3, f"{key} Π̂(Δ≥3)")
        close(out["grad_norm"], math.sqrt(G2), f"{key} ‖Ĝ‖")
        summary[key] = (long3, kappa)
    agree = res.primary["agreement"]
    for q, i in (("long_share_ge3", 0), ("kappa", 1)):
        a, b = summary["A"][i], summary["B"][i]
        close(agree[q]["diff"], a - b, f"{q} diff")
        close(agree[q]["spread"], abs(a - b) / 2, f"{q} spread")


# --------------------------------------------------------------------------- MJ-4: weighted aggregates
def test_MJ4_g1_population_weighted_quantiles_per_start_and_pooled(setup) -> None:
    """diag §4.2 (ii): population-weighted quantiles of log φ per (start s, Δ) and pooled per Δ, with
    weights w_β = N_ℓ/n_ℓ and log φ from explicit reference Jacobian products."""
    _, seqs, probe, m, p = setup
    res = dg.g1_products(dg.float64_copy(m), probe, seqs, spectral=False)
    H = m.hidden_dim
    w = probe.weights().tolist()
    cells = defaultdict(lambda: ([], [], []))
    for b, r in enumerate(probe.rows):
        x = seqs.inputs[r, : int(seqs.lengths[r])].tolist()
        f = ref.forward_name(p, x)
        for s in range(1, len(x)):
            for t in range(s + 1, len(x) + 1):
                phi = math.log(float(ref.jacobian_product(p, f, t, s).norm()) / math.sqrt(H))
                key = (probe.strata[b], r, t, s)
                for ck in (("s", s, t - s), ("pool", t - s)):
                    cells[ck][0].append(phi), cells[ck][1].append(w[b]), cells[ck][2].append(key)
    for row in res.primary["per_start_delta_weighted"]:
        v, ws, ks = cells[("s", row["start"], row["delta"])]
        assert row["pairs"] == len(v)
        for a, e in zip(row["log_phi_q"], wq(v, ws, ks)):
            close(a, e, f"G1 per-start s={row['start']} Δ={row['delta']}")
    for row in res.primary["pooled_delta_weighted"]:
        v, ws, ks = cells[("pool", row["delta"])]
        for a, e in zip(row["log_phi_q"], wq(v, ws, ks)):
            close(a, e, f"G1 pooled Δ={row['delta']}")


def test_MJ4_g2_rbar_and_h1_weighted_against_reference(setup) -> None:
    """G2 R̄ per (stratum, Δ) and population-weighted per Δ (diag §4.3), and H1 per-t weighted (diag §4.1)."""
    _, seqs, probe, m, p = setup
    cm = dg.float64_copy(m)
    r2 = dg.g2_attenuation(cm, probe, seqs)
    w = probe.weights()
    num, den, numw, denw = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    for b, r in enumerate(probe.rows):
        x = seqs.inputs[r, : int(seqs.lengths[r])].tolist()
        f = ref.forward_name(p, x)
        for t in range(1, len(x) + 1):
            rn = float(ref.hidden_residual(p, f, t).norm())
            for s in range(1, t + 1):
                u = float(ref.loss_projected_vector(p, f, t, s).norm())
                k = (probe.strata[b], t - s)
                num[k] += u
                den[k] += rn
                numw[t - s] += float(w[b]) * u
                denw[t - s] += float(w[b]) * rn
    for row in r2.primary["per_stratum_delta"]:
        k = (row["stratum"], row["delta"])
        close(row["R_bar"], num[k] / den[k], f"R̄ {k}")
    for row in r2.primary["pooled_delta_weighted"]:
        close(row["R_bar_weighted"], numw[row["delta"]] / denw[row["delta"]], f"R̄_w {row['delta']}")
    h1 = dg.h1_statistics(cm, probe, seqs)
    H = m.hidden_dim
    for row in h1.primary["per_t_weighted"]:
        t = row["t"]
        sw = s_rms = s_sat = s_d = s_d2 = 0.0
        for b, r in enumerate(probe.rows):
            x = seqs.inputs[r, : int(seqs.lengths[r])].tolist()
            if len(x) < t:
                continue
            h = ref.forward_name(p, x).states[t - 1]
            wb = float(w[b])
            sw += wb * H
            s_rms += wb * float((h * h).sum())
            s_sat += wb * float((h.abs() > 0.97).double().sum())
            s_d += wb * float((1 - h * h).sum())
            s_d2 += wb * float(((1 - h * h) ** 2).sum())
        close(row["rms"], math.sqrt(s_rms / sw), f"rms t={t}")
        close(row["mean_dtanh"], s_d / sw, f"E[tanh'] t={t}")
        close(row["mean_sq_dtanh"], s_d2 / sw, f"E[tanh'^2] t={t}")
        close(row["saturation"], s_sat / sw, f"saturation t={t}")


def test_MJ4_g2_exclusions_counted_per_stratum(setup) -> None:
    """With V = 0 every r_t = Vᵀ(p − e_y) is exactly zero: every target is excluded and counted in its
    own probe stratum (diag §4.3 step 3)."""
    _, seqs, probe, m, _ = setup
    m0 = dg.float64_copy(m)
    with torch.no_grad():
        m0.output_weight.zero_()
    res = dg.g2_attenuation(m0, probe, seqs)
    expect = defaultdict(int)
    for b, r in enumerate(probe.rows):
        expect[str(probe.strata[b])] += int(seqs.lengths[r])
    assert res.primary["excluded_exact_zero_by_stratum"] == dict(sorted(expect.items()))
    assert res.raw["g2_t"].size == 0


# --------------------------------------------------------------------------- MJ-5: clustered SE
def test_MJ5_clustered_se_nondegenerate_hand_computation() -> None:
    """diag §4.5: SE² = (N_c/(N_c − 1)) Σ_i (s_i − mean·m_i)² / (Σ m_i)², on a fixture whose residuals
    are not zero (the D8 fixture's residuals all vanish)."""
    by_name = {0: [1.0, 3.0], 1: [5.0], 2: [0.0, 0.5, 6.0], 3: [-2.0]}
    s, mm = [4.0, 5.0, 6.5, -2.0], [2, 1, 3, 1]
    mean = 13.5 / 7
    se = math.sqrt(4 / 3 * sum((si - mean * mi) ** 2 for si, mi in zip(s, mm)) / 49)
    out = dg._clustered(by_name)
    assert math.isclose(out["mean"], mean, rel_tol=1e-15) and math.isclose(out["se"], se, rel_tol=1e-14)
    assert se > 0.5  # far from degenerate
    assert dg._clustered({0: [1.0, 2.0]})["se"] is None  # one cluster: undefined


# --------------------------------------------------------------------------- MJ-6: run-time checks detect near misses
def test_MJ6_g3_identity_and_share_sum_detect_a_small_fault(setup, monkeypatch) -> None:
    """diag §10.2 rows 1 – 2: a residual off by a relative 1e‑6 (far above 1e‑9) fails the identity and
    the share-sum checks for the full probe and each half; the unperturbed decomposition passes."""
    _, seqs, probe, m, _ = setup
    ok = dg.g3_decomposition(dg.float64_copy(m), probe, seqs)
    assert all(ok.checks[f"{c}_{k}"]["pass"] for c in ("identity", "share_sum") for k in ("full", "A", "B"))
    original = dg._stable_residual
    monkeypatch.setattr(dg, "_stable_residual", lambda z, y: original(z, y) * (1 + 1e-6))
    bad = dg.g3_decomposition(dg.float64_copy(m), probe, seqs)
    for k in ("full", "A", "B"):
        assert bad.checks[f"identity_{k}"]["pass"] is False and 1e-7 < bad.checks[f"identity_{k}"]["rel_error"] < 1e-5
        assert bad.checks[f"share_sum_{k}"]["pass"] is False


def test_MJ6_mn3_c1_noop_check_uses_the_restart_builder(setup, monkeypatch) -> None:
    """diag §10.2 row 4 / review mn-3: the j = 2 no-op restart is built by the same builder as every
    j ≥ 3 restart; a builder that drops one token fails the check, the correct one passes."""
    names, seqs, _, m, _ = setup
    cm = dg.float64_copy(m)
    assert dg.c1_restart(cm, seqs).checks["noop_restart_reproduces_full"] is True
    original = dg._restart_batch

    def off_by_one(dev, lengths, part, pad):
        x = original(dev, lengths, part, pad)
        col = x[:, 1]
        x[:, 1] = torch.where(col != pad, (col % 26) + 1, col)  # a wrong (but valid) first restart token
        return x

    monkeypatch.setattr(dg, "_restart_batch", off_by_one)
    res = dg.c1_restart(cm, seqs)
    assert res.checks["noop_restart_reproduces_full"] is False and res.checks["noop_max_excess"] > 0


# --------------------------------------------------------------------------- MJ-1: R ≤ σ at E_sel
def test_MJ1_r_le_sigma_check_on_real_results_and_faults(setup) -> None:
    _, seqs, probe, m, _ = setup
    cm = dg.float64_copy(m)
    g1 = dg.g1_products(cm, probe, seqs, spectral=True)
    g2 = dg.g2_attenuation(cm, probe, seqs)
    ok = dg.r_le_sigma_check(g1.raw, g2.raw)
    assert ok["pass"] is True and ok["pairs"] == int((g2.raw["g2_s"] < g2.raw["g2_t"]).sum()) > 0
    assert ok["max_log_excess"] <= 0
    inflated = dict(g2.raw, g2_log_R=g2.raw["g2_log_R"] + math.log(1000.0))
    assert dg.r_le_sigma_check(g1.raw, inflated)["pass"] is False
    # one pair set just above σ(1 + 1e-12): the slack is the spec's, not a looser one
    sig = {(int(r), int(s), int(t)): v for r, s, t, v in
           zip(g1.raw["g1_row"], g1.raw["g1_s"], g1.raw["g1_t"], g1.raw["g1_log_sigma"])}
    i = int(np.flatnonzero(g2.raw["g2_s"] < g2.raw["g2_t"])[0])
    key = (int(g2.raw["g2_row"][i]), int(g2.raw["g2_s"][i]), int(g2.raw["g2_t"][i]))
    for excess, expect in ((3e-12, False), (-1e-13, True)):
        lr = g2.raw["g2_log_R"].copy()
        lr[i] = sig[key] + math.log1p(1e-12) + excess
        assert dg.r_le_sigma_check(g1.raw, dict(g2.raw, g2_log_R=lr))["pass"] is expect
    keep = g1.raw["g1_t"] != g1.raw["g1_t"].max()
    partial = {k: v[keep] for k, v in g1.raw.items()}
    miss = dg.r_le_sigma_check(partial, g2.raw)
    assert miss["pass"] is False and miss["missing_sigma"] > 0


def _plan_run(tmp_path, tag, probe, train, names, dev, epochs=2, steps=5):
    run = RunManager.create("rnn", config={"t": tag}, results_root=tmp_path / tag)
    plan = dg.RNNDiagnosticsPlan(probe=probe, train=train, train_names=names, dev=dev, steps_per_epoch=steps,
                                 max_epochs=epochs, recurrent_gain=1.0, out_dir=run.run_dir / dg.RESEARCH_DIAGNOSTICS_DIR)
    return run, plan


@pytest.fixture
def tiny(tmp_path):
    all_names = names_fixture(26, 17)
    names = all_names[:20]
    train = make_next_token_sequences(names, VOCAB)
    dev = make_next_token_sequences(all_names[20:], VOCAB)
    probe = dg.build_probe(train, names, info_for(train), tmp_path / "probe.json")
    cfg = rt.RNNTrainingConfig(lr=0.3, max_epochs=2, batch_size=4, eval_batch_size=7)
    return names, train, dev, probe, cfg


def _model(seed=5):
    return RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=4), generator=make_generator(seed))


def test_MJ1_MJ2_inflated_R_invalidates_spectral_points_and_lists_them(tmp_path, monkeypatch, tiny) -> None:
    """An inflated G2 (×1000) passes every G2-internal check but fails R ≤ σ at the spectral points
    (epoch ends 1 and E): those points are invalid and listed in the result and the metadata (C6);
    non-spectral points stay valid; training is unaffected."""
    names, train, dev, probe, cfg = tiny
    original = dg.g2_attenuation

    def inflated(*a, **k):
        res = original(*a, **k)
        res.raw["g2_log_R"] = res.raw["g2_log_R"] + math.log(1000.0)
        return res

    monkeypatch.setattr(dg, "g2_attenuation", inflated)
    run, plan = _plan_run(tmp_path, "inflated", probe, train, names, dev)
    with pytest.warns(UserWarning, match="invalid"):
        res = rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=run, diagnostics=plan)
    pts = dg.read_jsonl_complete(plan.out_dir / "points.jsonl")
    # spectral points with G2 (P0 runs H1 and G1 only, diag §5.2): epoch ends 1 and E
    spectral = sorted(p["point_id"] for p in pts if p["spectral"] and "G2" in p["diagnostics"])
    assert len(spectral) == 2 and sorted(p["point_id"] for p in pts if not p["valid"]) == spectral
    assert all(p["checks"]["G1_G2"]["R_le_sigma"]["pass"] is False for p in pts if p["point_id"] in spectral)
    assert all("G1_G2" not in p["checks"] for p in pts if p["point_id"] not in spectral)
    assert res.diagnostics_status == "incomplete" and list(res.invalid_diagnostic_points) == spectral
    assert json.loads(run.metadata_path.read_text())["invalid_diagnostic_points"] == spectral


def test_MJ2_complete_run_lists_no_invalid_points(tmp_path, tiny) -> None:
    names, train, dev, probe, cfg = tiny
    run, plan = _plan_run(tmp_path, "clean", probe, train, names, dev)
    res = rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=run, diagnostics=plan)
    pts = dg.read_jsonl_complete(plan.out_dir / "points.jsonl")
    assert res.diagnostics_status == "complete" and res.invalid_diagnostic_points == ()
    assert json.loads(run.metadata_path.read_text())["invalid_diagnostic_points"] == []
    with_g2 = [p for p in pts if p["spectral"] and "G2" in p["diagnostics"]]
    assert len(with_g2) == 2 and all(p["checks"]["G1_G2"]["R_le_sigma"]["pass"] for p in with_g2)


# --------------------------------------------------------------------------- mn-8, mn-7, N-5: resume safety
def test_mn8_completed_run_refuses_resume_before_any_truncation(tmp_path, tiny) -> None:
    names, train, dev, probe, cfg = tiny
    run, plan = _plan_run(tmp_path, "done", probe, train, names, dev)
    rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=run, diagnostics=plan)
    before = {p: p.read_bytes() for p in run.run_dir.rglob("*") if p.is_file()}
    plan2 = dg.RNNDiagnosticsPlan(probe=probe, train=train, train_names=names, dev=dev, steps_per_epoch=5,
                                  max_epochs=2, recurrent_gain=1.0, out_dir=plan.out_dir)
    with pytest.raises(rt.ResumeError, match="cannot be resumed"):
        rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=RunManager.open(run.run_dir), resume=True,
                     diagnostics=plan2)
    after = {p: p.read_bytes() for p in run.run_dir.rglob("*") if p.is_file()}
    assert after == before


def test_mn7_fresh_start_refuses_a_directory_with_diagnostic_points(tmp_path, tiny) -> None:
    names, train, dev, probe, cfg = tiny
    run, plan = _plan_run(tmp_path, "occupied", probe, train, names, dev)
    plan.at_start(_model())  # a point exists (e.g. a crash right after P0)
    with pytest.raises(rt.ResumeError, match="points.jsonl"):
        rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=run, diagnostics=plan)


def test_N5_truncate_moves_raw_files_never_deletes(tmp_path, tiny) -> None:
    """Plan §9.5: raw files of points removed by the resume truncation are moved, not deleted."""
    names, train, dev, probe, cfg = tiny
    run, plan = _plan_run(tmp_path, "trunc", probe, train, names, dev)
    rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=run, diagnostics=plan)
    raw_before = {p.name: p.read_bytes() for p in plan.raw_dir.glob("point_*.npz")}
    plan.truncate(sigma=5, completed_epochs=1)
    kept = {p.name for p in plan.raw_dir.glob("point_*.npz")}
    moved = {p.name: p.read_bytes() for p in (plan.out_dir / "truncated").rglob("point_*.npz")}
    assert kept | set(moved) == set(raw_before) and not kept & set(moved) and moved
    assert all(raw_before[n] == b for n, b in moved.items())


# --------------------------------------------------------------------------- mn-4: post-hoc isolation fields
def test_mn4_posthoc_point_records_process_state(tmp_path, tiny) -> None:
    names, train, dev, probe, _ = tiny
    cfg = rt.RNNTrainingConfig(lr=3.0, max_epochs=6, batch_size=4, eval_batch_size=7)  # best epoch 2 of 6
    run, plan = _plan_run(tmp_path, "posthoc", probe, train, names, dev, epochs=6)
    res = rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=run, diagnostics=plan)
    assert res.best_epoch not in (1, 6)
    pid = dg.posthoc_best_spectral(run.run_dir, probe, train)
    assert pid is not None
    rec = [p for p in dg.read_jsonl_complete(plan.out_dir / "points.jsonl") if p["point_id"] == pid][0]
    assert rec["rng_sha256_before"] == rec["rng_sha256_after"] is not None
    assert rec["param_sha256_before"] == rec["param_sha256_after"] and rec["valid"]


# --------------------------------------------------------------------------- arch §10 large gain; mn-15
@pytest.mark.parametrize("g", [5.0, 50.0])
def test_large_gain_fixture_stays_finite(g, setup) -> None:
    """arch §10: a large recurrent gain saturates tanh but forward, loss, gradients and the G1/G2/G3
    diagnostics stay finite (exact zeros of D are allowed; logs may be −inf, never NaN)."""
    _, seqs, probe, _, _ = setup
    m = perturbed(6, g, 3)
    x, y = seqs.inputs[:8], seqs.targets[:8]
    logits = m(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    cm = dg.float64_copy(m)
    g1 = dg.g1_products(cm, probe, seqs, spectral=True)
    g2 = dg.g2_attenuation(cm, probe, seqs)
    assert not np.isnan(g1.raw["g1_log_phi"]).any() and not np.isnan(g2.raw["g2_log_R"]).any()
    assert dg.r_le_sigma_check(g1.raw, g2.raw)["pass"]


def test_mn15_position_nll_totals_are_exact_fsums(setup) -> None:
    _, seqs, _, m, _ = setup
    pos = rt.evaluate_position_nll(m, seqs, batch_size=7)
    with torch.no_grad():
        logp = torch.log_softmax(m(seqs.inputs), dim=-1)
    vals = defaultdict(list)
    for r in range(seqs.inputs.shape[0]):
        for t in range(1, int(seqs.lengths[r]) + 1):
            vals[str(t)].append(-float(logp[r, t - 1, int(seqs.targets[r, t - 1])]))
    for t, v in vals.items():
        assert pos.by_t[t][1] == len(v) and abs(pos.by_t[t][0] - math.fsum(v)) <= 1e-12 * max(1, abs(math.fsum(v)))


# --------------------------------------------------------------------------- mn-12: X1 with a diverging run
def test_mn12_diverging_run_with_plan_records_points_up_to_its_end(tmp_path, tiny) -> None:
    """Plan R7 / integration X1: a Category D run (η = 10 diverges at epoch 1 in this fixture) with the
    full plan ends ``failed_diverged``; its diagnostic points stop at its last epoch, every recorded
    point is complete and hash-aligned with the history, and no later epoch-end point exists."""
    names, train, dev, probe, _ = tiny
    cfg = rt.RNNTrainingConfig(lr=10.0, max_epochs=4, batch_size=4, eval_batch_size=7)
    run, plan = _plan_run(tmp_path, "diverge", probe, train, names, dev, epochs=4)
    res = rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=run, diagnostics=plan)
    assert res.status == "failed_diverged"
    hist = [json.loads(x) for x in run.history_path.read_text().splitlines()]
    last = hist[-1]["completed_epochs"]
    pts = dg.read_jsonl_complete(plan.out_dir / "points.jsonl")
    ends = [p for p in pts if p["kind"] == "epoch_end"]
    assert [p["completed_epochs"] for p in ends] == list(range(1, last + 1))
    for p in ends:
        assert p["param_sha256_before"] == hist[p["completed_epochs"]]["param_sha256"] == p["param_sha256_after"]
    assert all((plan.raw_dir / f"point_{p['point_id']:04d}.npz").exists() for p in pts)
    assert json.loads(run.metadata_path.read_text())["outcome"] == "failed_diverged"


# --------------------------------------------------------------------------- mutation item b: mean field at g ≠ 1
def _expect_normal(f, q: float, n: int = 400_001, width: float = 12.0) -> float:
    """E[f(√q Z)], Z ~ N(0, 1), by a fine trapezoid rule (independent of the Gauss–Hermite rule)."""
    z = np.linspace(-width, width, n)
    return float(np.trapezoid(f(math.sqrt(q) * z) * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi), z))


@pytest.mark.parametrize("g,s2", [(1.0, 0.5), (1.3, 0.2), (0.7, 0.9)])
def test_mean_field_table_recursion_and_chi_carry_the_gain(g, s2) -> None:
    """arch §7.2 item 6 at g ≠ 1 (the change rule needs predictions at other gains): q_1 = s²,
    q_t = g² E[tanh²(√q_{t−1} Z)] + s², E[tanh′], χ_t = g² E[(1 − tanh²)²], saturation 2Φ(−atanh(0.97)/√q_t)."""
    q = s2
    for r in dg.mean_field_table(g, s2):
        if r["t"] > 1:
            q = g * g * _expect_normal(lambda v: np.tanh(v) ** 2, q) + s2
        close(r["q"], q, f"q t={r['t']}")
        close(r["mean_dtanh"], _expect_normal(lambda v: 1 - np.tanh(v) ** 2, q), f"E[tanh'] t={r['t']}")
        close(r["chi"], g * g * _expect_normal(lambda v: (1 - np.tanh(v) ** 2) ** 2, q), f"chi t={r['t']}")
        close(r["saturation"], math.erfc(math.atanh(0.97) / math.sqrt(q) / math.sqrt(2)), f"sat t={r['t']}")


def test_mn18_category_a_run_is_never_resumed(tmp_path, tiny, monkeypatch) -> None:
    """D-46 (13): a run ended by Category A is final; a direct resume is refused before anything changes."""
    names, train, dev, probe, cfg = tiny
    run, plan = _plan_run(tmp_path, "catA", probe, train, names, dev)
    calls = {"n": 0}
    original = rt.evaluate_nll

    def nan_at_epoch_2(model, seqs, *, batch_size):
        calls["n"] += 1
        if calls["n"] == 5:  # epoch-2 train evaluation (epochs 0 and 1 use calls 1 – 4)
            raise rt.CategoryAFailure("injected", where="evaluation", details={})
        return original(model, seqs, batch_size=batch_size)

    monkeypatch.setattr(rt, "evaluate_nll", nan_at_epoch_2)
    with pytest.raises(rt.CategoryAFailure):
        rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=run, diagnostics=plan)
    monkeypatch.setattr(rt, "evaluate_nll", original)
    assert json.loads(run.metadata_path.read_text())["outcome"] == "failed_mechanical"
    before = {p: p.read_bytes() for p in run.run_dir.rglob("*") if p.is_file()}
    plan2 = dg.RNNDiagnosticsPlan(probe=probe, train=train, train_names=names, dev=dev, steps_per_epoch=5,
                                  max_epochs=2, recurrent_gain=1.0, out_dir=plan.out_dir)
    with pytest.raises(rt.ResumeError, match="final"):
        rt.train_rnn(_model(), train, dev, training=cfg, seed=3, run=RunManager.open(run.run_dir), resume=True,
                     diagnostics=plan2)
    assert {p: p.read_bytes() for p in run.run_dir.rglob("*") if p.is_file()} == before
