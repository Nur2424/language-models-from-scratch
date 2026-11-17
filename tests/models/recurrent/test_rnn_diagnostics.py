"""MS3 tests of the mandatory diagnostic functions (``models/recurrent/rnn_diagnostics.py``).

Milestone MS3 of ``rnn_implementation_plan.md`` v0.2 (D-49). Labels (D-48, C2): ``D:`` diagnostics
spec §10 (D1, D2, D5 – D8, D11 parts), ``A:`` arch spec §13.2 (M6 – M11), ``X5`` integration.
Synthetic data only (plan §7.4): invented names, never the project's names file or probe.
Independent baselines: the accepted MS1 reference (``rnn_reference.py``), hand computation,
exhaustive enumeration. Tolerance A = 1e‑9 (float64).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct

import pytest
import torch

import tests.models.recurrent.rnn_reference as ref
from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from utils.data import make_next_token_sequences
from utils.reproducibility import make_generator

PAD = 27
PARAMS = dg.PARAM_ORDER


# =========================================================================== fixtures
def synthetic_names(count: int, seed: int, max_len: int = 15) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, max_len + 1, (count,), generator=g).tolist()
    return ["".join(chr(97 + int(c)) for c in torch.randint(0, 26, (L,), generator=g)) for L in lengths]


def split_info_for(seqs) -> dict:
    return {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1],
            "N": int(seqs.lengths.numel()), "M": int(seqs.lengths.sum())}


def perturbed_model(vocab, H: int, g: float, seed: int) -> RNNLanguageModel:
    """Float64 model with perturbed parameters (fixture rule M6-IMP9)."""
    gen = make_generator(seed)
    Q, R = torch.linalg.qr(torch.randn(H, H, generator=gen, dtype=torch.float64))
    W = g * (Q * torch.sign(torch.diagonal(R)).unsqueeze(0))
    values = (math.sqrt(0.5) * torch.randn(27, H, generator=gen, dtype=torch.float64), W,
              0.1 * torch.randn(H, generator=gen, dtype=torch.float64),
              torch.randn(27, H, generator=gen, dtype=torch.float64) / math.sqrt(H),
              0.1 * torch.randn(27, generator=gen, dtype=torch.float64))
    model = RNNLanguageModel(vocab, RNNConfig(hidden_dim=H), generator=make_generator(1)).to(torch.float64)
    with torch.no_grad():
        for name, v in zip(PARAMS, values):
            getattr(model, name).copy_(v)
    return model


def as_ref(model) -> ref.Params:
    return ref.Params(*(getattr(model, n).detach().to(torch.float64).contiguous() for n in PARAMS))


@pytest.fixture(scope="module")
def data(tmp_path_factory, request):
    from utils.vocabulary import Vocabulary
    import string
    vocab = Vocabulary(chars=tuple(string.ascii_lowercase))
    names = synthetic_names(700, seed=5)
    seqs = make_next_token_sequences(names, vocab)
    info = split_info_for(seqs)
    path = tmp_path_factory.mktemp("probe") / "probe.json"
    probe = dg.build_probe(seqs, names, info, path)
    return vocab, names, seqs, info, probe, path


@pytest.fixture(scope="module")
def small(data, tmp_path_factory):
    """A small probe (≤ 6 names per stratum) for the O(n²H³) checks against the reference."""
    vocab = data[0]
    names = synthetic_names(40, seed=9)
    seqs = make_next_token_sequences(names, vocab)
    info = split_info_for(seqs)
    probe = dg.build_probe(seqs, names, info, tmp_path_factory.mktemp("small") / "probe.json")
    return vocab, names, seqs, info, probe


def close(a, e, what=""):
    a, e = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(e, dtype=torch.float64)
    excess = float(((a - e).abs() - (1e-9 + 1e-9 * e.abs())).max())
    assert excess <= 0.0, f"{what}: tolerance A exceeded by {excess:.3e}"


# =========================================================================== shared definitions
def test_param_sha256_hand_fixture(full_vocab) -> None:
    """C3 (X1 hand fixture): digest of a hand-set H = 1 model equals SHA-256 over the struct-packed
    little-endian float32 values of U, W, b, V, c in that order (an independent byte route)."""
    model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=1), generator=make_generator(1))
    U = [0.5 * k - 3.0 for k in range(27)]
    V = [0.25 - 0.01 * k for k in range(27)]
    c = [(-1.0) ** k * 1e-3 * k for k in range(27)]
    with torch.no_grad():
        model.input_lookup.copy_(torch.tensor(U).unsqueeze(1))
        model.recurrent_weight.fill_(-0.75)
        model.hidden_bias.fill_(-0.0)  # signed zero is distinguished
        model.output_weight.copy_(torch.tensor(V).unsqueeze(1))
        model.output_bias.copy_(torch.tensor(c))
    expected = hashlib.sha256(struct.pack("<27f", *U) + struct.pack("<f", -0.75) + struct.pack("<f", -0.0)
                              + struct.pack("<27f", *V) + struct.pack("<27f", *c)).hexdigest()
    assert dg.param_sha256(model) == expected
    assert dg.param_sha256(model.to(torch.float64)) == expected  # float64 rounded to the same float32 values


def test_stratum_index() -> None:
    assert [dg.stratum_index(n) for n in (2, 5, 6, 7, 8, 9, 10, 16)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert dg.stratum_index(17) == dg.OVERFLOW_STRATUM
    with pytest.raises(ValueError):
        dg.stratum_index(1)


# =========================================================================== D1 probe
def test_D1_probe_construction_rules(data, tmp_path) -> None:
    """D:D1: deterministic rows and SHA-256; strata; census rule n_ℓ = min(256, N_ℓ); weights N_ℓ/n_ℓ;
    halves alternate within each stratum's sorted rows; the draw follows diag §3.1 exactly."""
    vocab, names, seqs, info, probe, path = data
    again = dg.build_probe(seqs, names, info, tmp_path / "again.json")
    assert again.rows == probe.rows and again.sha256 == probe.sha256
    lengths = seqs.lengths.tolist()
    gen = make_generator(dg.derive_seed(42, "m6-probe"))
    for s in range(4):
        members = [i for i, n in enumerate(lengths) if dg.stratum_index(n) == s]
        n_s = min(256, len(members))
        chosen = sorted(torch.tensor(members)[torch.randperm(len(members), generator=gen)[:n_s]].tolist())
        got = [r for r, st in zip(probe.rows, probe.strata) if st == s]
        assert got == chosen
        assert probe.population_by_stratum[s] == len(members) and probe.sample_by_stratum[s] == n_s
        assert probe.weights_by_stratum[s] == len(members) / n_s
        halves = [h for h, st in zip(probe.halves, probe.strata) if st == s]
        assert halves == ["A" if k % 2 == 0 else "B" for k in range(n_s)]
    assert max(probe.population_by_stratum) > 256 and 256 in probe.sample_by_stratum  # the cap is exercised
    assert dg.load_probe(path) == probe
    with pytest.raises(FileExistsError):
        dg.build_probe(seqs, names, info, path)


def test_D1_load_probe_rejects_tampering(data, tmp_path) -> None:
    *_, path = data
    blob = json.loads(path.read_text())
    blob["names"][0] = blob["names"][0] + "x"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(blob))
    with pytest.raises(dg.ProbeMismatchError):
        dg.load_probe(bad)


# =========================================================================== X5 probe identity (C5)
def test_X5_validate_probe_rejects_each_mismatch(data) -> None:
    """X5: hash, split information, a changed name, a changed length and a changed N_ℓ are each
    rejected before any computation; the matching data pass."""
    vocab, names, seqs, info, probe, _ = data
    dg.validate_probe(probe, seqs, names, info)
    from dataclasses import replace
    with pytest.raises(dg.ProbeMismatchError, match="SHA-256"):
        dg.validate_probe(replace(probe, sha256="0" * 64), seqs, names, info)
    with pytest.raises(dg.ProbeMismatchError, match="split_info"):
        dg.validate_probe(probe, seqs, names, {**info, "seed": 7})
    row = probe.rows[3]
    altered = list(names)
    altered[row] = "q" * len(names[row])
    with pytest.raises(dg.ProbeMismatchError, match="name"):
        dg.validate_probe(probe, seqs, altered, info)
    longer = list(names)
    longer[row] = names[row] + ("a" if len(names[row]) < 15 else "")
    longer[row] = longer[row] if len(longer[row]) != len(names[row]) else names[row][:-1]
    seqs2 = make_next_token_sequences(longer, vocab)
    with pytest.raises(dg.ProbeMismatchError):
        dg.validate_probe(probe, seqs2, longer, info)  # the name changed with it; the length check also bites
    outside = next(i for i in range(len(names)) if i not in set(probe.rows))
    fewer = [n for i, n in enumerate(names) if i != outside]
    with pytest.raises(dg.ProbeMismatchError):
        dg.validate_probe(probe, make_next_token_sequences(fewer, vocab), fewer, info)


def test_X5_validate_probe_length_check_alone(data) -> None:
    """X5: the stratum recomputed from a row's length is checked even when the name is unchanged."""
    vocab, names, seqs, info, probe, _ = data
    b = next(i for i, s in enumerate(probe.strata) if s == 0)
    row = probe.rows[b]
    lengths = seqs.lengths.clone()
    lengths[row] = 12  # stratum 3 instead of 0
    from dataclasses import replace
    with pytest.raises(dg.ProbeMismatchError, match="stratum"):
        dg.validate_probe(probe, replace(seqs, lengths=lengths), names, info)


def test_X5_initial_statistics_refuses_other_seeds(data, tmp_path) -> None:
    vocab, names, seqs, info, probe, _ = data
    cfg = RNNConfig(hidden_dim=4)
    for seeds in ((100, 101, 102), tuple(range(1000, 1009)), tuple(range(1001, 1011))):
        with pytest.raises(ValueError, match="1000 – 1009"):
            dg.initial_statistics(vocab, cfg, probe, seqs, names, info, seeds, out_dir=tmp_path)
    with pytest.raises(ValueError, match="g = 1"):
        dg.initial_statistics(vocab, RNNConfig(hidden_dim=4, recurrent_gain=1.2), probe, seqs, names, info,
                              out_dir=tmp_path)
    assert not any(tmp_path.iterdir())


# =========================================================================== D2 estimators
def test_D2_weighted_quantile_and_ties() -> None:
    """D:D2: smallest value with cumulative weight ≥ q·Σw; ties broken by the key tuple."""
    vals, wts = [3.0, 1.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]
    assert dg.weighted_quantiles(vals, wts, [(0,), (1,), (2,), (3,)]) == [1.0, 2.0, 2.0]
    assert dg.weighted_quantiles([5.0, 1.0], [3.0, 1.0], [(0,), (1,)], qs=(0.25, 0.26)) == [1.0, 5.0]
    assert dg.weighted_quantiles([], [], []) == [None, None, None]


def test_D2_horvitz_thompson_is_unbiased_by_enumeration() -> None:
    """D:D2: over every stratified sample (2 of 4 and 1 of 3), the mean of Σ w x equals the total."""
    strata = [[1.0, 4.0, 2.0, 7.0], [3.0, 5.0, 11.0]]
    sizes = [2, 1]
    totals = []
    for a in itertools.combinations(range(4), sizes[0]):
        for b in itertools.combinations(range(3), sizes[1]):
            totals.append(sum(strata[0][i] * 4 / 2 for i in a) + sum(strata[1][i] * 3 / 1 for i in b))
    assert math.isclose(sum(totals) / len(totals), sum(map(sum, strata)), rel_tol=1e-12)


def test_D11_half_weights_partition(data) -> None:
    """D:D11 (part): halves partition each stratum; half weight N_ℓ/m_{ℓ,h}."""
    *_, probe, _ = data
    wa, wb = probe.half_weights("A"), probe.half_weights("B")
    assert bool(((wa > 0) ^ (wb > 0)).all())
    for s in range(4):
        idx = [i for i, st in enumerate(probe.strata) if st == s]
        m_a = sum(1 for i in idx if probe.halves[i] == "A")
        assert all(math.isclose(float(wa[i]), probe.population_by_stratum[s] / m_a) for i in idx if probe.halves[i] == "A")
        # each half's weights sum to the stratum population
        assert math.isclose(float(wa[idx].sum()), probe.population_by_stratum[s])
        assert math.isclose(float(wb[idx].sum()), probe.population_by_stratum[s])


# =========================================================================== H1 (A:M7)
def test_M7_h1_matches_reference_states(small) -> None:
    """A:M7: H1 per (t, stratum) equals statistics computed from the MS1 reference's states with
    explicit masks (A); counts equal the valid cells."""
    vocab, names, seqs, info, probe = small
    model = perturbed_model(vocab, 8, 1.2, 11)
    res = dg.h1_statistics(dg.float64_copy(model), probe, seqs)
    assert res.checks["h1_counts_equal_valid_cells"]
    p = as_ref(model)
    states = {b: ref.forward_name(p, seqs.inputs[r, : int(seqs.lengths[r])].tolist()).states
              for b, r in enumerate(probe.rows)}
    for cell in res.primary["per_t_stratum"]:
        hs = torch.stack([states[b][cell["t"] - 1] for b in range(len(probe)) if probe.strata[b] == cell["stratum"]
                          and states[b].shape[0] >= cell["t"]])
        assert cell["names"] == hs.shape[0] and cell["elements"] == hs.numel()
        close(cell["rms"], math.sqrt(float((hs * hs).mean())), "rms")
        close(cell["saturation"], float((hs.abs() > 0.97).double().mean()), "saturation")
        close(cell["mean_dtanh"], float((1 - hs * hs).mean()), "d̄")
        close(cell["mean_sq_dtanh"], float(((1 - hs * hs) ** 2).mean()), "d²")


# =========================================================================== G1
@pytest.mark.parametrize("g", [1.0, 1.2])
def test_G1_products_match_reference(small, g) -> None:
    """φ_{t,s} and σ_{t,s} for every probe pair equal the norms of the reference's analytic P_{t,s}
    (A); ω(Δ) equals ‖W^Δ‖_F/√H; φ ≤ σ check passes."""
    vocab, names, seqs, info, probe = small
    model = perturbed_model(vocab, 6, g, 21)
    res = dg.g1_products(dg.float64_copy(model), probe, seqs, spectral=True)
    assert res.checks["phi_le_sigma"]
    p = as_ref(model)
    raw = res.raw
    forwards = {r: ref.forward_name(p, seqs.inputs[r, : int(seqs.lengths[r])].tolist()) for r in probe.rows}
    expected_pairs = sum(n * (n - 1) // 2 for n in (int(seqs.lengths[r]) for r in probe.rows))
    assert len(raw["g1_t"]) == expected_pairs
    for i in range(len(raw["g1_t"])):
        P = ref.jacobian_product(p, forwards[int(raw["g1_row"][i])], int(raw["g1_t"][i]), int(raw["g1_s"][i]))
        close(raw["g1_log_phi"][i], math.log(float(torch.linalg.matrix_norm(P)) / math.sqrt(6)), "log φ")
        close(raw["g1_log_sigma"][i], math.log(float(torch.linalg.matrix_norm(P, ord=2))), "log σ")
    W = p.W
    for d, om in enumerate(res.primary["omega"], start=1):
        close(om, float(torch.linalg.matrix_norm(torch.linalg.matrix_power(W, d))) / math.sqrt(6), "ω")


def test_D7_delta_hat_equals_level2_prediction_at_theta0(full_vocab, small) -> None:
    """D:D7: at θ₀ (W = g·Q exactly, b = V = c = 0) δ̂ equals the arch §7.2 Level-2 φ̂ (A)."""
    vocab, names, seqs, info, probe = small
    g = 1.0
    model = perturbed_model(vocab, 8, g, 31)
    with torch.no_grad():
        model.hidden_bias.zero_()
        model.output_weight.zero_()
        model.output_bias.zero_()
    cm = dg.float64_copy(model)
    res = dg.g1_products(cm, probe, seqs, spectral=False)
    with torch.no_grad():
        h = cm._recurrence(seqs.inputs[list(probe.rows)]).states
    raw = res.raw
    index = {r: b for b, r in enumerate(probe.rows)}
    for i in range(len(raw["g1_t"])):
        b, s, t = index[int(raw["g1_row"][i])], int(raw["g1_s"][i]), int(raw["g1_t"][i])
        level2 = 0.5 * sum(math.log(g * g * float(((1 - h[b, k - 1] ** 2) ** 2).mean())) for k in range(s + 1, t + 1))
        close(raw["g1_log_delta_hat"][i], level2, "δ̂ vs φ̂")


# =========================================================================== G2 (D6, A:M2 form)
def test_G2_matches_reference(small) -> None:
    """R_{t,s} = ‖P_{t,s}ᵀ r_t‖/‖r_t‖ equals the reference's u_{t,s} norm ratio for every pair (A)."""
    vocab, names, seqs, info, probe = small
    model = perturbed_model(vocab, 6, 1.2, 41)
    res = dg.g2_attenuation(dg.float64_copy(model), probe, seqs)
    p = as_ref(model)
    raw = res.raw
    forwards = {r: ref.forward_name(p, seqs.inputs[r, : int(seqs.lengths[r])].tolist()) for r in probe.rows}
    assert len(raw["g2_t"]) == sum(n * (n + 1) // 2 for n in (int(seqs.lengths[r]) for r in probe.rows))
    for i in range(len(raw["g2_t"])):
        f, t, s = forwards[int(raw["g2_row"][i])], int(raw["g2_t"][i]), int(raw["g2_s"][i])
        u = ref.loss_projected_vector(p, f, t, s)
        r = ref.hidden_residual(p, f, t)
        close(raw["g2_R"][i], float(u.norm() / r.norm()), "R")
        close(raw["g2_r_norm"][i], float(r.norm()), "‖r‖")


def test_D6_g2_numerical_rule(small) -> None:
    """D:D6: r = 0 exactly (V = 0) is excluded and counted; p_y = 1 − 1e‑20 gives a finite, exact g;
    V scaled by 1e‑200 gives the same R as at scale 1 (scale invariance through the scaled norm)."""
    vocab, names, seqs, info, probe = small
    model = perturbed_model(vocab, 6, 1.0, 51)
    zero = perturbed_model(vocab, 6, 1.0, 51)
    with torch.no_grad():
        zero.output_weight.zero_()
    res0 = dg.g2_attenuation(dg.float64_copy(zero), probe, seqs)
    assert sum(res0.primary["excluded_exact_zero_by_stratum"].values()) == int(seqs.lengths[list(probe.rows)].sum())
    assert len(res0.raw.get("g2_t", [])) == 0
    logits = torch.zeros(27, dtype=torch.float64)
    logits[3] = 46.0  # p_3 ≈ 1 − 26·e^{-46} ≈ 1 − 2.8e-19
    g = dg._stable_residual(logits, 3)
    others = float(torch.exp(torch.log_softmax(logits, 0))[torch.arange(27) != 3].sum())
    assert others > 0 and float(g[3]) == -others  # no cancellation to 0 although p_3 rounds to 1
    r = torch.linspace(-1.0, 1.0, 6, dtype=torch.float64) + 0.3
    u1, n1 = dg._unit_and_norm(r)
    u2, n2 = dg._unit_and_norm(r * 1e-200)  # components near 1e-200: no underflow, same direction
    close(u2, u1, "unit vector under rescaling")
    assert math.isclose(n2, n1 * 1e-200, rel_tol=1e-14) and n2 > 0
    W = model.recurrent_weight.detach()
    D = torch.full((6,), 0.5, dtype=torch.float64)
    close(torch.linalg.vector_norm(W.T @ (D * u2)), torch.linalg.vector_norm(W.T @ (D * u1)), "R under rescaling")
    assert dg._unit_and_norm(torch.zeros(6, dtype=torch.float64)) == (None, 0.0)


# =========================================================================== G3 (D5, D11)
def test_D5_g3_identity_and_reference(small) -> None:
    """D:D5: Σ T equals autograd ∂L̂/∂W with weights, full probe and each half (run-time checks pass);
    each Σπ̂ = 1. Independently, Ĝ equals Σ_β w_β × the MS1 reference's per-name G3 sum (A)."""
    vocab, names, seqs, info, probe = small
    model = perturbed_model(vocab, 6, 1.2, 61)
    res = dg.g3_decomposition(dg.float64_copy(model), probe, seqs)
    for key in ("full", "A", "B"):
        assert res.checks[f"identity_{key}"]["pass"], res.checks
        assert res.checks[f"share_sum_{key}"]["pass"], res.checks
    p = as_ref(model)
    w = probe.weights()
    G = sum(float(w[b]) * ref.g3_decomposition(p, seqs.inputs[[r], : int(seqs.lengths[r])]).G
            for b, r in enumerate(probe.rows))
    close(res.primary["full"]["grad_norm"], float(torch.linalg.matrix_norm(G)), "‖Ĝ‖")
    assert res.primary["label"] == "probe estimate"


def test_D11_g3_agreement_against_hand_computation(small) -> None:
    """D:D11: cos(Ĝ_A, Ĝ_B) and s(Δ) recomputed by hand from the per-half outputs and gradients."""
    vocab, names, seqs, info, probe = small
    model = perturbed_model(vocab, 6, 1.0, 71)
    cm = dg.float64_copy(model)
    res = dg.g3_decomposition(cm, probe, seqs)
    inputs, targets, _ = dg._probe_tensors(probe, seqs)
    grads = {}
    for h in ("A", "B"):
        W = cm.recurrent_weight
        (grads[h],) = torch.autograd.grad(dg._weighted_loss(cm, inputs, targets, probe.half_weights(h)), W)
    cos = float((grads["A"] * grads["B"]).sum() / (grads["A"].norm() * grads["B"].norm()))
    close(res.primary["agreement"]["cos_halves"], cos, "cos")
    for row in res.primary["agreement"]["per_delta"]:
        pa, pb = res.primary["A"]["share"][str(row["delta"])], res.primary["B"]["share"][str(row["delta"])]
        assert row["spread"] == abs(pa - pb) / 2 and row["diff"] == pa - pb


# =========================================================================== C1 (D8, A:M8, A:M9)
def test_M9_c1_shared_start_equals_definition(full_vocab) -> None:
    """A:M9 / D:D8: every ΔNLL_t(d) from the shared-start algorithm equals the definition run
    separately with the MS1 reference ([BOS] + the d kept characters), at tolerance A; F₈ and F₇
    membership and counts; matrix totals."""
    names = synthetic_names(30, seed=13)
    dev = make_next_token_sequences(names, full_vocab)
    model = perturbed_model(full_vocab, 6, 1.2, 81)
    res = dg.c1_restart(dg.float64_copy(model), dev, chunk=7)
    assert res.checks["noop_restart_reproduces_full"]
    p = as_ref(model)
    raw = res.raw
    for i in range(len(raw["c1_t"])):
        r, t, d = int(raw["c1_row"][i]), int(raw["c1_t"][i]), int(raw["c1_d"][i])
        x = dev.inputs[r, : int(dev.lengths[r])].tolist()
        y = int(dev.targets[r, t - 1])
        full = ref.forward_name(p, x)
        kept = [0] + x[t - d: t]  # BOS then x_{t-d+1} … x_t
        restart = ref.forward_name(p, kept, targets=[0] * d + [y])
        close(raw["c1_delta_nll"][i], float(restart.losses[-1]) - float(full.losses[t - 1]), f"row {r} t {t} d {d}")
        assert t >= d + 2
    count = sum(1 for n in dev.lengths.tolist() for t in range(1, n + 1) for d in range(1, t - 1))
    assert len(raw["c1_t"]) == count
    for label, tmin, dmax in (("F8", 10, 8), ("F7", 9, 7)):
        rows = res.primary[label]
        assert [row["d"] for row in rows] == list(range(1, dmax + 1))
        for row in rows:
            n_targets = sum(1 for n in dev.lengths.tolist() for t in range(tmin, n + 1))
            assert row["targets"] == n_targets
    assert sum(cell["count"] for cell in res.primary["matrix"]) == count


def test_M8_c1_noop_positions_reproduce_full_logits(full_vocab) -> None:
    """A:M8 (M6-E6): for t ≤ d + 1 the restart input is the full prefix, so the restart reproduces
    the full logits; checked with the reference on every name and such (t, d)."""
    names = synthetic_names(12, seed=17)
    dev = make_next_token_sequences(names, full_vocab)
    p = as_ref(perturbed_model(full_vocab, 6, 1.0, 91))
    for r in range(len(names)):
        x = dev.inputs[r, : int(dev.lengths[r])].tolist()
        full = ref.forward_name(p, x)
        for t in range(1, len(x) + 1):
            for d in range(max(1, t - 1), t + 1):
                if t > d + 1:
                    continue
                kept = [0] + x[max(1, t - d): t]
                if len(kept) != t:
                    continue
                close(ref.forward_name(p, kept).logits[-1], full.logits[t - 1], "no-op")


def test_D8_clustered_se_hand_computation() -> None:
    """D:D8: mean and name-clustered SE against a hand computation."""
    by_name = {0: [1.0, 3.0], 1: [2.0], 2: [0.0, 0.0, 6.0]}
    s, m = [4.0, 2.0, 6.0], [2, 1, 3]
    mean = 12.0 / 6
    se = math.sqrt(3 / 2 * sum((si - mean * mi) ** 2 for si, mi in zip(s, m)) / 36)
    out = dg._clustered(by_name)
    assert out["mean"] == mean and math.isclose(out["se"], se, rel_tol=1e-15) and out["targets"] == 6 and out["names"] == 3


# =========================================================================== A:M10 padding never enters
def test_M10_padding_never_enters_any_diagnostic(small) -> None:
    """A:M10: training sequences with heavy extra padding give identical diagnostics."""
    vocab, names, seqs, info, probe = small
    wide = make_next_token_sequences(names, vocab, max_len=30)
    model = perturbed_model(vocab, 6, 1.2, 101)
    for fn in (dg.h1_statistics, lambda c, p, s: dg.g1_products(c, p, s, spectral=True), dg.g2_attenuation,
               dg.g3_decomposition):
        a, b = fn(dg.float64_copy(model), probe, seqs), fn(dg.float64_copy(model), probe, wide)
        assert json.dumps(a.primary, sort_keys=True) == json.dumps(b.primary, sort_keys=True)
    c1a = dg.c1_restart(dg.float64_copy(model), make_next_token_sequences(names[:10], vocab))
    c1b = dg.c1_restart(dg.float64_copy(model), make_next_token_sequences(names[:10], vocab, max_len=30))
    assert json.dumps(c1a.primary, sort_keys=True) == json.dumps(c1b.primary, sort_keys=True)


# =========================================================================== A:M11 mean field
def test_M11_mean_field_table_reproduces_arch_7_2() -> None:
    """A:M11: the q_t / SD / saturation / E[tanh′] / χ_t table of arch §7.2 at g = 1, s² = 0.5.

    Criterion "|Δ| ≤ 1e‑4 (table rounding)" read as: within half a unit of the printed last digit,
    plus 1e‑4. (The printed E[tanh′] at t = 3, 0.636, is 0.63547 double-rounded via 0.6355; a
    literal 1e‑4 against 3-decimal entries cannot hold. Reported as a table-precision note.)"""
    table = dg.mean_field_table(1.0, 0.5)
    printed = {1: (0.5000, 0.707, 0.31, 0.726, 0.592), 2: (0.7737, 0.880, 1.74, 0.652, 0.511),
               3: (0.8481, 0.921, 2.31, 0.636, 0.494), 4: (0.8645, 0.930, 2.44, 0.632, 0.491),
               5: (0.8680, 0.932, 2.47, 0.631, 0.490), 16: (0.8689, 0.932, 2.48, 0.631, 0.490)}
    # The printed "≥ 6" row is the fixed point: q_6 = 0.86868 is still 2e-4 below it; it is
    # compared at t = 16, where the recursion has converged (q_t is monotone, checked below).
    def ok(value, printed_value, decimals):
        return abs(value - printed_value) <= 0.5 * 10 ** -decimals + 1e-4

    for t, (q, sd, sat, dt, chi) in printed.items():
        row = table[t - 1]
        assert ok(row["q"], q, 4) and ok(row["sd"], sd, 3) and ok(100 * row["saturation"], sat, 2), row
        assert ok(row["mean_dtanh"], dt, 3) and ok(row["chi"], chi, 3), row
    qs = [r["q"] for r in table]
    assert all(a <= b for a, b in zip(qs, qs[1:])) and all(ok(q, 0.8689, 4) for q in qs[7:])


# =========================================================================== §7.2 procedure (synthetic)
def test_initial_statistics_procedure_on_synthetic_probe(small, tmp_path) -> None:
    """The arch §7.2 procedure runs on seeds 1000 – 1009 only, takes no optimiser step, writes its
    report once, and its Level-2 decomposition is internally consistent."""
    vocab, names, seqs, info, probe = small
    rep = dg.initial_statistics(vocab, RNNConfig(hidden_dim=6), probe, seqs, names, info, out_dir=tmp_path)
    data = json.loads(rep.path.read_text())
    assert data["seeds"] == list(range(1000, 1010)) and len(data["per_seed"]) == 10
    assert hashlib.sha256(rep.path.read_bytes()).hexdigest() == rep.sha256
    for seed in data["per_seed"]:
        for pr in seed["pairs_by_delta_start"]:
            assert math.isclose(pr["obs_minus_mf"], pr["obs_minus_hat"] + pr["hat_minus_mf"], abs_tol=1e-12)
        model = RNNLanguageModel(vocab, RNNConfig(hidden_dim=6),
                                 generator=make_generator(dg.derive_seed(seed["seed"], "init")))
        assert seed["param_sha256"] == dg.param_sha256(model)  # untrained: no optimiser step
    with pytest.raises(FileExistsError):
        dg.initial_statistics(vocab, RNNConfig(hidden_dim=6), probe, seqs, names, info, out_dir=tmp_path)


def test_diagnostics_require_the_float64_copy(small) -> None:
    vocab, names, seqs, info, probe = small
    model = RNNLanguageModel(vocab, RNNConfig(hidden_dim=4), generator=make_generator(1))
    with pytest.raises(ValueError, match="float64"):
        dg.h1_statistics(model, probe, seqs)
