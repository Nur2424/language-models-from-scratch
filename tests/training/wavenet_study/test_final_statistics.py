"""Tests for training/wavenet_study/final_statistics.py (F1; D-31 and addendum).

Expected values are computed independently of the module: closed-form t distributions for
df = 1 and df = 2, numerical integration of the t density for df = 9, exact Decimal/Fraction
arithmetic for means and SDs, hand-worked Holm examples and brute-force sign-flip enumeration
by integer bitmask (a different enumeration from the module's).
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from training.wavenet_study import final_statistics as fs

getcontext().prec = 50
SEEDS = list(range(10))


# ------------------------------------------------------------------ independent references
def t_pdf(x: float, df: int) -> float:
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return c * (1 + x * x / df) ** (-(df + 1) / 2)


def t_p_two_sided_numeric(t: float, df: int, intervals: int = 20000) -> float:
    """1 − 2∫₀^|t| f(x) dx by composite Simpson's rule."""
    a, b = 0.0, abs(t)
    h = (b - a) / intervals
    s = t_pdf(a, df) + t_pdf(b, df)
    s += sum((4 if i % 2 else 2) * t_pdf(a + i * h, df) for i in range(1, intervals))
    return 1.0 - 2.0 * s * h / 3.0


def t_quantile_numeric(q: float, df: int) -> float:
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 - t_p_two_sided_numeric(mid, df, 4000)) / 2 + 0.5 < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def dec_mean_sd(d):
    xs = [Decimal(x) for x in d]          # exact binary values of the floats
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, var.sqrt()


def holm_reference(p: dict) -> dict:
    """Holm written as the textbook step-down rejection rule turned into adjusted p-values."""
    items = sorted(p.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, v) in enumerate(items):
        prev = max(prev, min(1.0, (m - i) * v))
        out[k] = prev
    return out


def sign_flip_reference(d) -> Fraction:
    xs = [Fraction(x) for x in d]
    n = len(xs)
    obs = abs(sum(xs))
    count = 0
    for mask in range(2 ** n):
        s = sum((-x if (mask >> i) & 1 else x) for i, x in enumerate(xs))
        count += abs(s) >= obs
    return Fraction(count, 2 ** n)


def arms_from_differences(base, ca, db):
    """A = base, C = A + ca, B = A + 0.1 (arbitrary), D = B + db; so C−A = ca, D−B = db exactly in reals."""
    A = {s: base[i] for i, s in enumerate(SEEDS)}
    B = {s: base[i] + 0.1 for i, s in enumerate(SEEDS)}
    C = {s: A[s] + ca[i] for i, s in enumerate(SEEDS)}
    D = {s: B[s] + db[i] for i, s in enumerate(SEEDS)}
    return {"A": A, "B": B, "C": C, "D": D}


def synthetic(seed=0, n=10, shift=None):
    rng = np.random.default_rng(seed)
    shift = shift or {"A": 1.60, "B": 1.58, "C": 1.55, "D": 1.52}
    common = rng.normal(0, 0.02, n)
    return {arm: {s: float(shift[arm] + common[s] + rng.normal(0, 0.004)) for s in range(n)} for arm in fs.ARMS}


# ------------------------------------------------------------------ t distribution
@pytest.mark.parametrize("df", [1, 2, 9])
def test_t_critical_matches_independent_quantile(df):
    if df == 1:
        expected = math.tan(math.pi * (0.975 - 0.5))
    elif df == 2:
        q = 0.975
        expected = (2 * q - 1) / math.sqrt(2 * q * (1 - q))
    else:
        expected = t_quantile_numeric(0.975, 9)
    d = [float(i) for i in range(df + 1)]
    assert fs.describe(d)["t_critical"] == pytest.approx(expected, rel=1e-9)


def test_t_critical_df9_reference_value():
    tc = fs.describe([float(i) for i in range(10)])["t_critical"]
    assert round(tc, 6) == 2.262157                                   # printed tables
    # 40-digit reference (mpmath quad + findroot): 2.26215716279820554…. SciPy 1.16.3's t.ppf is
    # accurate to ~2.5e-11 relative here (≈1e-13 nats on a CI bound); p-values (t.sf) to ~1e-15.
    assert tc == pytest.approx(2.2621571627982055, rel=1e-10)
    assert t_p_two_sided_numeric(tc, 9, intervals=20000) == pytest.approx(0.05, abs=1e-10)


def test_hand_worked_three_seed_example():
    d = [-0.02, -0.01, -0.03]
    desc = fs.describe(d)
    assert desc["n"] == 3 and desc["df"] == 2
    assert desc["mean"] == pytest.approx(-0.02, abs=1e-17)
    assert desc["sd"] == pytest.approx(0.01, rel=1e-14)
    assert desc["se"] == pytest.approx(0.01 / math.sqrt(3), rel=1e-14)
    tcrit = 0.95 / math.sqrt(2 * 0.975 * 0.025)
    half = tcrit * 0.01 / math.sqrt(3)
    assert desc["ci_low"] == pytest.approx(-0.02 - half, rel=1e-13)
    assert desc["ci_high"] == pytest.approx(-0.02 + half, rel=1e-12)
    tt = fs.paired_t_test(d)
    assert tt["t"] == pytest.approx(-2 * math.sqrt(3), rel=1e-13)
    assert tt["p_two_sided"] == pytest.approx(1 - math.sqrt(6 / 7), rel=1e-12)
    assert (desc["n_negative"], desc["n_positive"], desc["n_zero"], desc["n_sharing_sign_of_mean"]) == (3, 0, 0, 3)
    assert fs.sesoi_class(desc["ci_low"], desc["ci_high"]) == "inconclusive"


def test_df1_cauchy_p_value():
    d = [0.01, 0.03]                        # mean 0.02, sd √2·0.01, se 0.01, t = 2
    tt = fs.paired_t_test(d)
    assert tt["t"] == pytest.approx(2.0, rel=1e-13)
    assert tt["p_two_sided"] == pytest.approx(1 - 2 / math.pi * math.atan(2.0), rel=1e-12)


def test_ten_seed_p_value_against_numerical_integration():
    d = [-0.012, -0.004, -0.021, 0.003, -0.015, -0.009, -0.001, -0.018, -0.007, -0.011]
    mean, sd = dec_mean_sd(d)
    t_ref = float(mean / (sd / Decimal(10).sqrt()))
    tt = fs.paired_t_test(d)
    assert tt["df"] == 9
    assert tt["t"] == pytest.approx(t_ref, rel=1e-13)
    assert tt["p_two_sided"] == pytest.approx(t_p_two_sided_numeric(t_ref, 9), abs=1e-10)


def test_t_test_agrees_with_scipy_ttest_1samp_and_ttest_rel():
    v = synthetic(3)
    seeds = fs.analysed_seeds(v, SEEDS)
    for c in fs.INFERENTIAL:
        d = fs.paired_differences(v, c, seeds)
        ref = stats.ttest_1samp(d, 0.0)
        tt = fs.paired_t_test(d)
        assert tt["t"] == pytest.approx(float(ref.statistic), rel=1e-11)
        assert tt["p_two_sided"] == pytest.approx(float(ref.pvalue), rel=1e-9, abs=1e-15)
    x = [v["C"][s] for s in SEEDS]
    y = [v["A"][s] for s in SEEDS]
    rel = stats.ttest_rel(x, y)
    assert fs.paired_t_test(fs.paired_differences(v, "C-A", SEEDS))["p_two_sided"] == pytest.approx(
        float(rel.pvalue), rel=1e-9)


def test_mean_and_sd_exact_against_decimal():
    d = [0.1, 0.2, 0.3000000000000001, -0.7, 1e-9, 0.25, 0.125, -0.0625, 0.33, 0.01]
    mean, sd = dec_mean_sd(d)
    desc = fs.describe(d)
    assert desc["mean"] == pytest.approx(float(mean), rel=1e-15)
    assert desc["sd"] == pytest.approx(float(sd), rel=1e-14)
    assert desc["se"] == pytest.approx(float(sd / Decimal(10).sqrt()), rel=1e-14)


def test_sd_uses_n_minus_one_and_ci_is_symmetric():
    d = [1.0, 2.0, 3.0, 4.0]
    desc = fs.describe(d)
    assert desc["sd"] == pytest.approx(math.sqrt(5 / 3), rel=1e-15)     # ddof=0 would give √1.25
    assert desc["mean"] - desc["ci_low"] == pytest.approx(desc["ci_high"] - desc["mean"], rel=1e-15)


def test_interval_and_test_are_consistent():
    """p < 0.05 exactly when the 95 % interval excludes 0 (same mean/SE, same df)."""
    rng = np.random.default_rng(11)
    for _ in range(300):
        d = list(rng.normal(rng.uniform(-0.02, 0.02), 0.01, 10))
        desc, tt = fs.describe(d), fs.paired_t_test(d)
        excludes = desc["ci_high"] < 0 or desc["ci_low"] > 0
        if abs(tt["p_two_sided"] - 0.05) > 1e-9:
            assert excludes == (tt["p_two_sided"] < 0.05)


# ------------------------------------------------------------------ paired differences
def test_differences_paired_by_seed_and_signed():
    v = {"A": {0: 1.0, 1: 2.0}, "B": {0: 1.5, 1: 2.5}, "C": {0: 0.75, 1: 2.25}, "D": {0: 1.0, 1: 2.0}}
    assert fs.paired_differences(v, "C-A", [0, 1]) == [-0.25, 0.25]
    assert fs.paired_differences(v, "D-B", [0, 1]) == [-0.5, -0.5]
    assert fs.paired_differences(v, "B-A", [0, 1]) == [0.5, 0.5]
    assert fs.paired_differences(v, "D-C", [0, 1]) == [0.25, -0.25]
    assert fs.paired_differences(v, "C-B", [0, 1]) == [-0.75, -0.25]
    # interaction = (D−B) − (C−A) = (D−C) − (B−A)
    assert fs.paired_differences(v, "interaction", [0, 1]) == [-0.25, -0.75]
    assert fs.paired_differences(v, "interaction", [1, 0]) == [-0.75, -0.25]


def test_interaction_identity_and_sign_on_synthetic_data():
    v = synthetic(5)
    i = fs.paired_differences(v, "interaction", SEEDS)
    dc = fs.paired_differences(v, "D-C", SEEDS)
    ba = fs.paired_differences(v, "B-A", SEEDS)
    for a, b, c in zip(i, dc, ba, strict=True):
        assert a == pytest.approx(b - c, abs=1e-15)
    # D−B ≈ −0.06, C−A ≈ −0.05: BatchNorm helps the tree more → interaction negative
    assert fs.describe(i)["mean"] < 0


def test_pairing_is_not_order_dependent_on_dict_insertion():
    v = synthetic(1)
    shuffled = {arm: dict(reversed(list(v[arm].items()))) for arm in fs.ARMS}
    assert fs.to_json(fs.analyse(v, v)) == fs.to_json(fs.analyse(shuffled, shuffled))


# ------------------------------------------------------------------ Holm
def test_holm_textbook_example():
    adj = fs.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["status"] == "ok" and adj["m"] == 3
    assert adj["adjusted"] == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_holm_running_maximum_and_cap():
    adj = fs.holm({"x": 0.02, "y": 0.021, "z": 0.9})["adjusted"]
    assert adj["x"] == pytest.approx(0.06)
    assert adj["y"] == pytest.approx(0.06)            # 2·0.021 = 0.042 raised to 0.06 by monotonicity
    assert adj["z"] == 0.9                            # largest p: multiplier 1
    assert fs.holm({"x": 0.6, "y": 0.7})["adjusted"] == {"x": 1.0, "y": 1.0}


def test_holm_ties_and_m2():
    assert fs.holm({"C-A": 0.01, "D-B": 0.01})["adjusted"] == pytest.approx({"C-A": 0.02, "D-B": 0.02})
    assert fs.holm({"C-A": 0.03, "D-B": 0.001})["adjusted"] == pytest.approx({"C-A": 0.03, "D-B": 0.002})


def test_holm_against_reference_on_random_families():
    rng = np.random.default_rng(2)
    for m in (1, 2, 3, 5):
        for _ in range(200):
            p = {f"h{i}": float(x) for i, x in enumerate(rng.uniform(0, 0.5, m))}
            assert fs.holm(p)["adjusted"] == pytest.approx(holm_reference(p), abs=1e-15)


def test_holm_undefined_member_blocks_family_and_rejects_bad_p():
    adj = fs.holm({"a": 0.01, "b": None})
    assert adj["status"] == "not_computable_undefined_p" and adj["adjusted"] == {"a": None, "b": None}
    with pytest.raises(fs.StatisticsError):
        fs.holm({"a": 1.5})
    with pytest.raises(fs.StatisticsError):
        fs.holm({})


def test_families_adjusted_separately_and_from_t_tests_only():
    v = synthetic(7)
    r = fs.analyse(v, v)["primary"]
    cmp_ = r["comparisons"]
    for family, names in fs.FAMILIES.items():
        raw = {c: cmp_[c]["t_test"]["p_two_sided"] for c in names}
        expected = holm_reference(raw)
        assert r["families"][family]["m"] == len(names)
        for c in names:
            assert cmp_[c]["family"] == family
            assert cmp_[c]["t_test"]["p_holm"] == pytest.approx(expected[c], abs=1e-15)
    assert r["families"]["primary"]["m"] == 2 and r["families"]["secondary"]["m"] == 3
    # sign-flip p-values are never adjusted
    for c in fs.INFERENTIAL:
        assert set(cmp_[c]["sensitivity_sign_flip"]) == {"statistic", "alternative", "n", "n_patterns",
                                                          "n_extreme", "p_two_sided"}


# ------------------------------------------------------------------ sign flip
@pytest.mark.parametrize("n", range(1, 9))
def test_sign_flip_exhaustive_against_bitmask_reference(n):
    rng = np.random.default_rng(100 + n)
    for _ in range(5):
        d = [float(x) for x in rng.normal(-0.005, 0.01, n)]
        res = fs.sign_flip_exact(d)
        ref = sign_flip_reference(d)
        assert res["n_patterns"] == 2 ** n
        assert Fraction(res["n_extreme"], res["n_patterns"]) == ref
        assert res["p_two_sided"] == float(ref)


def test_sign_flip_exact_ties():
    assert fs.sign_flip_exact([1.0, 1.0, 1.0, -1.0])["n_extreme"] == 10      # |sum| >= 2 for 10 of 16
    # 0.1 + 0.2 − 0.3 is not exactly 0 for the stored binary values: the exact rational answer is used
    assert Fraction(fs.sign_flip_exact([0.1, 0.2, -0.3])["n_extreme"], 8) == sign_flip_reference([0.1, 0.2, -0.3])
    assert fs.sign_flip_exact([0.25, 0.5, -0.75])["p_two_sided"] == 1.0      # exact zero sum: all patterns


def test_sign_flip_sums_are_exact_not_floating_point():
    # exact |sum| = 1 + 2**-52 is reached only by the identity and the full flip; in float64
    # 1 + 2**-53 rounds to 1, so a floating-point enumeration miscounts the ties
    d = [1.0, 2.0 ** -53, 2.0 ** -53]
    assert sign_flip_reference(d) == Fraction(2, 8)
    assert fs.sign_flip_exact(d)["n_extreme"] == 2


def test_sign_flip_all_same_sign_distinct_ten():
    d = [-0.001 * (i + 1) for i in range(10)]
    res = fs.sign_flip_exact(d)
    assert (res["n_extreme"], res["n_patterns"], res["p_two_sided"]) == (2, 1024, 2 / 1024)


def test_sign_flip_with_zero_differences():
    res = fs.sign_flip_exact([0.0, 0.0, -0.01])
    assert res["n_extreme"] == 8                       # zeros flip to themselves; |±0.01| always extreme
    assert fs.sign_flip_exact([0.0, 0.0])["p_two_sided"] == 1.0


def test_sign_flip_ten_seed_reference():
    d = [-0.012, -0.004, -0.021, 0.003, -0.015, -0.009, -0.001, -0.018, -0.007, -0.011]
    assert Fraction(fs.sign_flip_exact(d)["n_extreme"], 1024) == sign_flip_reference(d)


def test_sign_flip_size_limit():
    with pytest.raises(fs.StatisticsError):
        fs.sign_flip_exact([0.1] * 17)
    with pytest.raises(fs.StatisticsError):
        fs.sign_flip_exact([])


# ------------------------------------------------------------------ SESOI
NEG = math.nextafter(-0.01, -math.inf)       # just below −0.01
POS = math.nextafter(0.01, math.inf)         # just above +0.01
IN_NEG = math.nextafter(-0.01, 0.0)          # just inside
IN_POS = math.nextafter(0.01, 0.0)


@pytest.mark.parametrize(("lo", "hi", "expected"), [
    (-0.05, NEG, "meaningful_negative"),
    (-0.05, -0.01, "inconclusive"),            # strict: upper bound exactly −SESOI is not meaningful
    (-0.01, -0.01, "practically_equivalent"),  # degenerate interval on the boundary: inclusive
    (POS, 0.05, "meaningful_positive"),
    (0.01, 0.05, "inconclusive"),
    (-0.01, 0.01, "practically_equivalent"),
    (NEG, 0.0, "inconclusive"),
    (0.0, POS, "inconclusive"),
    (IN_NEG, IN_POS, "practically_equivalent"),
    (-0.03, 0.03, "inconclusive"),
    (-0.002, 0.004, "practically_equivalent"),
])
def test_sesoi_boundaries(lo, hi, expected):
    assert fs.sesoi_class(lo, hi) == expected


def test_sesoi_invalid():
    with pytest.raises(fs.StatisticsError):
        fs.sesoi_class(0.02, 0.01)
    with pytest.raises(fs.StatisticsError):
        fs.sesoi_class(float("nan"), 0.01)
    assert fs.SESOI == 0.01


def test_sesoi_favoured_arm_in_report():
    base = [1.5 + 0.01 * i for i in range(10)]
    ca = [-0.05 + 0.001 * (i % 3) for i in range(10)]
    db = [0.04 + 0.001 * (i % 2) for i in range(10)]
    v = arms_from_differences(base, ca, db)
    cmp_ = fs.analyse(v, v)["primary"]["comparisons"]
    assert cmp_["C-A"]["sesoi"] == {"sesoi": 0.01, "class": "meaningful_negative", "favoured_arm": "C"}
    assert cmp_["D-B"]["sesoi"] == {"sesoi": 0.01, "class": "meaningful_positive", "favoured_arm": "B"}
    assert cmp_["interaction"]["sesoi"]["favoured_arm"] is None


# ------------------------------------------------------------------ zero variance
def test_zero_variance_differences():
    d = [-0.02] * 10
    desc = fs.describe(d)
    assert desc["sd"] == 0.0 and desc["se"] == 0.0 and desc["zero_variance"] is True
    assert desc["ci_low"] == desc["ci_high"] == -0.02
    tt = fs.paired_t_test(d)
    assert tt == {"t": None, "df": 9, "p_two_sided": None, "status": "undefined_zero_variance"}
    assert fs.sign_flip_exact(d)["p_two_sided"] == 2 / 1024
    assert fs.paired_t_test([0.0, 0.0, 0.0])["status"] == "undefined_zero_variance"
    assert fs.describe([0.0, 0.0])["n_sharing_sign_of_mean"] is None


def test_zero_variance_in_report_blocks_family_holm_but_reports_rest():
    v = synthetic(4)
    v["C"] = {s: v["A"][s] - 0.25 for s in SEEDS}      # C−A = −0.25 for every seed (exact in binary)
    seeds = fs.analysed_seeds(v, SEEDS)
    assert len(set(fs.paired_differences(v, "C-A", seeds))) == 1
    r = fs.analyse(v, v)
    prim = r["primary"]
    assert prim["comparisons"]["C-A"]["t_test"]["status"] == "undefined_zero_variance"
    assert prim["families"]["primary"]["status"] == "not_computable_undefined_p"
    assert prim["comparisons"]["D-B"]["t_test"]["p_holm"] is None
    assert prim["comparisons"]["D-B"]["t_test"]["p_two_sided"] is not None
    assert prim["families"]["secondary"]["status"] == "ok"
    assert prim["comparisons"]["C-A"]["sesoi"]["class"] == "meaningful_negative"
    json.loads(fs.to_json(r))


def test_near_constant_differences_raise_no_warning():
    d = [0.1, 0.1, 0.1 + 2 ** -55, 0.1]
    tt = fs.paired_t_test(d)        # runs under -W error
    assert tt["status"] == "ok" and math.isfinite(tt["t"])


# ------------------------------------------------------------------ seeds, missing data, exclusions
def test_missing_seed_is_an_error_not_dropped():
    v = synthetic(0)
    del v["D"][4]
    with pytest.raises(fs.StatisticsError, match="D/4"):
        fs.analyse(v, synthetic(0))
    with pytest.raises(fs.StatisticsError, match="D/4"):
        fs.analyse(synthetic(0), v)


def test_missing_arm_extra_seed_and_bad_values():
    v = synthetic(0)
    with pytest.raises(fs.StatisticsError):
        fs.analysed_seeds({k: v[k] for k in "ABC"}, SEEDS)
    w = synthetic(0)
    w["A"][10] = 1.0
    with pytest.raises(fs.StatisticsError, match="A/10"):
        fs.analysed_seeds(w, SEEDS)
    for bad in (float("nan"), float("inf"), True, "1.5", None):
        w = synthetic(0)
        w["B"][2] = bad
        with pytest.raises(fs.StatisticsError):
            fs.analysed_seeds(w, SEEDS)


def test_explicit_exclusion_applies_to_all_arms():
    v = synthetic(9)
    for arm in fs.ARMS:
        v[arm][3] = 99.0                                   # ignored once excluded
    del v["C"][3]                                          # and may be absent
    ex = [fs.Exclusion(3, "D-33", "permanent failure of arm C seed 3; human review")]
    r = fs.analyse(v, v, exclusions=ex)
    assert r["primary"]["seeds"] == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    assert r["exclusions"] == [{"seed": 3, "decision": "D-33",
                                "reason": "permanent failure of arm C seed 3; human review"}]
    for c in fs.CONTRASTS:
        assert r["primary"]["comparisons"][c]["describe"]["n"] == 9
        assert "3" not in r["primary"]["comparisons"][c]["differences"]
        assert r["secondary"]["comparisons"][c]["describe"]["n"] == 9
    assert r["primary"]["comparisons"]["C-A"]["sensitivity_sign_flip"]["n_patterns"] == 512
    assert r["primary"]["comparisons"]["C-A"]["describe"]["t_critical"] == pytest.approx(
        float(stats.t.ppf(0.975, 8)))
    # identical to analysing the nine seeds directly
    nine = [s for s in SEEDS if s != 3]
    sub = {arm: {s: v[arm][s] for s in nine} for arm in fs.ARMS}
    direct = fs.analyse(sub, sub, expected_seeds=nine)
    assert direct["primary"]["comparisons"] == r["primary"]["comparisons"]


def test_exclusion_validation():
    v = synthetic(0)
    for bad in [dict(seed=1, decision="", reason="x"), dict(seed=1, decision="D-33", reason="  "),
                dict(seed=True, decision="D-33", reason="x"), dict(seed="1", decision="D-33", reason="x")]:
        with pytest.raises(fs.StatisticsError):
            fs.Exclusion(**bad)
    with pytest.raises(fs.StatisticsError, match="twice"):
        fs.analysed_seeds(v, SEEDS, [fs.Exclusion(1, "D-33", "r"), fs.Exclusion(1, "D-34", "r")])
    with pytest.raises(fs.StatisticsError, match="not expected"):
        fs.analysed_seeds(v, SEEDS, [fs.Exclusion(12, "D-33", "r")])
    with pytest.raises(fs.StatisticsError, match="at least two"):
        fs.analysed_seeds(v, [0, 1], [fs.Exclusion(1, "D-33", "r")])
    with pytest.raises(fs.StatisticsError):
        fs.analysed_seeds(v, [0, 0, 1])


# ------------------------------------------------------------------ report structure
def _all_keys(obj, acc=None):
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for k, val in obj.items():
            acc.add(k)
            _all_keys(val, acc)
    elif isinstance(obj, list):
        for val in obj:
            _all_keys(val, acc)
    return acc


def test_report_scope():
    v = synthetic(2)
    r = fs.analyse(v, synthetic(8))
    keys = _all_keys({k: val for k, val in r.items() if k != "method"})
    assert not any("main" in k.lower() or "average" in k.lower() for k in keys)
    assert r["method"]["averaged_main_effects"] == "not computed"
    prim, sec = r["primary"], r["secondary"]
    assert prim["endpoint"] == "test_nll" and sec["endpoint"] == "best_dev_nll"
    assert set(prim["comparisons"]) == set(fs.CONTRASTS)
    cb = prim["comparisons"]["C-B"]
    assert cb["family"] == "descriptive"
    assert not {"t_test", "sesoi", "sensitivity_sign_flip"} & set(cb)
    assert "C-B" not in _all_keys(prim["families"]) and all("C-B" not in f["comparisons"]
                                                           for f in prim["families"].values())
    assert not {"t_test", "p_two_sided", "p_holm", "sesoi", "sensitivity_sign_flip", "families"} & _all_keys(sec)
    assert r["method"]["versions"]["scipy"]


def test_primary_and_secondary_use_their_own_endpoint():
    test_v, dev_v = synthetic(2), synthetic(8)
    r = fs.analyse(test_v, dev_v)
    assert r["primary"]["comparisons"]["C-A"]["differences"]["0"] == test_v["C"][0] - test_v["A"][0]
    assert r["secondary"]["comparisons"]["C-A"]["differences"]["0"] == dev_v["C"][0] - dev_v["A"][0]


def test_json_round_trip_exact_and_deterministic():
    v = synthetic(6)
    r = fs.analyse(v, synthetic(16))
    text = fs.to_json(r)
    assert json.loads(text) == r
    assert text == fs.to_json(fs.analyse(synthetic(6), synthetic(16)))
    back = json.loads(text)
    assert back["primary"]["comparisons"]["D-B"]["describe"]["mean"] == r["primary"]["comparisons"]["D-B"][
        "describe"]["mean"]
    with pytest.raises(ValueError):
        fs.to_json({"x": float("nan")})


def test_unknown_comparison():
    with pytest.raises(fs.StatisticsError):
        fs.paired_differences(synthetic(0), "B-C", SEEDS)
    with pytest.raises(fs.StatisticsError):
        fs.describe([0.1])
