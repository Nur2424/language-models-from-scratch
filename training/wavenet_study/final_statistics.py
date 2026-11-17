"""Paired statistical analysis of the final controlled WaveNet experiment (F1).

Implements ``PROJECT_RESEARCH_RECORD.md`` D-31 and its addendum, as specified in
``models/hierarchical/wavenet_spec.md`` §16.6.1. Pure functions over per-seed endpoint values;
no model, data or file access.

Endpoints:
  * ``test_nll`` (primary, inferential): per comparison the per-seed paired differences, mean,
    sample SD, SE, nominal two-sided 95 % paired t interval, two-sided paired t-test; Holm
    adjustment of the t-test p-values within the primary and within the secondary family,
    separately; the exact sign-flip test (mean statistic) as a separate sensitivity analysis,
    never adjusted together with the t-tests; SESOI classification of the interval.
  * ``best_dev_nll`` (secondary, descriptive only): the same descriptive statistics and the
    nominal interval; no p-values, no sign-flip test, no SESOI classification.

Comparisons (``d_s = NLL(X, s) − NLL(Y, s)``; negative favours X):
  primary family    C−A, D−B
  secondary family  B−A, D−C, interaction (D_s − B_s) − (C_s − A_s)
  descriptive       C−B (both factors change: no inference, ever)
No averaged main effect is computed.

Seeds: every expected seed must be present for every arm unless it is explicitly excluded; an
exclusion (a documented human decision) removes that seed from all four arms. Missing data is
an error, never silently dropped.

Numerics: inputs are float64. Paired differences are computed in float64 in the order written
above. Means and sums of squares use ``math.fsum`` (correctly rounded); SD uses the two-pass
formula with divisor n − 1. The t quantile and the t-test come from SciPy (pinned version,
D-31 addendum). The sign-flip test enumerates all 2ⁿ sign patterns with exact rational
arithmetic, so ties are exact. Results contain only JSON-native values; floats round-trip
exactly through ``json`` (shortest repr).
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import scipy
from scipy import stats

ARMS = ("A", "B", "C", "D")
SESOI = 0.01                      # nats per character (D-31, decision 5)
CONFIDENCE = 0.95                 # nominal, per comparison (not simultaneous)
CONTRASTS: dict[str, dict[str, int]] = {
    "C-A": {"C": 1, "A": -1},
    "D-B": {"D": 1, "B": -1},
    "B-A": {"B": 1, "A": -1},
    "D-C": {"D": 1, "C": -1},
    "interaction": {"D": 1, "B": -1, "C": -1, "A": 1},
    "C-B": {"C": 1, "B": -1},
}
FAMILIES: dict[str, tuple[str, ...]] = {"primary": ("C-A", "D-B"), "secondary": ("B-A", "D-C", "interaction")}
DESCRIPTIVE = ("C-B",)
INFERENTIAL = FAMILIES["primary"] + FAMILIES["secondary"]
FAVOURED = {"C-A": ("C", "A"), "D-B": ("D", "B"), "B-A": ("B", "A"), "D-C": ("D", "C"), "C-B": ("C", "B")}
MEANINGS = {
    "C-A": "architecture effect without BatchNorm (one-layer flat MLP vs three-level hierarchical tree)",
    "D-B": "architecture effect with BatchNorm (one-layer flat MLP vs three-level hierarchical tree)",
    "B-A": "BatchNorm effect in the flat MLP",
    "D-C": "BatchNorm effect in the hierarchical tree",
    "interaction": "(D-B) - (C-A) = (D-C) - (B-A): negative means BatchNorm lowers NLL more in the "
                   "hierarchical tree than in the flat MLP",
    "C-B": "descriptive only: architecture and normalisation both change; no causal reading",
}
MAX_SIGN_FLIP_N = 16              # 2**16 patterns; the final experiment has n <= 10
SESOI_CLASSES = ("meaningful_negative", "meaningful_positive", "practically_equivalent", "inconclusive")


class StatisticsError(ValueError):
    """Invalid input, missing data or an undefined quantity requested."""


@dataclass(frozen=True)
class Exclusion:
    """A seed excluded from all four arms by an explicit, documented human decision."""

    seed: int
    decision: str      # e.g. "D-33"
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise StatisticsError(f"exclusion seed must be an int, got {self.seed!r}")
        for name in ("decision", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise StatisticsError(f"an exclusion needs a non-empty {name}")

    def to_json(self) -> dict[str, Any]:
        return {"seed": self.seed, "decision": self.decision, "reason": self.reason}


# ------------------------------------------------------------------ input validation
def _finite_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise StatisticsError(f"{where}: expected a number, got {type(value).__name__}")
    x = float(value)
    if not math.isfinite(x):
        raise StatisticsError(f"{where}: non-finite value {value!r}")
    return x


def analysed_seeds(values: Mapping[str, Mapping[int, float]], expected_seeds: Iterable[int],
                   exclusions: Sequence[Exclusion] = ()) -> list[int]:
    """The seeds used for every comparison: the expected seeds minus the explicit exclusions.

    Raises unless every arm A–D has a finite value for every analysed seed and no arm has a value
    for a seed that is neither expected nor excluded. A value for an excluded seed is ignored.
    """
    expected = list(expected_seeds)
    if len(set(expected)) != len(expected) or any(isinstance(s, bool) or not isinstance(s, int) for s in expected):
        raise StatisticsError(f"expected seeds must be distinct ints, got {expected!r}")
    if set(values) != set(ARMS):
        raise StatisticsError(f"values must be given for exactly the arms {ARMS}, got {sorted(values)}")
    excluded = [e.seed for e in exclusions]
    if len(set(excluded)) != len(excluded):
        raise StatisticsError(f"a seed is excluded twice: {excluded}")
    unknown = sorted(set(excluded) - set(expected))
    if unknown:
        raise StatisticsError(f"excluded seeds {unknown} are not expected seeds")
    seeds = [s for s in expected if s not in set(excluded)]
    if len(seeds) < 2:
        raise StatisticsError(f"at least two paired seeds are needed, got {len(seeds)}")
    missing, extra = [], []
    for arm in ARMS:
        have = values[arm]
        missing += [f"{arm}/{s}" for s in seeds if s not in have]
        extra += [f"{arm}/{s}" for s in have if s not in set(expected)]
        for s in seeds:
            if s in have:
                _finite_float(have[s], f"{arm} seed {s}")
    if missing:
        raise StatisticsError("missing values (exclude a seed explicitly, for all arms, or supply it): "
                              + ", ".join(missing))
    if extra:
        raise StatisticsError("values for seeds that are not expected: " + ", ".join(extra))
    return seeds


def paired_differences(values: Mapping[str, Mapping[int, float]], comparison: str,
                       seeds: Sequence[int]) -> list[float]:
    """d_s per seed, in float64, in the order of the contrast's definition."""
    if comparison not in CONTRASTS:
        raise StatisticsError(f"unknown comparison {comparison!r}")
    v = {arm: {s: _finite_float(values[arm][s], f"{arm} seed {s}") for s in seeds} for arm in ARMS}
    if comparison == "interaction":
        return [(v["D"][s] - v["B"][s]) - (v["C"][s] - v["A"][s]) for s in seeds]
    x, y = FAVOURED[comparison]
    return [v[x][s] - v[y][s] for s in seeds]


# ------------------------------------------------------------------ descriptive statistics
def describe(d: Sequence[float], *, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """n, mean, sample SD (n − 1), SE, the nominal two-sided paired t interval, sign counts."""
    d = [_finite_float(x, "difference") for x in d]
    n = len(d)
    if n < 2:
        raise StatisticsError("at least two paired differences are needed")
    mean = math.fsum(d) / n
    var = math.fsum((x - mean) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    half = t_crit * se
    sign = (mean > 0) - (mean < 0)
    return {
        "n": n, "mean": mean, "sd": sd, "se": se, "df": n - 1, "confidence": confidence,
        "interval": "nominal_two_sided_paired_t_per_comparison",
        "t_critical": t_crit, "ci_low": mean - half, "ci_high": mean + half,
        "zero_variance": sd == 0.0,
        "n_negative": sum(x < 0 for x in d), "n_positive": sum(x > 0 for x in d), "n_zero": sum(x == 0 for x in d),
        "n_sharing_sign_of_mean": None if sign == 0 else sum(((x > 0) - (x < 0)) == sign for x in d),
    }


# ------------------------------------------------------------------ inference
def paired_t_test(d: Sequence[float]) -> dict[str, Any]:
    """Two-sided one-sample t-test of the differences against 0 (= the paired t-test).

    Undefined (t and p ``None``) when the differences have zero variance: then the statistic is
    0/0 or ±∞ and no p-value is reported rather than an invented one.
    """
    desc = describe(d)
    if desc["zero_variance"]:
        return {"t": None, "df": desc["df"], "p_two_sided": None, "status": "undefined_zero_variance"}
    # t from the same correctly rounded mean and SE as the interval (so CI and test cannot disagree);
    # the two-sided p-value from SciPy's t survival function. Agrees with scipy.stats.ttest_1samp
    # (verified in the tests) without its catastrophic-cancellation warning for near-constant data.
    t = desc["mean"] / desc["se"]
    p = float(2.0 * stats.t.sf(abs(t), desc["df"]))
    return {"t": t, "df": desc["df"], "p_two_sided": min(1.0, p), "status": "ok"}


def holm(pvalues: Mapping[str, float | None]) -> dict[str, Any]:
    """Holm step-down adjustment of one family's p-values.

    Sort ascending p₍₁₎ ≤ … ≤ p₍ₘ₎ (ties in the family's own order); adjusted
    p̃₍ᵢ₎ = max_{j≤i} min(1, (m − j + 1) · p₍ⱼ₎). If any p-value of the family is undefined, no
    adjusted value of that family is reported.
    """
    names = list(pvalues)
    if not names:
        raise StatisticsError("a family needs at least one p-value")
    if any(pvalues[n] is None for n in names):
        return {"status": "not_computable_undefined_p", "m": len(names), "adjusted": {n: None for n in names}}
    p = {n: _finite_float(pvalues[n], f"p-value {n}") for n in names}
    if any(not 0.0 <= v <= 1.0 for v in p.values()):
        raise StatisticsError(f"p-values must lie in [0, 1]: {p}")
    m = len(names)
    order = sorted(range(m), key=lambda i: (p[names[i]], i))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[names[i]]))
        adjusted[names[i]] = running
    return {"status": "ok", "m": m, "adjusted": {n: adjusted[n] for n in names}}


def sign_flip_exact(d: Sequence[float]) -> dict[str, Any]:
    """Exact two-sided sign-flip permutation test with the mean statistic (sensitivity analysis).

    p = #{ε ∈ {−1, 1}ⁿ : |Σ εᵢdᵢ| ≥ |Σ dᵢ|} / 2ⁿ, the identity pattern included; sums are exact
    rationals, so ties are counted exactly.
    """
    d = [_finite_float(x, "difference") for x in d]
    n = len(d)
    if not 1 <= n <= MAX_SIGN_FLIP_N:
        raise StatisticsError(f"sign-flip enumeration supports 1..{MAX_SIGN_FLIP_N} differences, got {n}")
    exact = [Fraction(x) for x in d]
    observed = abs(sum(exact, Fraction(0)))
    extreme = sum(1 for signs in itertools.product((1, -1), repeat=n)
                  if abs(sum((s * x for s, x in zip(signs, exact, strict=True)), Fraction(0))) >= observed)
    return {"statistic": "mean", "alternative": "two-sided", "n": n, "n_patterns": 2 ** n,
            "n_extreme": extreme, "p_two_sided": extreme / 2 ** n}


def sesoi_class(ci_low: float, ci_high: float, sesoi: float = SESOI) -> str:
    """Classify a nominal interval against ±SESOI (strict for meaningful, inclusive for equivalence)."""
    lo, hi, s = _finite_float(ci_low, "ci_low"), _finite_float(ci_high, "ci_high"), _finite_float(sesoi, "sesoi")
    if lo > hi or s <= 0:
        raise StatisticsError(f"invalid interval [{lo}, {hi}] or SESOI {s}")
    if hi < -s:
        return "meaningful_negative"
    if lo > s:
        return "meaningful_positive"
    if -s <= lo and hi <= s:
        return "practically_equivalent"
    return "inconclusive"


def _favoured(comparison: str, cls: str) -> str | None:
    if comparison not in FAVOURED or cls not in ("meaningful_negative", "meaningful_positive"):
        return None
    x, y = FAVOURED[comparison]
    return x if cls == "meaningful_negative" else y


# ------------------------------------------------------------------ endpoint analyses
def _comparison_block(values: Mapping[str, Mapping[int, float]], comparison: str, seeds: Sequence[int],
                      *, inferential: bool) -> dict[str, Any]:
    d = paired_differences(values, comparison, seeds)
    block: dict[str, Any] = {"comparison": comparison, "meaning": MEANINGS[comparison],
                             "differences": {str(s): x for s, x in zip(seeds, d, strict=True)},
                             "describe": describe(d)}
    if inferential:
        block["t_test"] = paired_t_test(d)
        cls = sesoi_class(block["describe"]["ci_low"], block["describe"]["ci_high"])
        block["sesoi"] = {"sesoi": SESOI, "class": cls, "favoured_arm": _favoured(comparison, cls)}
        block["sensitivity_sign_flip"] = sign_flip_exact(d)
    return block


def analyse_primary(test_nll: Mapping[str, Mapping[int, float]], *, expected_seeds: Iterable[int],
                    exclusions: Sequence[Exclusion] = ()) -> dict[str, Any]:
    """Primary endpoint (test NLL): inference for the two families, C−B descriptive only."""
    seeds = analysed_seeds(test_nll, expected_seeds, exclusions)
    comparisons = {c: _comparison_block(test_nll, c, seeds, inferential=True) for c in INFERENTIAL}
    comparisons.update({c: _comparison_block(test_nll, c, seeds, inferential=False) for c in DESCRIPTIVE})
    families = {}
    for family, names in FAMILIES.items():
        adj = holm({c: comparisons[c]["t_test"]["p_two_sided"] for c in names})
        families[family] = {"comparisons": list(names), "correction": "holm", "applies_to": "paired t-test p-values",
                            **adj}
        for c in names:
            comparisons[c]["t_test"]["p_holm"] = adj["adjusted"][c]
            comparisons[c]["family"] = family
    for c in DESCRIPTIVE:
        comparisons[c]["family"] = "descriptive"
    return {"endpoint": "test_nll", "role": "primary", "seeds": seeds, "comparisons": comparisons,
            "families": families}


def analyse_secondary(best_dev_nll: Mapping[str, Mapping[int, float]], *, expected_seeds: Iterable[int],
                      exclusions: Sequence[Exclusion] = ()) -> dict[str, Any]:
    """Secondary endpoint (best dev NLL): descriptive statistics and nominal intervals only."""
    seeds = analysed_seeds(best_dev_nll, expected_seeds, exclusions)
    comparisons = {c: _comparison_block(best_dev_nll, c, seeds, inferential=False) for c in CONTRASTS}
    return {"endpoint": "best_dev_nll", "role": "secondary_descriptive", "seeds": seeds, "comparisons": comparisons}


def analyse(test_nll: Mapping[str, Mapping[int, float]], best_dev_nll: Mapping[str, Mapping[int, float]], *,
            expected_seeds: Iterable[int] = range(10), exclusions: Sequence[Exclusion] = ()) -> dict[str, Any]:
    """The complete D-31 report: primary inference, secondary description, method metadata."""
    expected = list(expected_seeds)
    report = {
        "method": {
            "decision": "D-31 and D-31 addendum; wavenet_spec.md section 16.6.1",
            "difference": "d_s = NLL(X, s) - NLL(Y, s); negative favours X; interaction (D-B)-(C-A)",
            "sd": "sample, divisor n - 1", "interval": "nominal two-sided 95% paired t, per comparison, "
                                                      "not simultaneous",
            "inference": "two-sided paired t-test; Holm within the primary and the secondary family separately",
            "sensitivity": "exact sign-flip permutation, mean statistic, reported separately, not adjusted",
            "sesoi": {"value": SESOI, "units": "nats per character",
                      "meaningful": "strict: ci_high < -SESOI or ci_low > +SESOI",
                      "equivalent": "inclusive: -SESOI <= ci_low and ci_high <= +SESOI"},
            "descriptive_only": list(DESCRIPTIVE), "averaged_main_effects": "not computed",
            "versions": {"scipy": scipy.__version__, "numpy": np.__version__},
        },
        "expected_seeds": expected,
        "exclusions": [e.to_json() for e in exclusions],
        "primary": analyse_primary(test_nll, expected_seeds=expected, exclusions=exclusions),
        "secondary": analyse_secondary(best_dev_nll, expected_seeds=expected, exclusions=exclusions),
    }
    return report


def to_json(report: Mapping[str, Any]) -> str:
    """Canonical JSON (sorted keys, non-finite values refused); floats round-trip exactly."""
    return json.dumps(report, sort_keys=True, allow_nan=False, indent=2)
