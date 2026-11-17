"""Context stratification of per-target NLL for the final experiment (F2; RQ7, D-31 addendum).

Three **separate** stratifications of the prediction targets of a :class:`utils.data.ContextWindows`
with k = 8 (no cross-product is ever formed):

``F`` training frequency of the exact eight-position context
    ``c(w)`` = number of training windows whose context equals ``w`` exactly, all eight positions,
    BOS (id 0) included. Training windows are the only source of counts.
    * ``unseen`` (``c = 0``): its own nominal category; never merged with anything.
    * seen (``c ≥ 1``): dev-defined terciles. With the n dev targets of seen context sorted,
      ``c₍₁₎ ≤ … ≤ c₍ₙ₎`` (1-based nearest rank), ``t₁ = c₍⌈n/3⌉₎`` and ``t₂ = c₍⌈2n/3⌉₎``; the
      integer bins are ``low = [1, t₁]``, ``mid = [t₁+1, t₂]``, ``high = [t₂+1, ∞)``. Bins are
      defined on count *values*, so tied counts always fall in the same bin; ties can make bins
      unequal or empty.
``R`` real-context length
    the number of non-BOS tokens in the context, ``r = 8 − (leading BOS run) = min(t − 1, 8)`` for the
    target at 1-based position t of its name. Ordinal bins ``[0,0], …, [8,8]``.
``P`` target position
    ``first`` (t = 1), ``eos`` (the EOS target, t = L + 1), ``interior`` (2 ≤ t ≤ L). Nominal.

Finalisation on development data (deterministic; every merge is recorded). A category is
*under-supported* when it has fewer than ``MIN_SUPPORT = 500`` dev targets; an empty category
(0 dev targets) is the extreme case and follows the same rule, so ranges are absorbed rather than
dropped and every later test value still has a category. Ranges stay contiguous and ordered.
  * ``F`` (seen bins low < mid < high): scan from the highest-frequency category downwards; an
    under-supported category merges into its lower-frequency neighbour (the merged category is
    re-examined next). Finally, if ``low`` (or the merged category containing it) is still
    under-supported, it merges upward into its neighbour. One remaining category below 500 is kept
    and flagged. ``unseen`` is never merged; it is flagged if under-supported.
  * ``R`` (lengths 0 … 8): scan from 8 downwards; an under-supported category merges into the next
    lower category. Nothing merges upward: an under-supported lowest category is kept and flagged.
  * ``P`` (first, interior, eos): never merged; under-supported categories are flagged.

The definitions are a JSON document with a SHA-256 over its canonical form; every use verifies the
hash and the fingerprint of the training windows. Applying them to test windows only *assigns*
targets and reports test support (flag below 500); the definitions are never changed. Nothing in
this module reads files or knows split names: callers pass windows; ``define_buckets`` takes only
training and development windows.

Aggregation is token-weighted: per category the number of targets and ``math.fsum`` of their NLLs;
the categories of each stratification partition the targets, and the total is checked against a
direct ``math.fsum`` within the bound ``(K + 1)·(ε/2)·Σ|ℓ|`` (K category sums, each correctly
rounded, then one more correctly rounded sum).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from utils.data import ContextWindows

CONTEXT_LENGTH = 8
MIN_SUPPORT = 500
BOS_ID = 0
EOS_ID = 0
SCHEMA = "wavenet_study.context_buckets/1"
STRATIFICATIONS = ("frequency", "real_context_length", "target_position")
POSITIONS = ("first", "interior", "eos")
_EPS = 2.0 ** -52


class BucketError(ValueError):
    """Invalid windows, definitions or inputs; a tampered or mismatched definition."""


# ------------------------------------------------------------------ windows and fingerprints
def _check_windows(windows: ContextWindows, what: str) -> None:
    if not isinstance(windows, ContextWindows):
        raise BucketError(f"{what} must be a ContextWindows, got {type(windows).__name__}")
    if windows.context_length != CONTEXT_LENGTH:
        raise BucketError(f"{what}: context length must be {CONTEXT_LENGTH}, got {windows.context_length}")
    for name in ("contexts", "targets", "name_index"):
        t = getattr(windows, name)
        if t.dtype != torch.long or t.device.type != "cpu":
            raise BucketError(f"{what}.{name} must be int64 on CPU")
    if windows.num_examples == 0:
        raise BucketError(f"{what} has no windows")


def windows_fingerprint(windows: ContextWindows) -> str:
    """SHA-256 over shapes and bytes of contexts, targets and name_index."""
    _check_windows(windows, "windows")
    h = hashlib.sha256()
    for name in ("contexts", "targets", "name_index"):
        a = np.ascontiguousarray(getattr(windows, name).numpy().astype("<i8"))
        h.update(f"{name}:{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class TrainContextCounts:
    """Exact counts of eight-position training contexts and the fingerprint of the source windows."""

    counts: Mapping[tuple[int, ...], int]
    fingerprint: str
    num_windows: int

    def count(self, context: Sequence[int]) -> int:
        return self.counts.get(tuple(int(x) for x in context), 0)


def count_train_contexts(train: ContextWindows) -> TrainContextCounts:
    _check_windows(train, "train")
    counts = Counter(map(tuple, train.contexts.tolist()))
    return TrainContextCounts(counts=dict(counts), fingerprint=windows_fingerprint(train),
                              num_windows=train.num_examples)


def target_positions(windows: ContextWindows) -> list[int]:
    """1-based position t of every target within its name (rows ordered by name, then t)."""
    ni = windows.name_index.tolist()
    out, prev, t = [], None, 0
    for n in ni:
        if n != prev:
            if prev is not None and n < prev:
                raise BucketError("name_index must be non-decreasing")
            prev, t = n, 0
        t += 1
        out.append(t)
    return out


def target_features(windows: ContextWindows, counts: TrainContextCounts) -> dict[str, list]:
    """Per target: training count of its context, real-context length, position category."""
    _check_windows(windows, "windows")
    ctx = windows.contexts.tolist()
    tgt = windows.targets.tolist()
    pos = target_positions(windows)
    ni = windows.name_index.tolist()
    freq, real, place = [], [], []
    for i, (row, t) in enumerate(zip(ctx, pos, strict=True)):
        lead = 0
        while lead < CONTEXT_LENGTH and row[lead] == BOS_ID:
            lead += 1
        r = CONTEXT_LENGTH - lead
        if r != min(t - 1, CONTEXT_LENGTH) or any(x == BOS_ID for x in row[lead:]):
            raise BucketError(f"window {i}: context inconsistent with its position in the name")
        last_of_name = i + 1 == len(ni) or ni[i + 1] != ni[i]
        if (tgt[i] == EOS_ID) != last_of_name:
            raise BucketError(f"window {i}: EOS target must be exactly the last target of each name")
        freq.append(counts.count(row))
        real.append(r)
        place.append("first" if t == 1 else "eos" if tgt[i] == EOS_ID else "interior")
    return {"frequency": freq, "real_context_length": real, "target_position": place}


# ------------------------------------------------------------------ definitions
def tercile_thresholds(seen_counts: Sequence[int]) -> tuple[int, int] | None:
    """Nearest-rank terciles ``(c₍⌈n/3⌉₎, c₍⌈2n/3⌉₎)`` of positive integer counts; None if empty."""
    xs = sorted(int(c) for c in seen_counts)
    if not xs:
        return None
    if xs[0] < 1:
        raise BucketError("tercile thresholds are defined on seen (count >= 1) contexts only")
    n = len(xs)
    return xs[-(-n // 3) - 1], xs[-(-2 * n // 3) - 1]


def _in_range(v: int, lo: int, hi: int | None) -> bool:
    return lo <= v and (hi is None or v <= hi)


def _finalise_ordinal(initial: list[dict[str, Any]], values: Sequence[int], *, lowest_merges_upward: bool
                      ) -> tuple[list[dict], list[dict]]:
    """Merge under-supported contiguous ordinal bins downwards (see the module docstring).

    Scans from the highest bin to the lowest; an under-supported bin merges into the next lower bin
    and the merged bin is examined next. If ``lowest_merges_upward`` and the lowest bin is still
    under-supported afterwards, it merges into its upper neighbour.
    """
    bins = [dict(b, members=[b["label"]]) for b in initial]
    for b in bins:
        b["dev_support"] = sum(_in_range(v, b["lo"], b["hi"]) for v in values)
    steps: list[dict[str, Any]] = []

    def merge(lower: int, source: int) -> None:          # bins[lower] and bins[lower + 1]
        a, b = bins[lower], bins[lower + 1]
        src, dst = bins[source], bins[lower + 1 if source == lower else lower]
        steps.append({"step": "empty" if src["dev_support"] == 0 else "under_supported",
                      "category": src["label"], "into": dst["label"],
                      "direction": "upward" if source == lower else "downward",
                      "dev_support": [src["dev_support"], dst["dev_support"]]})
        bins[lower:lower + 2] = [{"label": f"{a['label']}+{b['label']}", "lo": a["lo"], "hi": b["hi"],
                                  "members": a["members"] + b["members"],
                                  "dev_support": a["dev_support"] + b["dev_support"]}]

    i = len(bins) - 1
    while i >= 1:
        if bins[i]["dev_support"] < MIN_SUPPORT:
            merge(i - 1, i)                               # into the next lower bin, then re-examine it
        i -= 1
    if lowest_merges_upward and len(bins) > 1 and bins[0]["dev_support"] < MIN_SUPPORT:
        merge(0, 0)
    return bins, steps


def _category(cid: str, kind: str, dev_support: int, **extra: Any) -> dict[str, Any]:
    return {"id": cid, "kind": kind, **extra, "dev_support": dev_support,
            "insufficient_dev_support": dev_support < MIN_SUPPORT}


def define_buckets(train: ContextWindows, dev: ContextWindows) -> dict[str, Any]:
    """Build the frozen definitions from training counts and development targets only."""
    counts = count_train_contexts(train)
    feats = target_features(dev, counts)
    freq = feats["frequency"]
    seen = [c for c in freq if c > 0]
    n_unseen = len(freq) - len(seen)

    thresholds = tercile_thresholds(seen)
    freq_cats = [_category("unseen", "nominal", n_unseen, lo=0, hi=0)]
    freq_steps: list[dict] = []
    if thresholds is not None:
        t1, t2 = thresholds
        initial = [{"label": "low", "lo": 1, "hi": t1}, {"label": "mid", "lo": t1 + 1, "hi": t2},
                   {"label": "high", "lo": t2 + 1, "hi": None}]
        bins, freq_steps = _finalise_ordinal(initial, seen, lowest_merges_upward=True)
        freq_cats += [_category(f"seen:{b['label']}", "ordinal", b["dev_support"], lo=b["lo"], hi=b["hi"],
                                members=b["members"]) for b in bins]
    else:
        # no dev target has a seen context: one seen category covering [1, ∞) keeps test coverage
        freq_cats.append(_category("seen:all", "ordinal", 0, lo=1, hi=None, members=["all"]))

    real = feats["real_context_length"]
    initial_r = [{"label": str(r), "lo": r, "hi": r} for r in range(CONTEXT_LENGTH + 1)]
    rbins, r_steps = _finalise_ordinal(initial_r, real, lowest_merges_upward=False)
    real_cats = [_category(f"r:{b['label']}", "ordinal", b["dev_support"], lo=b["lo"], hi=b["hi"],
                           members=b["members"]) for b in rbins]

    place = Counter(feats["target_position"])
    pos_cats = [_category(p, "nominal", place.get(p, 0)) for p in POSITIONS]

    body = {
        "schema": SCHEMA, "context_length": CONTEXT_LENGTH, "min_support": MIN_SUPPORT,
        "sources": {"train_fingerprint": counts.fingerprint, "train_windows": counts.num_windows,
                    "dev_fingerprint": windows_fingerprint(dev), "dev_windows": dev.num_examples},
        "frequency": {"tercile_thresholds": list(thresholds) if thresholds else None,
                      "dev_seen": len(seen), "dev_unseen": n_unseen,
                      "steps": freq_steps, "categories": freq_cats},
        "real_context_length": {"steps": r_steps, "categories": real_cats},
        "target_position": {"steps": [], "categories": pos_cats},
    }
    return {**body, "definition_sha256": _digest(body)}


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(body)).hexdigest()


def to_json(definitions: Mapping[str, Any]) -> str:
    verify_definitions(definitions)
    return json.dumps(definitions, sort_keys=True, indent=2, allow_nan=False)


def from_json(text: str) -> dict[str, Any]:
    d = json.loads(text)
    verify_definitions(d)
    return d


def verify_definitions(definitions: Mapping[str, Any]) -> None:
    """Raise unless the stored hash matches and the categories are structurally valid."""
    if not isinstance(definitions, Mapping) or "definition_sha256" not in definitions:
        raise BucketError("definitions lack definition_sha256")
    body = {k: v for k, v in definitions.items() if k != "definition_sha256"}
    if _digest(body) != definitions["definition_sha256"]:
        raise BucketError("definition hash mismatch: the frozen definitions were modified")
    if body.get("schema") != SCHEMA or body.get("context_length") != CONTEXT_LENGTH \
            or body.get("min_support") != MIN_SUPPORT:
        raise BucketError("unexpected schema, context length or minimum support")
    for strat, domain in (("frequency", (1, None)), ("real_context_length", (0, CONTEXT_LENGTH))):
        cats = [c for c in body[strat]["categories"] if c["kind"] == "ordinal"]
        if not cats or cats[0]["lo"] != domain[0] or cats[-1]["hi"] != domain[1] \
                or any(a["hi"] is None or b["lo"] != a["hi"] + 1 for a, b in zip(cats, cats[1:], strict=False)) \
                or any(c["hi"] is not None and c["lo"] > c["hi"] for c in cats):
            raise BucketError(f"{strat}: ordinal categories must partition {domain}")
    if [c["id"] for c in body["target_position"]["categories"]] != list(POSITIONS):
        raise BucketError("target_position categories must be exactly first, interior, eos")
    if [c["id"] for c in body["frequency"]["categories"] if c["kind"] == "nominal"] != ["unseen"]:
        raise BucketError("frequency must contain exactly one unseen category")


# ------------------------------------------------------------------ application
def assign(definitions: Mapping[str, Any], windows: ContextWindows, counts: TrainContextCounts) -> dict[str, Any]:
    """Assign every target of ``windows`` to one category per stratification (definitions unchanged).

    ``counts`` must come from the same training windows the definitions were built from.
    Returns per-stratification category ids (aligned with the rows of ``windows``), the support of
    each category in these windows and an ``insufficient_support`` flag (support < 500); the flags
    describe these windows only and never alter the definitions.
    """
    verify_definitions(definitions)
    if counts.fingerprint != definitions["sources"]["train_fingerprint"]:
        raise BucketError("training counts do not match the training windows of the definitions")
    feats = target_features(windows, counts)
    labels: dict[str, list[str]] = {}
    for strat in ("frequency", "real_context_length"):
        cats = definitions[strat]["categories"]
        out = []
        for v in feats[strat]:
            if strat == "frequency" and v == 0:
                out.append("unseen")
                continue
            hit = [c["id"] for c in cats if c["kind"] == "ordinal" and _in_range(v, c["lo"], c["hi"])]
            if len(hit) != 1:
                raise BucketError(f"{strat}: value {v} matches {len(hit)} categories")
            out.append(hit[0])
        labels[strat] = out
    labels["target_position"] = list(feats["target_position"])
    support = {s: {c["id"]: 0 for c in definitions[s]["categories"]} for s in STRATIFICATIONS}
    for s in STRATIFICATIONS:
        for cid in labels[s]:
            support[s][cid] += 1
    return {
        "definition_sha256": definitions["definition_sha256"],
        "windows_fingerprint": windows_fingerprint(windows),
        "num_targets": windows.num_examples,
        "labels": labels,
        "support": {s: {cid: {"n": n, "insufficient_support": n < MIN_SUPPORT} for cid, n in sup.items()}
                    for s, sup in support.items()},
    }


def aggregate(assignment: Mapping[str, Any], per_target_nll: Sequence[float] | torch.Tensor | np.ndarray
              ) -> dict[str, Any]:
    """Token-weighted per-category NLL sums, counts and means, checked against the overall total."""
    if isinstance(per_target_nll, torch.Tensor):
        per_target_nll = per_target_nll.detach().cpu().double().tolist()
    elif isinstance(per_target_nll, np.ndarray):
        per_target_nll = per_target_nll.astype(np.float64).tolist()
    xs = [float(x) for x in per_target_nll]
    n = assignment["num_targets"]
    if len(xs) != n:
        raise BucketError(f"expected {n} per-target NLLs, got {len(xs)}")
    if any(not math.isfinite(x) or x < 0 for x in xs):
        raise BucketError("per-target NLLs must be finite and non-negative")
    total = math.fsum(xs)
    abs_total = math.fsum(abs(x) for x in xs)
    out: dict[str, Any] = {"num_targets": n, "nll_sum": total, "mean_nll": total / n,
                           "definition_sha256": assignment["definition_sha256"], "stratifications": {}}
    for s in STRATIFICATIONS:
        groups: dict[str, list[float]] = {cid: [] for cid in assignment["support"][s]}
        for cid, x in zip(assignment["labels"][s], xs, strict=True):
            groups[cid].append(x)
        cats = {}
        for cid, g in groups.items():
            ssum = math.fsum(g)
            cats[cid] = {"n": len(g), "nll_sum": ssum, "mean_nll": ssum / len(g) if g else None,
                         "share": len(g) / n,
                         "insufficient_support": assignment["support"][s][cid]["insufficient_support"]}
        count = sum(c["n"] for c in cats.values())
        recombined = math.fsum(c["nll_sum"] for c in cats.values())
        bound = (len(cats) + 1) * (_EPS / 2) * abs_total
        if count != n or abs(recombined - total) > bound:
            raise BucketError(f"{s}: categories do not partition the targets")
        out["stratifications"][s] = {"categories": cats, "recombined_nll_sum": recombined,
                                     "consistency_bound": bound}
    return out
