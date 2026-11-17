"""Shared, model-agnostic research infrastructure for controlled studies.

Stage 1 of the implementation of ``training/wavenet_study/EPOCH_BUDGET_PILOT_PROTOCOL.md``
(v1.1; D-23, D-26): the **pure layer**. Nothing in this module trains a model, launches a
process, takes a lock or writes outside the paths it is explicitly given.

Contents
--------
* Non-finite handling (protocol §10.4). ``utils.io.to_jsonable`` stores NaN / ±∞ as the
  strings ``"nan"`` / ``"inf"`` / ``"-inf"``; every check here treats those strings as
  non-finite numbers, never as ordinary strings.
* Record flattening with **tuple paths** (field names such as ``units_ge_0.99`` and
  parameter names such as ``emb.weight`` contain dots, so dotted strings would be ambiguous).
* Bit-exact record comparison and ``(epoch, step)``-matched step-record comparison (§10.4).
* Deterministic model-state hashing (§10.2, clarification A).
* Field-path classification into categories A / B / C (§9.1, §9.3; approval A2), with
  fail-closed handling of unknown fields and an explicit schema check.
* The append-only, hash-chained category B ledger (§9.7; approval A5). The chain gives
  tamper *evidence*, not immutability.
* Integrity gates G1, G2, G3, G4 (with BatchNorm counter checks, approval A3) and G7, each
  computed independently (§10.3; I-C).

No numerical threshold is introduced anywhere. The only numeric comparisons are exact
equalities required by the protocol (bit patterns, ``== 0.0`` in the step-0 rule, counters
equal to 0 or to recorded step / epoch counts).

The LR-pilot scripts are neither imported nor modified (AM-1).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

FieldPath = tuple[str | int, ...]

# =============================================================================
# 1. Non-finite values (protocol §10.4; finding F1)
# =============================================================================
NONFINITE_STRINGS = frozenset({"nan", "inf", "-inf"})


def is_number(value: Any) -> bool:
    """True for int / float JSON values (bool excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_nonfinite(value: Any) -> bool:
    """True for a non-finite float or for one of the strings ``to_jsonable`` writes for NaN / ±∞."""
    if isinstance(value, float):
        return not math.isfinite(value)
    return isinstance(value, str) and value in NONFINITE_STRINGS


# =============================================================================
# 2. Flattening and bit-exact comparison (protocol §10.4)
# =============================================================================
def flatten(record: Any, prefix: FieldPath = ()) -> dict[FieldPath, Any]:
    """Map every leaf of a JSON-like record to its tuple path.

    Dict keys become ``str`` segments and list indices ``int`` segments, so the two can never
    collide. Empty dicts and empty lists are leaves (value ``{}`` / ``[]``).
    """
    out: dict[FieldPath, Any] = {}
    if isinstance(record, Mapping):
        if not record:
            out[prefix] = {}
        for key, value in record.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string key {key!r} at {format_path(prefix)}")
            out.update(flatten(value, prefix + (key,)))
    elif isinstance(record, list):
        if not record:
            out[prefix] = []
        for i, value in enumerate(record):
            out.update(flatten(value, prefix + (i,)))
    else:
        out[prefix] = record
    return out


def format_path(path: FieldPath) -> str:
    """Unambiguous display form of a path (a JSON list)."""
    return json.dumps(list(path))


def path_sort_key(path: FieldPath) -> tuple:
    return tuple((1, s) if isinstance(s, int) else (0, s) for s in path)


def float_bits(x: float) -> bytes:
    """IEEE-754 binary64 bit pattern (little-endian); distinguishes −0.0 from 0.0."""
    return struct.pack("<d", x)


def values_equal(a: Any, b: Any) -> bool:
    """Exact equality for non-null, finite JSON leaves: same Python type and same value / bits."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return float_bits(a) == float_bits(b)
    return a == b


@dataclass(frozen=True)
class Mismatch:
    """First difference found by a comparison (deterministic order)."""

    kind: Literal["missing_path", "extra_path", "nonfinite", "null", "value", "key", "duplicate_key"]
    path: FieldPath
    reference: Any = None
    candidate: Any = None
    where: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": list(self.path), "reference": _jsonable_leaf(self.reference),
                "candidate": _jsonable_leaf(self.candidate), "where": self.where}


def _jsonable_leaf(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    return v


NullPolicy = Callable[[FieldPath, Mapping[str, Any]], bool]


def no_nulls(_path: FieldPath, _record: Mapping[str, Any]) -> bool:
    return False


def compare_records(reference: Mapping[str, Any], candidate: Mapping[str, Any], *,
                    exclude: Iterable[FieldPath] = (), null_allowed: NullPolicy = no_nulls,
                    where: str = "") -> Mismatch | None:
    """Exact comparison of two records (protocol §10.4).

    * The sets of leaf paths (minus ``exclude``) must be identical.
    * Any non-finite value (float or ``"nan"``/``"inf"``/``"-inf"`` string) on either side fails;
      NaN never matches NaN.
    * ``null`` matches only when both sides are ``null`` and ``null_allowed(path, reference)``.
    * Everything else must have the same type and the same value (floats: same bit pattern).
    """
    excluded = set(exclude)
    ref = {p: v for p, v in flatten(reference).items() if p not in excluded}
    new = {p: v for p, v in flatten(candidate).items() if p not in excluded}
    missing = sorted(set(ref) - set(new), key=path_sort_key)
    if missing:
        return Mismatch("missing_path", missing[0], ref[missing[0]], None, where)
    extra = sorted(set(new) - set(ref), key=path_sort_key)
    if extra:
        return Mismatch("extra_path", extra[0], None, new[extra[0]], where)
    for path in sorted(ref, key=path_sort_key):
        a, b = ref[path], new[path]
        if is_nonfinite(a) or is_nonfinite(b):
            return Mismatch("nonfinite", path, a, b, where)
        if a is None or b is None:
            if a is None and b is None and null_allowed(path, reference):
                continue
            return Mismatch("null", path, a, b, where)
        if not values_equal(a, b):
            return Mismatch("value", path, a, b, where)
    return None


def compare_keyed_records(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], *,
                          key_fields: tuple[str, ...] = ("epoch", "step"), exclude: Iterable[FieldPath] = (),
                          null_allowed: NullPolicy = no_nulls, where: str = "") -> Mismatch | None:
    """Match records by explicit identifiers (clarification 8), never by position.

    Keys must be ints, unique on each side, and the two key sets identical; then every matched
    pair is compared with :func:`compare_records`.
    """
    def index(records: Sequence[Mapping[str, Any]], side: str) -> dict[tuple, Mapping[str, Any]] | Mismatch:
        out: dict[tuple, Mapping[str, Any]] = {}
        for rec in records:
            key = tuple(rec.get(f) for f in key_fields)
            if not all(isinstance(k, int) and not isinstance(k, bool) for k in key):
                return Mismatch("key", tuple(key_fields), key if side == "reference" else None,
                                key if side == "candidate" else None, where)
            if key in out:
                return Mismatch("duplicate_key", tuple(key_fields), key if side == "reference" else None,
                                key if side == "candidate" else None, where)
            out[key] = rec
        return out

    ref = index(reference, "reference")
    if isinstance(ref, Mismatch):
        return ref
    new = index(candidate, "candidate")
    if isinstance(new, Mismatch):
        return new
    missing = sorted(set(ref) - set(new))
    if missing:
        return Mismatch("missing_path", tuple(key_fields), missing[0], None, where)
    extra = sorted(set(new) - set(ref))
    if extra:
        return Mismatch("extra_path", tuple(key_fields), None, extra[0], where)
    for key in sorted(ref):
        m = compare_records(ref[key], new[key], exclude=exclude, null_allowed=null_allowed,
                            where=f"{where}{dict(zip(key_fields, key, strict=True))}")
        if m is not None:
            return m
    return None


def compare_run_prefix(*, reference_history: Sequence[Mapping[str, Any]], reference_steps: Sequence[Mapping[str, Any]],
                       reference_hash: str, candidate_history: Sequence[Mapping[str, Any]],
                       candidate_steps: Sequence[Mapping[str, Any]], candidate_hash: str, epochs: int,
                       history_exclude: Iterable[FieldPath] = (("seconds",),),
                       history_null_allowed: NullPolicy = no_nulls,
                       steps_null_allowed: NullPolicy = no_nulls) -> Mismatch | None:
    """G5 / G6 comparison of the first ``epochs`` epochs of two runs (protocol §10.4).

    Compares (1) history rows with ``epoch`` 0 … epochs − 1, matched by ``epoch``, every leaf
    except ``history_exclude``; (2) step records with ``epoch`` < epochs, matched by
    ``(epoch, step)``; (3) the state hashes after epoch ``epochs``. Records of later epochs on
    either side are ignored; fewer records on either side is a mismatch.
    """
    def prefix(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [r for r in rows if isinstance(r.get("epoch"), int) and r["epoch"] < epochs]

    ref_h, new_h = prefix(reference_history), prefix(candidate_history)
    for side, rows in (("reference", ref_h), ("candidate", new_h)):
        found = sorted(r["epoch"] for r in rows)
        if found != list(range(epochs)):
            return Mismatch("key", ("epoch",), list(range(epochs)), found, f"history ({side}) epochs")
    m = compare_keyed_records(ref_h, new_h, key_fields=("epoch",), exclude=history_exclude,
                              null_allowed=history_null_allowed, where="history ")
    if m is not None:
        return m
    m = compare_keyed_records(prefix(reference_steps), prefix(candidate_steps), key_fields=("epoch", "step"),
                              null_allowed=steps_null_allowed, where="steps ")
    if m is not None:
        return m
    if not (isinstance(reference_hash, str) and reference_hash == candidate_hash):
        return Mismatch("value", ("state_sha256",), reference_hash, candidate_hash, f"state after epoch {epochs}")
    return None


# =============================================================================
# 3. State hashing (protocol §10.2; clarification A)
# =============================================================================
STATE_HASH_FILE = "state_hashes.jsonl"


def state_hash(state: Mapping[str, Any]) -> str:
    """SHA-256 over a ``state_dict``-like mapping (protocol §10.2).

    Entries in ascending name order. Per entry: ``name``, ``str(dtype)`` and the JSON list of
    the shape, each UTF-8 encoded and NUL-terminated; then the byte length of the data as an
    8-byte little-endian integer; then the exact bytes of ``tensor.detach().cpu().contiguous()``
    in little-endian element order.
    """
    import torch

    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str):
            raise TypeError(f"state key {name!r} is not a string")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor ({type(tensor).__name__})")
        t = tensor.detach().to("cpu").contiguous()
        arr = t.numpy()
        arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
        data = arr.tobytes(order="C")
        h.update(name.encode("utf-8") + b"\0")
        h.update(str(t.dtype).encode("utf-8") + b"\0")
        h.update(json.dumps(list(t.shape)).encode("utf-8") + b"\0")
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def model_state_hash(model: Any) -> str:
    """Hash of ``model.state_dict()`` (all parameters and persistent buffers)."""
    return state_hash(model.state_dict())


def nonpersistent_buffers(model: Any) -> list[str]:
    """Names of buffers excluded from ``state_dict()``; the study requires none (protocol §10.2)."""
    in_state = set(model.state_dict())
    return sorted(name for name, _ in model.named_buffers() if name not in in_state)


def load_payload(path: str | Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    """Load a checkpoint written by ``utils.checkpoint`` (``weights_only=True``)."""
    from utils.checkpoint import load_checkpoint

    return load_checkpoint(path, map_location="cpu", weights_only=True, expected_kind=expected_kind)


def checkpoint_state_hash(path: str | Path) -> str:
    return state_hash(load_payload(path)["model_state"])


# =============================================================================
# 4. Field-path classification (protocol §9.1, §9.3; approval A2)
# =============================================================================
Category = Literal["A", "B", "C"]
ValueKind = Literal["number", "int", "bool", "counter", "buffer_counter", "empty_mapping", "never",
                    "structural_null"]

_ACT = r"(hidden|bn|act)[1-9]?"            # leaf modules observed by the activation monitor
_GRAD = r"(emb|hidden|bn|out)[1-9]?"       # gradient groups (group_by="module")
_PARAM = r"(emb|hidden|bn|out)[1-9]?\.(weight|bias)"
_LEVEL = r"level[1-9]"
_SUMMARY = r"mean|std|min|max|norm"


@dataclass(frozen=True)
class FieldRule:
    """One row of the classification table: a path pattern (one regex per segment, full match)."""

    record: Literal["history", "steps"]
    pattern: tuple[str, ...]
    category: Category
    kind: ValueKind
    meaning: str

    def matches(self, record: str, path: FieldPath) -> bool:
        return (record == self.record and len(path) == len(self.pattern)
                and all(isinstance(seg, str) and re.fullmatch(rx, seg)
                        for rx, seg in zip(self.pattern, path, strict=True)))


# Exhaustive table for the record formats written by the accepted trainers of Models 3–5
# (train_mlp, train_batchnorm_mlp, train_wavenet) through the shared diagnostics utilities,
# with step activations off (option B). Verified against all 12 LR-0.03 reference runs.
ACCEPTED_TRAINER_FIELD_RULES: tuple[FieldRule, ...] = (
    # ---------------- history rows ----------------
    FieldRule("history", ("epoch",), "C", "int", "0-based epoch index"),
    FieldRule("history", ("lr",), "C", "number", "learning rate of the epoch (configuration)"),
    FieldRule("history", ("steps",), "C", "int", "optimiser steps completed"),
    FieldRule("history", ("is_best",), "C", "bool", "strict dev improvement flag"),
    FieldRule("history", ("seconds",), "C", "number", "wall-clock epoch time"),
    FieldRule("history", ("running_train_nll",), "A", "number", "running train NLL"),
    FieldRule("history", ("train_nll",), "A", "number", "full-train NLL (fixed parameters)"),
    FieldRule("history", ("dev_nll",), "A", "number", "full-dev NLL (fixed parameters)"),
    FieldRule("history", ("activations", _ACT, _SUMMARY), "A", "number", "dev activation summary"),
    FieldRule("history", ("activations", _ACT, "nan_count|inf_count"), "A", "counter", "non-finite activations"),
    FieldRule("history", ("activations", _ACT, "numel|calls|skipped_calls"), "C", "int", "monitor bookkeeping"),
    FieldRule("history", ("activations", _ACT, "zero_fraction"), "B", "number", "fraction of exact zeros"),
    FieldRule("history", ("activations", _ACT, "saturation_fraction"), "B", "number", "tanh saturation fraction"),
    FieldRule("history", ("activations", "per_unit_saturation", "threshold"), "C", "number", "saturation threshold"),
    FieldRule("history", ("activations", "per_unit_saturation", _LEVEL, "threshold"), "C", "number",
              "saturation threshold"),
    FieldRule("history", ("activations", "per_unit_saturation", "min|median|max"), "B", "number",
              "per-unit saturation fractions"),
    FieldRule("history", ("activations", "per_unit_saturation", _LEVEL, "min|median|max"), "B", "number",
              "per-unit saturation fractions"),
    FieldRule("history", ("activations", "per_unit_saturation", r"units_ge_0\.99"), "B", "int",
              "units saturated ≥ 0.99"),
    FieldRule("history", ("activations", "per_unit_saturation", _LEVEL, r"units_ge_0\.99"), "B", "int",
              "units saturated ≥ 0.99"),
    FieldRule("history", ("output", "logit_mean|logit_std|mean_entropy|mean_max_prob|weight_norm|bias_norm"),
              "A", "number", "dev output summary"),
    FieldRule("history", ("levels", _LEVEL, "w_row_norm", "mean|min|max"), "A", "number", "W_ℓ row norms"),
    FieldRule("history", ("batchnorm",), "C", "structural_null", "no BatchNorm in this arm"),
    FieldRule("history", ("batchnorm", "gamma|beta", "mean|std|min|max"), "A", "number", "γ / β summary"),
    FieldRule("history", ("batchnorm", _LEVEL, "gamma|beta", "mean|std|min|max"), "A", "number", "γ / β summary"),
    FieldRule("history", ("batchnorm", "w1_row_norm", "mean|min|max"), "A", "number", "W₁ row norms"),
    FieldRule("history", ("batchnorm", "num_calibrations|num_batches_tracked"), "A", "buffer_counter",
              "BatchNorm counter buffer"),
    FieldRule("history", ("batchnorm", _LEVEL, "num_calibrations|num_batches_tracked"), "A", "buffer_counter",
              "BatchNorm counter buffer"),
    FieldRule("history", ("batchnorm", "ema_vs_inference",
                          "mean_abs_z|max_abs_z|median_abs_log_var_ratio|max_abs_log_var_ratio"),
              "B", "number", "EMA vs inference statistics"),
    FieldRule("history", ("batchnorm", _LEVEL, "ema_vs_inference",
                          "mean_abs_z|max_abs_z|median_abs_log_var_ratio|max_abs_log_var_ratio"),
              "B", "number", "EMA vs inference statistics"),
    # ---------------- step records ----------------
    FieldRule("steps", ("epoch",), "C", "int", "0-based epoch index"),
    FieldRule("steps", ("step",), "C", "int", "0-based global optimiser step"),
    FieldRule("steps", ("lr",), "C", "number", "learning rate (configuration)"),
    FieldRule("steps", ("batch_size",), "C", "int", "actual batch size"),
    FieldRule("steps", ("loss",), "A", "number", "mini-batch loss"),
    FieldRule("steps", ("grad_global_norm",), "A", "number", "global gradient norm"),
    FieldRule("steps", ("gradients", _GRAD, _SUMMARY), "A", "number", "gradient summary per module"),
    FieldRule("steps", ("gradients", _GRAD, "nan_count|inf_count"), "A", "counter", "non-finite gradients"),
    FieldRule("steps", ("gradients", _GRAD, "numel"), "C", "int", "gradient element count"),
    FieldRule("steps", ("gradients", _GRAD, "zero_fraction"), "B", "number", "fraction of exact zeros"),
    FieldRule("steps", ("nonfinite_gradients",), "A", "empty_mapping", "must be {}"),
    FieldRule("steps", ("nonfinite_gradients", r".+"), "A", "never", "a non-finite gradient was reported"),
    FieldRule("steps", ("updates", "per_parameter", _PARAM, "update_norm|param_norm|update_std|update_max_abs"),
              "A", "number", "parameter / update magnitude"),
    FieldRule("steps", ("updates", "per_parameter", _PARAM, "nonfinite_update"), "A", "counter",
              "non-finite update elements"),
    FieldRule("steps", ("updates", "per_parameter", _PARAM, "relative_update|log10_update_to_param_std"),
              "B", "number", "update ratio"),
    FieldRule("steps", ("updates", "global", "update_norm|param_norm"), "A", "number", "global magnitude"),
    FieldRule("steps", ("updates", "global", "relative_update"), "B", "number", "global update ratio"),
)


@dataclass(frozen=True)
class Finding:
    """A classified observation in one record (A evidence, B event candidate, notable C)."""

    category: Category
    kind: str
    record: Literal["history", "steps"]
    path: FieldPath
    value: Any
    reason: str
    epoch: int | None = None
    step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = list(self.path)
        d["value"] = _jsonable_leaf(self.value)
        return d


class ClassificationError(ValueError):
    """The rule table itself is inconsistent (a path matched by more than one rule)."""


def rule_for(record: str, path: FieldPath,
             rules: Sequence[FieldRule] = ACCEPTED_TRAINER_FIELD_RULES) -> FieldRule | None:
    """The unique rule matching ``path``; ``None`` if none; :class:`ClassificationError` if several."""
    return _rule_for_cached(tuple(rules), record, path)


@functools.lru_cache(maxsize=65536)
def _rule_for_cached(rules: tuple[FieldRule, ...], record: str, path: FieldPath) -> FieldRule | None:
    hits = [r for r in rules if r.matches(record, path)]
    if len(hits) > 1:
        raise ClassificationError(f"{record} {format_path(path)} matches {len(hits)} rules")
    return hits[0] if hits else None


def _step0_structural_null(path: FieldPath, rec: Mapping[str, Any]) -> bool:
    """Protocol §9.3 rule 1: at step 0 only, a null update ratio whose same-parameter record has
    ``update_std == 0.0`` or ``param_norm == 0.0`` exactly (zero-output initialisation)."""
    if not (len(path) == 4 and path[:2] == ("updates", "per_parameter") and path[3] == "log10_update_to_param_std"):
        return False
    if rec.get("step") != 0 or rec.get("epoch") != 0:
        return False
    entry = rec.get("updates", {}).get("per_parameter", {}).get(path[2], {})
    return _is_positive_zero(entry.get("update_std")) or _is_positive_zero(entry.get("param_norm"))


def _is_positive_zero(v: Any) -> bool:
    """Exactly ``0.0`` (a float with the bit pattern of +0.0)."""
    return isinstance(v, float) and float_bits(v) == float_bits(0.0)


def step_null_allowed(path: FieldPath, rec: Mapping[str, Any]) -> bool:
    """Null policy for step records in G5/G6: only the §9.3 step-0 structural case."""
    return _step0_structural_null(path, rec)


def history_null_allowed_for(normalization: str) -> NullPolicy:
    """Null policy for history rows in G5/G6: only ``batchnorm: null`` in an arm without BatchNorm."""
    def allowed(path: FieldPath, _rec: Mapping[str, Any]) -> bool:
        return normalization == "none" and path == ("batchnorm",)
    return allowed


def classify_record(record_kind: Literal["history", "steps"], record: Mapping[str, Any], *, normalization: str,
                    rules: Sequence[FieldRule] = ACCEPTED_TRAINER_FIELD_RULES) -> list[Finding]:
    """Classify every leaf of one record (fail-closed).

    Returns category A findings (evidence of mechanical instability), category B findings
    (diagnostic computation failures) and notable category C observations (structural nulls,
    ε-regularised ratios). Ordinary, valid leaves produce no finding. An unclassified path is
    category B; nothing passes silently.
    """
    epoch = record.get("epoch") if isinstance(record.get("epoch"), int) else None
    step = record.get("step") if isinstance(record.get("step"), int) else None
    out: list[Finding] = []

    def add(cat: Category, kind: str, path: FieldPath, value: Any, reason: str) -> None:
        out.append(Finding(cat, kind, record_kind, path, value, reason, epoch, step))

    for path, value in sorted(flatten(record).items(), key=lambda kv: path_sort_key(kv[0])):
        rule = rule_for(record_kind, path, rules)
        if rule is None:
            add("B", "unclassified_field", path, value, "no classification rule matches this field")
            continue
        if rule.kind == "never":
            add("A", "nonfinite_gradient_reported", path, value, rule.meaning)
            continue
        if rule.kind == "empty_mapping":
            if value != {}:
                add("A", "nonfinite_gradient_reported", path, value, "nonfinite_gradients must be {}")
            continue
        if rule.kind == "structural_null":
            if value is None and normalization == "none":
                add("C", "structural_null", path, value, "batchnorm is null in an arm without BatchNorm (§9.3 rule 2)")
            else:
                add("B", "unexpected_null_or_value", path, value, "batchnorm must be a mapping in a BatchNorm arm")
            continue
        if is_nonfinite(value):
            if rule.category == "A":
                add("A", "nonfinite_model_quantity", path, value, rule.meaning)
            else:
                add("B", "nonfinite_derived_or_structural", path, value, rule.meaning)
            continue
        if value is None:
            if rule_kind_allows_step0_null(rule, path) and _step0_structural_null(path, record):
                add("C", "structural_null", path, None, "step-0 update ratio undefined (§9.3 rule 1)")
            else:
                add("B", "undefined_value", path, None, f"null in {rule.category}-field ({rule.meaning})")
            continue
        if not _type_ok(rule.kind, value):
            add("B", "unexpected_type", path, value, f"expected {rule.kind}")
            continue
        if rule.kind == "counter" and value != 0:
            add("A", "nonfinite_count", path, value, rule.meaning)
            continue
        # ε-regularised ratios and zero parameter norms (§9.3 further rules)
        if (record_kind == "steps" and len(path) == 4 and path[:2] == ("updates", "per_parameter")
                and path[3] in ("param_norm", "relative_update")):
            entry = record["updates"]["per_parameter"][path[2]]
            zero_norm = _is_positive_zero(entry.get("param_norm"))
            if zero_norm and path[3] == "relative_update":
                if step == 0:
                    add("C", "epsilon_regularised", path, value,
                        "relative_update with param_norm == 0.0 (reported as recorded)")
                else:
                    add("B", "epsilon_regularised_after_step0", path, value, "param_norm == 0.0 at a step > 0")
            if zero_norm and path[3] == "param_norm" and step != 0:
                add("B", "zero_param_norm_after_step0", path, value, "param_norm == 0.0 at a step > 0 (§9.3)")
    return out


def rule_kind_allows_step0_null(rule: FieldRule, path: FieldPath) -> bool:
    return rule.record == "steps" and len(path) == 4 and path[3] == "log10_update_to_param_std"


def _type_ok(kind: ValueKind, value: Any) -> bool:
    if kind == "number":
        return is_number(value)
    if kind in ("int", "counter", "buffer_counter"):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if kind == "bool":
        return isinstance(value, bool)
    return False


def record_schema(records: Iterable[Mapping[str, Any]]) -> frozenset[FieldPath]:
    """Union of leaf paths; raises if the records do not share one schema."""
    schemas = {frozenset(flatten(r)) for r in records}
    if len(schemas) != 1:
        raise ValueError(f"records have {len(schemas)} different schemas")
    return next(iter(schemas))


def schema_findings(record_kind: Literal["history", "steps"], record: Mapping[str, Any],
                    schema: frozenset[FieldPath]) -> list[Finding]:
    """Category B findings for fields missing from or added to the frozen reference schema."""
    paths = set(flatten(record))
    epoch = record.get("epoch") if isinstance(record.get("epoch"), int) else None
    step = record.get("step") if isinstance(record.get("step"), int) else None
    out = [Finding("B", "missing_field", record_kind, p, None, "field of the reference schema is missing", epoch, step)
           for p in sorted(schema - paths, key=path_sort_key)]
    out += [Finding("B", "extra_field", record_kind, p, None, "field not in the reference schema", epoch, step)
            for p in sorted(paths - schema, key=path_sort_key)]
    return out


# =============================================================================
# 5. Category B ledger (protocol §9.7; approval A5)
# =============================================================================
GENESIS = "0" * 64
OUTCOMES = ("accepted_as_expected", "corrected", "rejected")
EVENT_FIELDS = ("event_id", "created_at", "attempt_id", "run_dir", "arm", "seed", "cap", "epoch_1based",
                "global_step", "record", "field", "observed", "reason", "evidence", "status")
RESOLUTION_FIELDS = ("event_id", "outcome", "reviewer", "timestamp", "rationale", "evidence")


class LedgerError(RuntimeError):
    """A ledger file is malformed, out of chain, or a request violates the procedure."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HashChainedLog:
    """Append-only JSONL file in which every line carries the SHA-256 of the previous line.

    Line *n* is ``canonical_json({**entry, "prev_sha256": sha256(line n−1)})``; line 1 uses
    64 zeros. Editing, reordering, inserting or deleting any line (except removing a suffix,
    which shortens the chain and changes its head) breaks verification. This is tamper
    evidence, not immutability: whoever can write the file can rebuild a consistent chain,
    which is why the chain head is also recorded elsewhere (attempt results, summaries).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> tuple[list[dict[str, Any]], str]:
        """Verified entries (without ``prev_sha256``) and the chain head."""
        if not self.path.exists():
            return [], GENESIS
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise LedgerError(f"{self.path.name}: last line is incomplete")
        entries, prev = [], GENESIS
        for n, line in enumerate(raw.decode("utf-8").split("\n")[:-1] if raw else [], start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"{self.path.name} line {n}: not JSON ({exc})") from exc
            if not isinstance(obj, dict) or canonical_json(obj) != line:
                raise LedgerError(f"{self.path.name} line {n}: not in canonical form")
            if obj.get("prev_sha256") != prev:
                raise LedgerError(f"{self.path.name} line {n}: hash chain broken")
            prev = _sha256_text(line)
            entries.append({k: v for k, v in obj.items() if k != "prev_sha256"})
        return entries, prev

    def append(self, entry: Mapping[str, Any]) -> str:
        """Verify the existing chain, append one entry (fsync) and return the new head."""
        if "prev_sha256" in entry:
            raise LedgerError("prev_sha256 is set by the log")
        _, head = self.read()
        line = canonical_json({**entry, "prev_sha256": head})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return _sha256_text(line)


class CategoryBLedger:
    """Category B events and their human resolutions (protocol §9.7), in two hash-chained logs."""

    EVENTS_FILE = "category_b_events.jsonl"
    RESOLUTIONS_FILE = "category_b_resolutions.jsonl"

    def __init__(self, root: str | Path) -> None:
        self.events_log = HashChainedLog(Path(root) / self.EVENTS_FILE)
        self.resolutions_log = HashChainedLog(Path(root) / self.RESOLUTIONS_FILE)

    # ---------------------------------------------------------------- reading
    def load(self) -> dict[str, Any]:
        """Verify both logs and their cross-references; return events, resolutions and heads."""
        events, events_head = self.events_log.read()
        resolutions, resolutions_head = self.resolutions_log.read()
        ids: set[str] = set()
        for e in events:
            _check_fields(e, EVENT_FIELDS, "event")
            if e["status"] != "unresolved":
                raise LedgerError(f"event {e['event_id']}: initial status must be 'unresolved'")
            if e["event_id"] in ids:
                raise LedgerError(f"duplicate event id {e['event_id']}")
            ids.add(e["event_id"])
        for r in resolutions:
            _check_resolution(r)
            if r["event_id"] not in ids:
                raise LedgerError(f"resolution for unknown event {r['event_id']}")
        return {"events": events, "resolutions": resolutions,
                "heads": {"events": events_head, "resolutions": resolutions_head}}

    def status(self) -> dict[str, dict[str, Any]]:
        """Current status per event: the latest resolution's outcome, or 'unresolved'; full history kept."""
        data = self.load()
        out = {e["event_id"]: {"event": e, "status": "unresolved", "resolutions": []} for e in data["events"]}
        for r in data["resolutions"]:
            out[r["event_id"]]["resolutions"].append(r)
            out[r["event_id"]]["status"] = r["outcome"]
        return out

    def blocking_events(self) -> list[str]:
        """Events that block acceptance: unresolved or rejected (§9.7)."""
        return sorted(eid for eid, s in self.status().items() if s["status"] in ("unresolved", "rejected"))

    def heads(self) -> dict[str, str]:
        return self.load()["heads"]

    # ---------------------------------------------------------------- writing
    def register(self, event: Mapping[str, Any]) -> str:
        """Append a new event (status 'unresolved'); returns the new events chain head."""
        event = dict(event)
        event.setdefault("status", "unresolved")
        _check_fields(event, EVENT_FIELDS, "event")
        if event["status"] != "unresolved":
            raise LedgerError("a new event must have status 'unresolved'")
        if event["event_id"] in {e["event_id"] for e in self.load()["events"]}:
            raise LedgerError(f"event id {event['event_id']} already exists")
        return self.events_log.append(event)

    def resolve(self, resolution: Mapping[str, Any]) -> str:
        """Append a human resolution; returns the new resolutions chain head."""
        resolution = dict(resolution)
        _check_resolution(resolution)
        if resolution["event_id"] not in {e["event_id"] for e in self.load()["events"]}:
            raise LedgerError(f"unknown event {resolution['event_id']}")
        return self.resolutions_log.append(resolution)


def _check_fields(obj: Mapping[str, Any], fields: Sequence[str], what: str) -> None:
    missing = [f for f in fields if f not in obj]
    extra = [f for f in obj if f not in fields]
    if missing or extra:
        raise LedgerError(f"{what} fields: missing {missing}, unexpected {extra}")
    if not isinstance(obj["event_id"], str) or not obj["event_id"]:
        raise LedgerError(f"{what}: event_id must be a non-empty string")


def _check_resolution(r: Mapping[str, Any]) -> None:
    _check_fields(r, RESOLUTION_FIELDS, "resolution")
    if r["outcome"] not in OUTCOMES:
        raise LedgerError(f"outcome must be one of {OUTCOMES}, got {r['outcome']!r}")
    for f in ("reviewer", "timestamp", "rationale", "evidence"):
        if not isinstance(r[f], str) or not r[f].strip():
            raise LedgerError(f"resolution field {f!r} must be a non-empty string")


# =============================================================================
# 6. Integrity gates (protocol §10.3; approvals A3, I-C)
# =============================================================================
GateStatus = Literal["pass", "fail", "unresolved", "not_applicable"]
CHECKPOINT_FILES = {"best": "model_best.pt", "final": "model_final.pt", "resume": "checkpoint_last.pt"}


@dataclass
class GateResult:
    gate: str
    status: GateStatus
    detail: str = ""
    first_mismatch: dict[str, Any] | None = None
    checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fail(gate: str, detail: str, checks: list[str], mismatch: Mismatch | None = None) -> GateResult:
    return GateResult(gate, "fail", detail, mismatch.to_dict() if mismatch else None, checks)


def _is_json_object(obj: Any) -> bool:
    return isinstance(obj, dict)


def read_jsonl_strict(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file; every line must be a JSON object (no blank lines inside)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError(f"{path.name}: last line is incomplete")
    out = []
    for n, line in enumerate(text.split("\n")[:-1] if text else [], start=1):
        obj = json.loads(line)
        if not _is_json_object(obj):
            raise ValueError(f"{path.name} line {n}: not a JSON object")
        out.append(obj)
    return out


# ---------------------------------------------------------------- G1
def gate_g1(history: Sequence[Mapping[str, Any]], cap: int) -> GateResult:
    """G1: exactly one history row per epoch, ``epoch`` = 0 … cap − 1 in order."""
    checks = [f"{len(history)} rows, cap {cap}"]
    epochs = [r.get("epoch") for r in history]
    if epochs != list(range(cap)):
        return _fail("G1", f"history epochs {epochs[:5]}… (n={len(epochs)}) != 0..{cap - 1}", checks)
    return GateResult("G1", "pass", "", None, checks)


# ---------------------------------------------------------------- G2
def diff_paths(expected: Any, actual: Any, prefix: FieldPath = ()) -> list[FieldPath]:
    """Tuple paths at which two JSON-like values differ (exact; floats by bit pattern)."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        out: list[FieldPath] = []
        for k in sorted(set(expected) | set(actual), key=str):
            p = prefix + (k,)
            out += [p] if (k not in expected or k not in actual) else diff_paths(expected[k], actual[k], p)
        return out
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        out = []
        for i, (a, b) in enumerate(zip(expected, actual, strict=True)):
            out += diff_paths(a, b, prefix + (i,))
        return out
    if expected is None or actual is None:
        return [] if expected is None and actual is None else [prefix]
    return [] if values_equal(expected, actual) else [prefix]


def gate_g2(*, actual_config: Mapping[str, Any], expected_config: Mapping[str, Any],
            actual_checksums: Mapping[str, str], frozen_checksums: Mapping[str, str],
            ignore: Iterable[FieldPath] = ()) -> GateResult:
    """G2: the run's configuration and the blocking checksums equal the frozen record."""
    ignored = set(ignore)
    checks = ["configuration", "blocking checksums"]
    diffs = [p for p in diff_paths(expected_config, actual_config) if not any(p[:len(i)] == i for i in ignored)]
    if diffs:
        return _fail("G2", f"configuration differs at {format_path(diffs[0])} (+{len(diffs) - 1} more)", checks)
    cdiff = sorted(k for k in set(frozen_checksums) | set(actual_checksums)
                   if frozen_checksums.get(k) != actual_checksums.get(k))
    if cdiff:
        return _fail("G2", f"blocking checksum differs for {cdiff[0]} (+{len(cdiff) - 1} more)", checks)
    return GateResult("G2", "pass", "", None, checks)


# ---------------------------------------------------------------- G3
# Attempt status → the RunManager metadata status it must coincide with.
ATTEMPT_TO_METADATA = {
    "completed": ("completed",),
    "mechanically_unstable": ("failed",),
    "integrity_blocked": ("failed",),
    "error": ("failed", "created"),      # a set-up error can precede run.track()
    "interrupted": ("interrupted",),
    "crashed": ("running", "created"),  # killed: the metadata never left 'running'
}


def gate_g3(*, attempt_status: str, metadata_status: str | None) -> GateResult:
    """G3: run metadata status consistent with the attempt record."""
    allowed = ATTEMPT_TO_METADATA.get(attempt_status)
    checks = [f"attempt {attempt_status!r}, metadata {metadata_status!r}"]
    if allowed is None:
        return _fail("G3", f"unknown attempt status {attempt_status!r}", checks)
    if metadata_status not in allowed:
        return _fail("G3", f"metadata status {metadata_status!r} not in {allowed}", checks)
    return GateResult("G3", "pass", "", None, checks)


# ---------------------------------------------------------------- G4
def _bn_counter_errors(payload: Mapping[str, Any], label: str) -> list[str]:
    """Approval A3: every BatchNorm counter in the checkpoint's own state equals its own step / epoch."""
    errors = []
    state = payload["model_state"]
    for name in sorted(state):
        if name.endswith(".num_batches_tracked") and int(state[name]) != payload["step"]:
            errors.append(f"{label}: {name} = {int(state[name])} != step {payload['step']}")
        elif name.endswith(".num_calibrations") and int(state[name]) != payload["epoch"]:
            errors.append(f"{label}: {name} = {int(state[name])} != epoch {payload['epoch']}")
    return errors


def gate_g4(run_dir: str | Path, *, cap: int) -> GateResult:
    """G4: checkpoint loading and consistency with the recorded training history.

    Uses only ``history.jsonl`` and checkpoint metadata (never state hashes). Checks:
    every checkpoint loads (``weights_only``) with the expected kind; ``model_final`` holds
    ``epoch == cap`` and the last step count; ``checkpoint_last`` holds the last completed epoch,
    its step count, the best metric so far and the best epoch; ``model_best`` holds the epoch,
    step and dev NLL of the last row whose ``is_best`` is true; ``is_best`` flags follow strict
    improvement; and (A3) BatchNorm counters in each checkpoint equal that checkpoint's own
    ``step`` (``num_batches_tracked``) and ``epoch`` (``num_calibrations``).
    """
    run_dir = Path(run_dir)
    checks: list[str] = []
    try:
        history = read_jsonl_strict(run_dir / "history.jsonl")
    except (OSError, ValueError) as exc:
        return _fail("G4", f"history unreadable: {exc}", checks)
    if not history or [r.get("epoch") for r in history] != list(range(len(history))):
        return _fail("G4", "history epochs are not 0..n-1", checks)
    payloads: dict[str, dict[str, Any]] = {}
    for kind, fname in CHECKPOINT_FILES.items():
        try:
            payloads[kind] = load_payload(run_dir / fname, expected_kind=kind)
            checks.append(f"{fname} loads")
        except Exception as exc:  # noqa: BLE001  (any load failure fails the gate)
            return _fail("G4", f"{fname}: {exc}", checks)

    # is_best must follow strict improvement (ties keep the earliest epoch)
    best_so_far: float | None = None
    best_idx = -1
    for i, row in enumerate(history):
        dev = row.get("dev_nll")
        if not is_number(dev) or is_nonfinite(dev):
            return _fail("G4", f"epoch {i}: dev_nll is not a finite number", checks)
        expected_flag = best_so_far is None or dev < best_so_far
        if row.get("is_best") is not expected_flag:
            return _fail("G4", f"epoch {i}: is_best {row.get('is_best')} != strict improvement {expected_flag}", checks)
        if expected_flag:
            best_so_far, best_idx = dev, i
    checks.append("is_best flags follow strict improvement")

    n, last_row, best_row = len(history), history[-1], history[best_idx]
    expectations = [
        ("final", "epoch", cap), ("final", "step", last_row["steps"]),
        ("resume", "epoch", n), ("resume", "step", last_row["steps"]),
        ("resume", "best_metric", best_row["dev_nll"]), ("resume", "best_metric_name", "dev_nll"),
        ("best", "epoch", best_idx + 1), ("best", "step", best_row["steps"]),
        ("best", "best_metric", best_row["dev_nll"]), ("best", "best_metric_name", "dev_nll"),
    ]
    if n != cap:
        return _fail("G4", f"history has {n} rows, cap is {cap}", checks)
    for kind, key, want in expectations:
        got = payloads[kind].get(key)
        if got is None or not values_equal(want, got):
            return _fail("G4", f"{CHECKPOINT_FILES[kind]}: {key} = {got!r}, history implies {want!r}", checks)
    for kind in ("resume", "best"):
        got = (payloads[kind].get("extra") or {}).get("best_epoch")
        if got != best_idx:
            return _fail("G4", f"{CHECKPOINT_FILES[kind]}: extra.best_epoch = {got!r}, "
                               f"history implies {best_idx}", checks)
    checks.append("epoch / step / best metric / best epoch agree with history")
    errors = [e for kind in CHECKPOINT_FILES for e in _bn_counter_errors(payloads[kind], CHECKPOINT_FILES[kind])]
    if errors:
        return _fail("G4", errors[0] + (f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""), checks)
    checks.append("BatchNorm counters equal each checkpoint's own step / epoch (A3)")
    return GateResult("G4", "pass", "", None, checks)


# ---------------------------------------------------------------- G7
def gate_g7(run_dir: str | Path) -> GateResult:
    """G7: state-hash records exist and match the corresponding checkpoint states (I-C).

    Independent of G4: reads only ``state_hashes.jsonl`` and the checkpoints' own ``epoch``,
    ``extra`` and ``model_state``; never ``history.jsonl`` and never a G4 result.

    * ``checkpoint_last`` states its completed-epoch count E; the hash log must contain exactly
      one record for each epoch 0 … E − 1, in order.
    * Each checkpoint's epoch index is derived from its **own** metadata, ``epoch − 1``
      (``epoch`` counts completed epochs). For ``model_best`` the checkpoint must also be
      internally consistent: ``extra.best_epoch == epoch − 1``.
    * The hash of each checkpoint's ``model_state`` must equal the record at that index.
    """
    run_dir = Path(run_dir)
    checks: list[str] = []
    try:
        records = read_jsonl_strict(run_dir / STATE_HASH_FILE)
    except (OSError, ValueError) as exc:
        return _fail("G7", f"{STATE_HASH_FILE} unreadable: {exc}", checks)
    for r in records:
        if set(r) != {"epoch", "sha256"} or not isinstance(r["epoch"], int) or not isinstance(r["sha256"], str) \
                or not re.fullmatch(r"[0-9a-f]{64}", r["sha256"]):
            return _fail("G7", f"malformed state-hash record {r!r}", checks)
    payloads: dict[str, dict[str, Any]] = {}
    for kind, fname in CHECKPOINT_FILES.items():
        try:
            payloads[kind] = load_payload(run_dir / fname, expected_kind=kind)
        except Exception as exc:  # noqa: BLE001
            return _fail("G7", f"{fname}: {exc}", checks)
    completed = payloads["resume"].get("epoch")
    if not isinstance(completed, int) or completed < 1:
        return _fail("G7", f"checkpoint_last.pt: epoch {completed!r} is not a positive int", checks)
    if [r["epoch"] for r in records] != list(range(completed)):
        return _fail("G7", f"state-hash epochs {[r['epoch'] for r in records][:5]}… (n={len(records)}) "
                           f"!= 0..{completed - 1} (checkpoint_last epoch {completed})", checks)
    checks.append(f"one state-hash record per epoch 0..{completed - 1}")
    by_epoch = {r["epoch"]: r["sha256"] for r in records}
    best_extra = (payloads["best"].get("extra") or {}).get("best_epoch")
    if not isinstance(payloads["best"].get("epoch"), int) or best_extra != payloads["best"]["epoch"] - 1:
        return _fail("G7", f"model_best.pt: extra.best_epoch {best_extra!r} != epoch − 1 "
                           f"({payloads['best'].get('epoch')!r} − 1)", checks)
    for kind, fname in CHECKPOINT_FILES.items():
        epoch = payloads[kind].get("epoch")
        idx = epoch - 1 if isinstance(epoch, int) else None
        if idx not in by_epoch:
            return _fail("G7", f"{fname}: epoch {epoch!r} has no state-hash record", checks)
        got = state_hash(payloads[kind]["model_state"])
        if got != by_epoch[idx]:
            return _fail("G7", f"{fname}: state hash differs from record of epoch {idx}", checks,
                         Mismatch("value", ("sha256",), by_epoch[idx], got, fname))
        checks.append(f"{fname} matches the record of epoch {idx}")
    return GateResult("G7", "pass", "", None, checks)


# =============================================================================
# 7. Read-only finiteness monitor (protocol §9.2; I-B; clarification B)   — Stage 2
# =============================================================================
# Two kinds of hook, both read-only (they only call ``torch.isfinite(...).all()`` on detached
# tensors and return ``None``):
#   * a global optimiser post-step hook: every model parameter, immediately after every
#     ``optimizer.step()`` of the optimiser that owns exactly this model's parameters;
#   * a forward hook on every leaf module (``emb``, ``hidden*``, ``bn*``, ``act*``, ``out``):
#     the module's output, in every forward, training and evaluation alike, *before* the next
#     module consumes it (``tanh(±∞) = ±1`` would otherwise mask a non-finite pre-activation).
# Forwards with gradients enabled are SGD forwards ("training"); forwards under ``no_grad``
# (recalibration, full-train / dev evaluation, epoch-end diagnostics) are "evaluation".
FinitenessKind = Literal["parameter_after_step", "module_output"]


@dataclass(frozen=True)
class FinitenessFailure:
    """First non-finite value seen by the monitor (protocol §9.2 item 5).

    ``global_step`` is 0-based: the step just applied (``parameter_after_step``), the step being
    computed (training forward), or the last applied step whose state is being evaluated
    (evaluation forward; ``None`` before the first step). ``steps_completed`` counts the
    optimiser steps that finished with finite parameters.
    """

    kind: FinitenessKind
    name: str
    epoch_1based: int
    global_step: int | None
    steps_completed: int
    forward: Literal["training", "evaluation"] | None
    module_training: bool | None
    nonfinite_elements: int
    numel: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def message(self) -> str:
        where = f"epoch {self.epoch_1based} (1-based), global step {self.global_step}"
        what = (f"parameter {self.name!r} after optimizer.step()" if self.kind == "parameter_after_step"
                else f"output of module {self.name!r} in a {self.forward} forward")
        return f"non-finite {what} at {where}: {self.nonfinite_elements} of {self.numel} elements"


class NonFiniteDetected(FloatingPointError):
    """Raised by the monitor; carries the structured :class:`FinitenessFailure`."""

    def __init__(self, failure: FinitenessFailure) -> None:
        super().__init__(failure.message())
        self.failure = failure


def leaf_modules(model: Any) -> list[tuple[str, Any]]:
    """``(name, module)`` for every module without children, in registration order."""
    return [(name, m) for name, m in model.named_modules() if name and next(m.children(), None) is None]


def _output_tensors(output: Any, name: str) -> list[Any]:
    import torch

    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in output):
        return list(output)
    raise TypeError(f"module {name!r} returned {type(output).__name__}; the finiteness monitor checks tensors only")


class FinitenessMonitor:
    """Read-only per-step finiteness checks with their own cost accounting (protocol §9.2).

    Use :meth:`attach` before training and :meth:`detach` afterwards (or the context manager).
    The owner calls :meth:`end_epoch` once per epoch, at the history write, so that costs are
    reported per epoch and the step count is cross-checked against the trainer's.
    """

    def __init__(self, model: Any, *, points: Sequence[tuple[str, Any]] | None = None) -> None:
        self.model = model
        self.points = list(points) if points is not None else leaf_modules(model)
        if not self.points:
            raise ValueError("no modules to monitor")
        self.parameters = list(model.named_parameters())
        if not self.parameters:
            raise ValueError("model has no parameters")
        self._param_ids = frozenset(id(p) for _, p in self.parameters)
        self.steps_completed = 0
        self.epochs_completed = 0
        self.failure: FinitenessFailure | None = None
        self.epoch_costs: list[dict[str, Any]] = []
        self._bucket = self._new_bucket()
        self._handles: list[Any] = []
        self._owners: dict[int, bool] = {}
        import time

        self._clock = time.perf_counter

    @staticmethod
    def _new_bucket() -> dict[str, Any]:
        return {"parameter_checks": 0, "parameter_check_seconds": 0.0, "module_checks_training": 0,
                "module_checks_evaluation": 0, "module_check_seconds": 0.0}

    # ------------------------------------------------------------- lifecycle
    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def attach(self) -> FinitenessMonitor:
        from torch.optim.optimizer import register_optimizer_step_post_hook

        if self._handles:
            raise RuntimeError("finiteness monitor is already attached")
        try:
            for name, module in self.points:
                self._handles.append(module.register_forward_hook(self._module_hook(name)))
            self._handles.append(register_optimizer_step_post_hook(self._after_step))
        except BaseException:
            self.detach()
            raise
        return self

    def detach(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def __enter__(self) -> FinitenessMonitor:  # noqa: PYI034
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()

    # ------------------------------------------------------------- epochs
    def end_epoch(self, *, steps_reported: int) -> dict[str, Any]:
        """Close the cost bucket of the epoch just written; the trainer's step count must agree."""
        if steps_reported != self.steps_completed:
            raise RuntimeError(f"step count mismatch: trainer reports {steps_reported}, "
                               f"monitor saw {self.steps_completed} optimizer steps")
        record = {"epoch": self.epochs_completed + 1, **self._bucket}
        self.epoch_costs.append(record)
        self._bucket = self._new_bucket()
        self.epochs_completed += 1
        return record

    def open_bucket(self) -> dict[str, Any]:
        """Costs accumulated since the last :meth:`end_epoch` (a partial epoch, or after the last)."""
        return dict(self._bucket)

    # ------------------------------------------------------------- checks
    def _fail(self, failure: FinitenessFailure) -> None:
        if self.failure is None:
            self.failure = failure
        raise NonFiniteDetected(failure)

    def _owns(self, optimizer: Any) -> bool:
        key = id(optimizer)
        if key not in self._owners:
            ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
            self._owners[key] = len(ids) == len(self._param_ids) and set(ids) == self._param_ids
        return self._owners[key]

    def _after_step(self, optimizer: Any, _args: Any, _kwargs: Any) -> None:
        if not self._owns(optimizer):
            return
        import torch

        t0 = self._clock()
        with torch.no_grad():
            finite = bool(torch.stack([torch.isfinite(p.detach()).all() for _, p in self.parameters]).all())
            bad = None
            if not finite:
                for name, p in self.parameters:
                    count = int((~torch.isfinite(p.detach())).sum())
                    if count:
                        bad = (name, count, p.numel())
                        break
        self._bucket["parameter_checks"] += 1
        self._bucket["parameter_check_seconds"] += self._clock() - t0
        if bad is not None:
            step = self.steps_completed
            self._fail(FinitenessFailure("parameter_after_step", bad[0], self.epochs_completed + 1, step, step,
                                         None, None, bad[1], bad[2]))
        self.steps_completed += 1

    def _module_hook(self, name: str) -> Callable[..., None]:
        def hook(module: Any, _inputs: Any, output: Any) -> None:
            import torch

            t0 = self._clock()
            training = torch.is_grad_enabled()
            bad = numel = 0
            with torch.no_grad():
                for t in _output_tensors(output, name):
                    d = t.detach()
                    numel += d.numel()
                    if not bool(torch.isfinite(d).all()):
                        bad += int((~torch.isfinite(d)).sum())
            self._bucket["module_checks_training" if training else "module_checks_evaluation"] += 1
            self._bucket["module_check_seconds"] += self._clock() - t0
            if bad:
                done = self.steps_completed
                step = done if training else (done - 1 if done else None)
                self._fail(FinitenessFailure("module_output", name, self.epochs_completed + 1, step, done,
                                             "training" if training else "evaluation", bool(module.training),
                                             bad, numel))

        return hook


# =============================================================================
# 8. Step timer — copied from lr_pilot.py (SHA-256 b77d7684…, AM-1: copy, never import)
# =============================================================================
class StepTimer:
    """Timestamps from a global optimiser post-step hook and from ``run.append_history``.

    Copied from ``training/wavenet_study/lr_pilot.py`` (accepted, SHA-256
    ``b77d76844a6c09dfd905e97a764bc929ab7591417ce08b6c55d2b1dfa95e27bc``) without numerical
    change; only the clock and ``statistics`` imports are local. The accepted trainers record
    one wall-clock number per epoch; these timestamps split it without touching the trainers.
    They never affect the numerics. Step ``s+1`` also pays for the update statistics and the
    JSONL write of diagnostic step ``s`` and, in the budget pilot, for the post-step
    finiteness check of step ``s`` (measured separately by :class:`FinitenessMonitor`).
    """

    def __init__(self, steps_per_epoch: int, every_n_steps: int | None, num_examples: int) -> None:
        import time

        self._clock = time.perf_counter
        self.steps_per_epoch = steps_per_epoch
        self.every = every_n_steps
        self.num_examples = num_examples
        self.start: float | None = None
        self.step_times: list[float] = []  # t[s] = end of optimizer.step() for global step s
        self.epoch_ends: list[float] = []  # right after the history row of each epoch has been written

    def on_step(self, _optimizer: Any, _args: Any, _kwargs: Any) -> None:
        self.step_times.append(self._clock())

    def epochs(self) -> list[dict]:
        import statistics

        out, n, t, m = [], self.steps_per_epoch, self.step_times, self.num_examples
        diag = set(range(0, len(t), self.every)) if self.every else set()
        affected = diag | {s + 1 for s in diag}
        for e, end in enumerate(self.epoch_ends):
            a, b = e * n, (e + 1) * n - 1
            if b >= len(t):
                break
            start = self.epoch_ends[e - 1] if e else self.start
            dur = {s: t[s] - t[s - 1] for s in range(a + 1, b + 1)}
            plain = sorted(d for s, d in dur.items() if s not in affected)
            if not plain:
                out.append({"epoch": e + 1, "error": "no non-diagnostic steps to time"})
                continue
            med = statistics.median(plain)
            overhead = sum(d - med for s, d in dur.items() if s in affected)
            sgd = t[b] - start
            clean = sgd - overhead

            def q(p: float, plain: list[float] = plain) -> float:
                return plain[min(len(plain) - 1, int(p * len(plain)))]

            out.append({
                "epoch": e + 1,
                "measured_sgd_phase_seconds": sgd,
                "measured_epoch_end_seconds": end - t[b],
                "diagnostic_steps": sum(1 for s in range(a, b + 1) if s in diag),
                "estimated_step_diagnostic_overhead_seconds": overhead,
                "estimated_sgd_seconds_without_diagnostics": clean,
                "measured_tokens_per_second_including_diagnostics": m / sgd,
                "estimated_tokens_per_second_without_diagnostics": m / clean,
                "idealized_steady_state_tokens_per_second": m / (med * n),
                "non_diagnostic_step_seconds": {"p10": q(0.10), "median": med, "p90": q(0.90), "max": plain[-1]},
            })
        return out


TIMING_DEFINITIONS = {
    "measured_sgd_phase_seconds": "as in the LR pilot: previous epoch's end timestamp (right after its history row "
                                  "and the epoch-end records below; epoch 1: the trainer call) to the end of this "
                                  "epoch's last optimizer.step(); includes step diagnostics, per-step finiteness "
                                  "checks and the previous epoch's checkpoint_last.pt write",
    "measured_epoch_end_seconds": "end of the last optimizer.step() to the timestamp taken right after this epoch's "
                                  "history row was written: [BN] recalibration, evaluation, epoch-end diagnostics "
                                  "(with module finiteness checks), best checkpoint, history write",
    "estimated_step_diagnostic_overhead_seconds": "LR-pilot definition (median-interval method); the post-step "
                                                  "finiteness check is in every step interval, not in the overhead",
    "finiteness": "per epoch: number of post-step parameter checks and module-output checks (training / "
                  "evaluation forwards) and the seconds spent inside them (protocol §9.2 item 6)",
    "state_hash_seconds": "per epoch: computing and writing the state hash at the history write (§10.2)",
    "record_check_seconds": "per epoch: reading back and classifying the epoch's history row and step records, "
                            "category B registration, runner-side prefix checks",
    "training_call_seconds": "the trainer call, including the final model_final.pt write",
}


# =============================================================================
# 9. Single-run executor — Stage 2
# =============================================================================
# One already-prepared run (model, windows, training configuration, accepted trainer, fresh
# RunManager) is trained by the accepted trainer, unchanged, with:
#   * the finiteness monitor attached (category A detection, §9.2);
#   * an instance-level wrapper of ``run.append_history`` that, right after the trainer's own
#     history write, records the state hash (§10.2), closes the epoch's cost bucket,
#     reads the epoch's records back from disk and classifies them (§9.1, §9.3; schema check),
#     registers category B events in the ledger (§9.7), raises on category A evidence and
#     runs the runner-side prefix checks (G5 / G6, §10.4);
#   * the outcome classified as completed / mechanically_unstable / integrity_blocked / error /
#     interrupted and written to ``pilot_result.json``, ``timing.json`` and ``integrity.json``.
# Set-up (configuration, data, model construction, run directory, Git guard, threads) is not
# part of this function; it never touches any data split.
RUN_RESULT_FILE = "pilot_result.json"
TIMING_FILE = "timing.json"
INTEGRITY_FILE = "integrity.json"
STEP_RECORDS_FILE = ("diagnostics", "steps.jsonl")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunContext:
    attempt_id: str
    arm: str
    seed: int
    cap: int
    normalization: Literal["none", "batchnorm"]
    labels: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedRun:
    """Everything a run needs, built by the caller through the accepted code; nothing trained yet."""

    model: Any
    train: Any
    dev: Any
    training: Any
    train_fn: Callable[..., Any]
    run: Any
    diagnostics: Any
    experiment_config: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RecordSchemas:
    """Frozen leaf-path schemas (from reference records) for one arm (finding F2)."""

    history: frozenset[FieldPath]
    steps: frozenset[FieldPath]


@dataclass(frozen=True)
class PrefixCheck:
    """Runner-side G5 / G6: the first ``epochs`` epochs must equal a reference (protocol §10.4)."""

    gate: str
    epochs: int
    reference_history: tuple[Mapping[str, Any], ...]
    reference_steps: tuple[Mapping[str, Any], ...]
    reference_hash: str
    reference: str


class IntegrityMismatch(RuntimeError):
    """Runner-side prefix check failed; the run stops and is ``integrity_blocked`` (§10.5)."""

    def __init__(self, gate: str, epochs: int, mismatch: Mismatch) -> None:
        super().__init__(f"{gate}: epochs 1-{epochs} differ from the reference ({mismatch.kind} at "
                         f"{format_path(mismatch.path)} {mismatch.where})")
        self.gate, self.epochs, self.mismatch = gate, epochs, mismatch


class RecordedNonFinite(FloatingPointError):
    """Category A evidence found in a written record (e.g. a non-finite recorded NLL)."""

    def __init__(self, findings: Sequence[Finding]) -> None:
        f = findings[0]
        super().__init__(f"category A in {f.record} record (epoch {f.epoch}, step {f.step}): "
                         f"{format_path(f.path)} = {f.value!r} ({f.reason}); {len(findings)} finding(s)")
        self.findings = list(findings)


def _utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def _repo_relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _plain(obj: Any) -> Any:
    """JSON-safe copy (non-finite floats → repr strings) for ledger and result records."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _append_line(path: Path, obj: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_plain(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _model_normalization(model: Any) -> str:
    import torch

    return "batchnorm" if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()) \
        else "none"


def _peak_rss_bytes() -> int | None:
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError):  # pragma: no cover
        return None


def _trainer_guard_code(message: str) -> str:
    """Category A sub-code of a FloatingPointError raised by an accepted trainer's own guard."""
    if "non-finite loss" in message:
        return "A1"
    if "non-finite gradients" in message:
        return "A2"
    if "recalibration" in message or "BatchNorm statistics" in message or "non-finite inference" in message:
        return "A4"
    return "unattributed"


class _EpochRecorder:
    """State of the ``append_history`` wrapper for one run."""

    def __init__(self, prepared: PreparedRun, context: RunContext, *, schemas: RecordSchemas,
                 ledger: CategoryBLedger, prefix_checks: Sequence[PrefixCheck], monitor: FinitenessMonitor,
                 timer: StepTimer, on_epoch: Callable[[dict[str, Any]], None] | None) -> None:
        self.prepared, self.context, self.schemas, self.ledger = prepared, context, schemas, ledger
        self.monitor, self.timer, self.on_epoch = monitor, timer, on_epoch
        self.run_dir = Path(prepared.run.run_dir)
        self.history_path = self.run_dir / "history.jsonl"
        self.steps_path = self.run_dir.joinpath(*STEP_RECORDS_FILE)
        self.hash_path = self.run_dir / STATE_HASH_FILE
        self.steps_offset = 0
        self.steps_lines = 0
        self.step_records: list[dict[str, Any]] = []
        self.event_ids: list[str] = []
        self.category_c: list[dict[str, Any]] = []
        self.epoch_times: list[dict[str, Any]] = []
        self.prefix = {c.gate: c for c in prefix_checks}
        self.integrity = {c.gate: {"gate": c.gate, "epochs": c.epochs, "reference": c.reference,
                                   "status": "not_applicable" if c.epochs > context.cap else "not_reached",
                                   "first_mismatch": None} for c in prefix_checks}
        self.original_append = prepared.run.append_history

    # ------------------------------------------------------------- wrapper
    def append_history(self, record: Mapping[str, Any]) -> None:
        self.original_append(record)
        self.timer.epoch_ends.append(self.timer._clock())
        self.after_history_write(record)

    def after_history_write(self, record: Mapping[str, Any]) -> None:
        clock = self.monitor._clock
        epoch = record["epoch"]
        if epoch != self.monitor.epochs_completed:
            raise RuntimeError(f"history epoch {epoch} != expected {self.monitor.epochs_completed}")
        t0 = clock()
        digest = state_hash(self.prepared.model.state_dict())  # exactly at the history write (§10.2)
        _append_line(self.hash_path, {"epoch": epoch, "sha256": digest})
        t1 = clock()
        cost = self.monitor.end_epoch(steps_reported=record["steps"])
        rows = read_jsonl_strict(self.history_path)
        if len(rows) != epoch + 1 or rows[-1].get("epoch") != epoch:
            raise RuntimeError(f"history.jsonl has {len(rows)} rows after epoch {epoch}")
        new_steps = self.read_new_steps()
        self.classify([("history", rows[-1], "history.jsonl", len(rows))] + new_steps)
        for check in self.prefix.values():
            if check.epochs == epoch + 1:
                self.run_prefix_check(check, rows, digest)
        self.epoch_times.append({"epoch": epoch + 1, "state_hash_seconds": t1 - t0,
                                 "record_check_seconds": clock() - t1, "finiteness": cost})
        if self.on_epoch is not None:
            self.on_epoch({"epoch": epoch + 1, "cap": self.context.cap, "seconds": record.get("seconds")})

    # ------------------------------------------------------------- records
    def read_new_steps(self) -> list[tuple[str, dict[str, Any], str, int]]:
        if not self.steps_path.exists():
            return []
        with self.steps_path.open("rb") as fh:
            fh.seek(self.steps_offset)
            data = fh.read()
        if data and not data.endswith(b"\n"):
            raise RuntimeError(f"{self.steps_path.name}: incomplete last line")
        self.steps_offset += len(data)
        out = []
        for line in data.decode("utf-8").split("\n")[:-1] if data else []:
            self.steps_lines += 1
            obj = json.loads(line)
            if not _is_json_object(obj):
                raise RuntimeError(f"steps.jsonl line {self.steps_lines}: not a JSON object")
            self.step_records.append(obj)
            out.append(("steps", obj, "/".join(STEP_RECORDS_FILE), self.steps_lines))
        return out

    def classify(self, items: Sequence[tuple[str, dict[str, Any], str, int]]) -> None:
        category_a: list[Finding] = []
        for kind, rec, fname, line in items:
            schema = self.schemas.history if kind == "history" else self.schemas.steps
            findings = classify_record(kind, rec, normalization=self.context.normalization) \
                + schema_findings(kind, rec, schema)
            for f in findings:
                if f.category == "A":
                    category_a.append(f)
                elif f.category == "B":
                    self.register_b(f, rec, fname, line)
                else:
                    self.category_c.append({"kind": f.kind, "record": f.record, "epoch_1based":
                                            None if f.epoch is None else f.epoch + 1, "global_step": f.step,
                                            "field": list(f.path), "value": _plain(f.value)})
        if category_a:
            raise RecordedNonFinite(category_a)

    def register_b(self, f: Finding, rec: Mapping[str, Any], fname: str, line: int) -> None:
        related = None
        if f.record == "steps" and len(f.path) >= 3 and f.path[:2] == ("updates", "per_parameter"):
            related = rec.get("updates", {}).get("per_parameter", {}).get(f.path[2])
        ctx = self.context
        event = {
            "event_id": f"B-{ctx.attempt_id}-{len(self.event_ids) + 1:04d}",
            "created_at": _utc_now(), "attempt_id": ctx.attempt_id, "run_dir": _repo_relative(self.run_dir),
            "arm": ctx.arm, "seed": ctx.seed, "cap": ctx.cap,
            "epoch_1based": None if f.epoch is None else f.epoch + 1,
            "global_step": f.step if f.record == "steps" else None,
            "record": f.record, "field": list(f.path), "observed": _plain(f.value),
            "reason": f"{f.kind}: {f.reason}",
            "evidence": _plain({"file": fname, "line": line, "related_fields": related}),
            "status": "unresolved",
        }
        self.ledger.register(event)
        self.event_ids.append(event["event_id"])

    def run_prefix_check(self, check: PrefixCheck, rows: list[dict[str, Any]], digest: str) -> None:
        mismatch = compare_run_prefix(
            reference_history=check.reference_history, reference_steps=check.reference_steps,
            reference_hash=check.reference_hash, candidate_history=rows, candidate_steps=self.step_records,
            candidate_hash=digest, epochs=check.epochs,
            history_null_allowed=history_null_allowed_for(self.context.normalization),
            steps_null_allowed=step_null_allowed)
        self.integrity[check.gate].update(status="pass" if mismatch is None else "fail",
                                          first_mismatch=None if mismatch is None else _plain(mismatch.to_dict()))
        if mismatch is not None:
            raise IntegrityMismatch(check.gate, check.epochs, mismatch)

    def final_sweep(self) -> None:
        """Step records written after the last history row (none with the accepted trainers)."""
        leftover = self.read_new_steps()
        if leftover:
            self.classify(leftover)


def _validate_execution(prepared: PreparedRun, context: RunContext, schemas: RecordSchemas, ledger: Any,
                        prefix_checks: Sequence[PrefixCheck]) -> None:
    if not isinstance(ledger, CategoryBLedger):
        raise TypeError("a CategoryBLedger is required: category B events must never be dropped")
    if not isinstance(schemas, RecordSchemas):
        raise TypeError("schemas must be RecordSchemas")
    if not schemas.history or not schemas.steps:
        raise ValueError("RecordSchemas with non-empty history and step schemas are required")
    if context.normalization not in ("none", "batchnorm"):
        raise ValueError(f"normalization must be 'none' or 'batchnorm', got {context.normalization!r}")
    if _model_normalization(prepared.model) != context.normalization:
        raise ValueError(f"context says normalization={context.normalization!r}, the model is "
                         f"{_model_normalization(prepared.model)!r}")
    if prepared.training.max_epochs != context.cap:
        raise ValueError(f"training.max_epochs={prepared.training.max_epochs} != cap {context.cap}")
    gates = [c.gate for c in prefix_checks]
    if len(set(gates)) != len(gates):
        raise ValueError(f"duplicate prefix-check gates {gates}")
    for c in prefix_checks:
        if not isinstance(c.epochs, int) or c.epochs < 1:
            raise ValueError(f"{c.gate}: epochs must be a positive int")
    run = prepared.run
    if run.status.value != "created" or run.history_path.exists() or (Path(run.run_dir) / STATE_HASH_FILE).exists():
        raise ValueError(f"{run.run_dir} is not a fresh run directory (status {run.status.value})")
    if "append_history" in vars(run):
        raise ValueError("run.append_history is already wrapped")


def execute_run(prepared: PreparedRun, context: RunContext, *, schemas: RecordSchemas, ledger: CategoryBLedger,
                prefix_checks: Sequence[PrefixCheck] = (), on_epoch: Callable[[dict[str, Any]], None] | None = None,
                finiteness_points: Sequence[tuple[str, Any]] | None = None) -> dict[str, Any]:
    """Train one prepared run with the accepted trainer and classify its outcome (Stage 2).

    Statuses (protocol §12.3): ``completed``; ``mechanically_unstable`` (category A during
    training: the monitor, an accepted trainer guard, or a non-finite recorded value);
    ``integrity_blocked`` (runner-side prefix check failed); ``error`` (any other exception,
    including a ``FloatingPointError`` outside training); ``interrupted`` (``KeyboardInterrupt``,
    returned, not re-raised, as in the LR pilot). ``crashed`` is assigned by the parent process.
    Category B never changes the status. Invalid arguments raise before anything is attached.
    """
    import time
    import traceback

    from utils.batching import EpochSampler
    from utils.reproducibility import derive_seed

    _validate_execution(prepared, context, schemas, ledger, prefix_checks)
    started_at = _utc_now()
    t_start = time.perf_counter()
    run, training = prepared.run, prepared.training
    steps_per_epoch = len(EpochSampler(prepared.train.num_examples, training.batch_size,
                                       seed=derive_seed(context.seed, "data")).batch_sizes())
    schedule = prepared.diagnostics
    timer = StepTimer(steps_per_epoch, schedule.every_n_steps if schedule is not None and schedule.enabled else None,
                      prepared.train.num_examples)
    monitor = FinitenessMonitor(prepared.model, points=finiteness_points)
    recorder = _EpochRecorder(prepared, context, schemas=schemas, ledger=ledger, prefix_checks=prefix_checks,
                              monitor=monitor, timer=timer, on_epoch=on_epoch)
    status, phase, failure, category_a = "completed", "attach", None, None
    timer_handle = None
    t_train = t_train_end = None
    try:
        from torch.optim.optimizer import register_optimizer_step_post_hook

        timer_handle = register_optimizer_step_post_hook(timer.on_step)  # timer first, then the check
        monitor.attach()
        run.append_history = recorder.append_history  # instance attribute; removed in `finally`
        phase = "training"
        timer.start = t_train = time.perf_counter()
        with run.track():
            prepared.train_fn(prepared.model, prepared.train, prepared.dev, training=training, seed=context.seed,
                              run=run, diagnostics=schedule, config=prepared.experiment_config)
            recorder.final_sweep()
        t_train_end = time.perf_counter()
        phase = "finished"
    except BaseException as exc:  # noqa: BLE001  (every outcome is classified and recorded)
        status, category_a = _classify_outcome(exc, phase, monitor)
        failure = {"type": type(exc).__name__, "message": str(exc)[:2000], "phase": phase,
                   "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]}
        if isinstance(exc, IntegrityMismatch):
            failure["integrity"] = {"gate": exc.gate, "epochs": exc.epochs,
                                    "first_mismatch": _plain(exc.mismatch.to_dict())}
    finally:
        monitor.detach()
        if timer_handle is not None:
            timer_handle.remove()
        if "append_history" in vars(run):
            del run.append_history
    result = _write_run_records(prepared, context, recorder, monitor, timer, ledger, status=status, phase=phase,
                                failure=failure, category_a=category_a, started_at=started_at,
                                wall=time.perf_counter() - t_start,
                                training_call=(t_train_end or time.perf_counter()) - t_train if t_train else None)
    return result


def _classify_outcome(exc: BaseException, phase: str, monitor: FinitenessMonitor) -> tuple[str, dict | None]:
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted", None
    if phase != "training":
        return "error", None  # D-18: nothing outside training is category A
    if monitor.failure is not None:  # the monitor's own detection, even if a trainer re-wrapped it
        code = "A3" if monitor.failure.kind == "parameter_after_step" else "A5"
        return "mechanically_unstable", {"code": code, "source": "finiteness_monitor",
                                         "detail": monitor.failure.to_dict()}
    if isinstance(exc, IntegrityMismatch):
        return "integrity_blocked", None
    if isinstance(exc, RecordedNonFinite):
        return "mechanically_unstable", {"code": "recorded_value", "source": "record_classification",
                                         "detail": [f.to_dict() for f in exc.findings]}
    if isinstance(exc, FloatingPointError):
        return "mechanically_unstable", {"code": _trainer_guard_code(str(exc)), "source": "trainer_guard",
                                         "detail": {"message": str(exc)[:2000]}}
    return "error", None


def _write_run_records(prepared: PreparedRun, context: RunContext, recorder: _EpochRecorder,
                       monitor: FinitenessMonitor, timer: StepTimer, ledger: CategoryBLedger, *, status: str,
                       phase: str, failure: dict | None, category_a: dict | None, started_at: str, wall: float,
                       training_call: float | None) -> dict[str, Any]:
    run_dir = Path(prepared.run.run_dir)
    try:
        heads: Any = ledger.heads()
    except Exception as exc:  # noqa: BLE001  (a broken ledger is reported, never hidden)
        heads = {"error": f"{type(exc).__name__}: {exc}"}
    result = {
        **dict(context.labels), "attempt_id": context.attempt_id, "arm": context.arm, "seed": context.seed,
        "cap": context.cap, "normalization": context.normalization, "status": status, "phase": phase,
        "category_a": category_a, "failure": failure, "run_dir": _repo_relative(run_dir),
        "epochs_completed": monitor.epochs_completed, "steps_completed": monitor.steps_completed,
        "category_b_events": list(recorder.event_ids), "category_b_ledger_heads": heads,
        "category_c": recorder.category_c, "integrity": list(recorder.integrity.values()),
        "peak_rss_bytes": _peak_rss_bytes(), "started_at": started_at, "ended_at": _utc_now(),
    }
    timing = {"definitions": TIMING_DEFINITIONS, "wall_seconds": wall, "training_call_seconds": training_call,
              "epochs": timer.epochs(), "epoch_records": recorder.epoch_times,
              "finiteness_open_bucket": monitor.open_bucket(),
              "finiteness_totals": {k: sum(c[k] for c in monitor.epoch_costs) + monitor.open_bucket()[k]
                                    for k in FinitenessMonitor._new_bucket()}}
    try:
        _write_json(run_dir / INTEGRITY_FILE, {"attempt_id": context.attempt_id, "checks": result["integrity"]})
        _write_json(run_dir / TIMING_FILE, timing)
        _write_json(run_dir / RUN_RESULT_FILE, result)
    except Exception as exc:  # noqa: BLE001  (never lose the primary result over a record write)
        result["record_error"] = f"{type(exc).__name__}: {exc}"
    return result


# =============================================================================
# 10. Frozen conditions, exclusive lock and attempt store — Stage 3
# =============================================================================
# Generalised from the accepted LR pilot (``lr_pilot.py``, SHA-256 b77d7684…; copied and
# adapted, never imported, AM-1): the LR pilot's module-global paths become a ``PilotPaths``
# object and its ``(arm, lr, seed)`` attempt key becomes a configurable key (``(arm, cap, seed)``
# in the budget pilot, protocol §12.3). Changes relative to the LR pilot:
#   * blocking vs recorded checksums (DEC-1): differences under ``checksums.recorded`` are
#     reported, never blocking; everything else in the frozen conditions blocks;
#   * ``conditions.json`` carries the SHA-256 of its own conditions and is verified on read;
#   * ``manifest.jsonl`` and ``invocations.jsonl`` are hash-chained (Stage 1 ``HashChainedLog``)
#     and the manifest's event sequence is validated on every read;
#   * new status ``integrity_blocked`` (§10.5): blocks the sweep; retry needs ``authorized_by``
#     (approval A4: audit provenance, not authentication) in addition to the reason;
#   * category A halts the pilot (§9.4): after any ``mechanically_unstable`` attempt no run and
#     no retry is accepted;
#   * attempt result files are written once and must agree with the manifest; result files
#     and run directories without a known attempt block the pilot (orphans).
FINAL_STATUSES = ("completed", "mechanically_unstable")
RETRYABLE_STATUSES = ("integrity_blocked", "error", "crashed", "interrupted")
AUTHORIZATION_REQUIRED_STATUSES = ("integrity_blocked",)
ATTEMPT_STATUSES = FINAL_STATUSES + RETRYABLE_STATUSES
NONBLOCKING_CONDITION_PATHS: tuple[FieldPath, ...] = (("checksums", "recorded"),)
MANIFEST_EVENTS = ("launched", "finished", "registered")


class Blocked(SystemExit):
    """A rule of the pilot forbids the requested action. Nothing was started (as in the LR pilot)."""

    def __init__(self, message: str) -> None:
        super().__init__(f"BLOCKED: {message}")
        self.message = message


@dataclass(frozen=True)
class PilotPaths:
    """Locations of one pilot's records (protocol §12.2)."""

    root: Path

    @property
    def conditions(self) -> Path:
        return self.root / "conditions.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def invocations(self) -> Path:
        return self.root / "invocations.jsonl"

    @property
    def attempts_dir(self) -> Path:
        return self.root / "attempts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def lock_file(self) -> Path:
        return self.root / ".runner.lock"

    def attempt_result(self, attempt_id: str) -> Path:
        return self.attempts_dir / f"{attempt_id}.json"


# ----------------------------------------------------------------- checksums and environment
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 of the canonical JSON form (sorted keys, no whitespace, no NaN)."""
    return _sha256_text(canonical_json(obj))


def python_files_under(repo: str | Path, directory: str) -> list[str]:
    """Repository-relative paths of every ``*.py`` below ``directory`` (``__pycache__`` excluded)."""
    repo = Path(repo)
    return sorted(p.relative_to(repo).as_posix() for p in (repo / directory).rglob("*.py")
                  if "__pycache__" not in p.parts)


def checksum_record(repo: str | Path, *, blocking: Sequence[str], recorded: Sequence[str]) -> dict[str, Any]:
    """``{"blocking": {path: sha256}, "recorded": {path: sha256}}`` (DEC-1). A missing file blocks."""
    repo = Path(repo)
    overlap = sorted(set(blocking) & set(recorded))
    if overlap:
        raise ValueError(f"files listed as both blocking and recorded: {overlap}")
    out: dict[str, Any] = {"blocking": {}, "recorded": {}}
    for cls, files in (("blocking", blocking), ("recorded", recorded)):
        for rel in sorted(set(files)):
            path = repo / rel
            if not path.is_file():
                raise Blocked(f"{cls} checksum file missing: {rel}")
            out[cls][rel] = sha256_file(path)
    return out


def cpu_description() -> str | None:
    """CPU model name. macOS: ``sysctl``; Linux: ``/proc/cpuinfo`` (copied from the LR pilot)."""
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=5, check=True).stdout.strip() or None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001  (best-effort description only)
        return None
    return None


def package_version(name: str) -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_record() -> dict[str, Any]:
    """Software, hardware and thread configuration that must stay identical (protocol §10.1)."""
    import platform
    import sys

    import torch

    return {
        "python": sys.version, "python_executable": sys.executable,
        "packages": {p: package_version(p) for p in ("torch", "numpy", "PyYAML")},
        "platform": platform.platform(), "machine": platform.machine(), "cpu": cpu_description(),
        "intra_op_threads": torch.get_num_threads(), "inter_op_threads": torch.get_num_interop_threads(),
    }


# ----------------------------------------------------------------- frozen conditions
def _normalised(obj: Any) -> Any:
    """JSON round trip: tuples become lists, keys strings; non-finite floats are refused."""
    return json.loads(json.dumps(obj, allow_nan=False))


def _is_nonblocking(path: FieldPath, nonblocking: Sequence[FieldPath]) -> bool:
    return any(path[:len(p)] == p for p in nonblocking)


def load_frozen(paths: PilotPaths) -> dict[str, Any] | None:
    """Read and self-verify ``conditions.json`` (``None`` if absent)."""
    if not paths.conditions.is_file():
        return None
    try:
        frozen = json.loads(paths.conditions.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise Blocked(f"conditions.json is not valid JSON: {exc}") from None
    if not isinstance(frozen, dict) or {"conditions", "conditions_sha256"} - set(frozen):
        raise Blocked("conditions.json lacks 'conditions' or 'conditions_sha256'")
    if sha256_json(frozen["conditions"]) != frozen["conditions_sha256"]:
        raise Blocked("conditions.json does not match its own conditions_sha256 (edited after freezing?)")
    return frozen


def verify_or_freeze(paths: PilotPaths, current: Mapping[str, Any], *, may_create: bool, lock: PilotLock | None,
                     nonblocking: Sequence[FieldPath] = NONBLOCKING_CONDITION_PATHS,
                     note: str = "") -> tuple[dict[str, Any], list[FieldPath]]:
    """Compare ``current`` with the frozen conditions, or freeze them on the first ``run``.

    Returns ``(frozen, nonblocking_differences)``. Any other difference raises :class:`Blocked`
    listing every differing path. Freezing requires the exclusive lock and an empty manifest.
    """
    current = _normalised(current)
    frozen = load_frozen(paths)
    if frozen is None:
        if paths.manifest.exists() and _read_chain(paths.manifest):
            raise Blocked("manifest.jsonl exists but conditions.json is missing; the pilot state is inconsistent")
        if not may_create:
            raise Blocked("the pilot has not been started (no conditions.json)")
        _require_lock(lock, "freezing the conditions")
        frozen = {"created_at": _utc_now(), "conditions": current, "conditions_sha256": sha256_json(current),
                  "note": note or "Frozen at the first run. Any blocking difference stops the pilot until a "
                                  "documented amendment is approved."}
        paths.root.mkdir(parents=True, exist_ok=True)
        _write_json(paths.conditions, frozen)
        return frozen, []
    diffs = diff_paths(frozen["conditions"], current)
    blocking = [p for p in diffs if not _is_nonblocking(p, nonblocking)]
    if blocking:
        shown = "\n  ".join(format_path(p) for p in blocking[:40]) + ("\n  ..." if len(blocking) > 40 else "")
        raise Blocked(f"current conditions differ from conditions.json (documented amendment required):\n  {shown}")
    return frozen, [p for p in diffs if _is_nonblocking(p, nonblocking)]


# ----------------------------------------------------------------- exclusive lock
class PilotLock:
    """Exclusive, non-blocking ``flock`` on ``<root>/.runner.lock``; released if the process dies."""

    def __init__(self, paths: PilotPaths, command: str) -> None:
        self.paths, self.command = paths, command
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def __enter__(self) -> PilotLock:  # noqa: PYI034
        import fcntl

        if self._fd is not None:
            raise RuntimeError("lock already held by this object")
        self.paths.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.paths.lock_file, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise Blocked("another pilot command is active (the exclusive lock is held)") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} command {self.command} since {_utc_now()}\n".encode())
        self._fd = fd
        return self

    def __exit__(self, *exc: object) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)  # closing releases the lock


def lock_held_elsewhere(paths: PilotPaths) -> bool:
    """True if some other open lock (another process or another :class:`PilotLock`) holds the lock."""
    import fcntl

    if not paths.lock_file.is_file():
        return False
    fd = os.open(paths.lock_file, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _require_lock(lock: PilotLock | None, what: str) -> None:
    if not isinstance(lock, PilotLock) or not lock.held:
        raise RuntimeError(f"{what} requires the pilot's exclusive lock")


def _read_chain(path: Path) -> list[dict[str, Any]]:
    try:
        return HashChainedLog(path).read()[0]
    except LedgerError as exc:
        raise Blocked(f"{path.name} failed verification: {exc}") from None


def log_invocation(paths: PilotPaths, lock: PilotLock, command: str, **fields: Any) -> str:
    """Append one hash-chained invocation record; returns the new chain head."""
    _require_lock(lock, "logging an invocation")
    return HashChainedLog(paths.invocations).append(_plain({"command": command, "at": _utc_now(), **fields}))


# ----------------------------------------------------------------- attempt store
class AttemptStore:
    """Append-only attempt records (protocol §12.3) over a hash-chained ``manifest.jsonl``.

    Manifest events per attempt: exactly one ``launched``, then at most one ``finished`` (written
    by the launching invocation) or ``registered`` (written by a later invocation for an attempt
    whose runner died). Every read validates this sequence; any violation is :class:`Blocked`.
    Writing methods require the pilot's exclusive lock.
    """

    def __init__(self, paths: PilotPaths, *, key_fields: tuple[str, ...],
                 attempt_id: Callable[[Mapping[str, Any], int], str], lock: PilotLock | None = None,
                 run_attempt_id_path: tuple[str, ...] = ("pilot", "attempt_id")) -> None:
        if not key_fields:
            raise ValueError("key_fields must not be empty")
        self.paths, self.key_fields, self._attempt_id = paths, tuple(key_fields), attempt_id
        self.lock, self.run_attempt_id_path = lock, run_attempt_id_path
        self.manifest = HashChainedLog(paths.manifest)

    # ------------------------------------------------------------- keys
    def key_of(self, record: Mapping[str, Any]) -> tuple:
        missing = [f for f in self.key_fields if f not in record]
        if missing:
            raise Blocked(f"record lacks key fields {missing}")
        key = tuple(record[f] for f in self.key_fields)
        if not all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in key):
            raise Blocked(f"key values must be strings or ints, got {key!r}")
        return key

    def key_dict(self, key: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
        if isinstance(key, Mapping):
            values = self.key_of(key)
        else:
            values = tuple(key)
            if len(values) != len(self.key_fields):
                raise ValueError(f"key must have {len(self.key_fields)} values")
            self.key_of(dict(zip(self.key_fields, values, strict=True)))
        return dict(zip(self.key_fields, values, strict=True))

    # ------------------------------------------------------------- reading
    def attempts(self) -> dict[str, dict[str, Any]]:
        """One merged, validated record per attempt id, in launch order."""
        out: dict[str, dict[str, Any]] = {}
        for n, row in enumerate(_read_chain(self.paths.manifest), start=1):
            event, aid = row.get("event"), row.get("attempt_id")
            if event not in MANIFEST_EVENTS or not isinstance(aid, str) or not aid:
                raise Blocked(f"manifest line {n}: invalid event {event!r} / attempt id {aid!r}")
            if event == "launched":
                if aid in out:
                    raise Blocked(f"manifest line {n}: attempt {aid} launched twice")
                out[aid] = {**row, "status": "launched", "events": ["launched"], "key": self.key_of(row)}
                continue
            rec = out.get(aid)
            if rec is None:
                raise Blocked(f"manifest line {n}: {event} for unknown attempt {aid}")
            if rec["status"] != "launched":
                raise Blocked(f"manifest line {n}: attempt {aid} already closed ({rec['status']})")
            if self.key_of(row) != rec["key"]:
                raise Blocked(f"manifest line {n}: attempt {aid} key changed")
            if row.get("status") not in ATTEMPT_STATUSES:
                raise Blocked(f"manifest line {n}: unknown status {row.get('status')!r}")
            rec.update({k: v for k, v in row.items() if k != "event"})
            rec["events"].append(event)
        return out

    def by_key(self) -> dict[tuple, list[dict[str, Any]]]:
        out: dict[tuple, list[dict[str, Any]]] = {}
        for rec in self.attempts().values():
            out.setdefault(rec["key"], []).append(rec)
        return out

    def read_result(self, attempt_id: str, key: tuple) -> dict[str, Any] | None:
        """The attempt's own result file, validated against the manifest (``None`` if absent)."""
        path = self.paths.attempt_result(attempt_id)
        if not path.is_file():
            return None
        try:
            res = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise Blocked(f"{path.name} is not valid JSON: {exc}") from None
        if not isinstance(res, dict) or res.get("attempt_id") != attempt_id or self.key_of(res) != key:
            raise Blocked(f"{path.name} does not belong to attempt {attempt_id} {key}")
        if res.get("status") not in ATTEMPT_STATUSES:
            raise Blocked(f"{path.name}: unknown status {res.get('status')!r}")
        return res

    def run_dirs_by_attempt(self) -> dict[str | None, list[str]]:
        """Run directories grouped by the attempt id their own ``metadata.json`` records."""
        out: dict[str | None, list[str]] = {}
        for p in sorted(self.paths.runs_dir.glob("*/run_*")):
            if not p.is_dir():
                continue
            try:
                value: Any = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
                for part in self.run_attempt_id_path:
                    value = value[part]
                aid = value if isinstance(value, str) else None
            except (OSError, ValueError, KeyError, TypeError):
                aid = None
            out.setdefault(aid, []).append(_repo_relative(p))
        return out

    def run_dir_of(self, attempt_id: str) -> str | None:
        dirs = self.run_dirs_by_attempt().get(attempt_id, [])
        return dirs[0] if len(dirs) == 1 else None

    def orphans(self) -> list[str]:
        """Run directories and result files that do not belong to exactly one known attempt."""
        known = self.attempts()
        bad = []
        for aid, dirs in self.run_dirs_by_attempt().items():
            if aid not in known or len(dirs) > 1:
                bad += [f"run directory {d}" for d in dirs]
        if self.paths.attempts_dir.is_dir():
            bad += [f"result file {p.name}" for p in sorted(self.paths.attempts_dir.glob("*.json"))
                    if p.stem not in known]
        return bad

    def sweep_state(self) -> dict[str, list[str]]:
        """Latest attempt per key, grouped: halted (category A), integrity_blocked, unresolved, launched."""
        state: dict[str, list[str]] = {"halted": [], "integrity_blocked": [], "unresolved": [], "launched": []}
        for recs in self.by_key().values():
            last = recs[-1]
            status, aid = last["status"], last["attempt_id"]
            if status == "mechanically_unstable":
                state["halted"].append(aid)
            elif status == "integrity_blocked":
                state["integrity_blocked"].append(aid)
            elif status in ("error", "crashed", "interrupted"):
                state["unresolved"].append(f"{aid} ({status})")
            elif status == "launched":
                state["launched"].append(aid)
        return state

    # ------------------------------------------------------------- state gates
    def check_state(self) -> list[str]:
        """Common gate for every run / retry: register unfinished attempts, refuse on orphans."""
        registered = self.register_unfinished()
        orphans = self.orphans()
        if orphans:
            raise Blocked("records without an attempt (inspect them before continuing):\n  " + "\n  ".join(orphans))
        return registered

    def require_runnable(self) -> None:
        """No new run while the pilot is halted, integrity-blocked or has unresolved attempts."""
        state = self.sweep_state()
        if state["halted"]:
            raise Blocked("the pilot is halted after mechanical instability (protocol §9.4): "
                          + ", ".join(state["halted"]))
        if state["integrity_blocked"]:
            raise Blocked("the sweep is integrity_blocked (protocol §10.5); human review and an authorised "
                          "retry are required: " + ", ".join(state["integrity_blocked"]))
        if state["unresolved"] or state["launched"]:
            raise Blocked("unresolved attempts must be inspected and explicitly retried first: "
                          + ", ".join(state["unresolved"] + state["launched"]))

    def pending(self, keys: Iterable[Mapping[str, Any] | Sequence[Any]]) -> list[dict[str, Any]]:
        """Requested keys (duplicates removed, order kept) that have no attempt yet."""
        existing = self.by_key()
        out, seen = [], set()
        for key in keys:
            d = self.key_dict(key)
            k = self.key_of(d)
            if k not in seen and k not in existing:
                out.append(d)
            seen.add(k)
        return out

    # ------------------------------------------------------------- writing
    def _append(self, row: Mapping[str, Any]) -> None:
        _require_lock(self.lock, "writing the manifest")
        self.manifest.append(_plain(row))

    def launch(self, key: Mapping[str, Any] | Sequence[Any], *, retry: Mapping[str, Any] | None = None,
               **fields: Any) -> str:
        """Register a new attempt for ``key`` (``launched``); returns its attempt id.

        A key without attempts is launched directly. A key with attempts only through ``retry``,
        the token returned by :meth:`authorize_retry`, whose rules are checked again here.
        """
        _require_lock(self.lock, "launching an attempt")
        kd = self.key_dict(key)
        reserved = {"event", "attempt_id", "status", "retry_of", "retry_reason", "retry_authorized_by",
                    *self.key_fields} & set(fields)
        if reserved:
            raise ValueError(f"fields may not set {sorted(reserved)}")
        recs = self.by_key().get(self.key_of(kd), [])
        if recs and recs[-1]["status"] == "launched":
            raise Blocked(f"attempt {recs[-1]['attempt_id']} for {kd} is still unfinished")
        if recs and recs[-1]["status"] in FINAL_STATUSES:
            raise Blocked(f"{recs[-1]['attempt_id']} for {kd} is final ({recs[-1]['status']})")
        retry_fields: dict[str, Any] = {}
        if recs:
            if retry is None:
                raise Blocked(f"{kd} already has attempts; a new attempt must be an authorised retry")
            again = self.authorize_retry(kd, reason=retry.get("retry_reason"),
                                         authorized_by=retry.get("retry_authorized_by"),
                                         conditions_sha256=retry.get("conditions_sha256"))
            if again != dict(retry):
                raise Blocked("the retry authorisation does not match the current attempt records")
            retry_fields = {k: v for k, v in again.items() if k != "conditions_sha256"}
        elif retry is not None:
            raise Blocked(f"{kd} has no attempt to retry")
        aid = self._attempt_id(kd, len(recs) + 1)
        if aid in self.attempts() or self.paths.attempt_result(aid).exists() or self.run_dir_of(aid):
            raise Blocked(f"attempt id {aid} is already in use")
        self._append({"event": "launched", "attempt_id": aid, **kd, **retry_fields, **fields,
                      "launched_at": _utc_now()})
        return aid

    def _close_row(self, aid: str, rec: Mapping[str, Any], *, interrupted: bool) -> dict[str, Any]:
        res = self.read_result(aid, rec["key"])
        if res is not None:
            failure = res.get("failure")
            return {"status": res["status"], "phase": res.get("phase"), "run_dir": res.get("run_dir"),
                    "failure": None if failure is None else {k: v for k, v in failure.items() if k != "traceback"},
                    "result_sha256": sha256_file(self.paths.attempt_result(aid))}
        return {"status": "interrupted" if interrupted else "crashed", "phase": "unknown",
                "run_dir": self.run_dir_of(aid),
                "failure": {"type": "no result file", "message": "the attempt ended without recording a result"},
                "result_sha256": None}

    def finish(self, attempt_id: str, *, returncode: int | None = None, interrupted: bool = False,
               **fields: Any) -> dict[str, Any]:
        """Close a launched attempt from its result file (``crashed`` / ``interrupted`` without one)."""
        rec = self.attempts().get(attempt_id)
        if rec is None or rec["status"] != "launched":
            raise Blocked(f"attempt {attempt_id} is not an unfinished attempt")
        row = {"event": "finished", "attempt_id": attempt_id, **self.key_dict(rec["key"]),
               **self._close_row(attempt_id, rec, interrupted=interrupted), "returncode": returncode,
               "ended_at": _utc_now(), **fields}
        self._append(row)
        return self.attempts()[attempt_id]

    def register_unfinished(self) -> list[str]:
        """Close attempts whose runner died (``launched`` without ``finished``): ``registered`` rows."""
        _require_lock(self.lock, "registering unfinished attempts")
        done = []
        for aid, rec in self.attempts().items():
            if rec["status"] != "launched":
                continue
            row = {"event": "registered", "attempt_id": aid, **self.key_dict(rec["key"]),
                   **self._close_row(aid, rec, interrupted=False), "registered_at": _utc_now(),
                   "note": "closing record written by a later invocation"}
            self._append(row)
            done.append(f"{aid} -> {row['status']}")
        return done

    def authorize_retry(self, key: Mapping[str, Any] | Sequence[Any], *, reason: str, authorized_by: str | None,
                        conditions_sha256: str) -> dict[str, Any]:
        """Check the retry rules (§10.5, §12.3; A4) and return the fields for :meth:`launch`."""
        _require_lock(self.lock, "authorising a retry")
        state = self.sweep_state()
        if state["halted"]:
            raise Blocked("the pilot is halted after mechanical instability; retries are refused (§9.4)")
        kd = self.key_dict(key)
        recs = self.by_key().get(self.key_of(kd))
        if not recs:
            raise Blocked(f"no attempt exists for {kd}")
        last = recs[-1]
        if last["status"] not in RETRYABLE_STATUSES:
            raise Blocked(f"{last['attempt_id']} is {last['status']}; only {', '.join(RETRYABLE_STATUSES)} "
                          "attempts can be retried (completed and mechanically unstable results are final)")
        if not isinstance(reason, str) or not reason.strip():
            raise Blocked("a retry needs a non-empty written reason")
        if last["status"] in AUTHORIZATION_REQUIRED_STATUSES and \
                (not isinstance(authorized_by, str) or not authorized_by.strip()):
            raise Blocked(f"{last['attempt_id']} is {last['status']}: a retry requires --authorized-by "
                          "(recorded as audit provenance, not authentication)")
        res = self.read_result(last["attempt_id"], last["key"])
        recorded = res.get("conditions_sha256") if res else None
        if recorded is not None and recorded != conditions_sha256:
            raise Blocked(f"{last['attempt_id']} ran under different conditions ({recorded[:12]}); retry refused")
        return {"retry_of": last["attempt_id"], "retry_reason": reason.strip(),
                "retry_authorized_by": authorized_by.strip() if isinstance(authorized_by, str) else None,
                "conditions_sha256": conditions_sha256}

    # ------------------------------------------------------------- child side
    def verify_launched(self, attempt_id: str, key: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
        """For the child process: the attempt is launched by an active invocation holding the lock."""
        rec = self.attempts().get(attempt_id)
        if not lock_held_elsewhere(self.paths) or rec is None or rec["status"] != "launched" \
                or rec["key"] != self.key_of(self.key_dict(key)):
            raise Blocked(f"attempt {attempt_id!r} is not registered as launched by an active invocation")
        return rec


def write_attempt_result(paths: PilotPaths, attempt_id: str, result: Mapping[str, Any]) -> Path:
    """Write an attempt's result file once (atomic); an existing file is never replaced."""
    path = paths.attempt_result(attempt_id)
    if path.exists():
        raise Blocked(f"result file for {attempt_id} already exists; attempt results are written once")
    if result.get("attempt_id") != attempt_id:
        raise ValueError("result attempt_id does not match")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, result)
    return path


# =============================================================================
# 11. Production Git guard, invocation sessions and record-head verification — Stage 4
# =============================================================================
GIT_DISABLED = {"available": False, "reason": "git disabled by project policy"}
_GIT_WORD = re.compile(r"(?:^|[\s;&|()`'\"/=])git(?:\.exe)?(?=$|[\s;&|()`'\"])")


class GitBlockedError(PermissionError):
    """Raised instead of starting a ``git`` process."""


def _mentions_git(*parts: Any) -> bool:
    for part in parts:
        if part is None:
            continue
        items = [part] if isinstance(part, (str, bytes, os.PathLike)) else list(part)
        for item in items:
            text = os.fsdecode(item) if isinstance(item, (bytes, os.PathLike)) else str(item)
            if os.path.basename(text.strip()) in ("git", "git.exe") or _GIT_WORD.search(text):
                return True
    return False


def disable_git() -> None:
    """Make Git impossible in this process (project policy; D-15).

    1. ``utils.run_manager.git_info`` returns :data:`GIT_DISABLED` instead of running ``git``.
    2. Starting any program named ``git`` is refused before a process exists: every
       ``subprocess`` entry point (at ``Popen._execute_child``), ``os.system``,
       ``os.posix_spawn*``, ``os.spawn*`` and ``os.exec*``, including shell strings and
       wrappers such as ``sh -c 'git …'``. Idempotent. Nothing on disk is modified.
    """
    import subprocess

    from utils import run_manager

    run_manager.git_info = lambda *args, **kwargs: dict(GIT_DISABLED)
    if getattr(subprocess.Popen, "_study_git_guard", False):
        return
    original_execute_child = subprocess.Popen._execute_child

    def _execute_child(self: Any, args: Any, executable: Any, *rest: Any, **kwargs: Any) -> Any:
        if _mentions_git(args, executable):
            raise GitBlockedError(f"git disabled by project policy (refused subprocess {args!r})")
        return original_execute_child(self, args, executable, *rest, **kwargs)

    subprocess.Popen._execute_child = _execute_child
    subprocess.Popen._study_git_guard = True

    def wrap(name: str, picks: Callable[[tuple], tuple]) -> None:
        original = getattr(os, name, None)
        if original is None:
            return

        def guarded(*args: Any, **kwargs: Any) -> Any:
            if _mentions_git(*picks(args)):
                raise GitBlockedError(f"git disabled by project policy (refused os.{name}: {args!r})")
            return original(*args, **kwargs)

        guarded.__name__ = name
        setattr(os, name, guarded)

    wrap("system", lambda a: a[:1])
    for name in ("posix_spawn", "posix_spawnp"):
        wrap(name, lambda a: a[:2])
    for name in ("execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe"):
        wrap(name, lambda a: a)
    for name in ("spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe"):
        wrap(name, lambda a: a[1:])


def manifest_head(paths: PilotPaths) -> str:
    """Current, verified chain head of ``manifest.jsonl`` (``GENESIS`` if empty)."""
    if not paths.manifest.exists():
        return GENESIS
    try:
        return HashChainedLog(paths.manifest).read()[1]
    except LedgerError as exc:
        raise Blocked(f"manifest.jsonl failed verification: {exc}") from None


def _line_heads(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Verified entries and the chain head after each line (index 0: ``GENESIS``)."""
    entries = _read_chain(path) if path.exists() else []
    heads = [GENESIS]
    if path.exists():
        heads += [_sha256_text(line) for line in path.read_text(encoding="utf-8").split("\n")[:-1]]
    return entries, heads


def verify_record_heads(paths: PilotPaths, store: AttemptStore) -> dict[str, Any]:
    """Cross-verify ``manifest.jsonl`` against the heads recorded elsewhere (approval A5).

    The hash chain alone cannot show an edit or truncation of the manifest's *last* line. Every
    manifest write is therefore followed by an invocation record holding the new manifest head,
    and every attempt result holds the head written right after its ``launched`` row. This
    function, independent of the writers:

    * requires every head recorded in ``invocations.jsonl`` to be a head of the current manifest,
      in non-decreasing order;
    * requires every manifest line to be covered by a recorded head, except a tail written by an
      invocation that never recorded its ``end`` (the runner died between a manifest write and
      the next invocation record; such a tail is reported and adopted by the next invocation);
    * requires each attempt result's ``manifest_head`` to equal the head right after that
      attempt's ``launched`` row, and each result file's SHA-256 to equal the one in its closing
      manifest row.

    Tamper *evidence*, not immutability: a writer able to rewrite every record consistently is
    not detected (documented limitation).
    """
    rows, heads = _line_heads(paths.manifest)
    position = {h: i for i, h in enumerate(heads)}
    invocations = _read_chain(paths.invocations) if paths.invocations.exists() else []
    last, last_invocation, ended = 0, None, set()
    for n, rec in enumerate(invocations, start=1):
        h = rec.get("manifest_head")
        if h not in position:
            raise Blocked(f"invocations.jsonl record {n} names manifest head {str(h)[:12]}… which is not a head "
                          "of the current manifest.jsonl (manifest edited or truncated)")
        if position[h] < last:
            raise Blocked(f"invocations.jsonl record {n}: the recorded manifest head goes backwards")
        last, last_invocation = position[h], rec.get("invocation_id")
        if rec.get("event") == "end":
            ended.add(last_invocation)
    tail = len(heads) - 1 - last
    if tail and (last_invocation is None or last_invocation in ended):
        raise Blocked(f"{tail} manifest line(s) are not covered by any invocation record although the last "
                      "invocation ended (manifest edited or extended outside an invocation)")
    launch_line = {row["attempt_id"]: i + 1 for i, row in enumerate(rows) if row.get("event") == "launched"}
    for aid, rec in store.attempts().items():
        res = store.read_result(aid, rec["key"])
        if res is None:
            continue
        if res.get("manifest_head") != heads[launch_line[aid]]:
            raise Blocked(f"result of {aid} records manifest head {str(res.get('manifest_head'))[:12]}…, not the "
                          "head after its launched row")
        if rec["status"] != "launched" and rec.get("result_sha256") not in (None,
                                                                          sha256_file(paths.attempt_result(aid))):
            raise Blocked(f"result file of {aid} changed after the attempt was closed")
        if rec["status"] != "launched" and rec.get("result_sha256") is None:
            raise Blocked(f"result file of {aid} appeared after the attempt was closed without one")
    return {"manifest_lines": len(heads) - 1, "manifest_head": heads[-1], "unrecorded_tail": tail,
            "invocation_records": len(invocations)}


class InvocationSession:
    """One command's invocation records; every record carries the current manifest head.

    Created under the exclusive lock, after the frozen conditions were verified. Verifies the
    recorded heads first (:func:`verify_record_heads`), then writes ``start``; the owner calls
    :meth:`note` after every manifest write and :meth:`end` when the command finishes.
    """

    def __init__(self, paths: PilotPaths, lock: PilotLock, store: AttemptStore, command: str,
                 **fields: Any) -> None:
        import uuid

        _require_lock(lock, "an invocation session")
        self.paths, self.lock, self.store, self.command = paths, lock, store, command
        self.verification = verify_record_heads(paths, store)
        self.invocation_id = uuid.uuid4().hex
        self.ended = False
        self.note("start", adopted_unrecorded_tail=self.verification["unrecorded_tail"], **fields)

    def note(self, event: str, **fields: Any) -> str:
        return log_invocation(self.paths, self.lock, self.command, invocation_id=self.invocation_id, event=event,
                              manifest_head=manifest_head(self.paths), **fields)

    def end(self, **fields: Any) -> None:
        if not self.ended:
            self.ended = True
            self.note("end", **fields)
