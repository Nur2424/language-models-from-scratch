"""Stage 1: non-finite handling, flattening and bit-exact comparison (protocol §10.4)."""

from __future__ import annotations

import copy
import math

import pytest

from training.wavenet_study import study_common as sc


# ----------------------------------------------------------------- non-finite values (F1)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "nan", "inf", "-inf"])
def test_nonfinite_values_and_their_jsonable_strings_are_nonfinite(value):
    assert sc.is_nonfinite(value)


@pytest.mark.parametrize("value", [0.0, -0.0, 1e308, 5, "NaN", "Inf", "wavenet", None, True, {}])
def test_finite_values_and_other_strings_are_not_nonfinite(value):
    assert not sc.is_nonfinite(value)


def test_is_number_excludes_bool():
    assert sc.is_number(1) and sc.is_number(1.5) and not sc.is_number(True)


# ----------------------------------------------------------------- flatten
def test_flatten_uses_tuple_paths_so_dotted_keys_are_unambiguous():
    rec = {"a": {"b.c": 1}, "a.b": {"c": 2}, "e": {}, "l": [3, {"x": 4}], "z": []}
    flat = sc.flatten(rec)
    assert flat == {("a", "b.c"): 1, ("a.b", "c"): 2, ("e",): {}, ("l", 0): 3, ("l", 1, "x"): 4, ("z",): []}


def test_flatten_rejects_non_string_keys():
    with pytest.raises(TypeError):
        sc.flatten({1: 2})


def test_float_bits_distinguishes_signed_zero():
    assert sc.float_bits(0.0) != sc.float_bits(-0.0)


# ----------------------------------------------------------------- compare_records
REF = {"epoch": 0, "dev_nll": 2.5, "is_best": True, "name": "x", "seconds": 1.0,
       "activations": {"act": {"mean": 0.1, "nan_count": 0}}, "nested": {"empty": {}}}


def test_identical_records_match():
    assert sc.compare_records(REF, copy.deepcopy(REF)) is None


def test_one_ulp_difference_fails():
    new = copy.deepcopy(REF)
    new["dev_nll"] = math.nextafter(2.5, 3.0)
    m = sc.compare_records(REF, new)
    assert m.kind == "value" and m.path == ("dev_nll",)


def test_signed_zero_and_int_float_type_differences_fail():
    a = {"v": 0.0}
    assert sc.compare_records(a, {"v": -0.0}).kind == "value"
    assert sc.compare_records({"v": 1.0}, {"v": 1}).kind == "value"
    assert sc.compare_records({"v": True}, {"v": 1}).kind == "value"


@pytest.mark.parametrize("bad", [math.nan, "nan", math.inf, "-inf"])
def test_nonfinite_values_never_match_even_themselves(bad):
    a = {"v": bad}
    m = sc.compare_records(a, {"v": bad})
    assert m is not None and m.kind == "nonfinite"


def test_null_matches_only_under_the_null_policy():
    a, b = {"v": None, "w": 1}, {"v": None, "w": 1}
    assert sc.compare_records(a, b).kind == "null"
    assert sc.compare_records(a, b, null_allowed=lambda p, r: p == ("v",)) is None
    assert sc.compare_records({"v": None}, {"v": 1.0}, null_allowed=lambda p, r: True).kind == "null"


def test_missing_and_extra_fields_fail():
    new = copy.deepcopy(REF)
    del new["activations"]["act"]["mean"]
    assert sc.compare_records(REF, new).kind == "missing_path"
    new = copy.deepcopy(REF)
    new["extra"] = 1
    assert sc.compare_records(REF, new).kind == "extra_path"


def test_exclude_removes_only_the_named_wall_clock_field():
    new = copy.deepcopy(REF)
    new["seconds"] = 99.0
    assert sc.compare_records(REF, new, exclude=[("seconds",)]) is None
    new["dev_nll"] = 0.0
    assert sc.compare_records(REF, new, exclude=[("seconds",)]).path == ("dev_nll",)


def test_first_mismatch_is_deterministic():
    new = copy.deepcopy(REF)
    new["dev_nll"], new["epoch"] = 0.0, 5
    assert sc.compare_records(REF, new).path == ("dev_nll",)  # sorted path order


# ----------------------------------------------------------------- keyed records
def _steps(n=4):
    return [{"epoch": i // 2, "step": i * 10, "loss": float(i)} for i in range(n)]


def test_steps_are_matched_by_identifiers_not_by_position():
    ref = _steps()
    assert sc.compare_keyed_records(ref, list(reversed(copy.deepcopy(ref)))) is None


def test_missing_extra_duplicate_and_non_int_keys_fail():
    ref = _steps()
    assert sc.compare_keyed_records(ref, ref[:-1]).kind == "missing_path"
    assert sc.compare_keyed_records(ref, ref + [{"epoch": 9, "step": 900, "loss": 0.0}]).kind == "extra_path"
    assert sc.compare_keyed_records(ref, ref + [copy.deepcopy(ref[0])]).kind == "duplicate_key"
    bad = copy.deepcopy(ref)
    bad[0]["step"] = "0"
    assert sc.compare_keyed_records(ref, bad).kind == "key"


def test_a_shifted_step_identifier_fails_even_with_equal_counts():
    ref = _steps()
    new = copy.deepcopy(ref)
    new[1]["step"] = 11
    assert sc.compare_keyed_records(ref, new) is not None


# ----------------------------------------------------------------- run prefix (G5 / G6 comparison)
def _history(n, seconds=1.0):
    return [{"epoch": e, "dev_nll": 2.0 - e / 10, "seconds": seconds} for e in range(n)]


def test_run_prefix_ignores_later_epochs_and_wall_clock():
    ok = sc.compare_run_prefix(reference_history=_history(2), reference_steps=_steps(4), reference_hash="h",
                               candidate_history=_history(5, seconds=7.0), candidate_steps=_steps(8),
                               candidate_hash="h", epochs=2)
    assert ok is None


def test_run_prefix_requires_every_prefix_epoch_and_the_state_hash():
    kw = {"reference_history": _history(2), "reference_steps": _steps(4), "reference_hash": "h",
          "candidate_steps": _steps(4), "epochs": 2}
    assert sc.compare_run_prefix(candidate_history=_history(1), candidate_hash="h", **kw).kind == "key"
    assert sc.compare_run_prefix(candidate_history=_history(2), candidate_hash="g", **kw).path == ("state_sha256",)
    shorter = {**kw, "candidate_steps": _steps(3)}
    assert sc.compare_run_prefix(candidate_history=_history(2), candidate_hash="h", **shorter).kind == "missing_path"
