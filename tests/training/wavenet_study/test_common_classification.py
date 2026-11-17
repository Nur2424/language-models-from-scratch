"""Stage 1: field-path classification into categories A / B / C (protocol §9.1, §9.3; approval A2).

The records come from the accepted trainers run on synthetic data, so their schemas are the
real ones. The expected path counts were measured on the 12 LR-0.03 reference runs (read-only
inspection, 2026-11-11); the synthetic records must reproduce them exactly.
"""

from __future__ import annotations

import collections
import copy

import pytest

from training.wavenet_study import study_common as sc

REFERENCE_PATH_COUNTS = {  # (history, steps) leaf paths per arm in the LR-0.03 reference runs
    "A": (36, 72), "B": (64, 88), "C": (108, 97), "D": (182, 166),
}


def _records(run):
    return (sc.read_jsonl_strict(run.run_dir / "history.jsonl"),
            sc.read_jsonl_strict(run.run_dir / "diagnostics" / "steps.jsonl"))


def _findings(kind, rec, norm):
    return sc.classify_record(kind, rec, normalization=norm)


@pytest.mark.parametrize("arm", list("ABCD"))
def test_synthetic_records_reproduce_the_reference_schemas_and_every_path_has_one_rule(synthetic_runs, arm):
    history, steps = _records(synthetic_runs[arm])
    h_schema, s_schema = sc.record_schema(history), sc.record_schema(steps)
    assert (len(h_schema), len(s_schema)) == REFERENCE_PATH_COUNTS[arm]
    for kind, schema in (("history", h_schema), ("steps", s_schema)):
        for path in schema:
            assert sc.rule_for(kind, path) is not None, (kind, path)  # raises if two rules match


@pytest.mark.parametrize("arm", list("ABCD"))
def test_clean_records_give_only_the_expected_category_c_observations(synthetic_runs, arm):
    run = synthetic_runs[arm]
    history, steps = _records(run)
    found = collections.Counter()
    for kind, recs in (("history", history), ("steps", steps)):
        for rec in recs:
            for f in _findings(kind, rec, run.normalization):
                found[(f.category, f.kind, f.step == 0 if kind == "steps" else "history")] += 1
    assert {k[0] for k in found} <= {"C"}
    n_params = len(list(run.model.parameters()))
    assert found[("C", "structural_null", True)] == n_params  # every parameter at step 0 (zero-output init)
    assert found[("C", "structural_null", "history")] == (len(history) if arm == "C" else 0)
    assert found[("C", "epsilon_regularised", True)] >= 2  # out.weight and out.bias start at zero norm
    assert not [k for k in found if k[2] is False]  # nothing at a later step


def _step(run, index):
    return copy.deepcopy(_records(run)[1][index])


def _hist(run, index=0):
    return copy.deepcopy(_records(run)[0][index])


def _kinds(findings):
    return {(f.category, f.kind) for f in findings}


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", float("nan")])
def test_nonfinite_model_quantity_is_category_a(synthetic_runs, bad):
    run = synthetic_runs["A"]
    rec = _hist(run)
    rec["dev_nll"] = bad
    assert ("A", "nonfinite_model_quantity") in _kinds(_findings("history", rec, "none"))


def test_nonfinite_counters_and_reported_gradients_are_category_a(synthetic_runs):
    run = synthetic_runs["D"]
    rec = _hist(run)
    rec["activations"]["hidden2"]["inf_count"] = 1
    assert ("A", "nonfinite_count") in _kinds(_findings("history", rec, "batchnorm"))
    s = _step(run, 3)
    s["nonfinite_gradients"] = {"hidden1.weight": 2}
    assert ("A", "nonfinite_gradient_reported") in _kinds(_findings("steps", s, "batchnorm"))
    s = _step(run, 3)
    s["updates"]["per_parameter"]["emb.weight"]["nonfinite_update"] = 1
    assert ("A", "nonfinite_count") in _kinds(_findings("steps", s, "batchnorm"))


def test_nonfinite_derived_ratio_is_category_b_not_a(synthetic_runs):
    run = synthetic_runs["C"]
    s = _step(run, 3)
    s["updates"]["per_parameter"]["hidden1.weight"]["relative_update"] = "inf"
    kinds = _kinds(_findings("steps", s, "none"))
    assert ("B", "nonfinite_derived_or_structural") in kinds and not any(c == "A" for c, _ in kinds)


def test_step0_exception_applies_only_at_step_0_and_only_under_its_exact_condition(synthetic_runs):
    run = synthetic_runs["A"]
    step0, later = _step(run, 0), _step(run, 2)
    assert step0["step"] == 0 and later["step"] > 0
    # a null update ratio after step 0 is category B
    later["updates"]["per_parameter"]["hidden.weight"]["log10_update_to_param_std"] = None
    assert ("B", "undefined_value") in _kinds(_findings("steps", later, "none"))
    # at step 0, the same null is category B when neither update_std nor param_norm is exactly 0.0
    entry = step0["updates"]["per_parameter"]["hidden.weight"]
    assert entry["log10_update_to_param_std"] is None and entry["update_std"] == 0.0
    entry["update_std"] = 1e-30
    assert ("B", "undefined_value") in _kinds(_findings("steps", step0, "none"))
    # -0.0 is not the exact +0.0 of the rule
    entry["update_std"] = -0.0
    assert ("B", "undefined_value") in _kinds(_findings("steps", step0, "none"))


def test_zero_param_norm_after_step0_is_category_b(synthetic_runs):
    run = synthetic_runs["A"]
    s = _step(run, 2)
    s["updates"]["per_parameter"]["out.weight"]["param_norm"] = 0.0
    kinds = _kinds(_findings("steps", s, "none"))
    assert ("B", "zero_param_norm_after_step0") in kinds and ("B", "epsilon_regularised_after_step0") in kinds


def test_epsilon_regularised_values_are_preserved_as_recorded(synthetic_runs):
    run = synthetic_runs["A"]
    s = _step(run, 0)
    found = [f for f in _findings("steps", s, "none") if f.kind == "epsilon_regularised"]
    assert found and all(f.value == s["updates"]["per_parameter"][f.path[2]]["relative_update"] for f in found)


def test_batchnorm_null_is_structural_only_in_arms_without_batchnorm(synthetic_runs):
    rec = _hist(synthetic_runs["D"])
    rec["batchnorm"] = None
    assert ("B", "unexpected_null_or_value") in _kinds(_findings("history", rec, "batchnorm"))
    rec_c = _hist(synthetic_runs["C"])
    assert ("C", "structural_null") in _kinds(_findings("history", rec_c, "none"))


def test_unknown_fields_wrong_types_and_nulls_never_pass_silently(synthetic_runs):
    run = synthetic_runs["B"]
    rec = _hist(run)
    rec["new_metric"] = 1.0
    assert ("B", "unclassified_field") in _kinds(_findings("history", rec, "batchnorm"))
    rec = _hist(run)
    rec["train_nll"] = "2.0"
    assert ("B", "unexpected_type") in _kinds(_findings("history", rec, "batchnorm"))
    rec = _hist(run)
    rec["activations"]["act"]["mean"] = None
    assert ("B", "undefined_value") in _kinds(_findings("history", rec, "batchnorm"))


def test_schema_check_reports_missing_and_extra_fields(synthetic_runs):
    history, _ = _records(synthetic_runs["B"])
    schema = sc.record_schema(history)
    rec = copy.deepcopy(history[0])
    del rec["batchnorm"]["gamma"]["mean"]
    rec["output"] = {"x": 1.0}
    kinds = {f.kind for f in sc.schema_findings("history", rec, schema)}
    assert kinds == {"missing_field", "extra_field"}


def test_record_schema_rejects_mixed_schemas():
    with pytest.raises(ValueError):
        sc.record_schema([{"a": 1}, {"b": 1}])


def test_overlapping_rules_are_rejected():
    rules = sc.ACCEPTED_TRAINER_FIELD_RULES + (sc.FieldRule("history", ("dev_nll",), "B", "number", "dup"),)
    with pytest.raises(sc.ClassificationError):
        sc.rule_for("history", ("dev_nll",), rules)
