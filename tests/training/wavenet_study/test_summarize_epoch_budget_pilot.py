"""Stage 5: the independent epoch-budget summarizer, on synthetic data only.

Part 1 tests the §16.5 statistics and decisions as pure functions against independently computed
values (exact rationals, 80-digit decimals, the spec's worked illustration, the protocol's derived
consequences). Part 2 runs a small synthetic pilot (arms A and B, cap 3 → 6) with the Stage 4
runner, then verifies the summary on it and on in-place tampered copies of single files, each
restored afterwards. The Stage 4 test helpers (synthetic names, LR-style reference runs, spec) are
reused, not duplicated.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import re
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from training.wavenet_study import epoch_budget_pilot as ebp
from training.wavenet_study import study_common as sc
from training.wavenet_study import summarize_epoch_budget_pilot as sm

REPO = ebp.REPO
_spec = importlib.util.spec_from_file_location("stage4_helpers", Path(__file__).with_name("test_epoch_budget_pilot.py"))
t4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t4)


# ============================================================ part 1: statistics (pure)
def _independent(values):
    """Exact mean; SD and SE from 80-digit decimal square roots of the exact variance."""
    exact = [Fraction(v) for v in values]
    n = len(exact)
    mean = sum(exact, Fraction(0)) / n
    var = sum((x - mean) ** 2 for x in exact) / (n - 1)
    with localcontext() as ctx:
        ctx.prec = 80
        sd = float((Decimal(var.numerator) / Decimal(var.denominator)).sqrt())
        se = float((Decimal((var / n).numerator) / Decimal((var / n).denominator)).sqrt())
    return float(mean), sd, se


def _curves(per_epoch):
    """per_epoch: list of (v_seed100, v_seed101, v_seed102) for epochs 1..E."""
    return {s: [row[i] for row in per_epoch] for i, s in enumerate((100, 101, 102))}


def test_statistics_match_an_independent_high_precision_computation():
    rows = [(2.3011, 2.2987, 2.3104), (2.1203, 2.1289, 2.1177), (2.0901, 2.0905, 2.0899), (2.08, 2.08, 2.08)]
    stats = sm.epoch_statistics(_curves(rows), 4)
    for st, row in zip(stats, rows, strict=True):
        mean, sd, se = _independent(row)
        assert (st["mean"], st["sd"], st["se"]) == (mean, sd, se)
        assert st["epsilon"] == max(1e-3, se) and st["values"] == {"100": row[0], "101": row[1], "102": row[2]}
    assert stats[3]["sd"] == 0.0 and stats[3]["epsilon_source"] == "floor-derived"
    assert stats[0]["epsilon_source"] == "SE-derived"


def test_epsilon_source_labels_use_exact_float_comparison():
    assert sm.epsilon_source(0.001) == "floor (SE equal)"
    assert sm.epsilon_source(math.nextafter(0.001, 1.0)) == "SE-derived"
    assert sm.epsilon_source(math.nextafter(0.001, 0.0)) == "floor-derived"
    assert sm.epsilon_source(0.0) == "floor-derived"


def test_worked_illustration_of_the_specification():
    """wavenet_spec §16.5: D̄* = 2.050 at ê = 24; epoch 18 has D̄ = 2.0508, SE = 0.0006 → e* = 18."""
    delta = 0.0006 * math.sqrt(3)  # values m − δ, m, m + δ have SE = δ/√3
    rows = []
    for e in range(1, 31):
        if e < 18:
            rows.append((2.10, 2.10, 2.10))
        elif e == 18:
            rows.append((2.0508 - delta, 2.0508, 2.0508 + delta))
        elif e == 24:
            rows.append((2.050, 2.050, 2.050))
        else:
            rows.append((2.0505, 2.0505, 2.0505))
    stats = sm.epoch_statistics(_curves(rows), 30)
    assert math.isclose(stats[17]["se"], 0.0006, rel_tol=1e-12) and stats[17]["epsilon"] == 0.001
    sel = sm.select_epoch(stats, 30)
    assert (sel["best_mean"], sel["best_epoch"], sel["selected_epoch"]) == (2.050, 24, 18)
    assert sel["epsilon_source_at_selected"] == "floor-derived" and not sel["still_improving_at_cap"]


def test_earliest_epoch_wins_exact_ties_for_the_best_and_for_the_selection():
    rows = [(3.0, 3.0, 3.0), (2.5, 2.5, 2.5), (2.0, 2.0, 2.0), (2.0, 2.0, 2.0), (2.25, 2.25, 2.25)]
    sel = sm.select_epoch(sm.epoch_statistics(_curves(rows), 5), 5)
    assert (sel["best_epoch"], sel["selected_epoch"]) == (3, 3)


def test_the_condition_is_less_or_equal_exactly_at_the_boundary():
    eps = 1e-3
    on_boundary = [(eps, eps, eps), (0.0, 0.0, 0.0)]  # exact mean of epoch 1 = D̄* + ε(1)
    assert sm.select_epoch(sm.epoch_statistics(_curves(on_boundary), 2), 2)["selected_epoch"] == 1
    above = math.nextafter(eps, 1.0)
    just_above = [(above, above, above), (0.0, 0.0, 0.0)]
    assert sm.select_epoch(sm.epoch_statistics(_curves(just_above), 2), 2)["selected_epoch"] == 2


def test_a_plateau_still_falling_below_the_tolerance_needs_no_extension():
    d = 2.0 ** -14  # exact dyadic step, 6.1e-5 nats per epoch (protocol §7 example, exact version)
    rows = [(1.0 + (30 - e) * d,) * 3 for e in range(1, 31)]
    sel = sm.select_epoch(sm.epoch_statistics(_curves(rows), 30), 30)
    assert (sel["best_epoch"], sel["selected_epoch"], sel["still_improving_at_cap"]) == (30, 14, False)


def test_still_improving_at_the_cap_requires_extension():
    rows = [(2.0 - 0.01 * e,) * 3 for e in range(1, 31)]
    sel = sm.select_epoch(sm.epoch_statistics(_curves(rows), 30), 30)
    assert sel["selected_epoch"] == 30 and sel["still_improving_at_cap"]


def test_a_single_deviating_seed_makes_an_epoch_qualify_through_se():
    """Protocol §8 item 4: two seeds at x, one at x + Δ with Δ/3 > 1e-3 → D̄ ≤ D̄* + SE iff x ≤ D̄*."""
    # (x, x, x + Δ) has D̄ = x + Δ/3 and SE = Δ/3 exactly, so D̄ − SE = x; Δ = 0.3, D̄* = 2.0.
    qualifies = [(1.99, 1.99, 2.29), (2.0, 2.0, 2.0)]  # x < D̄*
    sel = sm.select_epoch(sm.epoch_statistics(_curves(qualifies), 2), 2)
    assert sel["selected_epoch"] == 1 and sel["epsilon_source_at_selected"] == "SE-derived"
    fails = [(2.01, 2.01, 2.31), (2.0, 2.0, 2.0)]  # x > D̄*
    assert sm.select_epoch(sm.epoch_statistics(_curves(fails), 2), 2)["selected_epoch"] == 2


def test_statistics_refuse_non_finite_values_and_partial_curves():
    with pytest.raises(ValueError):
        sm.epoch_statistics(_curves([(2.0, float("nan"), 2.0)]), 1)
    with pytest.raises(ValueError):
        sm.select_epoch(sm.epoch_statistics(_curves([(2.0, 2.0, 2.0)] * 3), 3), 4)


def _arm(selected, source="floor-derived", status="selected"):
    return {"status": status, "selection": {"selected_epoch": selected, "epsilon_source_at_selected": source}}


@pytest.mark.parametrize("kwargs,outcome,budget", [
    ({"halted": True, "integrity_problems": ["x"]}, sm.OUTCOME_HALTED, None),
    ({"integrity_problems": ["G4"]}, sm.OUTCOME_INTEGRITY, None),
    ({"incomplete": ["A cap 30 seed 101"]}, sm.OUTCOME_INCOMPLETE, None),
    ({"arms": {"A": _arm(12), "B": {"status": "extension_required"}}}, sm.OUTCOME_EXTENSION, None),
    ({"arms": {"A": _arm(12), "B": {"status": "inconclusive"}}}, sm.OUTCOME_INCONCLUSIVE, None),
    ({"arms": {"A": _arm(12), "B": _arm(20)}, "category_b_blocking": ["B-x-0001"]}, sm.OUTCOME_CATEGORY_B, None),
    ({"arms": {"A": _arm(12), "B": _arm(20), "C": _arm(7)}}, sm.OUTCOME_BUDGET, 20),
    ({"arms": {"A": _arm(12, "SE-derived"), "B": _arm(20)}}, sm.OUTCOME_REVIEW, 20),
    ({"arms": {"A": _arm(12, "floor (SE equal)"), "B": _arm(20)}}, sm.OUTCOME_BUDGET, 20),
])
def test_outcome_order_and_common_budget(kwargs, outcome, budget):
    args = {"halted": False, "integrity_problems": [], "incomplete": [], "arms": {}, "category_b_blocking": []}
    res = sm.decide_outcome(**{**args, **kwargs})
    assert res["outcome"] == outcome and res["max_epochs"] == budget
    if outcome == sm.OUTCOME_REVIEW:
        assert res["requires_human_review"] == ["A"]


def test_optimisation_flags_are_descriptive():
    hist = [{"dev_nll": d, "running_train_nll": r} for d, r in ((2.0, 2.5), (1.9, 2.4), (1.95, 2.45), (1.8, 2.3),
                                                                  (1.85, 2.2))]
    f = sm.optimisation_flags(hist)
    assert [x["epoch"] for x in f["dev_nll_rises"]] == [3, 5] and f["repeated_dev_nll_rise"]
    assert f["running_train_nll_rises"] == [3] and f["best_epoch"] == 4 and not f["best_epoch_is_last"]


def test_extension_provenance_rules():
    spec = type("S", (), {"caps": [30, 60]})()
    attempts = {
        "A-c60-s100-a1": {"key": ("A", 60, 100), "stage": "extension", "authorized_by": "Nur"},
        "A-c60-s101-a1": {"key": ("A", 60, 101), "stage": "extension", "authorized_by": " "},
        "A-c60-s102-a1": {"key": ("A", 60, 102), "stage": "completion"},
        "A-c60-s100-a2": {"key": ("A", 60, 100), "stage": "retry", "retry_of": "A-c60-s100-a1"},
        "A-c30-s100-a1": {"key": ("A", 30, 100), "stage": "validation"},
    }
    problems = sm._extension_provenance(spec, attempts)
    assert sorted(p.split(":")[0] for p in problems) == ["A-c60-s101-a1", "A-c60-s102-a1"]


# ============================================================ part 2: a synthetic pilot
@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    world = _world(tmp_path_factory)
    spec = t4.make_spec(world, tmp_path_factory.mktemp("summary_pilot") / "pilot", arms="AB")
    quiet = []
    assert ebp.run_stage(spec, "validation", out=quiet.append) == 0
    status_before_completion = sm.evaluate_pilot(spec.root, selection=True)
    assert ebp.run_stage(spec, "completion", out=quiet.append) == 0
    after_completion = sm.evaluate_pilot(spec.root, selection=True)
    decision = sm.extension_decision(spec, list(spec.arms))
    to_extend = [a for a, need in decision.items() if need]
    refused_without_rule = None
    not_needed = [a for a, need in decision.items() if not need]
    if not_needed:
        with pytest.raises(sc.Blocked) as info:
            ebp.run_stage(spec, "extension", arms=not_needed, authorized_by="Nur",
                          extension_decision=sm.extension_decision, out=quiet.append)
        refused_without_rule = info.value.message
    if to_extend:
        assert ebp.run_stage(spec, "extension", arms=to_extend, authorized_by="Nur",
                             extension_decision=sm.extension_decision, out=quiet.append) == 0
    return {"spec": spec, "world": world, "before_completion": status_before_completion,
            "after_completion": after_completion, "decision": decision, "extended": to_extend,
            "refused_without_rule": refused_without_rule}


def _world(tmp_path_factory):
    base = tmp_path_factory.mktemp("summary_world")
    data = base / "names.txt"
    data.write_text("\n".join(t4._names()) + "\n")
    ref_root = base / "lr_pilot_reference"
    runs = {}
    for arm in "AB":
        for seed in ebp.SEEDS:
            d = t4.lr_style_reference(arm, seed, data=data, root=ref_root, epochs=t4.G5)
            runs[f"{arm}/{seed}"] = d.relative_to(ref_root).as_posix()
    (ref_root / "summary.json").write_text(json.dumps({"synthetic": True}))
    doc = base / "protocol_doc.md"
    doc.write_text("recorded, non-blocking document\n")
    return {"base": base, "data": data, "ref_root": ref_root, "runs": runs, "doc": doc}


def _evaluate(pilot, selection=True):
    return sm.evaluate_pilot(pilot["spec"].root, selection=selection)


def _gates(report):
    return {e["attempt_id"]: {g: r["status"] for g, r in e["gates"].items()} for e in report["attempts"]}


def _run_dir(pilot, aid):
    return ebp._run_path(ebp.make_store(pilot["spec"]).attempts()[aid]["run_dir"])


@contextlib.contextmanager
def tampered(path: Path, mutate):
    """Change one file in place; always restore its exact bytes."""
    original = path.read_bytes()
    try:
        mutate(path)
        yield
    finally:
        path.write_bytes(original)


def _jsonl_edit(fn):
    def mutate(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = fn(rows)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return mutate


def _ckpt_edit(fn):
    def mutate(path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        fn(payload)
        torch.save(payload, path)
    return mutate


# ---------------------------------------------------------------- clean pilot
def test_selection_is_refused_before_all_initial_runs_exist(pilot):
    r = pilot["before_completion"]
    assert not r["selection_prerequisites"]["met"] and "incomplete cap-3 runs" in r["selection_prerequisites"]["failed"]
    assert r["outcome"]["outcome"] == sm.OUTCOME_INCOMPLETE and "arms" not in r


def test_all_gates_pass_independently_on_the_clean_pilot(pilot):
    r = _evaluate(pilot)
    assert r["blocking"] == [] and r["integrity_problems"] == [] and r["conditions"]["status"] == "pass"
    assert r["records"]["unrecorded_tail"] == 0 and r["records"]["orphans"] == []
    for aid, gates in _gates(r).items():
        expected_g6 = "pass" if "-c6-" in aid else "not_applicable"
        assert gates == {"G1": "pass", "G2": "pass", "G3": "pass", "G4": "pass", "G5": "pass", "G6": expected_g6,
                         "G7": "pass"}, aid


def test_selection_statistics_equal_an_independent_recomputation_from_raw_histories(pilot):
    r = pilot["after_completion"]
    assert r["selection_prerequisites"]["met"]
    store = ebp.make_store(pilot["spec"])
    for arm, res in r["arms"].items():
        stats = res["caps"]["3"]["statistics"]
        curves = {}
        for s in ebp.SEEDS:
            rec = store.by_key()[(arm, 3, s)][0]
            hist = sc.read_jsonl_strict(ebp._run_path(rec["run_dir"]) / "history.jsonl")
            curves[s] = [row["dev_nll"] for row in hist]
        for e, st in enumerate(stats, 1):
            mean, sd, se = _independent([curves[s][e - 1] for s in ebp.SEEDS])
            assert (st["mean"], st["sd"], st["se"], st["epsilon"]) == (mean, sd, se, max(1e-3, se))
        means = [Fraction(sum(Fraction(curves[s][e]) for s in ebp.SEEDS), 3) for e in range(3)]
        best = min(means)
        eps = [Fraction(max(1e-3, st["se"])) for st in stats]
        expected_e = next(e + 1 for e in range(3) if means[e] <= best + eps[e])
        assert res["caps"]["3"]["selection"]["selected_epoch"] == expected_e
        assert res["extension_required"] == (expected_e == 3) == pilot["decision"][arm]


def test_extension_decisions_drive_the_runner_and_extended_arms_use_their_cap_6_curves(pilot):
    r = _evaluate(pilot)
    if pilot["refused_without_rule"]:
        assert "does not require an extension" in pilot["refused_without_rule"]
    for arm, res in r["arms"].items():
        if arm in pilot["extended"]:
            assert res["extension_used"] and "6" in res["caps"]
            assert res["status"] in ("selected", "inconclusive")
            assert res["caps"]["6"]["selection"]["selected_epoch"] >= 3  # protocol §8 item 2
        else:
            assert not res["extension_used"] and res["status"] == "selected"
    outcome = r["outcome"]["outcome"]
    if any(r["arms"][a]["status"] == "inconclusive" for a in r["arms"]):
        assert outcome == sm.OUTCOME_INCONCLUSIVE and r["outcome"]["max_epochs"] is None
    else:
        assert r["outcome"]["max_epochs"] == max(a["selection"]["selected_epoch"] for a in r["arms"].values())
    assert r["acceptance"]["accepted"] is False


def test_status_only_outputs_contain_no_learning_metrics(pilot):
    report = sm.summarize(pilot["spec"].root, selection=False)
    root = pilot["spec"].paths.root
    texts = {name: (root / name).read_text() for name in sm.OUTPUT_FILES}
    dev_values = [repr(row["dev_nll"]) for e in report["attempts"] if e["latest"]
                  for row in sc.read_jsonl_strict(_run_dir(pilot, e["attempt_id"]) / "history.jsonl")]
    for name, text in texts.items():
        assert "dev_nll" not in text and "train_nll" not in text, name
        assert not any(v in text for v in dev_values), name
    assert texts["epochs.csv"].splitlines()[0] == "arm,cap,seed,attempt_id,epoch,seconds"


# Learning-curve information that status-only mode (protocol §13.1) must never reveal: every NLL
# field and every §16.5 statistic or selection result derived from the dev curves.
_LEARNING_KEYS = {"dev_nll", "train_nll", "running_train_nll", "values", "mean", "sd", "se", "epsilon",
                  "epsilon_source", "best_mean", "best_epoch", "selected_epoch", "epsilon_at_selected",
                  "epsilon_source_at_selected", "still_improving_at_cap", "statistics", "selection",
                  "extension_required", "dev_nll_rises", "running_train_nll_rises", "repeated_dev_nll_rise"}


def _learning_numbers(pilot, report):
    """Every NLL in the latest runs' histories and every statistic of a selection report."""
    nums = {row[k] for e in report["attempts"] if e["latest"]
            for row in sc.read_jsonl_strict(_run_dir(pilot, e["attempt_id"]) / "history.jsonl")
            for k in ("dev_nll", "train_nll", "running_train_nll")}
    for arm in report.get("arms", {}).values():
        for cap in arm["caps"].values():
            nums.add(cap["selection"]["best_mean"])
            for st in cap["statistics"]:
                nums |= {st["mean"], st["sd"], st["se"], *st["values"].values()}
                if st["epsilon_source"] == "SE-derived":
                    nums.add(st["epsilon"])
    return {v for v in nums if v not in (0.0, 0.001)}


def _leaks(obj, numbers, where="report"):
    """Recursive scan: prohibited keys anywhere, prohibited numbers as values or inside strings."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _LEARNING_KEYS:
                found.append(f"{where}.{k}")
            found += _leaks(v, numbers, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += _leaks(v, numbers, f"{where}[{i}]")
    elif isinstance(obj, float) and obj in numbers:
        found.append(f"{where} = {obj!r}")
    elif isinstance(obj, str) and any(repr(n) in obj for n in numbers):
        found.append(f"{where} (string contains a learning value)")
    return found


def test_status_only_stripping_removes_every_learning_statistic_from_a_full_selection_report(pilot, tmp_path):
    """Mutant 16: the production status-only transformation, applied to a report that *does* hold
    the §16.5 statistics (as it would if a later change let them reach status-only mode)."""
    full = sm.evaluate_pilot(pilot["spec"].root, selection=True)
    numbers = _learning_numbers(pilot, full)
    assert full["arms"] and numbers and _leaks(full, numbers)  # the input really carries the statistics
    assert full["outcome"]["outcome"] != sm.OUTCOME_STATUS_ONLY  # and a selection decision
    before = json.dumps(full, sort_keys=True)
    stripped = sm._finish(full, selection=False)
    assert json.dumps(full, sort_keys=True) == before  # the transformation copies; the input is unchanged
    assert _leaks(stripped, numbers) == []
    assert stripped["mode"] == "status-only" and "arms" not in stripped
    assert stripped["outcome"] == {"outcome": sm.OUTCOME_STATUS_ONLY, "max_epochs": None, "requires_human_review": []}
    assert stripped["acceptance"]["ready_for_human_approval"] is False
    # status and integrity information is preserved
    assert stripped["selection_prerequisites"] == full["selection_prerequisites"]
    assert _gates(stripped) == _gates(full) and stripped["integrity_problems"] == full["integrity_problems"]
    for e in stripped["attempts"]:
        for g in ("G4", "G7"):
            assert not any(re.search(r"\b\d+\b", c) for c in e["gates"][g]["checks"]), (e["attempt_id"], g)
    sm.write_outputs(stripped, tmp_path)
    texts = {name: (tmp_path / name).read_text() for name in sm.OUTPUT_FILES}
    # the console line of the command, built from the same report
    texts["console"] = f"{stripped['mode']}: {stripped['outcome']['outcome']}"
    for name, text in texts.items():
        assert not any(repr(n) in text for n in numbers), name
        assert not any(f'"{k}"' in text for k in _LEARNING_KEYS), name
        assert "## Selection" not in text and "max_epochs =" not in text, name
    assert texts["epochs.csv"].splitlines()[0] == "arm,cap,seed,attempt_id,epoch,seconds"


def test_status_only_command_reveals_nothing_even_when_gates_fail(pilot, monkeypatch, capsys):
    """The real status-only path (command line → summarize → outputs and console line), with a
    tampered history so that failing-gate details are also exercised."""
    numbers = _learning_numbers(pilot, sm.evaluate_pilot(pilot["spec"].root, selection=True))
    d = _run_dir(pilot, _first(pilot))
    new_value = 1.2345678901234567
    monkeypatch.setattr(ebp, "RESULTS_ROOT", Path(pilot["spec"].root))
    with tampered(d / "history.jsonl", _jsonl_edit(lambda rows: [*rows[:-1], {**rows[-1], "dev_nll": new_value}])):
        unsanitized = next(e for e in sm.evaluate_pilot(pilot["spec"].root, selection=True)["attempts"]
                           if e["attempt_id"] == _first(pilot))["gates"]["G4"]
        assert sm.main([]) == 0
        console = capsys.readouterr().out
        root = pilot["spec"].paths.root
        texts = {name: (root / name).read_text() for name in sm.OUTPUT_FILES} | {"console": console}
        report = json.loads(texts["summary.json"])
    # the accepted Stage 1 detail really carries compared learning values
    assert unsanitized["status"] == "fail" and " = " in unsanitized["detail"]
    assert any(repr(n) in unsanitized["detail"] for n in numbers | {new_value})
    assert report["mode"] == "status-only" and _gates(report)[_first(pilot)]["G4"] == "fail"
    assert _leaks(report, numbers | {new_value}) == []
    for name, text in texts.items():
        assert not any(repr(n) in text for n in numbers | {new_value}), name
    # gate identity, status and location survive; the values do not
    g4 = next(e for e in report["attempts"] if e["attempt_id"] == _first(pilot))["gates"]["G4"]
    location = unsanitized["detail"].split(":", 1)[0]
    assert (g4["gate"], g4["status"], g4["detail"]) == ("G4", "fail", f"{location} — {sm.WITHHELD}")
    assert f"- {_first(pilot)} G4: {location} — {sm.WITHHELD}" in texts["summary.md"]
    assert f"{_first(pilot)}: gate(s) G4 not passed" in report["integrity_problems"]
    assert console.strip() == f"status-only: {sm.OUTCOME_INTEGRITY}"
    assert texts["epochs.csv"].splitlines()[0] == "arm,cap,seed,attempt_id,epoch,seconds"
    # no G4/G7 string names an epoch (best epochs are curve-derived, §13.1), on any attempt
    for e in report["attempts"]:
        for g in ("G4", "G7"):
            assert not any(re.search(r"\b\d+\b", c) for c in e["gates"][g]["checks"]), (e["attempt_id"], g)


def test_selection_outputs_preserve_full_precision_evidence(pilot):
    report = sm.summarize(pilot["spec"].root, selection=True)
    root = pilot["spec"].paths.root
    data = json.loads((root / "summary.json").read_text())
    st = data["arms"]["A"]["caps"]["3"]["statistics"][0]
    assert st == report["arms"]["A"]["caps"]["3"]["statistics"][0]  # JSON keeps every float exactly
    rows = (root / "epochs.csv").read_text().splitlines()
    assert rows[0].endswith("running_train_nll,train_nll,dev_nll") and len(rows) > 1
    first = rows[1].split(",")
    hist = sc.read_jsonl_strict(_run_dir(pilot, first[3]) / "history.jsonl")
    assert float(first[-1]) == hist[int(first[4]) - 1]["dev_nll"]
    assert "## Selection" in (root / "summary.md").read_text()


def test_summary_never_launches_training_and_writes_only_its_outputs(pilot, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("the summary must not run training")

    for name in ("run_stage", "retry", "_launch_child", "build_arm", "child_main"):
        monkeypatch.setattr(ebp, name, forbidden)
    monkeypatch.setattr(sc, "execute_run", forbidden)
    root = pilot["spec"].paths.root
    allowed = {root / n for n in sm.OUTPUT_FILES} | {root / ".runner.lock"}
    before = {p: sc.sha256_file(p) for p in root.rglob("*") if p.is_file() and p not in allowed}
    sm.summarize(root, selection=True)
    after = {p: sc.sha256_file(p) for p in root.rglob("*") if p.is_file() and p not in allowed}
    assert before == after
    assert {p.name for p in root.iterdir() if p.is_file()} <= {"conditions.json", "manifest.jsonl", "invocations.jsonl",
                                                              ".runner.lock", *sm.OUTPUT_FILES}


def test_summary_never_reads_or_builds_data_splits(pilot, monkeypatch):
    from utils import data

    def forbidden(*_a, **_k):
        raise AssertionError("the summary must not load names, split them or build windows")

    for name in ("load_names", "split_names", "make_context_windows", "check_vocab_coverage"):
        monkeypatch.setattr(data, name, forbidden)
    assert _evaluate(pilot)["integrity_problems"] == []
    source = (REPO / "training/wavenet_study/summarize_epoch_budget_pilot.py").read_text()
    for token in ("load_names", "split_names", "make_context_windows", "build_arm(", ".test", "test_nll", '"test"'):
        assert token not in source, token


def test_cli_without_a_started_pilot_refuses_and_creates_nothing(tmp_path, monkeypatch):
    # Point the command line at an empty temporary root: the test must never run against the real pilot.
    root = tmp_path / "epoch_budget_pilot"
    monkeypatch.setattr(ebp, "RESULTS_ROOT", root)
    with pytest.raises(sc.Blocked, match="not been started"):
        sm.main(["--selection"])
    assert not root.exists()


# ---------------------------------------------------------------- each gate, independently
def _first(pilot, cap=3, arm="A"):
    return f"{arm}-c{cap}-s100-a1"


def _cases(pilot):
    a3 = _run_dir(pilot, _first(pilot))
    cases = {
        "G1": (a3 / "history.jsonl", _jsonl_edit(lambda rows: rows + [rows[-1]]), "G1"),
        "G2": (a3 / "config.json", lambda p: p.write_text(p.read_text().replace('"lr": 0.03', '"lr": 0.1', 1)), "G2"),
        "G3": (a3 / "metadata.json",
               lambda p: p.write_text(json.dumps({**json.loads(p.read_text()), "status": "failed"})), "G3"),
        "G4": (a3 / "checkpoint_last.pt", _ckpt_edit(lambda c: c.__setitem__("best_metric", c["best_metric"] + 1e-9)),
               "G4"),
        "G5": (a3 / "diagnostics" / "steps.jsonl",
               _jsonl_edit(lambda rows: [{**rows[0], "loss": math.nextafter(rows[0]["loss"], 9.0)}, *rows[1:]]), "G5"),
        # the record checkpoint_last points to (G7 can verify only records a checkpoint refers to)
        "G7": (a3 / sc.STATE_HASH_FILE, _jsonl_edit(lambda rows: [*rows[:-1], {**rows[-1], "sha256": "0" * 64}]), "G7"),
    }
    if pilot["extended"]:
        a6 = _run_dir(pilot, _first(pilot, 6, pilot["extended"][0]))
        steps = sc.read_jsonl_strict(a6 / "diagnostics" / "steps.jsonl")
        target = next(i for i, r in enumerate(steps) if r["epoch"] == 2)  # epoch 3: inside G6, outside G5
        cases["G6"] = (a6 / "diagnostics" / "steps.jsonl",
                       _jsonl_edit(lambda rows: [*rows[:target], {**rows[target], "loss": rows[target]["loss"] + 1e-6},
                                                 *rows[target + 1:]]), "G6")
    return cases


@pytest.mark.parametrize("gate", ["G1", "G2", "G3", "G4", "G5", "G6", "G7"])
def test_each_gate_fails_alone_and_blocks_selection(pilot, gate):
    cases = _cases(pilot)
    if gate not in cases:
        pytest.skip("no arm required an extension in this synthetic pilot")
    path, mutate, expected = cases[gate]
    with tampered(path, mutate):
        r = _evaluate(pilot)
        bad = {aid: [g for g, s in gates.items() if s not in ("pass", "not_applicable")]
               for aid, gates in _gates(r).items()}
        bad = {aid: g for aid, g in bad.items() if g}
        target = path.parent.name if gate not in ("G5", "G6") else path.parent.parent.name
        target = next(aid for aid in bad if _run_dir(pilot, aid).name == target and
                      _run_dir(pilot, aid) in (path.parent, path.parent.parent))
        failed = bad.pop(target)
        # a tampered cap-3 run is also the G6 reference of the same arm/seed's cap-6 run, if any
        dependent = target.replace("-c3-", "-c6-")
        assert all(aid == dependent and g == ["G6"] for aid, g in bad.items()), bad
        if gate == "G1":
            assert failed == ["G1", "G4"]  # an extra history row also breaks history/checkpoint agreement
        else:
            assert failed == [expected]
        if gate in ("G5", "G6"):
            assert any(f"runner-side {gate} pass but summary-side fail" in p for p in r["integrity_problems"])
        assert not r["selection_prerequisites"]["met"] and r["outcome"]["outcome"] == sm.OUTCOME_INTEGRITY
        with pytest.raises(sc.Blocked, match="extension decision refused"):
            sm.extension_decision(pilot["spec"], list(pilot["spec"].arms))
    assert _evaluate(pilot)["integrity_problems"] == []


def test_corrupted_and_missing_checkpoints_fail_g4_and_g7(pilot):
    d = _run_dir(pilot, _first(pilot))
    with tampered(d / "model_best.pt", lambda p: p.write_bytes(p.read_bytes()[:100])):
        gates = _gates(_evaluate(pilot))[_first(pilot)]
        assert gates["G4"] == "fail" and gates["G7"] == "fail" and gates["G5"] == "pass"
    with tampered(d / "model_final.pt", lambda p: p.unlink()):
        gates = _gates(_evaluate(pilot))[_first(pilot)]
        assert gates["G4"] == "fail" and gates["G7"] == "fail"


def test_missing_history_leaves_gates_failed_or_unresolved(pilot):
    d = _run_dir(pilot, _first(pilot))
    with tampered(d / "history.jsonl", lambda p: p.unlink()):
        gates = _gates(_evaluate(pilot))[_first(pilot)]
        assert gates["G1"] == "fail" and gates["G4"] == "fail" and gates["G5"] in ("unresolved", "not_applicable")
        assert gates["G7"] == "pass"  # G7 never reads the history


# ---------------------------------------------------------------- conditions, records, categories
def test_blocking_file_change_fails_the_conditions_but_documents_do_not(pilot):
    data = pilot["world"]["data"]
    with tampered(data, lambda p: p.write_text(p.read_text() + "zz\n")):
        r = _evaluate(pilot)
        assert r["conditions"]["status"] == "fail" and r["outcome"]["outcome"] == sm.OUTCOME_INTEGRITY
        assert all(g["G2"] == "fail" for g in _gates(r).values())  # every run's blocking checksums differ
    with tampered(pilot["world"]["doc"], lambda p: p.write_text("amended\n")):
        r = _evaluate(pilot)
        assert r["conditions"]["status"] == "pass" and r["integrity_problems"] == []
        assert r["conditions"]["recorded_document_changes"] == [["checksums", "recorded", str(pilot["world"]["doc"])]]


def test_edited_manifest_or_attempt_result_is_an_integrity_failure(pilot):
    paths = pilot["spec"].paths

    def edit_last(p):
        lines = p.read_text().splitlines()
        obj = json.loads(lines[-1])
        obj["status"] = "error"
        p.write_text("\n".join([*lines[:-1], sc.canonical_json(obj)]) + "\n")

    with tampered(paths.manifest, edit_last):
        r = _evaluate(pilot)
        assert r["records"]["status"] == "fail" and r["outcome"]["outcome"] == sm.OUTCOME_INTEGRITY
    result = paths.attempt_result(_first(pilot))
    with tampered(result, lambda p: p.write_text(p.read_text().replace('"completed"', '"error"', 1))):
        assert "changed after the attempt was closed" in _evaluate(pilot)["records"]["detail"]


def test_category_b_must_match_the_ledger_and_unresolved_events_block_acceptance(pilot):
    aid = _first(pilot)
    d = _run_dir(pilot, aid)
    steps = sc.read_jsonl_strict(d / "diagnostics" / "steps.jsonl")
    i = next(k for k, r in enumerate(steps) if r["epoch"] == 2)  # epoch 3: outside the G5 epochs
    param = min(steps[i]["updates"]["per_parameter"])

    def null_ratio(rows):
        rows = json.loads(json.dumps(rows))
        rows[i]["updates"]["per_parameter"][param]["relative_update"] = None
        return rows

    paths = pilot["spec"].paths
    ledger_files = [paths.root / sc.CategoryBLedger.EVENTS_FILE, paths.root / sc.CategoryBLedger.RESOLUTIONS_FILE]
    saved = {p: p.read_bytes() if p.exists() else None for p in ledger_files}
    try:
        with tampered(d / "diagnostics" / "steps.jsonl", _jsonl_edit(null_ratio)):
            r = _evaluate(pilot)
            assert any("re-derived category B 1 vs registered 0" in p for p in r["integrity_problems"])
            ledger = sc.CategoryBLedger(paths.root)
            ledger.register({"event_id": "B-test-0001", "created_at": "t", "attempt_id": aid, "run_dir": str(d),
                             "arm": "A", "seed": 100, "cap": 3, "epoch_1based": 3, "global_step": steps[i]["step"],
                             "record": "steps", "field": ["updates", "per_parameter", param, "relative_update"],
                             "observed": None, "reason": "undefined_value", "evidence": {}, "status": "unresolved"})
            r = _evaluate(pilot)
            assert not any("category B" in p for p in r["integrity_problems"])
            assert r["acceptance"]["criteria"]["3_no_unresolved_or_rejected_category_B"] is False
            assert [e["current_status"] for e in r["category_b"]["events"]] == ["unresolved"]
            assert sm.decide_outcome(halted=False, integrity_problems=[], incomplete=[],
                                     arms={"A": _arm(2)}, category_b_blocking=ledger.blocking_events())["outcome"] \
                == sm.OUTCOME_CATEGORY_B
            ledger.resolve({"event_id": "B-test-0001", "outcome": "accepted_as_expected", "reviewer": "Nur",
                            "timestamp": "t", "rationale": "synthetic test", "evidence": "this test"})
            r = _evaluate(pilot)
            assert r["acceptance"]["criteria"]["3_no_unresolved_or_rejected_category_B"] is True
            assert r["category_b"]["events"][0]["resolutions"][0]["outcome"] == "accepted_as_expected"
        # the step record is restored: the registered event no longer matches anything (orphan event)
        assert any("registered 1" in p for p in _evaluate(pilot)["integrity_problems"])
    finally:
        for p, b in saved.items():
            if b is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(b)
    assert _evaluate(pilot)["integrity_problems"] == []


def test_non_finite_recorded_nll_halts_the_pilot(pilot):
    d = _run_dir(pilot, _first(pilot))
    with tampered(d / "history.jsonl", _jsonl_edit(lambda rows: [*rows[:2], {**rows[2], "dev_nll": "nan"}])):
        r = _evaluate(pilot)
        assert r["halted"] and r["outcome"]["outcome"] == sm.OUTCOME_HALTED
        entry = next(e for e in r["attempts"] if e["attempt_id"] == _first(pilot))
        assert entry["category_a_records"][0]["path"] == ["dev_nll"]
