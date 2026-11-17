"""Independent summary of the common epoch-budget pilot — Stage 5.

Canonical protocol: ``training/wavenet_study/EPOCH_BUDGET_PILOT_PROTOCOL.md`` v1.1 (§6, §7, §10,
§13, §14); scientific rule: ``models/hierarchical/wavenet_spec.md`` v0.5 §16.5.

The summary trusts no success flag written by the runner. From the raw files it re-verifies

* the frozen conditions and every blocking checksum (§10.1);
* the hash-chained manifest and invocation records, the record heads they hold and every
  attempt result (Stage 3/4 records), orphans, and the category B ledger chains (§9.7);
* the integrity gates G1–G7 of every attempt, each computed separately (§10.3); G5 and G6 are
  recomputed from the saved histories, step records, state hashes and the frozen-verified
  reference files, and compared with the runner-side results (§10.4, R6);
* the category A/B/C classification of every written record (§9.1, §9.3), compared with the
  category B events registered by the runner;
* the provenance of every extension run (extension stage, ``authorized_by``, or a retry of one).

Then, in **selection mode only** (explicit ``--selection``, protocol §13.1), it computes the
§16.5 statistics and decisions. It never launches, retries or extends anything, never builds
data windows and never reads the test split: its only inputs are the pilot's own records, the
frozen-verified LR-pilot reference files and the configuration and code files hashed for G2.
It writes exactly ``summary.json``, ``summary.md`` and ``epochs.csv`` in the pilot folder.

Numerics (protocol §6.1: full precision, ``≤`` exactly as written). Dev NLLs are taken from
``history.jsonl`` as parsed float64 values and converted to exact rationals. ``D̄`` and the
sample variance are computed exactly; ``SD`` and ``SE = SD/√3`` are the square roots of the exact
variance and variance/3, rounded once to float64 (via a 60-digit decimal square root);
``ε = max(1e-3, SE)`` in float64; the ε label compares SE with the float 0.001 exactly. ``D̄*``,
the earliest-tie ``ê`` and the condition ``D̄(e) ≤ D̄* + ε(e)`` use the exact rational means and
the exact value of the float ε, so no decision depends on summation order or rounding of ``D̄``.
Reported means are the exact means rounded once to float64.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sc = importlib.import_module("training.wavenet_study.study_common")
ebp = importlib.import_module("training.wavenet_study.epoch_budget_pilot")  # spec, store, conditions: read only

OUTPUT_FILES = ("summary.json", "summary.md", "epochs.csv")
EPSILON_FLOOR = 1e-3
GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
OUTCOME_HALTED = "halted: mechanical instability"
OUTCOME_INTEGRITY = "withheld: integrity"
OUTCOME_INCOMPLETE = "pending: incomplete runs"
OUTCOME_EXTENSION = "pending: extension required"
OUTCOME_INCONCLUSIVE = "inconclusive"
OUTCOME_CATEGORY_B = "withheld: unresolved or rejected category B"
OUTCOME_REVIEW = "requires human review: SE-derived e*"
OUTCOME_BUDGET = "max_epochs"
OUTCOME_STATUS_ONLY = "not evaluated (status-only mode)"


# ============================================================ §16.5 statistics (pure functions)
def _sqrt_fraction(x: Fraction) -> float:
    """√x rounded to float64 through a 60-significant-digit decimal square root."""
    from decimal import Decimal, localcontext

    if x < 0:
        raise ValueError("negative variance")
    with localcontext() as ctx:
        ctx.prec = 60
        return float((Decimal(x.numerator) / Decimal(x.denominator)).sqrt())


def epsilon_source(se: float) -> str:
    """Protocol §6.1 (R3): exact float comparison with 0.001."""
    if se > 0.001:
        return "SE-derived"
    if se < 0.001:
        return "floor-derived"
    return "floor (SE equal)"


def epoch_statistics(curves: Mapping[int, Sequence[float]], epochs: int) -> list[dict[str, Any]]:
    """Step 2 for one arm: per epoch e = 1 … ``epochs``, D̄, SD (n − 1), SE = SD/√n, ε, ε source."""
    seeds = sorted(curves)
    n = len(seeds)
    if n < 2:
        raise ValueError("at least two seeds are needed for a sample standard deviation")
    out = []
    for e in range(1, epochs + 1):
        values = [curves[s][e - 1] for s in seeds]
        if not all(isinstance(v, float) and math.isfinite(v) for v in values):
            raise ValueError(f"epoch {e}: non-finite or non-float dev NLL")
        exact = [Fraction(v) for v in values]
        mean = sum(exact, Fraction(0)) / n
        var = sum(((x - mean) ** 2 for x in exact), Fraction(0)) / (n - 1)
        sd, se = _sqrt_fraction(var), _sqrt_fraction(var / n)
        eps = max(EPSILON_FLOOR, se)
        out.append({"epoch": e, "values": {str(s): v for s, v in zip(seeds, values, strict=True)},
                    "mean": float(mean), "sd": sd, "se": se, "epsilon": eps, "epsilon_source": epsilon_source(se),
                    "_mean_exact": mean, "_epsilon_exact": Fraction(eps)})
    return out


def select_epoch(stats: Sequence[Mapping[str, Any]], cap: int) -> dict[str, Any]:
    """Steps 3–4 at one cap: D̄*, ê (earliest tie), e* (earliest within ε(e) of D̄*), extension need."""
    if [s["epoch"] for s in stats] != list(range(1, cap + 1)):
        raise ValueError(f"statistics must cover epochs 1..{cap}")
    best = min(s["_mean_exact"] for s in stats)
    best_epoch = next(s["epoch"] for s in stats if s["_mean_exact"] == best)
    selected = next(s["epoch"] for s in stats if s["_mean_exact"] <= best + s["_epsilon_exact"])
    at = stats[selected - 1]
    return {"cap": cap, "best_mean": float(best), "best_epoch": best_epoch, "selected_epoch": selected,
            "epsilon_at_selected": at["epsilon"], "epsilon_source_at_selected": at["epsilon_source"],
            "still_improving_at_cap": selected == cap}


def decide_outcome(*, halted: bool, integrity_problems: Sequence[str], incomplete: Sequence[str],
                   arms: Mapping[str, Mapping[str, Any]], category_b_blocking: Sequence[str]) -> dict[str, Any]:
    """Step 5 with the withholding rules (§13.2, §13.3). ``arms[a]["status"]`` is one of
    ``selected`` (with ``selection``), ``extension_required``, ``extension_pending`` or ``inconclusive``."""
    def result(outcome: str, **extra: Any) -> dict[str, Any]:
        return {"outcome": outcome, "max_epochs": None, "requires_human_review": [], **extra}

    if halted:
        return result(OUTCOME_HALTED)
    if integrity_problems:
        return result(OUTCOME_INTEGRITY, reasons=list(integrity_problems))
    if incomplete:
        return result(OUTCOME_INCOMPLETE, reasons=list(incomplete))
    pending = sorted(a for a, r in arms.items() if r["status"] in ("extension_required", "extension_pending"))
    if pending:
        return result(OUTCOME_EXTENSION, arms=pending)
    inconclusive = sorted(a for a, r in arms.items() if r["status"] == "inconclusive")
    if inconclusive:
        return result(OUTCOME_INCONCLUSIVE, arms=inconclusive)
    if category_b_blocking:
        return result(OUTCOME_CATEGORY_B, events=list(category_b_blocking))
    budget = max(r["selection"]["selected_epoch"] for r in arms.values())
    review = sorted(a for a, r in arms.items() if r["selection"]["epsilon_source_at_selected"] == "SE-derived")
    return {"outcome": OUTCOME_REVIEW if review else OUTCOME_BUDGET, "max_epochs": budget,
            "requires_human_review": review}


def optimisation_flags(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive only (DEC-7); no part in the selection."""
    dev = [r["dev_nll"] for r in history]
    run = [r["running_train_nll"] for r in history]
    rises = [{"epoch": i + 1, "increase": dev[i] - dev[i - 1]} for i in range(1, len(dev)) if dev[i] > dev[i - 1]]
    best = min(range(len(dev)), key=lambda i: (dev[i], i)) + 1
    return {"dev_nll_rises": rises, "repeated_dev_nll_rise": len(rises) >= 2,
            "running_train_nll_rises": [i + 1 for i in range(1, len(run)) if run[i] > run[i - 1]],
            "best_epoch": best, "best_epoch_is_last": best == len(dev)}


# ============================================================ independent verification
def _run_path(run_dir: str | None) -> Path | None:
    if not run_dir:
        return None
    p = Path(run_dir)
    return p if p.is_absolute() else REPO / p


def _norm(spec: Any, arm: str) -> str:
    return "batchnorm" if spec.arms[arm]["config"] in ("batchnorm_mlp", "wavenet_batchnorm") else "none"


def _read(path: Path) -> list[dict[str, Any]]:
    return sc.read_jsonl_strict(path)


def _reference(spec: Any, frozen: Mapping[str, Any], arm: str, seed: int) -> dict[str, Any]:
    """LR-pilot reference records, each file verified against its frozen SHA-256 (summary-side)."""
    rec = frozen["conditions"]["references"]["runs"][f"{arm}/{seed}"]
    d = Path(spec.reference_root) / spec.reference_runs[f"{arm}/{seed}"]
    for rel, digest in rec["files"].items():
        if sc.sha256_file(d / rel) != digest:
            raise sc.Blocked(f"reference file {d / rel} differs from its frozen SHA-256")
    return {"history": _read(d / "history.jsonl"), "steps": _read(d / "diagnostics" / "steps.jsonl"),
            "hash": sc.checkpoint_state_hash(d / "model_final.pt")}


def _expected_config(spec: Any, frozen: Mapping[str, Any], arm: str, cap: int, seed: int) -> dict[str, Any]:
    experiment = json.loads(json.dumps(frozen["conditions"]["arms"][arm]["config"]))
    experiment["reproducibility"]["seed"] = seed
    return {"experiment": experiment,
            "training": {**frozen["conditions"]["expected_training"], "lr": spec.learning_rates[arm],
                         "max_epochs": cap}}


def _gate(name: str, fn: Any) -> dict[str, Any]:
    try:
        return fn().to_dict()
    except (Exception, sc.Blocked) as exc:  # noqa: BLE001  (unevaluable gate: unresolved, §9.5; Blocked: SystemExit)
        return {"gate": name, "status": "unresolved", "detail": f"{type(exc).__name__}: {exc}", "first_mismatch": None,
                "checks": []}


def _prefix_gate(name: str, *, epochs: int, reference: Mapping[str, Any], run_dir: Path, norm: str,
                 reached: int) -> sc.GateResult:
    if reached < epochs:
        return sc.GateResult(name, "not_applicable", f"run has {reached} of {epochs} epochs")
    hashes = {r["epoch"]: r["sha256"] for r in _read(run_dir / sc.STATE_HASH_FILE)}
    mismatch = sc.compare_run_prefix(
        reference_history=reference["history"], reference_steps=reference["steps"], reference_hash=reference["hash"],
        candidate_history=_read(run_dir / "history.jsonl"),
        candidate_steps=_read(run_dir / "diagnostics" / "steps.jsonl"),
        candidate_hash=hashes.get(epochs - 1, ""), epochs=epochs,
        history_null_allowed=sc.history_null_allowed_for(norm), steps_null_allowed=sc.step_null_allowed)
    if mismatch is None:
        return sc.GateResult(name, "pass", "", None, [f"epochs 1-{epochs}"])
    return sc.GateResult(name, "fail", f"{mismatch.kind} at {sc.format_path(mismatch.path)} {mismatch.where}",
                         mismatch.to_dict(), [f"epochs 1-{epochs}"])


def _g6_reference(spec: Any, store: Any, by_key: Mapping[tuple, list], arm: str, seed: int) -> dict[str, Any]:
    recs = by_key.get((arm, spec.caps[0], seed), [])
    if not recs or recs[-1]["status"] != "completed":
        raise sc.Blocked(f"no completed cap-{spec.caps[0]} run for arm {arm}, seed {seed}")
    d = _run_path(recs[-1].get("run_dir"))
    hashes = {r["epoch"]: r["sha256"] for r in _read(d / sc.STATE_HASH_FILE)}
    digest = hashes.get(spec.caps[0] - 1)
    if digest is None or digest != sc.checkpoint_state_hash(d / "model_final.pt"):
        raise sc.Blocked(f"G6 reference {d}: epoch-{spec.caps[0]} state hash missing or not equal to model_final.pt")
    return {"history": _read(d / "history.jsonl"), "steps": _read(d / "diagnostics" / "steps.jsonl"), "hash": digest}


def _attempt_gates(spec: Any, frozen: Mapping[str, Any], current: Mapping[str, Any] | None, store: Any,
                   by_key: Mapping[tuple, list], rec: Mapping[str, Any]) -> dict[str, Any]:
    arm, cap, seed = rec["key"]
    run_dir = _run_path(rec.get("run_dir")) or _run_path(store.run_dir_of(rec["attempt_id"]))
    out: dict[str, Any] = {}
    if run_dir is None or not run_dir.is_dir():
        return {g: {"gate": g, "status": "unresolved", "detail": "no run directory", "first_mismatch": None,
                    "checks": []} for g in GATES}
    history: list[dict[str, Any]] = []
    try:
        history = _read(run_dir / "history.jsonl")
    except Exception:  # noqa: BLE001
        history = []
    out["G1"] = _gate("G1", lambda: sc.gate_g1(history, cap))

    def g2() -> sc.GateResult:
        if current is None:
            return sc.GateResult("G2", "unresolved", "current conditions could not be computed")
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        actual = {"experiment": config.get("experiment"), "training": config.get("training")}
        result = sc.gate_g2(actual_config=actual, expected_config=_expected_config(spec, frozen, arm, cap, seed),
                            actual_checksums=current["checksums"]["blocking"],
                            frozen_checksums=frozen["conditions"]["checksums"]["blocking"])
        if result.status != "pass":
            return result
        pilot = config.get("pilot") or {}
        checks = [("model block", config.get("model"), (config.get("experiment") or {}).get("model", {}).get("params")),
                  ("pilot attempt", pilot.get("attempt_id"), rec["attempt_id"]),
                  ("pilot key", [pilot.get("arm"), pilot.get("cap"), pilot.get("seed")], [arm, cap, seed]),
                  ("metadata conditions", meta.get("conditions_sha256"), frozen["conditions_sha256"])]
        for label, got, want in checks:
            if got != want:
                return sc.GateResult("G2", "fail", f"{label}: {got!r} != {want!r}")
        return result

    out["G2"] = _gate("G2", g2)
    out["G3"] = _gate("G3", lambda: sc.gate_g3(
        attempt_status=rec["status"],
        metadata_status=json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")).get("status")))
    out["G4"] = _gate("G4", lambda: sc.gate_g4(run_dir, cap=cap))
    norm = _norm(spec, arm)
    out["G5"] = _gate("G5", lambda: _prefix_gate("G5", epochs=spec.g5_epochs,
                                                 reference=_reference(spec, frozen, arm, seed), run_dir=run_dir,
                                                 norm=norm, reached=len(history)))
    if cap == spec.caps[1]:
        out["G6"] = _gate("G6", lambda: _prefix_gate("G6", epochs=spec.caps[0],
                                                     reference=_g6_reference(spec, store, by_key, arm, seed),
                                                     run_dir=run_dir, norm=norm, reached=len(history)))
    else:
        out["G6"] = {"gate": "G6", "status": "not_applicable", "detail": f"cap {cap} run", "first_mismatch": None,
                     "checks": []}
    out["G7"] = _gate("G7", lambda: sc.gate_g7(run_dir))
    return out


def _schemas(spec: Any, frozen: Mapping[str, Any], arm: str, seed: int) -> sc.RecordSchemas:
    ref = _reference(spec, frozen, arm, seed)
    return sc.RecordSchemas(sc.record_schema(ref["history"]), sc.record_schema(ref["steps"]))


def _rederive_categories(spec: Any, frozen: Mapping[str, Any], rec: Mapping[str, Any],
                         run_dir: Path) -> dict[str, Any]:
    """Re-classify every record up to the last written history row (where the runner classified)."""
    arm, _cap, seed = rec["key"]
    schemas = _schemas(spec, frozen, arm, seed)
    history = _read(run_dir / "history.jsonl")
    steps_path = run_dir / "diagnostics" / "steps.jsonl"
    steps = [s for s in (_read(steps_path) if steps_path.exists() else []) if s.get("epoch", 0) < len(history)]
    a, b, c = [], Counter(), Counter()
    eps_values = []
    norm = _norm(spec, arm)
    for kind, records, schema in (("history", history, schemas.history), ("steps", steps, schemas.steps)):
        for r in records:
            for f in sc.classify_record(kind, r, normalization=norm) + sc.schema_findings(kind, r, schema):
                if f.category == "A":
                    a.append(f.to_dict())
                elif f.category == "B":
                    b[(f.record, None if f.epoch is None else f.epoch + 1, f.step if f.record == "steps" else None,
                       tuple(f.path))] += 1
                else:
                    c[f.kind] += 1
                    if f.kind == "epsilon_regularised":
                        eps_values.append({"field": list(f.path), "step": f.step, "value": f.value})
    return {"category_a": a, "category_b": b, "category_c": dict(c), "epsilon_regularised": eps_values}


def _ledger_keys(events: Sequence[Mapping[str, Any]], attempt_id: str) -> Counter:
    return Counter((e["record"], e["epoch_1based"], e["global_step"], tuple(e["field"])) for e in events
                   if e["attempt_id"] == attempt_id)


def _extension_provenance(spec: Any, attempts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Every cap-60 attempt must come from an authorised extension stage or a retry of one."""
    problems = []
    for aid, rec in attempts.items():
        if rec["key"][1] != spec.caps[1]:
            continue
        if rec.get("stage") == "extension" and isinstance(rec.get("authorized_by"), str) \
                and rec["authorized_by"].strip():
            continue
        origin = attempts.get(rec.get("retry_of") or "")
        if rec.get("stage") == "retry" and origin is not None and origin["key"] == rec["key"]:
            continue
        problems.append(f"{aid}: cap-{spec.caps[1]} attempt without an authorised extension")
    return problems


def evaluate_pilot(root: str | Path, *, selection: bool, expected_spec: Any | None = None) -> dict[str, Any]:
    """Build the summary of the pilot at ``root`` (read only; no lock taken, no file written)."""
    paths = sc.PilotPaths(Path(root))
    report: dict[str, Any] = {"mode": "selection" if selection else "status-only", "root": str(paths.root),
                              "generated_at": sc._utc_now(), "blocking": [], "integrity_problems": []}
    try:
        frozen = sc.load_frozen(paths)
        if frozen is None:
            raise sc.Blocked("the pilot has not been started (no conditions.json)")
    except sc.Blocked as exc:
        report["blocking"].append(exc.message)
        report["outcome"] = {"outcome": OUTCOME_INTEGRITY, "max_epochs": None, "requires_human_review": []}
        return report
    spec = ebp.PilotSpec.from_json(frozen["conditions"]["spec"])
    if expected_spec is not None and expected_spec.to_json() != spec.to_json():
        report["blocking"].append("the frozen spec differs from the caller's spec")
    if Path(spec.root) != paths.root:
        report["blocking"].append("the frozen spec belongs to another pilot root")
    report["conditions_sha256"] = frozen["conditions_sha256"]
    integrity = report["integrity_problems"]

    # ---- frozen conditions and checksums (§10.1)
    current = None
    try:
        ebp.prepare_threads(spec)
        current = ebp.current_conditions(spec)
        _, nonblocking = sc.verify_or_freeze(paths, current, may_create=False, lock=None)
        report["conditions"] = {"status": "pass", "recorded_document_changes": [list(p) for p in nonblocking]}
    except sc.Blocked as exc:
        report["conditions"] = {"status": "fail", "detail": exc.message}
        integrity.append(f"conditions: {exc.message}")

    # ---- records (manifest, invocations, results, ledger, orphans)
    store = ebp.make_store(spec)
    try:
        attempts = store.attempts()
        report["records"] = {**sc.verify_record_heads(paths, store), "orphans": store.orphans()}
        if report["records"]["orphans"]:
            integrity.append("orphan records: " + ", ".join(report["records"]["orphans"]))
    except sc.Blocked as exc:
        report["records"] = {"status": "fail", "detail": exc.message}
        integrity.append(f"records: {exc.message}")
        report["outcome"] = decide_outcome(halted=False, integrity_problems=integrity, incomplete=[], arms={},
                                           category_b_blocking=[])
        return _finish(report, selection)
    ledger = sc.CategoryBLedger(paths.root)
    try:
        ledger_status = ledger.status()
        events = [s["event"] for s in ledger_status.values()]
        report["category_b"] = {"heads": ledger.heads(), "events": [
            {**s["event"], "current_status": s["status"], "resolutions": s["resolutions"]}
            for s in ledger_status.values()]}
        b_blocking = ledger.blocking_events()
    except (sc.LedgerError, sc.Blocked) as exc:
        report["category_b"] = {"status": "fail", "detail": str(exc)}
        integrity.append(f"category B ledger: {exc}")
        events, b_blocking = [], []
    integrity += _extension_provenance(spec, attempts)

    # ---- every attempt: gates, re-derived categories, runner/summary agreement
    by_key = store.by_key()
    halted = False
    report["attempts"] = []
    for aid, rec in attempts.items():
        entry: dict[str, Any] = {"attempt_id": aid, "arm": rec["key"][0], "cap": rec["key"][1],
                                 "seed": rec["key"][2], "status": rec["status"], "stage": rec.get("stage"),
                                 "retry_of": rec.get("retry_of"), "retry_reason": rec.get("retry_reason"),
                                 "retry_authorized_by": rec.get("retry_authorized_by"), "run_dir": rec.get("run_dir"),
                                 "latest": by_key[rec["key"]][-1]["attempt_id"] == aid}
        if rec["status"] == "mechanically_unstable":
            halted = True
        entry["gates"] = _attempt_gates(spec, frozen, current, store, by_key, rec)
        result = store.read_result(aid, rec["key"])
        if result is not None:
            runner = {i["gate"]: i["status"] for i in result.get("integrity", [])}
            for g in ("G5", "G6"):
                mine = entry["gates"][g]["status"]
                if mine in ("pass", "fail") and runner.get(g) in ("pass", "fail") and runner[g] != mine:
                    integrity.append(f"{aid}: runner-side {g} {runner[g]} but summary-side {mine}")
                if rec["status"] == "completed" and mine == "pass" and runner.get(g) != "pass":
                    integrity.append(f"{aid}: summary-side {g} pass but no runner-side pass recorded")
            entry["timing"] = _timing(_run_path(result.get("run_dir")))
            entry["peak_rss_bytes"] = result.get("peak_rss_bytes")
            run_dir = _run_path(result.get("run_dir"))
            if run_dir is not None and (run_dir / "history.jsonl").exists():
                try:
                    derived = _rederive_categories(spec, frozen, rec, run_dir)
                except (Exception, sc.Blocked) as exc:  # noqa: BLE001
                    integrity.append(f"{aid}: categories could not be re-derived ({type(exc).__name__}: {exc})")
                    derived = {"category_a": [], "category_b": None, "category_c": {}, "epsilon_regularised": []}
                registered = _ledger_keys(events, aid)
                if derived["category_a"]:
                    halted = True
                if derived["category_b"] is not None and derived["category_b"] != registered:
                    integrity.append(f"{aid}: re-derived category B {sum(derived['category_b'].values())} vs "
                                     f"registered {sum(registered.values())} (unregistered or orphan events)")
                entry["category_a_records"] = derived["category_a"]
                entry["category_c"] = derived["category_c"]
                entry["epsilon_regularised"] = derived["epsilon_regularised"]
        entry["category_a"] = (result or {}).get("category_a")
        report["attempts"].append(entry)

    state = store.sweep_state()
    report["sweep_state"] = state
    incomplete = [f"{a} cap {spec.caps[0]} seed {s}" for a in spec.arms for s in spec.seeds
                  if not by_key.get((a, spec.caps[0], s)) or by_key[(a, spec.caps[0], s)][-1]["status"] != "completed"]
    used = {(e["arm"], e["cap"], e["seed"]): e for e in report["attempts"] if e["latest"]}
    for e in used.values():
        if e["status"] == "completed":
            failed = [g for g, r in e["gates"].items() if r["status"] not in ("pass", "not_applicable")]
            if failed:
                integrity.append(f"{e['attempt_id']}: gate(s) {', '.join(failed)} not passed")
    if state["unresolved"] or state["launched"] or state["integrity_blocked"]:
        integrity.append("unresolved attempts: " + ", ".join(state["unresolved"] + state["launched"]
                                                            + state["integrity_blocked"]))
    report["halted"] = halted
    prerequisites = []
    if halted:
        prerequisites.append("category A event")
    if integrity:
        prerequisites.append("integrity")
    if incomplete:
        prerequisites.append(f"incomplete cap-{spec.caps[0]} runs")
    report["selection_prerequisites"] = {"met": not prerequisites, "failed": prerequisites}

    # ---- statistics and decisions (selection mode only, §13.1)
    arms: dict[str, dict[str, Any]] = {}
    if selection and not prerequisites:
        for arm in spec.arms:
            arms[arm] = _arm_selection(spec, arm, by_key, used)
            if arms[arm].get("problem"):
                integrity.append(arms[arm]["problem"])
        report["arms"] = arms
    report["outcome"] = (decide_outcome(halted=halted, integrity_problems=integrity, incomplete=incomplete, arms=arms,
                                        category_b_blocking=b_blocking)
                         if selection and not prerequisites else
                         decide_outcome(halted=halted, integrity_problems=integrity, incomplete=incomplete,
                                        arms={}, category_b_blocking=b_blocking)
                         if (halted or integrity or incomplete) else
                         {"outcome": OUTCOME_STATUS_ONLY, "max_epochs": None, "requires_human_review": []})
    report["acceptance"] = _acceptance(report, b_blocking, ledger_status if events else {})
    return _finish(report, selection)


def _arm_selection(spec: Any, arm: str, by_key: Mapping[tuple, list], used: Mapping[tuple, dict]) -> dict[str, Any]:
    initial, extended = spec.caps

    def curves(cap: int) -> tuple[dict[int, list[float]], dict[int, Any]] | None:
        out, flags = {}, {}
        for s in spec.seeds:
            e = used.get((arm, cap, s))
            if e is None or e["status"] != "completed":
                return None
            history = _read(_run_path(by_key[(arm, cap, s)][-1]["run_dir"]) / "history.jsonl")
            out[s] = [r["dev_nll"] for r in history]
            flags[s] = optimisation_flags(history)
        return out, flags

    first = curves(initial)
    stats = epoch_statistics(first[0], initial)
    sel = select_epoch(stats, initial)
    res: dict[str, Any] = {"learning_rate": spec.learning_rates[arm], "caps": {str(initial): {
        "statistics": _public(stats), "selection": sel, "flags": {str(k): v for k, v in first[1].items()}}},
        "extension_required": sel["still_improving_at_cap"], "extension_used": False}
    any_extension_run = any((arm, extended, s) in used for s in spec.seeds)
    if not sel["still_improving_at_cap"]:
        res.update(status="selected", selection=sel)
        if any_extension_run:
            res["problem"] = f"arm {arm}: cap-{extended} runs exist although the rule requires no extension"
        return res
    second = curves(extended)
    if second is None:
        res["status"] = "extension_pending" if any_extension_run else "extension_required"
        return res
    stats60 = epoch_statistics(second[0], extended)
    sel60 = select_epoch(stats60, extended)
    res["caps"][str(extended)] = {"statistics": _public(stats60), "selection": sel60,
                                  "flags": {str(k): v for k, v in second[1].items()}}
    res["extension_used"] = True
    res.update(status="inconclusive" if sel60["still_improving_at_cap"] else "selected", selection=sel60)
    return res


def _public(stats: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in s.items() if not k.startswith("_")} for s in stats]


def _timing(run_dir: Path | None) -> dict[str, Any] | None:
    try:
        t = json.loads((run_dir / ebp.sc.TIMING_FILE).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  (timing is operational evidence only)
        return None
    return {"wall_seconds": t.get("wall_seconds"), "training_call_seconds": t.get("training_call_seconds"),
            "finiteness_totals": t.get("finiteness_totals")}


def _acceptance(report: Mapping[str, Any], b_blocking: Sequence[str], ledger_status: Mapping[str, Any]) -> dict:
    """Protocol §14. Criteria 5 and 8 always need a human; this lists them, it never accepts."""
    outcome = report["outcome"]
    corrected = sorted(eid for eid, s in ledger_status.items() if s["status"] == "corrected")
    criteria = {
        "1_runs_completed_no_unresolved_no_category_A": not report["halted"] and not report["sweep_state"]["unresolved"]
        and not report["sweep_state"]["launched"] and not report["sweep_state"]["integrity_blocked"],
        "2_gates_pass_runner_summary_agree": not report["integrity_problems"],
        "3_no_unresolved_or_rejected_category_B": not b_blocking,
        "4_budget_computed_not_inconclusive": outcome.get("max_epochs") is not None,
        "5_se_derived_reviewed": "human review required" if outcome.get("requires_human_review") else "not required",
        "6_test_split_protection": "verified by automated tests and code inspection (Stage 4/5 test suites)",
        "7_finiteness_cost_reported": all(e.get("timing") and e["timing"].get("finiteness_totals")
                                          for e in report["attempts"] if e["status"] == "completed"),
        "8_human_approval": "required: record a new decision in PROJECT_RESEARCH_RECORD.md §11",
    }
    if corrected:
        criteria["note_corrected_category_B"] = (f"{corrected}: confirm that no correction requires new evidence "
                                                 "(protocol §9.7)")
    automated = [criteria[k] for k in criteria if k[0] in "12347"]
    return {"criteria": criteria, "ready_for_human_approval": all(v is True for v in automated),
            "accepted": False}


def _redact(obj: Any) -> Any:
    """Status-only mode: drop every learning-curve value (mismatch values, NLL fields)."""
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if k in ("reference", "candidate") and not isinstance(v, (str, type(None))):
                out[k] = "<redacted in status-only mode>"
            elif k in ("dev_nll", "train_nll", "running_train_nll", "best_mean", "mean", "values"):
                continue
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


# Status-only mode (protocol §13.1): "No dev or train NLL, curves, curve-derived flags, best epochs,
# statistics or selections in any output." Three layers, each sufficient for what it covers:
#   1. the whole per-arm selection section and every selection-derived outcome are removed, not
#      filtered field by field;
#   2. G4 and G7 compare learning values and best epochs (accepted Stage 1 messages such as
#      "checkpoint_last.pt: best_metric = …, history implies …" and "model_best.pt matches the
#      record of epoch …"): their details keep only the location before the first ":", and every
#      number in their check descriptions is masked;
#   3. NLL fields and mismatch values are dropped wherever they occur (_redact).
SELECTION_OUTCOMES = (OUTCOME_EXTENSION, OUTCOME_INCONCLUSIVE, OUTCOME_CATEGORY_B, OUTCOME_REVIEW, OUTCOME_BUDGET)
VALUE_GATES = ("G4", "G7")
WITHHELD = "values withheld in status-only mode"
_NUMBER = re.compile(r"(?<![A-Za-z_\d])(?:\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf)(?![A-Za-z_\d])")


def _status_only_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(gate)
    mismatch = gate.get("first_mismatch")
    if isinstance(mismatch, Mapping):  # location only: kind, field path, epoch/step
        out["first_mismatch"] = {k: mismatch.get(k) for k in ("kind", "path", "where")}
    if gate.get("gate") in VALUE_GATES:
        detail = str(gate.get("detail") or "")
        if gate.get("status") in ("fail", "unresolved") and detail:
            location = detail.split(":", 1)[0].strip() if ":" in detail else ""
            out["detail"] = f"{location} — {WITHHELD}" if location else WITHHELD
        out["checks"] = [_NUMBER.sub("#", str(c)) for c in gate.get("checks") or []]
    return out


def _status_only(report: Mapping[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(report))  # a copy: the caller's report is never changed
    out["mode"] = "status-only"
    out.pop("arms", None)
    outcome = out.get("outcome") or {}
    if outcome.get("outcome") in SELECTION_OUTCOMES or outcome.get("max_epochs") is not None:
        out["outcome"] = {"outcome": OUTCOME_STATUS_ONLY, "max_epochs": None, "requires_human_review": []}
    else:
        out["outcome"] = {**outcome, "max_epochs": None, "requires_human_review": []}
        out["outcome"].pop("arms", None)
    acceptance = out.get("acceptance")
    if isinstance(acceptance, dict):
        criteria = acceptance.get("criteria", {})
        for key in ("4_budget_computed_not_inconclusive", "5_se_derived_reviewed"):
            if key in criteria:
                criteria[key] = OUTCOME_STATUS_ONLY
        acceptance["ready_for_human_approval"] = False
        acceptance["accepted"] = False
    for entry in out.get("attempts", []):
        entry["gates"] = {g: _status_only_gate(r) for g, r in entry.get("gates", {}).items()}
    return _redact(out)


def _finish(report: dict[str, Any], selection: bool) -> dict[str, Any]:
    return report if selection else _status_only(report)


# ============================================================ extension decision for the runner
def extension_decision(spec: Any, arms: Sequence[str]) -> dict[str, bool]:
    """For ``epoch_budget_pilot.run_stage(..., extension_decision=...)``: which arms the §16.5 rule
    sends to cap 60. Refuses unless every selection prerequisite holds (§7.1, §13.1)."""
    report = evaluate_pilot(spec.root, selection=True, expected_spec=spec)
    if not report["selection_prerequisites"]["met"] or report["integrity_problems"] or report["blocking"]:
        raise sc.Blocked("extension decision refused: selection prerequisites not met ("
                         + "; ".join(report["selection_prerequisites"]["failed"] + report["integrity_problems"]
                                     + report["blocking"]) + ")")
    return {a: bool(report["arms"][a]["extension_required"]) for a in arms if a in report["arms"]}


# ============================================================ outputs
def epochs_csv(report: Mapping[str, Any], root: Path) -> str:
    """One row per arm, cap, seed and epoch (§13.2). Status-only mode: progress and time only."""
    selection = report["mode"] == "selection"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    cols = ["arm", "cap", "seed", "attempt_id", "epoch", "seconds"]
    if selection:
        cols += ["running_train_nll", "train_nll", "dev_nll"]
    w.writerow(cols)
    for e in report.get("attempts", []):
        run_dir = _run_path(e.get("run_dir"))
        if not e["latest"] or run_dir is None or not (run_dir / "history.jsonl").exists():
            continue
        for r in _read(run_dir / "history.jsonl"):
            row = [e["arm"], e["cap"], e["seed"], e["attempt_id"], r["epoch"] + 1, repr(r["seconds"])]
            if selection:
                row += [repr(r["running_train_nll"]), repr(r["train_nll"]), repr(r["dev_nll"])]
            w.writerow(row)
    return buf.getvalue()


def summary_md(report: Mapping[str, Any]) -> str:
    o = report["outcome"]
    lines = ["# Common epoch-budget pilot — summary", "",
             (f"Mode: **{report['mode']}** · generated {report['generated_at']} · conditions "
              f"`{str(report.get('conditions_sha256', ''))[:12]}`"), "",
             "## Outcome", "", f"**{o['outcome']}**" + (f": max_epochs = **{o['max_epochs']}**"
                                                      if o.get("max_epochs") is not None else ""), ""]
    for key in ("reasons", "arms", "events"):
        if o.get(key):
            lines += [f"- {key}: {', '.join(map(str, o[key]))}"]
    if o.get("requires_human_review"):
        lines.append(f"- SE-derived e* (explicit human review required): {', '.join(o['requires_human_review'])}")
    lines += ["", "## Blocking and integrity problems", ""]
    lines += [f"- {p}" for p in report.get("blocking", []) + report.get("integrity_problems", [])] or ["- none"]
    lines += ["", "## Attempts and gates", "", "| attempt | status | latest | " + " | ".join(GATES) + " |",
              "|---|---|---|" + "---|" * len(GATES)]
    for e in report.get("attempts", []):
        lines.append(f"| {e['attempt_id']} | {e['status']} | {'yes' if e['latest'] else 'no'} | "
                     + " | ".join(e["gates"][g]["status"] for g in GATES) + " |")
    fails = [(e["attempt_id"], g, r) for e in report.get("attempts", []) for g, r in e["gates"].items()
             if r["status"] in ("fail", "unresolved")]
    if fails:
        lines += ["", "First mismatch / reason per failed or unresolved gate:", ""]
        lines += [f"- {aid} {g}: {r['detail']}" for aid, g, r in fails]
    cb = report.get("category_b", {})
    lines += ["", "## Category B events (protocol §9.7)", ""]
    evs = cb.get("events", [])
    lines += [f"- `{ev['event_id']}` {ev['current_status']}: {ev['record']} epoch {ev['epoch_1based']} step "
              f"{ev['global_step']} `{'.'.join(map(str, ev['field']))}` — {ev['reason']}" for ev in evs] or ["- none"]
    if report["mode"] == "selection" and report.get("arms"):
        lines += ["", "## Selection (wavenet_spec §16.5)", ""]
        for arm, a in report["arms"].items():
            lines += [f"### Arm {arm} (η = {a['learning_rate']}) — {a['status']}", ""]
            for cap, c in a["caps"].items():
                s = c["selection"]
                lines += [(f"Cap {cap}: D̄* = {s['best_mean']!r} at ê = {s['best_epoch']}; e* = {s['selected_epoch']} "
                           f"(ε = {s['epsilon_at_selected']!r}, {s['epsilon_source_at_selected']}); still improving "
                           f"at cap: {s['still_improving_at_cap']}"), "",
                          "| epoch | " + " | ".join(f"seed {k}" for k in c["statistics"][0]["values"])
                          + " | D̄ | SD | SE | ε | ε source |",
                          "|---|" + "---|" * (len(c["statistics"][0]["values"]) + 5)]
                for st in c["statistics"]:
                    lines.append(f"| {st['epoch']} | " + " | ".join(repr(v) for v in st["values"].values())
                                 + f" | {st['mean']!r} | {st['sd']!r} | {st['se']!r} | {st['epsilon']!r} | "
                                 f"{st['epsilon_source']} |")
                lines.append("")
    lines += ["## Acceptance (protocol §14)", ""]
    for k, v in report.get("acceptance", {}).get("criteria", {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "The summary never accepts a budget: criteria 5 and 8 require a human decision.", ""]
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any], root: Path) -> list[Path]:
    root = Path(root)
    out = [root / name for name in OUTPUT_FILES]
    sc._write_json(out[0], report)
    for path, text in ((out[1], summary_md(report)), (out[2], epochs_csv(report, root))):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return out


def summarize(root: str | Path, *, selection: bool) -> dict[str, Any]:
    """Command: evaluate under the exclusive lock (no runner active) and write the three outputs."""
    sc.disable_git()
    paths = sc.PilotPaths(Path(root))
    if not paths.conditions.is_file():
        raise sc.Blocked("the pilot has not been started (no conditions.json)")
    with sc.PilotLock(paths, "summary"):
        report = evaluate_pilot(paths.root, selection=selection)
        write_outputs(report, paths.root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Independent summary of the common epoch-budget pilot (status-only by "
                                            "default; --selection for the §16.5 statistics and decisions)")
    p.add_argument("--selection", action="store_true", help="selection mode (protocol §13.1)")
    args = p.parse_args(argv)
    report = summarize(ebp.RESULTS_ROOT, selection=args.selection)
    print(f"{report['mode']}: {report['outcome']['outcome']}"
          + (f" (max_epochs = {report['outcome']['max_epochs']})" if report["outcome"].get("max_epochs") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
