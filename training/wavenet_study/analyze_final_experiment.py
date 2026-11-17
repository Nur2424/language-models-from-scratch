"""Final experiment analysis and research report (F5).

Two commands, in this order::

    python training/wavenet_study/analyze_final_experiment.py dev       before the test freeze
    python training/wavenet_study/analyze_final_experiment.py report    after every test evaluation

``dev`` (no test data): requires the independent verifier to pass. Per seed and arm: best dev NLL,
selected epoch and learning curves (from ``history.jsonl``); the selected checkpoints are loaded,
checked (selection, BatchNorm calibration, dev reproduction) and their per-target dev NLL computed;
the context definitions are built from training counts and dev targets only (F2) and written once
(``analysis/dev/bucket_definitions.json``); dev context aggregates and paired descriptive
differences per category. Writes ``analysis/dev/`` (JSON + CSV). ``dev_summary.json`` is written
last and marks the dev analysis complete; the freeze (F4) records its hash.

``report`` (after the protected evaluation): verifies the freeze and every committed output, then
D-31 statistics (F1): primary endpoint test NLL (paired t intervals and tests, Holm within the
primary and the secondary family, exact sign-flip sensitivity, SESOI ±0.01), secondary endpoint
best dev NLL (descriptive), C−B descriptive, context-stratified test results (exploratory,
descriptive only, frozen definitions), per-seed results, exclusions, protocol deviations,
runtime, limitations. Writes ``analysis/final/``: ``report.json``, CSV tables and ``REPORT.md``.

No averaged main effect is computed anywhere; nothing here changes a model, a checkpoint, a
hyperparameter, a selection or a frozen definition.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import math
import os
import statistics
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sc = importlib.import_module("training.wavenet_study.study_common")
fe = importlib.import_module("training.wavenet_study.final_experiment")
vf = importlib.import_module("training.wavenet_study.verify_final_runs")
ev = importlib.import_module("training.wavenet_study.evaluate_final_test")
cb = importlib.import_module("training.wavenet_study.context_buckets")
fs = importlib.import_module("training.wavenet_study.final_statistics")

FINAL_DIR = "analysis/final"
COMPARISONS = tuple(fs.CONTRASTS)                       # C-A, D-B, B-A, D-C, interaction, C-B
LIMITATIONS = (
    "Each arm differs from its comparison arm by a bundle of design features (for C−A and D−B: hierarchical "
    "tree vs one flat layer, depth, weight sharing, width, parameter count and compute). A difference is "
    "attributable to the bundle as implemented, not to any single feature such as 'hierarchy'.",
    "C−B changes both architecture and normalisation; it is descriptive only and supports no inference.",
    "All conclusions are conditional on the selected learning rate (0.03 for every arm, coarse grid, D-22), the "
    "common 26-epoch budget (D-32), plain SGD with batch size 32, this dataset and split, and CPU float32.",
    "Confidence intervals are nominal 95% per comparison, not simultaneous; Holm adjusts only the paired t-test "
    "p-values within each declared family. Sign-flip p-values are a separate sensitivity analysis.",
    "With n paired seeds the exact sign-flip test cannot go below 2/2^n; seeds measure initialisation and "
    "data-order variability only, not variability over datasets or hyperparameters.",
    "Dev NLL was used for learning-rate, budget and checkpoint selection, so best dev NLL is optimistically "
    "biased; it is descriptive only. Test NLL was evaluated once per run after the freeze.",
    "Context-stratified results are exploratory: definitions were fixed on dev data, test support may be small, "
    "categories are not crossed, and no inferential adjustment is made across categories.",
    "SESOI = 0.01 nats per character is a chosen practical convention, not an empirically derived threshold.",
    "Runtime figures describe this hardware, thread setting and measurement protocol only; no inferential "
    "efficiency claim is made.",
)


# ============================================================ small writers
def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _csv_text(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
    return buf.getvalue()


def _keys(spec: fe.FinalSpec, excluded: Sequence[int]) -> list[tuple[str, int]]:
    return [(a, s) for s in spec.seeds if s not in excluded for a in spec.arms]


def _paired_describe(values: Mapping[str, Mapping[int, float]], seeds: Sequence[int]) -> dict[str, Any]:
    """Descriptive paired statistics for every comparison (no p-values)."""
    out = {}
    for c in COMPARISONS:
        d = fs.paired_differences(values, c, seeds)
        out[c] = {"differences": {str(s): x for s, x in zip(seeds, d, strict=True)}, "describe": fs.describe(d)}
    return out


def _category_contrasts(per_run: Mapping[tuple[str, int], Mapping[str, Any]], spec: fe.FinalSpec,
                        seeds: Sequence[int]) -> dict[str, Any]:
    """Per stratification and category: paired descriptive contrasts of category mean NLL."""
    out: dict[str, Any] = {}
    first = per_run[next(iter(per_run))]
    for strat in cb.STRATIFICATIONS:
        out[strat] = {}
        for cat in first["stratifications"][strat]["categories"]:
            cells = {(a, s): per_run[(a, s)]["stratifications"][strat]["categories"][cat] for a, s in per_run}
            entry: dict[str, Any] = {"n_per_run": cells[next(iter(cells))]["n"],
                                     "insufficient_support": any(c["insufficient_support"] for c in cells.values())}
            if any(c["mean_nll"] is None for c in cells.values()) or len(seeds) < 2:
                entry["contrasts"] = None
                entry["note"] = "no targets in this category" if len(seeds) >= 2 else "fewer than two seeds"
            else:
                values = {a: {s: cells[(a, s)]["mean_nll"] for s in seeds} for a in spec.arms}
                entry["contrasts"] = _paired_describe(values, seeds)
            out[strat][cat] = entry
    return out


# ============================================================ dev analysis (before the freeze)
def dev_analysis(root: str | Path, *, out: Callable[[str], None] = fe.console) -> dict[str, Any]:
    """Dev metrics, learning curves, frozen context definitions and dev context analysis."""
    sc.disable_git()
    spec, frozen = ev._spec(root)
    if fe.freeze_path(root).exists():
        raise sc.Blocked("the test freeze exists: the dev analysis and its definitions are frozen")
    fe.prepare_threads(spec)
    if fe.status(spec)["state"] != "COMPLETE":
        raise sc.Blocked("training is not complete")
    report = vf.verify(root)
    selected = vf.selected_checkpoints(report)
    excluded = report["excluded_seeds"]
    seeds = [s for s in spec.seeds if s not in excluded]
    dev_dir = Path(root) / ev.DEV_ANALYSIS
    data = ev.dev_windows_and_counts(spec)
    defs = cb.define_buckets(data["train"], data["dev"])
    defs_path = dev_dir / ev.BUCKETS_FILE
    if defs_path.exists():
        if cb.from_json(defs_path.read_text(encoding="utf-8")) != defs:
            raise sc.Blocked("existing bucket definitions differ from a deterministic rebuild; inspect them")
    else:
        _atomic_text(defs_path, cb.to_json(defs))
    assignment = cb.assign(defs, data["dev"], data["counts"])
    batch = spec.expected_training["eval_batch_size"]
    per_seed, curves, buckets, checks = {}, [], {}, {}
    for arm, seed in _keys(spec, excluded):
        key = f"{arm}/{seed}"
        sel = selected[key]
        loaded = ev.load_selected_model(spec, arm, seed, sel)
        checks[key] = ev.selection_checks(loaded, sel, batch)
        res = ev.nll_pass(loaded["built"]["module"], loaded["model"], data["dev"], batch)
        history = loaded["history"]
        best = sel["best_epoch"]
        if not sc.values_equal(res["mean_nll"], history[best]["dev_nll"]):
            raise sc.Blocked(f"{key}: dev pass does not reproduce the recorded dev NLL")
        buckets[(arm, seed)] = cb.aggregate(assignment, res["per_target"])
        per_seed[key] = {"arm": arm, "seed": seed, "best_epoch": best, "best_dev_nll": history[best]["dev_nll"],
                         "train_nll_at_best": history[best]["train_nll"], "final_dev_nll": history[-1]["dev_nll"],
                         "epochs": len(history)}
        curves += [[arm, seed, r["epoch"], r["train_nll"], r["dev_nll"], r["is_best"]] for r in history]
        out(f"  dev {key}: done")
    best_dev = {a: {s: per_seed[f"{a}/{s}"]["best_dev_nll"] for s in seeds} for a in spec.arms}
    summary = {
        "complete": True, "conditions_sha256": frozen["conditions_sha256"], "selected_checkpoints": selected,
        "excluded_seeds": excluded, "seeds": seeds, "bucket_definition_sha256": defs["definition_sha256"],
        "bucket_definitions_file_sha256": sc.sha256_file(defs_path), "per_seed": per_seed, "checks": checks,
        "best_dev_nll_paired": _paired_describe(best_dev, seeds) if len(seeds) >= 2 else None,
        "dev_buckets": {f"{a}/{s}": v for (a, s), v in buckets.items()},
        "dev_bucket_contrasts": _category_contrasts(buckets, spec, seeds),
        "role": "descriptive; dev data defined the context categories and was used for selection",
    }
    _atomic_text(dev_dir / "per_seed_dev.csv", _csv_text(
        ["arm", "seed", "best_epoch", "best_dev_nll", "train_nll_at_best", "final_dev_nll", "epochs"],
        [[v[k] for k in ("arm", "seed", "best_epoch", "best_dev_nll", "train_nll_at_best", "final_dev_nll",
                         "epochs")] for v in per_seed.values()]))
    _atomic_text(dev_dir / "learning_curves.csv", _csv_text(["arm", "seed", "epoch", "train_nll", "dev_nll",
                                                             "is_best"], curves))
    _atomic_text(dev_dir / "dev_buckets.csv", _bucket_csv(buckets))
    _atomic_text(dev_dir / ev.DEV_SUMMARY_FILE, _json_text(summary))      # last: marks completeness
    return summary


def _bucket_csv(buckets: Mapping[tuple[str, int], Mapping[str, Any]]) -> str:
    rows = []
    for (arm, seed), agg in buckets.items():
        for strat, block in agg["stratifications"].items():
            for cat, c in block["categories"].items():
                rows.append([arm, seed, strat, cat, c["n"], c["nll_sum"], c["mean_nll"], c["share"],
                             c["insufficient_support"]])
    return _csv_text(["arm", "seed", "stratification", "category", "n", "nll_sum", "mean_nll", "share",
                      "insufficient_support"], rows)


# ============================================================ final report (after the evaluation)
def _deviations(root: str | Path, spec: fe.FinalSpec, states: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    store = fe.make_store(spec)
    for aid, rec in store.attempts().items():
        if rec.get("retry_of"):
            out.append({"kind": "training_retry", "attempt_id": aid, "retry_of": rec["retry_of"],
                        "reason": rec.get("retry_reason"), "authorized_by": rec.get("retry_authorized_by")})
    for eid, st in sc.CategoryBLedger(Path(root)).status().items():
        out.append({"kind": "category_b", "event_id": eid, "status": st["status"],
                    "attempt_id": st["event"].get("attempt_id"), "reason": st["event"].get("reason")})
    for key, st in states.items():
        for d in st["deviations"]:
            out.append({"kind": f"test_{d['event']}", "key": key, "deviation_id": d["deviation_id"],
                        "reason": d["reason"], "authorized_by": d["authorized_by"]})
    return out


def _runtime(spec: fe.FinalSpec, selected: Mapping[str, Any]) -> dict[str, Any]:
    per_run, per_arm = {}, {}
    for key, sel in selected.items():
        d = vf._run_path(sel["run_dir"])
        timing = json.loads((d / sc.TIMING_FILE).read_text(encoding="utf-8"))
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        per_run[key] = {"wall_seconds": timing["wall_seconds"],
                        "training_call_seconds": timing["training_call_seconds"],
                        "parameters": meta[fe.METADATA_BLOCK]["parameters"]}
    for arm in spec.arms:
        walls = [v["training_call_seconds"] for k, v in per_run.items() if k.split("/")[0] == arm]
        per_arm[arm] = {"runs": len(walls), "mean_training_call_seconds": statistics.fmean(walls),
                        "sd_training_call_seconds": statistics.stdev(walls) if len(walls) > 1 else None,
                        "min": min(walls), "max": max(walls)}
    return {"per_run": per_run, "per_arm": per_arm,
            "note": "descriptive only; includes diagnostics, finiteness checks and recalibration (timing.json)"}


def final_report(root: str | Path, *, out: Callable[[str], None] = fe.console) -> dict[str, Any]:
    """Statistics, tables and the Markdown report, after the protected test evaluation."""
    sc.disable_git()
    freeze = ev.load_freeze(root)
    spec = ev.verify_freeze_inputs(root, freeze)
    check = ev.verify_outputs(root)
    if check["status"] != "pass":
        raise sc.Blocked(f"the test evaluation is not complete and valid: {check['problems'] or check['states']}")
    states = ev.ledger_states(ev.read_ledger(root), freeze)
    dev_summary = json.loads((Path(root) / freeze["dev_analysis"]["path"]).read_text(encoding="utf-8"))
    excluded = freeze["excluded_seeds"]
    seeds = [s for s in spec.seeds if s not in excluded]
    results = {}
    for key in freeze["order"]:
        d = ev._paths(root)["evaluations"] / ev._dir_name(key)
        results[key] = json.loads((d / ev.RESULT_FILE).read_text(encoding="utf-8"))
    test_nll = {a: {s: results[f"{a}/{s}"]["test_nll"] for s in seeds} for a in spec.arms}
    best_dev = {a: {s: dev_summary["per_seed"][f"{a}/{s}"]["best_dev_nll"] for s in seeds} for a in spec.arms}
    exclusions = [fs.Exclusion(r["seed"], r["decision"], r["reason"]) for r in freeze["exclusions"]]
    stats = fs.analyse(test_nll, best_dev, expected_seeds=spec.seeds, exclusions=exclusions)
    context = _category_contrasts({(r["arm"], r["seed"]): r["buckets"] for r in results.values()}, spec, seeds)
    per_seed = [{"arm": a, "seed": s, "best_epoch": dev_summary["per_seed"][f"{a}/{s}"]["best_epoch"],
                 "best_dev_nll": best_dev[a][s], "test_nll": test_nll[a][s], "test_tokens":
                 results[f"{a}/{s}"]["num_tokens"], "evaluation_state": states[f"{a}/{s}"]["state"]}
                for s in seeds for a in spec.arms]
    report = {
        "format": "wavenet_study.final_report/1", "freeze_sha256": freeze["freeze_sha256"],
        "conditions_sha256": freeze["conditions_sha256"], "bucket_definition_sha256":
            freeze["bucket_definitions"]["definition_sha256"], "seeds": seeds, "excluded_seeds": excluded,
        "exclusions": freeze["exclusions"], "statistics": stats, "context_test": context, "per_seed": per_seed,
        "deviations": _deviations(root, spec, states), "runtime": _runtime(spec, freeze["checkpoints"]),
        "limitations": list(LIMITATIONS), "evaluation_states": {k: v["state"] for k, v in states.items()},
        "design": {"arms": {a: v["label"] for a, v in spec.arms.items()}, "learning_rates": spec.learning_rates,
                   "epochs": spec.epochs, "batch_size": spec.expected_training["batch_size"],
                   "eval_batch_size": freeze["evaluation"]["batch_size"], "seeds_planned": spec.seeds,
                   "decisions": spec.decisions},
    }
    final = Path(root) / FINAL_DIR
    _atomic_text(final / "report.json", _json_text(report))
    _atomic_text(final / "comparisons.csv", _comparisons_csv(stats))
    _atomic_text(final / "per_seed.csv", _csv_text(list(per_seed[0]), [list(r.values()) for r in per_seed]))
    _atomic_text(final / "context_test.csv", _context_csv(context))
    _atomic_text(final / "runtime.csv", _csv_text(["run", "wall_seconds", "training_call_seconds", "parameters"], [
        [k, v["wall_seconds"], v["training_call_seconds"], v["parameters"]]
        for k, v in report["runtime"]["per_run"].items()]))
    _atomic_text(final / "REPORT.md", render_markdown(report))
    out(f"report written to {final}")
    return report


def _comparisons_csv(stats: Mapping[str, Any]) -> str:
    rows = []
    for role in ("primary", "secondary"):
        block = stats[role]
        for name, c in block["comparisons"].items():
            d, t, s = c["describe"], c.get("t_test") or {}, c.get("sesoi") or {}
            f = c.get("sensitivity_sign_flip") or {}
            rows.append([block["endpoint"], name, c.get("family", "descriptive"), d["n"], d["mean"], d["sd"], d["se"],
                         d["ci_low"], d["ci_high"], t.get("t"), t.get("p_two_sided"), t.get("p_holm"),
                         f.get("p_two_sided"), s.get("class"), s.get("favoured_arm"), d["n_sharing_sign_of_mean"]])
    return _csv_text(["endpoint", "comparison", "family", "n", "mean_difference", "sd", "se", "ci95_low",
                      "ci95_high", "t", "p_two_sided", "p_holm", "p_sign_flip", "sesoi_class", "favoured_arm",
                      "seeds_sharing_sign"], rows)


def _context_csv(context: Mapping[str, Any]) -> str:
    rows = []
    for strat, cats in context.items():
        for cat, e in cats.items():
            for name, c in (e["contrasts"] or {}).items():
                d = c["describe"]
                rows.append([strat, cat, e["n_per_run"], e["insufficient_support"], name, d["n"], d["mean"],
                             d["ci_low"], d["ci_high"]])
    return _csv_text(["stratification", "category", "targets", "insufficient_support", "comparison", "seeds",
                      "mean_difference", "ci95_low", "ci95_high"], rows)


# ============================================================ Markdown
def _f(x: Any, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return "—" if not math.isfinite(x) else f"{x:.{digits}f}" if abs(x) >= 1e-4 or x == 0 else f"{x:.2e}"
    return str(x)


SESOI_TEXT = {
    "meaningful_negative": "interval entirely below −SESOI: lower NLL for {x} than {y} by more than 0.01 (observed)",
    "meaningful_positive": "interval entirely above +SESOI: lower NLL for {y} than {x} by more than 0.01 (observed)",
    "practically_equivalent": "interval within ±SESOI: no practically meaningful difference (observed)",
    "inconclusive": "interval crosses a SESOI boundary: inconclusive at this sample size",
}


def _finding(name: str, c: Mapping[str, Any]) -> str:
    d, s = c["describe"], c["sesoi"]["class"]
    if name == "interaction":
        text = {"meaningful_negative": "BatchNorm lowered NLL more in the hierarchical tree than in the flat MLP, by "
                                       "more than 0.01 (observed)",
                "meaningful_positive": "BatchNorm lowered NLL more in the flat MLP than in the hierarchical tree, by "
                                       "more than 0.01 (observed)"}.get(s, SESOI_TEXT[s])
    else:
        x, y = fs.FAVOURED[name]
        text = SESOI_TEXT[s].format(x=x, y=y)
    t = c["t_test"]
    return (f"- **{name}** ({fs.MEANINGS[name]}): mean paired difference {_f(d['mean'])} nats/char, 95% CI "
            f"[{_f(d['ci_low'])}, {_f(d['ci_high'])}], Holm-adjusted p = {_f(t.get('p_holm'))}, "
            f"{d['n_sharing_sign_of_mean'] if d['n_sharing_sign_of_mean'] is not None else '—'}/{d['n']} seeds share "
            f"the sign of the mean — {text}.")


def render_markdown(report: Mapping[str, Any]) -> str:
    st = report["statistics"]
    prim, sec = st["primary"], st["secondary"]
    L = ["# WaveNet final controlled experiment — research report", "",
         "Generated by `training/wavenet_study/analyze_final_experiment.py report` from verified records only.", "",
         "## Provenance", "",
         f"- Test freeze: `{report['freeze_sha256']}`",
         f"- Frozen conditions: `{report['conditions_sha256']}`",
         f"- Context definitions (dev-defined, frozen): `{report['bucket_definition_sha256']}`",
         f"- Seeds analysed: {', '.join(map(str, report['seeds']))}; excluded: "
         f"{', '.join(map(str, report['excluded_seeds'])) or 'none'}",
         f"- SciPy {st['method']['versions']['scipy']}, NumPy {st['method']['versions']['numpy']}", "",
         "## Design", "",
         "Arms: " + "; ".join(f"{a} {v}" for a, v in report["design"]["arms"].items()) + ". Learning rate "
         + ", ".join(f"{a} {v}" for a, v in report["design"]["learning_rates"].items())
         + f"; {report['design']['epochs']} epochs; plain SGD; batch {report['design']['batch_size']}; paired seeds "
         + ", ".join(map(str, report["design"]["seeds_planned"])) + ". Checkpoints selected by dev "
         "NLL (earliest epoch on a tie); test NLL evaluated once per run after the freeze. Differences are "
         "d = NLL(X) − NLL(Y): negative favours X. Primary family {C−A, D−B}; secondary family {B−A, D−C, "
         "interaction}; C−B descriptive only. No averaged main effect is reported.", "",
         "## Primary endpoint: test NLL (nats per character)", "",
         "| Comparison | Family | Mean diff | SD | 95% CI | t | p | Holm p | Sign-flip p | SESOI | Favoured |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in prim["comparisons"].items():
        d = c["describe"]
        t, s, f = c.get("t_test") or {}, c.get("sesoi") or {}, c.get("sensitivity_sign_flip") or {}
        L.append(f"| {name} | {c['family']} | {_f(d['mean'])} | {_f(d['sd'])} | [{_f(d['ci_low'])}, "
                 f"{_f(d['ci_high'])}] | {_f(t.get('t'), 2)} | {_f(t.get('p_two_sided'))} | {_f(t.get('p_holm'))} | "
                 f"{_f(f.get('p_two_sided'))} | {s.get('class', 'descriptive')} | {s.get('favoured_arm') or '—'} |")
    L += ["", "Holm correction is applied separately within the primary and the secondary family to the paired "
          "t-test p-values; the sign-flip p-values are an unadjusted sensitivity analysis.", "",
          "## Observed findings (primary endpoint)", ""]
    L += [_finding(n, prim["comparisons"][n]) for n in fs.INFERENTIAL]
    L += ["", "These statements describe what was observed under this design. They are not claims about any "
          "single architectural feature (see Limitations).", "",
          "## Secondary endpoint: best dev NLL (descriptive only)", "",
          "| Comparison | Mean diff | 95% CI (nominal) |", "|---|---|---|"]
    for name, c in sec["comparisons"].items():
        d = c["describe"]
        L.append(f"| {name} | {_f(d['mean'])} | [{_f(d['ci_low'])}, {_f(d['ci_high'])}] |")
    L += ["", "## Context-stratified test results (exploratory, descriptive)", "",
          "Frozen dev-defined categories; each stratification is reported separately (no cross-product).", "",
          "| Stratification | Category | Targets per run | Support | C−A mean diff [95% CI] | D−B mean diff [95% CI] |",
          "|---|---|---|---|---|---|"]
    for strat, cats in report["context_test"].items():
        for cat, e in cats.items():
            cells = []
            for name in ("C-A", "D-B"):
                c = (e["contrasts"] or {}).get(name)
                cells.append("—" if c is None else f"{_f(c['describe']['mean'])} [{_f(c['describe']['ci_low'])}, "
                                                   f"{_f(c['describe']['ci_high'])}]")
            L.append(f"| {strat} | {cat} | {e['n_per_run']} | {'INSUFFICIENT' if e['insufficient_support'] else 'ok'} "
                     f"| {cells[0]} | {cells[1]} |")
    L += ["", "## Per-seed results", "", "| Seed | Arm | Selected epoch | Best dev NLL | Test NLL | Evaluation |",
          "|---|---|---|---|---|---|"]
    L += [f"| {r['seed']} | {r['arm']} | {r['best_epoch'] + 1} | {_f(r['best_dev_nll'])} | {_f(r['test_nll'])} | "
          f"{r['evaluation_state']} |" for r in report["per_seed"]]
    L += ["", "## Exclusions and protocol deviations", ""]
    L += [f"- Seed {e['seed']} excluded for all arms ({e['decision']}): {e['reason']} (authorised by "
          f"{e['authorized_by']})" for e in report["exclusions"]] or ["- No seed was excluded."]
    L += [f"- {d['kind']}: " + ", ".join(f"{k}={v}" for k, v in d.items() if k != "kind")
          for d in report["deviations"]] or ["- No protocol deviation was recorded."]
    L += ["", "## Runtime (descriptive)", "", "| Arm | Runs | Mean training seconds | SD | Min | Max |",
          "|---|---|---|---|---|---|"]
    L += [f"| {a} | {v['runs']} | {_f(v['mean_training_call_seconds'], 1)} | {_f(v['sd_training_call_seconds'], 1)} | "
          f"{_f(v['min'], 1)} | {_f(v['max'], 1)} |" for a, v in report["runtime"]["per_arm"].items()]
    L += ["", "## Hypotheses (not tested by this experiment)", "",
          "- Any mechanism proposed for an observed difference (e.g. how hierarchical merging uses long contexts, or "
          "why BatchNorm interacts with depth) is a hypothesis for further work, not a finding of this report.",
          "- Whether the observed differences hold at other learning rates, budgets, model sizes or datasets.", "",
          "## Limitations and interpretation warnings", ""]
    L += [f"- {x}" for x in report["limitations"]]
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("dev", "report"))
    ap.add_argument("--root", default=str(fe.RESULTS_ROOT))
    args = ap.parse_args(argv)
    sc.disable_git()
    if args.command == "dev":
        dev_analysis(args.root)
    else:
        final_report(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
