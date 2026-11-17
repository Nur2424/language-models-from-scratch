"""F5: dev analysis and final report, on synthetic data only.

One module-level synthetic experiment (four arms, seeds 0–2, three epochs): runner (F3), dev
analysis, freeze and protected evaluation (F4), then the report. Expected values are recomputed
independently from the run histories and the committed evaluation outputs.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import pytest

from tests.training.wavenet_study.test_final_experiment import approve, make_spec, make_world, quiet, tampered
from training.wavenet_study import analyze_final_experiment as an
from training.wavenet_study import context_buckets as cb
from training.wavenet_study import evaluate_final_test as ev
from training.wavenet_study import final_experiment as fe
from training.wavenet_study import final_statistics as fs
from training.wavenet_study import study_common as sc
from training.wavenet_study import verify_final_runs as vf

SEEDS = (0, 1, 2)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("f5_world")
    w = make_world(base)
    spec = make_spec(w, base / "final", seeds=SEEDS)
    assert fe.run_stage(spec, "validation", out=quiet([])) == 0
    approve(spec)
    assert fe.run_stage(spec, "completion", out=quiet([])) == 0
    root = Path(spec.root)
    out = {"spec": spec, "root": root}
    out["dev"] = an.dev_analysis(root, out=quiet([]))
    out["dev_hashes"] = {p.name: sc.sha256_file(p) for p in (root / ev.DEV_ANALYSIS).iterdir()}
    out["dev_again"] = an.dev_analysis(root, out=quiet([]))
    ev.create_freeze(root, authorized_by="Nur", note="frozen")
    with pytest.raises(sc.Blocked, match="not complete"):
        an.final_report(root, out=quiet([]))
    assert ev.evaluate(root, out=quiet([])) == 0
    out["report"] = an.final_report(root, out=quiet([]))
    return out


def _csv(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_text())))


# ============================================================ dev analysis
def test_dev_metrics_and_learning_curves(world):
    spec, root, dev = world["spec"], world["root"], world["dev"]
    sel = vf.selected_checkpoints(vf.verify(root))
    assert dev["complete"] is True and dev["selected_checkpoints"] == sel
    for a in "ABCD":
        for s in SEEDS:
            history = sc.read_jsonl_strict(Path(sel[f"{a}/{s}"]["run_dir"]) / "history.jsonl")
            devs = [r["dev_nll"] for r in history]
            row = dev["per_seed"][f"{a}/{s}"]
            assert row["best_epoch"] == devs.index(min(devs)) and row["best_dev_nll"] == min(devs)
            assert row["final_dev_nll"] == devs[-1] and row["epochs"] == spec.epochs
    curves = _csv(root / ev.DEV_ANALYSIS / "learning_curves.csv")
    assert len(curves) == 4 * len(SEEDS) * spec.epochs
    assert float(curves[0]["dev_nll"]) == sc.read_jsonl_strict(Path(sel["A/0"]["run_dir"]) / "history.jsonl")[0][
        "dev_nll"]
    per_seed = _csv(root / ev.DEV_ANALYSIS / "per_seed_dev.csv")
    assert {(r["arm"], int(r["seed"])) for r in per_seed} == {(a, s) for a in "ABCD" for s in SEEDS}


def test_dev_definitions_are_independent_and_frozen(world):
    root, spec = world["root"], world["spec"]
    data = ev.dev_windows_and_counts(spec)
    defs = cb.from_json((root / ev.DEV_ANALYSIS / ev.BUCKETS_FILE).read_text())
    assert defs == cb.define_buckets(data["train"], data["dev"])
    assert world["dev"]["bucket_definition_sha256"] == defs["definition_sha256"]
    for key, agg in world["dev"]["dev_buckets"].items():
        assert agg["num_targets"] == data["dev"].num_examples, key
        for block in agg["stratifications"].values():
            assert sum(c["n"] for c in block["categories"].values()) == data["dev"].num_examples
    assert world["dev_again"] == world["dev"]           # deterministic re-run before the freeze
    assert set(world["dev"]["dev_bucket_contrasts"]) == set(cb.STRATIFICATIONS)


def test_dev_analysis_refuses_after_freeze_and_with_altered_definitions(world, monkeypatch):
    with pytest.raises(sc.Blocked, match="frozen"):
        an.dev_analysis(world["root"], out=quiet([]))
    monkeypatch.setattr(fe, "freeze_path", lambda root: Path(root) / "no_such_freeze")
    path = world["root"] / ev.DEV_ANALYSIS / ev.BUCKETS_FILE
    doc = json.loads(path.read_text())
    doc["min_support"] = 400
    body = {k: v for k, v in doc.items() if k != "definition_sha256"}
    doc["definition_sha256"] = cb._digest(body)
    with tampered(path, lambda b: json.dumps(doc).encode()):
        with pytest.raises((sc.Blocked, cb.BucketError)):
            an.dev_analysis(world["root"], out=quiet([]))


# ============================================================ final statistics
def test_report_statistics_match_independent_recomputation(world):
    root, report = world["root"], world["report"]
    results = {k: json.loads((ev._paths(root)["evaluations"] / ev._dir_name(k) / ev.RESULT_FILE).read_text())
               for k in [f"{a}/{s}" for a in "ABCD" for s in SEEDS]}
    test = {a: {s: results[f"{a}/{s}"]["test_nll"] for s in SEEDS} for a in "ABCD"}
    dev = {a: {s: world["dev"]["per_seed"][f"{a}/{s}"]["best_dev_nll"] for s in SEEDS} for a in "ABCD"}
    expected = fs.analyse(test, dev, expected_seeds=list(SEEDS))
    assert report["statistics"] == expected
    prim = report["statistics"]["primary"]
    assert set(prim["families"]) == {"primary", "secondary"} and prim["comparisons"]["C-B"]["family"] == "descriptive"
    assert "t_test" not in prim["comparisons"]["C-B"]
    for c in fs.CONTRASTS:
        d = [test[x][s] - test[y][s] for s in SEEDS for x, y in [fs.FAVOURED.get(c, ("D", "B"))]] \
            if c != "interaction" else [(test["D"][s] - test["B"][s]) - (test["C"][s] - test["A"][s]) for s in SEEDS]
        assert [prim["comparisons"][c]["differences"][str(s)] for s in SEEDS] == d
    assert report["seeds"] == list(SEEDS) and report["excluded_seeds"] == []


def test_context_test_results_are_paired_category_means(world):
    root, report = world["root"], world["report"]
    results = {k: json.loads((ev._paths(root)["evaluations"] / ev._dir_name(k) / ev.RESULT_FILE).read_text())
               for k in [f"{a}/{s}" for a in "ABCD" for s in SEEDS]}
    for strat, cats in report["context_test"].items():
        for cat, entry in cats.items():
            means = {a: {s: results[f"{a}/{s}"]["buckets"]["stratifications"][strat]["categories"][cat]["mean_nll"]
                         for s in SEEDS} for a in "ABCD"}
            if entry["contrasts"] is None:
                assert any(v is None for m in means.values() for v in m.values())
                continue
            ca = [means["C"][s] - means["A"][s] for s in SEEDS]
            assert entry["contrasts"]["C-A"]["describe"] == fs.describe(ca)
            assert "t_test" not in entry["contrasts"]["C-A"]           # descriptive only


def test_csv_tables_round_trip(world):
    final = world["root"] / an.FINAL_DIR
    rows = _csv(final / "comparisons.csv")
    assert len(rows) == 12 and {r["endpoint"] for r in rows} == {"test_nll", "best_dev_nll"}
    prim = world["report"]["statistics"]["primary"]["comparisons"]
    for r in rows:
        if r["endpoint"] == "test_nll":
            c = prim[r["comparison"]]
            assert float(r["mean_difference"]) == c["describe"]["mean"]
            assert float(r["ci95_low"]) == c["describe"]["ci_low"]
            if r["comparison"] != "C-B":
                assert float(r["p_holm"]) == c["t_test"]["p_holm"] and r["sesoi_class"] == c["sesoi"]["class"]
            else:
                assert r["p_holm"] == "" and r["sesoi_class"] == ""
        else:
            assert r["p_two_sided"] == "" and r["p_holm"] == "" and r["sesoi_class"] == ""
    per_seed = _csv(final / "per_seed.csv")
    assert len(per_seed) == 12
    assert {r["evaluation_state"] for r in per_seed} == {"COMPLETED"}
    assert _csv(final / "context_test.csv") and len(_csv(final / "runtime.csv")) == 12
    assert json.loads((final / "report.json").read_text()) == world["report"]


def test_markdown_report_content(world):
    md = (world["root"] / an.FINAL_DIR / "REPORT.md").read_text()
    for heading in ("## Provenance", "## Primary endpoint: test NLL", "## Observed findings",
                    "## Secondary endpoint: best dev NLL (descriptive only)", "## Context-stratified test results",
                    "## Per-seed results", "## Exclusions and protocol deviations", "## Runtime (descriptive)",
                    "## Hypotheses (not tested by this experiment)", "## Limitations and interpretation warnings"):
        assert heading in md, heading
    assert world["report"]["freeze_sha256"] in md
    assert f"{world['spec'].epochs} epochs" in md
    lowered = md.lower()
    for claim in ("proves", "causes", "caused by", "demonstrates that hierarchy"):
        assert claim not in lowered
    assert "main effect is reported" in md and not re.search(r"main effect[^.\n]*\d", md)
    assert md.count("(observed)") + md.count("inconclusive at this sample size") >= 5


def test_report_requires_valid_outputs(world):
    out = ev._paths(world["root"])["evaluations"] / "D-s2" / ev.RESULT_FILE
    with tampered(out, lambda b: b.replace(b'"test_nll": ', b'"test_nll": 1', 1)):
        with pytest.raises(sc.Blocked, match="not complete and valid"):
            an.final_report(world["root"], out=quiet([]))
