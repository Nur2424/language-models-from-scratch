"""F3–F5 integration: the complete final-experiment workflow on synthetic data only.

1. seed-0 validation across all four arms, with a crash and an authorised restart from scratch;
2. the status-only human validation review;
3. completion runs, with a permanent mechanical instability that halts the experiment and a
   documented exclusion of that seed for all four arms;
4. independent verification;
5. dev analysis and frozen context definitions (with a runtime guard against test windows);
6. the protected test freeze (training closed);
7. synthetic test evaluation, with a crash after an output commit resolved by a documented adoption;
8. statistical analysis and report generation.

Nothing outside pytest's temporary folder is written; the real experiment folder is never created.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.training.wavenet_study.test_final_experiment import (
    SITECUSTOMIZE,
    approve,
    make_spec,
    make_world,
    quiet,
)
from training.wavenet_study import analyze_final_experiment as an
from training.wavenet_study import evaluate_final_test as ev
from training.wavenet_study import final_experiment as fe
from training.wavenet_study import final_statistics as fs
from training.wavenet_study import study_common as sc
from training.wavenet_study import verify_final_runs as vf

SEEDS = (0, 1, 2, 3)


class SimulatedCrash(BaseException):
    pass


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("integration")
    site = base / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    world = make_world(base)
    spec = make_spec(world, base / "final", seeds=SEEDS)
    root = Path(spec.root)
    log: dict = {"spec": spec, "root": root, "states": [], "codes": {}}
    real_root_existed = fe.RESULTS_ROOT.exists()

    def fault(mode, attempt, step=5):
        mp.setenv("PYTHONPATH", str(site))
        mp.setenv("FINAL_FAULT", mode)
        mp.setenv("FINAL_FAULT_ATTEMPT", attempt)
        mp.setenv("FINAL_FAULT_STEP", str(step))

    try:
        # 1. validation with a crash, then an authorised restart from scratch
        fault("crash", "C-s0-a1")
        log["codes"]["validation"] = fe.run_stage(spec, "validation", out=quiet([]))
        mp.delenv("FINAL_FAULT")
        log["codes"]["retry"] = fe.retry(spec, "C", 0, reason="child crash inspected; restart from scratch",
                                         authorized_by="Nur", out=quiet([]))
        log["codes"]["validation_rest"] = fe.run_stage(spec, "validation", out=quiet([]))
        log["states"].append(fe.status(spec)["state"])
        # 2. status-only review
        log["review_check"] = vf.verify_validation(root)
        approve(spec)
        # 3. completion with a permanent mechanical instability in seed 2
        fault("nan", "A-s2-a1", step=4)
        log["codes"]["completion"] = fe.run_stage(spec, "completion", out=quiet([]))
        mp.delenv("FINAL_FAULT")
        log["states"].append(fe.status(spec)["state"])
        fe.exclude_seed(spec, 2, decision="D-SYN-1", reason="category A in arm A, seed 2 (synthetic)",
                        authorized_by="Nur")
        log["codes"]["completion_rest"] = fe.run_stage(spec, "completion", out=quiet([]))
        log["states"].append(fe.status(spec)["state"])
        # 4. independent verification
        log["verification"] = vf.verify(root)
        # 5. dev analysis, with every window construction recorded
        from utils import data as udata
        from utils.data import load_names, split_names

        splits = split_names(load_names(world["data"]), fractions=(0.8, 0.1, 0.1), seed=42,
                             duplicate_policy="group")
        built_from: list[tuple] = []
        original = udata.make_context_windows

        def recording(names, vocab, *, context_length):
            built_from.append(tuple(names))
            return original(names, vocab, context_length=context_length)

        mp.setattr(udata, "make_context_windows", recording)
        log["dev"] = an.dev_analysis(root, out=quiet([]))
        mp.setattr(udata, "make_context_windows", original)
        log["test_windows_before_freeze"] = tuple(splits.test) in built_from
        log["windows_built_before_freeze"] = len(built_from)
        # 6. freeze
        log["freeze"] = ev.create_freeze(root, authorized_by="Nur", note="all training and analysis decisions frozen")
        log["states"].append(fe.status(spec)["state"])
        # 7. evaluation with a crash after one commit, resolved by a documented adoption
        commit = ev._commit

        def crashing(tmp, final):
            commit(tmp, final)
            if final.name == "B-s1":
                raise SimulatedCrash("evaluator died after committing B/1")

        mp.setattr(ev, "_commit", crashing)
        with pytest.raises(SimulatedCrash):
            ev.evaluate(root, out=quiet([]))
        mp.setattr(ev, "_commit", commit)
        log["codes"]["evaluate_blocked"] = ev.evaluate(root, out=quiet([]))
        ev.adopt(root, "B/1", deviation_id="DEV-SYN-1", reason="committed output re-validated", authorized_by="Nur")
        log["codes"]["evaluate"] = ev.evaluate(root, out=quiet([]))
        log["outputs"] = ev.verify_outputs(root)
        # 8. report
        log["report"] = an.final_report(root, out=quiet([]))
        log["records"] = fe.verify_records(spec)
    finally:
        mp.undo()
    log["real_root_created"] = fe.RESULTS_ROOT.exists() and not real_root_existed
    return log


def test_training_workflow_states_and_codes(flow):
    assert flow["codes"] == {"validation": 3, "retry": 0, "validation_rest": 0, "completion": 4,
                             "completion_rest": 0, "evaluate_blocked": ev.EXIT["blocked"], "evaluate": 0}
    assert flow["states"] == ["AWAITING_VALIDATION_REVIEW", "HALTED", "COMPLETE", "CLOSED"]
    assert flow["review_check"]["status"] == "pass"
    attempts = fe.make_store(flow["spec"]).attempts()
    assert attempts["C-s0-a1"]["status"] == "crashed" and attempts["C-s0-a2"]["retry_authorized_by"] == "Nur"
    assert attempts["A-s2-a1"]["status"] == "mechanically_unstable"
    assert not any(k.endswith(("-s2-a1",)) and k[0] != "A" for k in attempts)   # seed 2 stopped at the halt
    assert all(r["status"] == "completed" for k, r in attempts.items() if k not in ("C-s0-a1", "A-s2-a1"))


def test_verification_and_pairing(flow):
    v = flow["verification"]
    assert v["status"] == "pass" and v["excluded_seeds"] == [2]
    assert v["pairing"] == {"0": "pass", "1": "pass", "3": "pass"}
    assert v["runs"]["C/0"]["attempt_id"] == "C-s0-a2"
    assert flow["records"]["exclusions"] == 1 and flow["records"]["orphans"] == []


def test_test_data_isolated_until_the_freeze(flow):
    assert flow["test_windows_before_freeze"] is False and flow["windows_built_before_freeze"] > 0
    assert flow["real_root_created"] is False
    defs = json.loads((flow["root"] / ev.DEV_ANALYSIS / ev.BUCKETS_FILE).read_text())
    assert defs["definition_sha256"] == flow["freeze"]["bucket_definitions"]["definition_sha256"]
    with pytest.raises(sc.Blocked, match="training is closed"):
        fe.retry(flow["spec"], "A", 1, reason="x", authorized_by="Nur", out=quiet([]))


def test_once_only_evaluation_with_documented_adoption(flow):
    assert flow["outputs"]["status"] == "pass"
    states = ev.ledger_status(flow["root"])
    assert states["states"] == {**{f"{a}/{s}": "COMPLETED" for s in (0, 1, 3) for a in "ABCD"}, "B/1": "ADOPTED"}
    assert states["deviations"]["B/1"][0]["deviation_id"] == "DEV-SYN-1"
    started = [e["key"] for e in ev.read_ledger(flow["root"]) if e["event"] == "started"]
    assert len(started) == len(set(started)) == 12


def test_report_reflects_exclusions_deviations_and_statistics(flow):
    r = flow["report"]
    assert r["seeds"] == [0, 1, 3] and r["excluded_seeds"] == [2]
    assert r["exclusions"][0]["decision"] == "D-SYN-1"
    kinds = {d["kind"] for d in r["deviations"]}
    assert {"training_retry", "test_adopted"} <= kinds
    prim = r["statistics"]["primary"]
    assert prim["seeds"] == [0, 1, 3]
    assert all(prim["comparisons"][c]["describe"]["n"] == 3 for c in fs.CONTRASTS)
    assert prim["comparisons"]["C-A"]["sensitivity_sign_flip"]["n_patterns"] == 8
    assert r["statistics"]["exclusions"] == [{"seed": 2, "decision": "D-SYN-1",
                                              "reason": "category A in arm A, seed 2 (synthetic)"}]
    md = (flow["root"] / an.FINAL_DIR / "REPORT.md").read_text()
    assert "Seed 2 excluded for all arms (D-SYN-1)" in md and "test_adopted" in md and "training_retry" in md
    assert {row["seed"] for row in r["per_seed"]} == {0, 1, 3}
    assert next(row for row in r["per_seed"] if row["arm"] == "B" and row["seed"] == 1)["evaluation_state"] == \
        "ADOPTED"
