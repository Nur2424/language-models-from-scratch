"""Independent verifier of the final experiment's training records (F3).

Read-only. Recomputes every applicable integrity gate from the files on disk and never trusts a
runner-written summary: the runner's ``pilot_result.json`` / attempt result is only cross-checked,
its ``integrity`` and gate fields are never used as evidence.

Per expected run (every arm of every non-excluded seed), from the latest attempt:
  * attempt closed ``completed`` in the hash-chained manifest; result file hash = manifest record;
  * G1 history has exactly one row per epoch 0 … E − 1 (E = frozen epochs);
  * G2 the run's stored experiment configuration equals the frozen arm configuration plus the
    seed; learning rate, epochs and training defaults equal the frozen values; the blocking
    source checksums still equal the frozen ones;
  * G3 run metadata status is ``completed``;
  * G4 checkpoint consistency with the history (accepted ``study_common.gate_g4``);
  * G7 state hashes match the checkpoint states (accepted ``study_common.gate_g7``);
  * schemas of every history row and step record equal the frozen record schemas;
  * every recorded value re-classified: no category A; category B events of the attempt resolved;
  * the selected checkpoint recomputed independently: earliest epoch of minimum dev NLL; it must be
    ``model_best.pt`` (epoch, step, metric, state hash);
  * parameter count and dataset / split fingerprint equal the frozen expectations.
Pairing, per seed: all four arms present and passing, the same conditions, the same dataset and
split fingerprints and window counts. No test window is built and no test metric is read.

``python training/wavenet_study/verify_final_runs.py [--root PATH] [--status-only]``
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sc = importlib.import_module("training.wavenet_study.study_common")
fe = importlib.import_module("training.wavenet_study.final_experiment")

GATES = ("attempt", "G1", "G2", "G3", "G4", "G7", "schema", "category_a", "category_b", "selection", "identity")


def _gate(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"gate": name, "status": "pass" if ok else "fail", "detail": "" if ok else detail}


def _run_path(run_dir: str) -> Path:
    p = Path(run_dir)
    return p if p.is_absolute() else REPO / p


def selected_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    """Earliest epoch index with the minimum dev NLL (strict improvement keeps the earliest)."""
    devs = [r["dev_nll"] for r in history]
    if not devs or any(not sc.is_number(d) or sc.is_nonfinite(d) for d in devs):
        raise ValueError("dev_nll missing or non-finite")
    best = min(devs)
    return devs.index(best)


def check_run(frozen: Mapping[str, Any], arm: str, seed: int, attempt: Mapping[str, Any],
              store: sc.AttemptStore, blocking_now: Mapping[str, str]) -> dict[str, Any]:
    """All per-run gates for the latest attempt of (arm, seed)."""
    cond = frozen["conditions"]
    epochs = cond["protocol"]["epochs"]
    gates: list[dict[str, Any]] = []
    out: dict[str, Any] = {"arm": arm, "seed": seed, "attempt_id": attempt["attempt_id"], "gates": gates}
    res = store.read_result(attempt["attempt_id"], attempt["key"])
    ok = attempt["status"] == "completed" and res is not None and res.get("status") == "completed" \
        and res.get("conditions_sha256") == frozen["conditions_sha256"] and bool(attempt.get("run_dir"))
    gates.append(_gate("attempt", ok, f"latest attempt {attempt['attempt_id']} is {attempt['status']} or its result "
                                      "file disagrees with the frozen conditions"))
    if not ok:
        return out
    run_dir = _run_path(attempt["run_dir"])
    out["run_dir"] = attempt["run_dir"]
    try:
        history = sc.read_jsonl_strict(run_dir / "history.jsonl")
        steps = sc.read_jsonl_strict(run_dir.joinpath(*sc.STEP_RECORDS_FILE))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        gates.append(_gate("G1", False, f"run records unreadable: {exc}"))
        return out
    gates.append(sc.gate_g1(history, epochs).to_dict())

    # G2: configuration and code
    expected_cfg = cond["arms"][arm]["config"]
    exp = config.get("experiment") or {}
    actual_cfg = {**exp, "reproducibility": {k: v for k, v in (exp.get("reproducibility") or {}).items()
                                             if k != "seed"}}
    training = config.get("training") or {}
    want_training = {**cond["expected_training"], "max_epochs": epochs, "lr": cond["learning_rates"][arm]["lr"]}
    diffs = sc.diff_paths(expected_cfg, actual_cfg)
    problems = [f"configuration differs at {sc.format_path(diffs[0])}"] if diffs else []
    if (exp.get("reproducibility") or {}).get("seed") != seed:
        problems.append("stored seed differs")
    problems += [f"training.{k} = {training.get(k)!r} != {v!r}" for k, v in want_training.items()
                 if not sc.values_equal(v, training.get(k))]
    changed = sorted(k for k in set(cond["checksums"]["blocking"]) | set(blocking_now)
                     if cond["checksums"]["blocking"].get(k) != blocking_now.get(k))
    problems += [f"blocking checksum changed: {changed[0]} (+{len(changed) - 1})"] if changed else []
    gates.append(_gate("G2", not problems, "; ".join(problems)))
    gates.append(sc.gate_g3(attempt_status="completed", metadata_status=meta.get("status")).to_dict())
    gates.append(sc.gate_g4(run_dir, cap=epochs).to_dict())
    gates.append(sc.gate_g7(run_dir).to_dict())

    # schemas and recorded values (recomputed; never the runner's classification)
    doc = cond["record_schemas"]["document"]
    schemas = fe.schemas_for(doc, arm)
    normalization = "batchnorm" if cond["spec"]["arms"][arm]["config"] in ("batchnorm_mlp", "wavenet_batchnorm") \
        else "none"
    schema_bad = [f for kind, rows, schema in (("history", history, schemas.history), ("steps", steps, schemas.steps))
                  for r in rows for f in sc.schema_findings(kind, r, schema)]
    gates.append(_gate("schema", not schema_bad, f"{len(schema_bad)} schema difference(s), first "
                       f"{sc.format_path(schema_bad[0].path) if schema_bad else ''}"))
    findings = [f for kind, rows in (("history", history), ("steps", steps)) for r in rows
                for f in sc.classify_record(kind, r, normalization=normalization)]
    cat_a = [f for f in findings if f.category == "A"]
    gates.append(_gate("category_a", not cat_a, f"{len(cat_a)} category A value(s)"))
    ledger = sc.CategoryBLedger(Path(frozen["conditions"]["spec"]["root"])).status()
    open_b = sorted(e for e, s in ledger.items() if s["event"].get("attempt_id") == attempt["attempt_id"]
                    and s["status"] in ("unresolved", "rejected"))
    gates.append(_gate("category_b", not open_b, f"unresolved or rejected category B events: {open_b[:3]}"))

    # independent selection of the checkpoint
    try:
        best = selected_epoch(history)
        payload = sc.load_payload(run_dir / "model_best.pt", expected_kind="best")
        hashes = sc.read_jsonl_strict(run_dir / sc.STATE_HASH_FILE)
        state = sc.state_hash(payload["model_state"])
        sel_problems = []
        if payload.get("epoch") != best + 1 or payload.get("step") != history[best]["steps"]:
            sel_problems.append(f"model_best.pt epoch/step {payload.get('epoch')}/{payload.get('step')} != selected "
                                f"epoch {best + 1}")
        if not sc.values_equal(payload.get("best_metric"), history[best]["dev_nll"]):
            sel_problems.append("model_best.pt best_metric != selected dev NLL")
        if len(hashes) <= best or hashes[best].get("sha256") != state:
            sel_problems.append("model_best.pt state hash != the state hash recorded at the selected epoch")
        gates.append(_gate("selection", not sel_problems, "; ".join(sel_problems)))
        out["selection"] = {"best_epoch": best, "model_best_sha256": sc.sha256_file(run_dir / "model_best.pt"),
                            "state_sha256": state}
    except Exception as exc:  # noqa: BLE001  (any failure fails the gate)
        gates.append(_gate("selection", False, f"{type(exc).__name__}: {exc}"))

    block = meta.get(fe.METADATA_BLOCK) or {}
    id_problems = []
    if block.get("parameters") != cond["arms"][arm]["expected_parameters"]:
        id_problems.append(f"parameters {block.get('parameters')} != {cond['arms'][arm]['expected_parameters']}")
    if block.get("attempt_id") != attempt["attempt_id"] or block.get("arm") != arm or block.get("seed") != seed:
        id_problems.append("run metadata names another attempt")
    if block.get("epochs") != epochs:
        id_problems.append("run metadata epochs differ")
    gates.append(_gate("identity", not id_problems, "; ".join(id_problems)))
    out["pairing_identity"] = {"dataset": sc.sha256_json(meta.get("dataset")),
                               "train_windows": block.get("train_windows"), "dev_windows": block.get("dev_windows"),
                               "conditions_sha256": res.get("conditions_sha256")}
    return out


def _passed(run: Mapping[str, Any]) -> bool:
    names = [g["gate"] for g in run["gates"]]
    return all(g["status"] == "pass" for g in run["gates"]) and set(GATES) <= set(names)


def verify(root: str | Path, *, keys: Sequence[tuple[str, int]] | None = None) -> dict[str, Any]:
    """Verify every expected run (or ``keys``) and the pairing; read-only."""
    paths = sc.PilotPaths(Path(root))
    frozen = sc.load_frozen(paths)
    if frozen is None:
        return {"status": "fail", "problems": ["no conditions.json"], "runs": {}}
    spec = fe.FinalSpec.from_json(frozen["conditions"]["spec"])
    problems: list[str] = []
    try:
        store = fe.make_store(spec)
        sc.verify_record_heads(paths, store)
        orphans = store.orphans()
        if orphans:
            problems.append(f"orphan records: {orphans[:3]}")
        excluded = fe.excluded_seeds(root)
        fe.read_reviews(root)
    except SystemExit as exc:  # sc.Blocked
        return {"status": "fail", "problems": [str(exc)], "runs": {}}
    blocking_now = sc.checksum_record(spec.repo, blocking=frozen["conditions"]["checksums"]["blocking"],
                                      recorded=[])["blocking"] if all(
        (Path(spec.repo) / p).is_file() for p in frozen["conditions"]["checksums"]["blocking"]) else {}
    latest = fe.latest_attempts(store)
    wanted = list(keys) if keys is not None else [(a, s) for s in spec.seeds if s not in excluded for a in spec.arms]
    runs: dict[str, Any] = {}
    incomplete = []
    for arm, seed in wanted:
        rec = latest.get((arm, seed))
        if rec is None:
            incomplete.append(f"{arm}/{seed}")
            continue
        runs[f"{arm}/{seed}"] = r = check_run(frozen, arm, seed, rec, store, blocking_now)
        r["status"] = "pass" if _passed(r) else "fail"
    pairing = {}
    for seed in sorted({s for _, s in wanted}):
        members = [runs.get(f"{a}/{seed}") for a in spec.arms if (a, seed) in wanted]
        if any(m is None or m["status"] != "pass" for m in members) or len(members) != len(spec.arms):
            pairing[str(seed)] = "incomplete"
            continue
        identities = {json.dumps(m["pairing_identity"], sort_keys=True) for m in members}
        pairing[str(seed)] = "pass" if len(identities) == 1 else "fail"
        if len(identities) != 1:
            problems.append(f"seed {seed}: arms differ in data, windows or conditions")
    datasets = {runs[k]["pairing_identity"]["dataset"] for k in runs if "pairing_identity" in runs[k]}
    if len(datasets) > 1:
        problems.append("runs use different datasets or splits")
    failed = sorted(k for k, r in runs.items() if r["status"] != "pass")
    problems += [f"{k}: " + "; ".join(f"{g['gate']}: {g['detail']}" for g in runs[k]["gates"]
                                      if g["status"] != "pass") for k in failed]
    if incomplete:
        status = "incomplete"
    else:
        status = "pass" if not problems and all(v == "pass" for v in pairing.values()) else "fail"
    return {"status": status, "problems": problems, "incomplete": incomplete, "excluded_seeds": excluded,
            "conditions_sha256": frozen["conditions_sha256"], "runs": runs, "pairing": pairing,
            "manifest_head": sc.manifest_head(paths)}


def status_only(report: Mapping[str, Any]) -> dict[str, Any]:
    """Statuses and gate outcomes only (no hashes of results, no learning metric)."""
    return {"status": report["status"], "problems": [str(p) for p in report.get("problems", [])],
            "incomplete": list(report.get("incomplete", [])), "pairing": dict(report.get("pairing", {})),
            "runs": {k: {"status": r.get("status"), "gates": {g["gate"]: g["status"] for g in r["gates"]}}
                     for k, r in report.get("runs", {}).items()}}


def verify_validation(root: str | Path) -> dict[str, Any]:
    """Status-only verification of the validation stage (seed ``seeds[0]``, every arm)."""
    frozen = sc.load_frozen(sc.PilotPaths(Path(root)))
    if frozen is None:
        return {"status": "fail", "problems": ["no conditions.json"]}
    spec = fe.FinalSpec.from_json(frozen["conditions"]["spec"])
    return status_only(verify(root, keys=[(a, spec.validation_seed) for a in spec.arms]))


def selected_checkpoints(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{"A/0": {run_dir, attempt_id, best_epoch, model_best_sha256, state_sha256}}`` of a passing report."""
    if report.get("status") != "pass":
        raise sc.Blocked(f"verification does not pass: {report.get('problems')}")
    return {k: {"run_dir": r["run_dir"], "attempt_id": r["attempt_id"], **r["selection"]}
            for k, r in report["runs"].items()}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=str(fe.RESULTS_ROOT))
    p.add_argument("--status-only", action="store_true")
    args = p.parse_args(argv)
    sc.disable_git()
    report = verify(args.root)
    print(json.dumps(status_only(report) if args.status_only else report, indent=2, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
