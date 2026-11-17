"""Protected one-time test evaluation of the final experiment (F4).

D-31 and addendum; F0 decision 10. This is the **only** module of the study that turns the test
split into windows, and it does so only inside :func:`evaluate`, after an explicit freeze.

Commands (``python training/wavenet_study/evaluate_final_test.py <command> …``)::

    freeze --authorized-by NAME --note TEXT        once; closes training (final_experiment.py)
    status                                         freeze and ledger state (no test metric)
    evaluate                                       every frozen checkpoint, each exactly once
    adopt --key A/0 --deviation ID --reason TEXT --authorized-by NAME
    authorize-reevaluation --key A/0 --deviation ID --reason TEXT --authorized-by NAME
    verify                                         ledger, freeze and committed outputs (read-only)

Freeze (``test_freeze/freeze.json``, written once): requires the independent verifier to pass for
every run of every non-excluded seed, the frozen dev analysis (``analysis/dev``) with its context
bucket definitions, and records the SHA-256 of every selected checkpoint (file and state), of the
conditions, exclusions, bucket definitions, dev analysis and all training / evaluation / analysis
code. Any later change of any of them blocks evaluation.

Evaluation of one frozen checkpoint (``A/0`` …), in one pass:
  1. the checkpoint file and state hash equal the freeze; strict state-dict load; eval mode;
  2. BatchNorm arms: every ``num_calibrations`` equals the checkpoint epoch; inference statistics
     finite with non-negative variance; recalibration on the training windows (a copy of the
     model) reproduces the stored inference statistics bit for bit;
  3. the dev NLL of the loaded checkpoint reproduces the recorded selection-time dev NLL exactly;
  4. test NLL through the accepted ``evaluate_nll`` path (batch 4096, ``NLLAccumulator``), with
     the finiteness monitor attached; the logits of the same forwards are captured and give the
     per-target NLL; their per-batch sums reproduce the accumulator bit for bit;
  5. the frozen context definitions assign every test target in all three stratifications;
     token counts, bucket counts and sums are checked against the accumulator;
  6. outputs are written to a temporary directory, validated, then committed atomically.

Ledger (``test_freeze/evaluation_ledger.jsonl``, append-only, hash-chained; every head is also
recorded in ``invocations.jsonl``). States per checkpoint::

    PENDING → STARTED → COMPLETED                      (terminal: never evaluated again)
    STARTED → INTERRUPTED (registered by a later run)  → BLOCKED: human review required
    STARTED → FAILED                                   → BLOCKED: human review required
    BLOCKED → AUTHORISED_REEVAL (documented protocol deviation; outputs quarantined) → STARTED
    BLOCKED → ADOPTED (documented deviation; only a committed, re-validated output)  (terminal)
    anything else                                      → INTEGRITY_FAILURE (refuses to proceed)

Interrupted outputs are never adopted silently. Test results are never written anywhere a
training, selection or analysis-definition step reads.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sc = importlib.import_module("training.wavenet_study.study_common")
fe = importlib.import_module("training.wavenet_study.final_experiment")
vf = importlib.import_module("training.wavenet_study.verify_final_runs")
cb = importlib.import_module("training.wavenet_study.context_buckets")

FORMAT = "wavenet_study.final_test_freeze/1"
RESULT_FORMAT = "wavenet_study.final_test_result/1"
LEDGER = f"{fe.FREEZE_DIR}/evaluation_ledger.jsonl"
EVALUATIONS = "evaluations"
QUARANTINE = "quarantine"
DEV_ANALYSIS = "analysis/dev"
BUCKETS_FILE = "bucket_definitions.json"
DEV_SUMMARY_FILE = "dev_summary.json"
RESULT_FILE = "result.json"
PER_TARGET_FILE = "per_target_nll.npy"
CODE_FILES = ("training/wavenet_study/study_common.py", "training/wavenet_study/final_experiment.py",
              "training/wavenet_study/verify_final_runs.py", "training/wavenet_study/evaluate_final_test.py",
              "training/wavenet_study/analyze_final_experiment.py", "training/wavenet_study/final_statistics.py",
              "training/wavenet_study/context_buckets.py", fe.RECORD_SCHEMAS)
TERMINAL = ("COMPLETED", "ADOPTED")
EXIT = {"blocked": 6, "failed": 7, "interrupted": 130}
U32 = 2.0 ** -24
U64 = 2.0 ** -53


class EvaluationIntegrityError(RuntimeError):
    """A selection, calibration, reproduction or consistency check failed."""


def _gamma(n: int, u: float) -> float:
    return n * u / (1 - n * u)


# ============================================================ shared: model loading and NLL passes
def load_selected_model(spec: fe.FinalSpec, arm: str, seed: int, selection: Mapping[str, Any]) -> dict[str, Any]:
    """Build the arm through the accepted set-up and load its selected ``model_best.pt`` (strict)."""
    run_dir = vf._run_path(selection["run_dir"])
    path = run_dir / "model_best.pt"
    if sc.sha256_file(path) != selection["model_best_sha256"]:
        raise EvaluationIntegrityError(f"{arm}/{seed}: model_best.pt differs from its recorded SHA-256")
    built = fe.build_arm(spec, arm, seed, spec.epochs, ["setup"])
    payload = sc.load_payload(path, expected_kind="best")
    if sc.state_hash(payload["model_state"]) != selection["state_sha256"]:
        raise EvaluationIntegrityError(f"{arm}/{seed}: checkpoint state hash differs from the recorded one")
    model = built["model"]
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    history = sc.read_jsonl_strict(run_dir / "history.jsonl")
    return {"built": built, "model": model, "payload": payload, "history": history, "run_dir": run_dir}


def selection_checks(loaded: Mapping[str, Any], selection: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    """Selected epoch, BatchNorm calibration and inference state, dev reproduction (raises on failure)."""
    import torch

    built, model, payload, history = loaded["built"], loaded["model"], loaded["payload"], loaded["history"]
    best = selection["best_epoch"]
    if payload["epoch"] != best + 1 or vf.selected_epoch(history) != best:
        raise EvaluationIntegrityError("the checkpoint is not the earliest minimum-dev-NLL epoch")
    if model.training:
        raise EvaluationIntegrityError("model is not in evaluation mode")
    checks: dict[str, Any] = {"selected_epoch": best, "mode": "eval"}
    bns = [(n, m) for n, m in model.named_modules() if hasattr(m, "inference_mean")]
    if built["normalization"] == "batchnorm":
        if not bns:
            raise EvaluationIntegrityError("BatchNorm arm without calibrated BatchNorm modules")
        for name, m in bns:
            if int(m.num_calibrations) != payload["epoch"]:
                raise EvaluationIntegrityError(f"{name}.num_calibrations {int(m.num_calibrations)} != epoch "
                                               f"{payload['epoch']}")
            if not (bool(torch.isfinite(m.inference_mean).all()) and bool(torch.isfinite(m.inference_var).all())
                    and bool((m.inference_var >= 0).all())):
                raise EvaluationIntegrityError(f"{name}: invalid inference statistics")
        twin = copy.deepcopy(model)
        built["module"].calibrate_batchnorm(twin, built["train"], batch_size=batch_size)
        for (name, m), (_, t) in zip(bns, [(n, x) for n, x in twin.named_modules() if hasattr(x, "inference_mean")],
                                     strict=True):
            if not (torch.equal(m.inference_mean, t.inference_mean) and torch.equal(m.inference_var, t.inference_var)):
                raise EvaluationIntegrityError(f"{name}: recalibration does not reproduce the stored statistics")
        checks["batchnorm"] = {"modules": [n for n, _ in bns], "num_calibrations": payload["epoch"],
                               "recalibration_reproduced": True}
    elif bns:
        raise EvaluationIntegrityError("non-BatchNorm arm with BatchNorm modules")
    dev = built["module"].evaluate_nll(model, built["dev"], batch_size=batch_size)
    if not sc.values_equal(dev.mean_nll, history[best]["dev_nll"]):
        raise EvaluationIntegrityError(f"dev NLL {dev.mean_nll!r} does not reproduce the recorded "
                                       f"{history[best]['dev_nll']!r}")
    checks["dev_reproduced"] = True
    return checks


def nll_pass(module: Any, model: Any, windows: Any, batch_size: int) -> dict[str, Any]:
    """Accepted ``evaluate_nll`` with the finiteness monitor and per-target NLL from the same logits."""
    from utils.metrics import token_nll

    captured: list[Any] = []
    handle = model.register_forward_hook(lambda _m, _i, out: captured.append(out.detach().clone()))
    try:
        with sc.FinitenessMonitor(model):
            acc = module.evaluate_nll(model, windows, batch_size=batch_size)
    finally:
        handle.remove()
    n = windows.num_examples
    starts = list(range(0, n, batch_size))
    if len(captured) != len(starts):
        raise EvaluationIntegrityError(f"{len(captured)} forwards captured for {len(starts)} batches")
    per_target = np.empty(n, dtype=np.float64)
    running, bound = 0.0, 0.0
    for start, logits in zip(starts, captured, strict=True):
        yb = windows.targets[start:start + batch_size]
        if logits.shape[0] != yb.shape[0]:
            raise EvaluationIntegrityError("captured logits do not match the batch")
        batch_sum = float(token_nll(logits, yb, reduction="sum").item())
        running += batch_sum
        values = token_nll(logits, yb, reduction="none").double().numpy()
        per_target[start:start + yb.shape[0]] = values
        exact = math.fsum(values.tolist())
        b = _gamma(len(values), U32) * math.fsum(abs(v) for v in values.tolist()) + U32 * abs(batch_sum)
        if abs(exact - batch_sum) > b:
            raise EvaluationIntegrityError("per-target NLL does not sum to the batch NLL")
        bound += b
    if running != acc.total_nll or acc.num_tokens != n:
        raise EvaluationIntegrityError("the captured logits do not reproduce the accumulator")
    if not np.all(np.isfinite(per_target)) or np.any(per_target < 0):
        raise EvaluationIntegrityError("non-finite or negative per-target NLL")
    total_fsum = math.fsum(per_target.tolist())
    bound += _gamma(len(starts) + 1, U64) * math.fsum(np.abs(per_target).tolist())
    if abs(total_fsum - acc.total_nll) > bound:
        raise EvaluationIntegrityError("per-target total differs from the accumulator beyond the summation bound")
    return {"accumulator": acc, "mean_nll": acc.mean_nll, "total_nll": acc.total_nll, "num_tokens": acc.num_tokens,
            "per_target": per_target, "per_target_fsum": total_fsum, "consistency_bound": bound,
            "batches": len(starts)}


def dev_windows_and_counts(spec: fe.FinalSpec) -> dict[str, Any]:
    """Training and dev windows (any arm: data and windows are shared) and the exact training counts."""
    built = fe.build_arm(spec, next(iter(spec.arms)), spec.validation_seed, spec.epochs, ["setup"])
    return {"train": built["train"], "dev": built["dev"], "counts": cb.count_train_contexts(built["train"])}


def _test_windows(spec: fe.FinalSpec, built: Mapping[str, Any]) -> Any:
    """The test windows (the study's only construction of them), with the arm's vocabulary."""
    from utils.data import load_names, make_context_windows, split_names

    cfg = built["cfg"]
    splits = split_names(load_names(cfg.data.path), fractions=cfg.data.split_fractions, seed=cfg.data.split_seed,
                         duplicate_policy=cfg.data.duplicate_policy)
    splits.assert_disjoint()
    return make_context_windows(splits.test, built["vocab"], context_length=cfg.model.params.context_length)


# ============================================================ freeze
def _paths(root: str | Path) -> dict[str, Path]:
    d = Path(root) / fe.FREEZE_DIR
    return {"dir": d, "freeze": d / fe.FREEZE_FILE, "ledger": Path(root) / LEDGER, "evaluations": d / EVALUATIONS,
            "quarantine": d / QUARANTINE, "dev": Path(root) / DEV_ANALYSIS}


def _spec(root: str | Path) -> tuple[fe.FinalSpec, dict[str, Any]]:
    frozen = sc.load_frozen(sc.PilotPaths(Path(root)))
    if frozen is None:
        raise sc.Blocked("no conditions.json: the experiment was never started")
    return fe.FinalSpec.from_json(frozen["conditions"]["spec"]), frozen


def code_hashes(repo: str | Path) -> dict[str, str]:
    files = sorted({*CODE_FILES, "models/feedforward/mlp.py", "models/feedforward/batchnorm_mlp.py",
                    "models/hierarchical/wavenet.py", *sc.python_files_under(repo, "utils")})
    return {p: sc.sha256_file(Path(repo) / p) for p in files}


def _key(arm: str, seed: int) -> str:
    return f"{arm}/{seed}"


def _dir_name(key: str) -> str:
    arm, seed = key.split("/")
    return f"{arm}-s{seed}"


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def create_freeze(root: str | Path, *, authorized_by: str, note: str) -> dict[str, Any]:
    """Freeze every input of the test evaluation (once). Closes training."""
    sc.disable_git()
    authorized_by, note = fe._nonempty(authorized_by, "--authorized-by"), fe._nonempty(note, "--note")
    spec, frozen = _spec(root)
    p = _paths(root)
    with sc.PilotLock(spec.paths, "test-freeze") as lock:
        if p["freeze"].exists() or p["ledger"].exists():
            raise sc.Blocked("the test freeze already exists; it is created exactly once")
        sc.verify_or_freeze(spec.paths, fe.current_conditions(spec), may_create=False, lock=lock)
        state = fe.status(spec)
        if state["state"] != "COMPLETE":
            raise sc.Blocked(f"training is not complete (state {state['state']})")
        report = vf.verify(root)
        selected = vf.selected_checkpoints(report)
        defs = cb.from_json((p["dev"] / BUCKETS_FILE).read_text(encoding="utf-8")) \
            if (p["dev"] / BUCKETS_FILE).is_file() else None
        if defs is None or not (p["dev"] / DEV_SUMMARY_FILE).is_file():
            raise sc.Blocked("the dev analysis and the frozen context definitions must exist before the freeze")
        dev_summary = json.loads((p["dev"] / DEV_SUMMARY_FILE).read_text(encoding="utf-8"))
        if dev_summary.get("complete") is not True or dev_summary.get("selected_checkpoints") != selected \
                or dev_summary.get("bucket_definition_sha256") != defs["definition_sha256"]:
            raise sc.Blocked("the dev analysis does not cover exactly the verified checkpoints and definitions")
        data = dev_windows_and_counts(spec)
        if defs["sources"]["train_fingerprint"] != data["counts"].fingerprint \
                or defs["sources"]["dev_fingerprint"] != cb.windows_fingerprint(data["dev"]):
            raise sc.Blocked("the context definitions were not built from this experiment's training and dev windows")
        body = {
            "format": FORMAT, "created_at": sc._utc_now(), "authorized_by": authorized_by, "note": note,
            "conditions_sha256": frozen["conditions_sha256"], "manifest_head": report["manifest_head"],
            "exclusions": fe.read_exclusions(root), "excluded_seeds": report["excluded_seeds"],
            "checkpoints": selected, "order": list(selected),
            "bucket_definitions": {"path": f"{DEV_ANALYSIS}/{BUCKETS_FILE}",
                                   "definition_sha256": defs["definition_sha256"],
                                   "file_sha256": sc.sha256_file(p["dev"] / BUCKETS_FILE)},
            "dev_analysis": {"path": f"{DEV_ANALYSIS}/{DEV_SUMMARY_FILE}",
                             "sha256": sc.sha256_file(p["dev"] / DEV_SUMMARY_FILE)},
            "code": code_hashes(spec.repo),
            "evaluation": {"batch_size": spec.expected_training["eval_batch_size"], "split": "test",
                           "endpoint": "test NLL (nats per character), NLLAccumulator via evaluate_nll",
                           "per_target": "token_nll(reduction='none') of the same logits",
                           "stratifications": list(cb.STRATIFICATIONS)},
        }
        doc = {**body, "freeze_sha256": sc.sha256_json(body)}
        p["dir"].mkdir(parents=True, exist_ok=True)
        _write_json_atomic(p["freeze"], doc)
        store = fe.make_store(spec, lock)
        session = sc.InvocationSession(spec.paths, lock, store, "test-freeze", authorized_by=authorized_by)
        try:
            head = sc.HashChainedLog(p["ledger"]).append({"event": "freeze", "freeze_sha256": doc["freeze_sha256"],
                                                          "at": sc._utc_now(), "authorized_by": authorized_by})
            session.note("evaluation_ledger", ledger_head=head, freeze_sha256=doc["freeze_sha256"])
        finally:
            session.end()
    return doc


def load_freeze(root: str | Path) -> dict[str, Any]:
    """Read and self-verify ``freeze.json``; :class:`sc.Blocked` if it is absent or altered."""
    p = _paths(root)
    if not p["freeze"].is_file():
        raise sc.Blocked("no test freeze: the test split cannot be evaluated before an explicit freeze")
    try:
        doc = json.loads(p["freeze"].read_text(encoding="utf-8"))
    except ValueError as exc:
        raise sc.Blocked(f"freeze.json is not valid JSON: {exc}") from None
    body = {k: v for k, v in doc.items() if k != "freeze_sha256"}
    if doc.get("format") != FORMAT or sc.sha256_json(body) != doc.get("freeze_sha256"):
        raise sc.Blocked("freeze.json does not match its own freeze_sha256 (edited after freezing?)")
    return doc


def verify_freeze_inputs(root: str | Path, freeze: Mapping[str, Any]) -> fe.FinalSpec:
    """Every frozen input is unchanged: conditions, exclusions, checkpoints, definitions, code."""
    spec, frozen = _spec(root)
    problems = []
    if frozen["conditions_sha256"] != freeze["conditions_sha256"]:
        problems.append("conditions changed")
    if fe.read_exclusions(root) != freeze["exclusions"]:
        problems.append("exclusions changed")
    for key, sel in freeze["checkpoints"].items():
        path = vf._run_path(sel["run_dir"]) / "model_best.pt"
        if not path.is_file() or sc.sha256_file(path) != sel["model_best_sha256"]:
            problems.append(f"checkpoint {key} changed")
    for what, rel, digest in (("bucket definitions", freeze["bucket_definitions"]["path"],
                               freeze["bucket_definitions"]["file_sha256"]),
                              ("dev analysis", freeze["dev_analysis"]["path"], freeze["dev_analysis"]["sha256"])):
        f = Path(root) / rel
        if not f.is_file() or sc.sha256_file(f) != digest:
            problems.append(f"{what} changed")
    now = code_hashes(spec.repo)
    changed = sorted(k for k in set(now) | set(freeze["code"]) if now.get(k) != freeze["code"].get(k))
    if changed:
        problems.append(f"code changed: {changed[:3]}")
    if problems:
        raise sc.Blocked("frozen evaluation inputs changed: " + "; ".join(problems))
    return spec


# ============================================================ ledger
def read_ledger(root: str | Path) -> list[dict[str, Any]]:
    """Verified ledger entries (hash chain and recorded heads)."""
    return fe._cross_checked_chain(root, LEDGER, "evaluation_ledger", "ledger_head")


def ledger_states(entries: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Replay the ledger into one state per frozen checkpoint; invalid sequences raise ``Blocked``."""
    if not entries or entries[0].get("event") != "freeze" or entries[0].get("freeze_sha256") != freeze["freeze_sha256"]:
        raise sc.Blocked("INTEGRITY_FAILURE: the ledger does not start with this freeze")
    states = {k: {"state": "PENDING", "evaluation_ids": [], "deviations": [], "last": None} for k in freeze["order"]}
    allowed = {"started": ("PENDING", "AUTHORISED_REEVAL"), "completed": ("STARTED",), "failed": ("STARTED",),
               "interrupted": ("STARTED",), "reevaluation_authorised": ("BLOCKED",), "adopted": ("BLOCKED",)}
    after = {"started": "STARTED", "completed": "COMPLETED", "failed": "BLOCKED", "interrupted": "BLOCKED",
             "reevaluation_authorised": "AUTHORISED_REEVAL", "adopted": "ADOPTED"}
    for n, e in enumerate(entries[1:], start=2):
        ev, key = e.get("event"), e.get("key")
        if ev not in allowed or key not in states:
            raise sc.Blocked(f"INTEGRITY_FAILURE: ledger line {n}: event {ev!r} for {key!r}")
        st = states[key]
        if st["state"] not in allowed[ev]:
            raise sc.Blocked(f"INTEGRITY_FAILURE: ledger line {n}: {ev} not allowed in state {st['state']} ({key})")
        if ev == "started":
            st["evaluation_ids"].append(e.get("evaluation_id"))
        elif e.get("evaluation_id") is not None and e.get("evaluation_id") != st["evaluation_ids"][-1]:
            raise sc.Blocked(f"INTEGRITY_FAILURE: ledger line {n}: {ev} names another evaluation")
        if ev == "adopted" and not (st["last"] or {}).get("output_present"):
            raise sc.Blocked(f"INTEGRITY_FAILURE: ledger line {n}: adoption without a committed output")
        if ev in ("reevaluation_authorised", "adopted"):
            st["deviations"].append({k: e.get(k) for k in ("event", "deviation_id", "reason", "authorized_by", "at")})
        st["state"] = after[ev]
        st["last"] = dict(e)
    return states


def _append(spec: fe.FinalSpec, session: sc.InvocationSession, root: str | Path, entry: Mapping[str, Any]) -> str:
    head = sc.HashChainedLog(_paths(root)["ledger"]).append(sc._plain({**entry, "at": sc._utc_now()}))
    session.note("evaluation_ledger", ledger_head=head, ledger_event=entry["event"], key=entry.get("key"))
    return head


# ============================================================ outputs
def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_output(out_dir: Path, *, key: str, evaluation_id: str, freeze: Mapping[str, Any],
                    assignment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Re-read a (temporary or committed) output and check it is complete and self-consistent."""
    try:
        result = json.loads((out_dir / RESULT_FILE).read_text(encoding="utf-8"))
        per_target = np.load(out_dir / PER_TARGET_FILE, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise EvaluationIntegrityError(f"{key}: output unreadable: {exc}") from None
    if result.get("format") != RESULT_FORMAT or result.get("key") != key \
            or result.get("evaluation_id") != evaluation_id or result.get("freeze_sha256") != freeze["freeze_sha256"]:
        raise EvaluationIntegrityError(f"{key}: output belongs to another evaluation or freeze")
    if result["files"][PER_TARGET_FILE] != sc.sha256_file(out_dir / PER_TARGET_FILE):
        raise EvaluationIntegrityError(f"{key}: per-target file differs from its recorded SHA-256")
    n = result["num_tokens"]
    if per_target.dtype != np.float64 or per_target.shape != (n,) or not np.all(np.isfinite(per_target)) \
            or np.any(per_target < 0):
        raise EvaluationIntegrityError(f"{key}: per-target array malformed")
    if not sc.values_equal(result["test_nll"], result["total_nll"] / n):
        raise EvaluationIntegrityError(f"{key}: test NLL is not total / tokens")
    if abs(math.fsum(per_target.tolist()) - result["total_nll"]) > result["consistency_bound"]:
        raise EvaluationIntegrityError(f"{key}: per-target sum differs from the total beyond the bound")
    if assignment is not None:
        if cb.aggregate(assignment, per_target) != result["buckets"]:
            raise EvaluationIntegrityError(f"{key}: bucket aggregates do not reproduce from the per-target NLL")
    return {"result_sha256": sc.sha256_file(out_dir / RESULT_FILE), **result["files"]}


def _commit(tmp: Path, final: Path) -> None:
    """Atomic directory commit (``os.replace``) followed by a directory fsync."""
    os.replace(tmp, final)
    _fsync_dir(final.parent)


def evaluate_one(spec: fe.FinalSpec, freeze: Mapping[str, Any], key: str, evaluation_id: str,
                 defs: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    """Evaluate one frozen checkpoint and commit its outputs; returns the committed file hashes."""
    t0 = time.perf_counter()
    arm, seed = key.split("/")[0], int(key.split("/")[1])
    sel = freeze["checkpoints"][key]
    batch = freeze["evaluation"]["batch_size"]
    loaded = load_selected_model(spec, arm, seed, sel)
    checks = selection_checks(loaded, sel, batch)
    built, model = loaded["built"], loaded["model"]
    test = _test_windows(spec, built)
    counts = cb.count_train_contexts(built["train"])
    res = nll_pass(built["module"], model, test, batch)
    assignment = cb.assign(defs, test, counts)
    buckets = cb.aggregate(assignment, res["per_target"])
    if buckets["num_targets"] != res["num_tokens"] or res["num_tokens"] != test.num_examples:
        raise EvaluationIntegrityError("token counts disagree")
    for s, block in buckets["stratifications"].items():
        if sum(c["n"] for c in block["categories"].values()) != res["num_tokens"]:
            raise EvaluationIntegrityError(f"{s}: bucket counts do not add up to the token count")
    if abs(buckets["nll_sum"] - res["total_nll"]) > res["consistency_bound"]:
        raise EvaluationIntegrityError("bucket total differs from the accumulator beyond the bound")
    cb.verify_definitions(defs)
    if defs["definition_sha256"] != freeze["bucket_definitions"]["definition_sha256"]:
        raise EvaluationIntegrityError("frozen definitions changed during evaluation")
    p = _paths(root)
    final = p["evaluations"] / _dir_name(key)
    if final.exists():
        raise sc.Blocked(f"{final} already exists; outputs are never overwritten")
    p["evaluations"].mkdir(parents=True, exist_ok=True)
    tmp = p["evaluations"] / f".tmp-{_dir_name(key)}-{evaluation_id.split('-e')[-1]}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    np.save(tmp / PER_TARGET_FILE, res["per_target"], allow_pickle=False)
    result = {
        "format": RESULT_FORMAT, "key": key, "arm": arm, "seed": seed, "evaluation_id": evaluation_id,
        "freeze_sha256": freeze["freeze_sha256"], "checkpoint": sel, "checks": checks,
        "test_nll": res["mean_nll"], "total_nll": res["total_nll"], "num_tokens": res["num_tokens"],
        "batches": res["batches"], "batch_size": batch, "per_target_fsum": res["per_target_fsum"],
        "consistency_bound": res["consistency_bound"], "test_windows_fingerprint": cb.windows_fingerprint(test),
        "definition_sha256": defs["definition_sha256"], "buckets": buckets,
        "files": {PER_TARGET_FILE: sc.sha256_file(tmp / PER_TARGET_FILE)},
        "seconds": time.perf_counter() - t0, "finished_at": sc._utc_now(),
    }
    with (tmp / RESULT_FILE).open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    with (tmp / PER_TARGET_FILE).open("rb") as fh:
        os.fsync(fh.fileno())
    validate_output(tmp, key=key, evaluation_id=evaluation_id, freeze=freeze, assignment=assignment)
    _commit(tmp, final)
    return validate_output(final, key=key, evaluation_id=evaluation_id, freeze=freeze)


# ============================================================ commands
def _session(spec: fe.FinalSpec, lock: sc.PilotLock, command: str, **fields: Any) -> sc.InvocationSession:
    return sc.InvocationSession(spec.paths, lock, fe.make_store(spec, lock), command, **fields)


def evaluate(root: str | Path, *, out: Callable[[str], None] = fe.console) -> int:
    """Evaluate every frozen checkpoint that is PENDING or AUTHORISED_REEVAL, each exactly once."""
    sc.disable_git()
    freeze = load_freeze(root)
    spec = verify_freeze_inputs(root, freeze)
    fe.prepare_threads(spec)
    p = _paths(root)
    defs = cb.from_json((Path(root) / freeze["bucket_definitions"]["path"]).read_text(encoding="utf-8"))
    with sc.PilotLock(spec.paths, "test-evaluate") as lock:
        freeze = load_freeze(root)
        verify_freeze_inputs(root, freeze)
        states = ledger_states(read_ledger(root), freeze)
        session = _session(spec, lock, "test-evaluate")
        code = 0
        try:
            for key, st in states.items():          # an earlier invocation died mid-evaluation
                if st["state"] == "STARTED":
                    eid = st["evaluation_ids"][-1]
                    final = p["evaluations"] / _dir_name(key)
                    present = final.is_dir()
                    _append(spec, session, root, {"event": "interrupted", "key": key, "evaluation_id": eid,
                                                  "output_present": present, "registered_by": "later invocation",
                                                  "temporary": sorted(x.name for x in p["evaluations"].glob(
                                                      f".tmp-{_dir_name(key)}-*")) if p["evaluations"].is_dir()
                                                  else []})
            states = ledger_states(read_ledger(root), freeze)
            blocked = [k for k, st in states.items() if st["state"] == "BLOCKED"]
            if blocked:
                out(f"BLOCKED: evaluation(s) {blocked} need human review (adopt or authorize-reevaluation, each a "
                    "documented protocol deviation); nothing was evaluated")
                return EXIT["blocked"]
            todo = [k for k, st in states.items() if st["state"] in ("PENDING", "AUTHORISED_REEVAL")]
            out(f"test evaluation: {len(todo)} checkpoint(s) to evaluate")
            for key in todo:
                eid = f"{key}-e{len(states[key]['evaluation_ids']) + 1}"
                if (p["evaluations"] / _dir_name(key)).exists():
                    raise sc.Blocked(f"an output for {key} exists although the ledger has none; inspect it")
                _append(spec, session, root, {"event": "started", "key": key, "evaluation_id": eid})
                try:
                    files = evaluate_one(spec, freeze, key, eid, defs, root)
                except KeyboardInterrupt:
                    _append(spec, session, root, {"event": "interrupted", "key": key, "evaluation_id": eid,
                                                  "output_present": (p["evaluations"] / _dir_name(key)).is_dir(),
                                                  "registered_by": "same invocation"})
                    out(f"  -> {key}: interrupted; human review required")
                    code = EXIT["interrupted"]
                    break
                except Exception as exc:  # noqa: BLE001  (every failure is recorded, then blocks)
                    _append(spec, session, root, {"event": "failed", "key": key, "evaluation_id": eid,
                                                  "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                                                  "output_present": (p["evaluations"] / _dir_name(key)).is_dir()})
                    out(f"  -> {key}: FAILED ({type(exc).__name__}); human review required")
                    code = EXIT["failed"]
                    break
                _append(spec, session, root, {"event": "completed", "key": key, "evaluation_id": eid,
                                              "output": f"{fe.FREEZE_DIR}/{EVALUATIONS}/{_dir_name(key)}",
                                              "files": files})
                out(f"  -> {key}: completed")
        finally:
            session.end(exit_code=code)
    return code


def _blocked_entry(root: str | Path, freeze: Mapping[str, Any], key: str) -> dict[str, Any]:
    states = ledger_states(read_ledger(root), freeze)
    if key not in states:
        raise sc.Blocked(f"{key} is not a frozen checkpoint")
    st = states[key]
    if st["state"] in TERMINAL:
        raise sc.Blocked(f"{key} is {st['state']}: a completed evaluation is never repeated")
    if st["state"] != "BLOCKED":
        raise sc.Blocked(f"{key} is {st['state']}; only a BLOCKED (interrupted or failed) evaluation needs a decision")
    return st


def authorize_reevaluation(root: str | Path, key: str, *, deviation_id: str, reason: str, authorized_by: str) -> str:
    """Documented protocol deviation: quarantine any output of a blocked evaluation; allow one new one."""
    sc.disable_git()
    deviation_id, reason, authorized_by = (fe._nonempty(deviation_id, "--deviation"), fe._nonempty(reason, "--reason"),
                                           fe._nonempty(authorized_by, "--authorized-by"))
    freeze = load_freeze(root)
    spec = verify_freeze_inputs(root, freeze)
    p = _paths(root)
    with sc.PilotLock(spec.paths, "authorize-reevaluation") as lock:
        st = _blocked_entry(root, freeze, key)
        session = _session(spec, lock, "authorize-reevaluation", key=key, deviation_id=deviation_id)
        try:
            moved = []
            dest = p["quarantine"] / f"{_dir_name(key)}-{deviation_id}"
            sources = [p["evaluations"] / _dir_name(key)] + (sorted(p["evaluations"].glob(f".tmp-{_dir_name(key)}-*"))
                                                             if p["evaluations"].is_dir() else [])
            for src in sources:
                if src.exists():
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest / src.name))
                    moved.append(f"{dest.name}/{src.name}")
            head = _append(spec, session, root, {"event": "reevaluation_authorised", "key": key,
                                                 "evaluation_id": st["evaluation_ids"][-1],
                                                 "deviation_id": deviation_id, "reason": reason,
                                                 "authorized_by": authorized_by, "quarantined": moved})
        finally:
            session.end()
    return head


def adopt(root: str | Path, key: str, *, deviation_id: str, reason: str, authorized_by: str) -> str:
    """Documented protocol deviation: adopt a committed output of an interrupted evaluation after re-validation."""
    sc.disable_git()
    deviation_id, reason, authorized_by = (fe._nonempty(deviation_id, "--deviation"), fe._nonempty(reason, "--reason"),
                                           fe._nonempty(authorized_by, "--authorized-by"))
    freeze = load_freeze(root)
    spec = verify_freeze_inputs(root, freeze)
    p = _paths(root)
    with sc.PilotLock(spec.paths, "adopt") as lock:
        st = _blocked_entry(root, freeze, key)
        eid = st["evaluation_ids"][-1]
        final = p["evaluations"] / _dir_name(key)
        if not final.is_dir() or not st["last"].get("output_present"):
            raise sc.Blocked(f"{key}: there is no committed output to adopt (authorize a re-evaluation instead)")
        arm, seed = key.split("/")[0], int(key.split("/")[1])
        built = fe.build_arm(spec, arm, seed, spec.epochs, ["setup"])
        defs = cb.from_json((Path(root) / freeze["bucket_definitions"]["path"]).read_text(encoding="utf-8"))
        assignment = cb.assign(defs, _test_windows(spec, built), cb.count_train_contexts(built["train"]))
        files = validate_output(final, key=key, evaluation_id=eid, freeze=freeze, assignment=assignment)
        session = _session(spec, lock, "adopt", key=key, deviation_id=deviation_id)
        try:
            head = _append(spec, session, root, {"event": "adopted", "key": key, "evaluation_id": eid,
                                                 "deviation_id": deviation_id, "reason": reason,
                                                 "authorized_by": authorized_by, "files": files})
        finally:
            session.end()
    return head


def ledger_status(root: str | Path) -> dict[str, Any]:
    """Freeze and per-checkpoint ledger states only (no test metric)."""
    freeze = load_freeze(root)
    states = ledger_states(read_ledger(root), freeze)
    return {"freeze_sha256": freeze["freeze_sha256"], "states": {k: v["state"] for k, v in states.items()},
            "deviations": {k: v["deviations"] for k, v in states.items() if v["deviations"]}}


def verify_outputs(root: str | Path) -> dict[str, Any]:
    """Every terminal evaluation's committed files still match the ledger (read-only)."""
    freeze = load_freeze(root)
    states = ledger_states(read_ledger(root), freeze)
    p = _paths(root)
    problems = []
    for key, st in states.items():
        if st["state"] not in TERMINAL:
            continue
        d = p["evaluations"] / _dir_name(key)
        try:
            files = validate_output(d, key=key, evaluation_id=st["evaluation_ids"][-1], freeze=freeze)
        except EvaluationIntegrityError as exc:
            problems.append(str(exc))
            continue
        if files != st["last"]["files"]:
            problems.append(f"{key}: committed files differ from the ledger")
    complete = all(st["state"] in TERMINAL for st in states.values())
    return {"status": "pass" if complete and not problems else ("fail" if problems else "incomplete"),
            "problems": problems, "states": {k: v["state"] for k, v in states.items()}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(fe.RESULTS_ROOT))
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--authorized-by", required=True)
    f.add_argument("--note", required=True)
    sub.add_parser("status")
    sub.add_parser("evaluate")
    sub.add_parser("verify")
    for name in ("adopt", "authorize-reevaluation"):
        a = sub.add_parser(name)
        for opt in ("--key", "--deviation", "--reason", "--authorized-by"):
            a.add_argument(opt, required=True)
    args = ap.parse_args(argv)
    sc.disable_git()
    if args.command == "freeze":
        print(create_freeze(args.root, authorized_by=args.authorized_by, note=args.note)["freeze_sha256"])
        return 0
    if args.command == "evaluate":
        return evaluate(args.root)
    if args.command in ("adopt", "authorize-reevaluation"):
        fn = adopt if args.command == "adopt" else authorize_reevaluation
        print(fn(args.root, args.key, deviation_id=args.deviation, reason=args.reason,
                 authorized_by=args.authorized_by))
        return 0
    report = ledger_status(args.root) if args.command == "status" else verify_outputs(args.root)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    from training.wavenet_study import evaluate_final_test as _module  # patchable module object (not __main__)

    sys.exit(_module.main())
