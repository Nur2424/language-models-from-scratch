"""Summarise the WaveNet-study learning-rate pilot (wavenet_spec.md v0.4, §16.4).

Reads only what ``lr_pilot.py`` wrote under ``results/experiments/wavenet_study/lr_pilot/``
and writes ``summary.json``, ``summary.md`` and ``epochs.csv`` there. It trains nothing,
never touches test data and never launches anything. It holds the pilot lock, so it
cannot run while a ``run`` / ``retry`` / ``benchmark`` is active.

Two modes, chosen automatically per arm:

* **status only** (default): every attempt, its status, integrity checks and per-run
  records. No candidate means, no ranking, no selection, no extension.
* **selection**: only when (a) nothing global blocks (frozen conditions intact, no
  unfinished attempts, no unrecorded run directories), (b) every initial-grid LR, and
  every extension LR already run for that arm, has a final, integrity-checked result for
  all three seeds, and (c) the extension rounds are in order.

Categories of a candidate (one LR of one arm):

* ``incomplete``: some seed has no final, valid result (not run, unresolved, integrity error).
* ``mechanically_unstable``: a non-finite loss / gradient / recalibration, or a non-finite
  recorded NLL, on any seed. Excluded automatically.
* ``flagged_for_review``: threshold-free optimisation observations on some seed: dev NLL rose
  between consecutive epochs (``repeated`` = in two or more epochs), running train NLL rose, or
  the epoch-10 dev NLL is above that seed's best. An observation, not evidence of instability.
* ``no_flags_raised``: none of the above. Stability is decided at human review.

Selection (unchanged protocol): lowest mean epoch-10 dev NLL across seeds 100–102 among
candidates that are not mechanically unstable (flagged ones stay eligible until review). The
boundary rule uses the evaluated grid including unstable points; at most two extension rounds.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.wavenet_study.lr_pilot import (  # noqa: E402, RUF100  (needs the sys.path set-up above)
    ARMS,
    ATTEMPTS_DIR,
    EXTENSION_DOWN,
    EXTENSION_UP,
    FINAL_STATUSES,
    INITIAL_GRID,
    MAX_EPOCHS,
    MAX_ROUNDS,
    PILOT_ROOT,
    SEEDS,
    attempts_by_id,
    attempts_by_key,
    diff_paths,
    load_frozen,
    lr_tag,
    pilot_lock,
    repo_relative,
    run_key,
    source_checksums,
    unrecorded_run_dirs,
)

HISTORY_FIELDS = ("epoch", "lr", "steps", "running_train_nll", "train_nll", "dev_nll", "is_best", "seconds")


def num(x) -> float:
    """History floats; the shared JSON writer stores non-finite values as "nan" / "inf" / "-inf"."""
    return float(x)


def finite(x) -> bool:
    return x is not None and math.isfinite(x)


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ================================================================== integrity (S1)
def validate_run(rec: dict, frozen: dict) -> tuple[list[str], list[dict]]:
    """Problems with one final attempt's records, and its parsed history (empty on failure)."""
    import torch

    from utils.checkpoint import load_checkpoint

    problems: list[str] = []
    arm, lr, seed, status = rec["arm"], float(rec["lr"]), int(rec["seed"]), rec["status"]
    cond = frozen["conditions"]
    attempt_file = ATTEMPTS_DIR / f"{rec['attempt_id']}.json"
    if not attempt_file.is_file():
        problems.append("attempt result file missing")
    else:
        res = read_json(attempt_file)
        if (res.get("status"), run_key(res["arm"], res["lr"], res["seed"])) != (status, run_key(arm, lr, seed)):
            problems.append("attempt result file disagrees with the manifest")
        if res.get("conditions_sha256") != frozen["conditions_sha256"]:
            problems.append("attempt ran under different frozen conditions")
    if not rec.get("run_dir") or not (REPO / rec["run_dir"]).is_dir():
        return problems + ["run directory missing"], []
    run_dir = REPO / rec["run_dir"]

    def load(name):
        try:
            return read_json(run_dir / name)
        except (OSError, ValueError) as exc:
            problems.append(f"{name} unreadable: {type(exc).__name__}")
            return None

    meta, cfg = load("metadata.json"), load("config.json")
    for name in ("timing.json", "pilot_result.json"):
        if not (run_dir / name).is_file():
            problems.append(f"{name} missing")
    if meta is not None:
        expected_status = "completed" if status == "completed" else "failed"
        if meta.get("status") != expected_status:
            problems.append(f"metadata status {meta.get('status')!r}, expected {expected_status!r}")
        pilot = meta.get("pilot") or {}
        if (pilot.get("attempt_id"), pilot.get("arm"), pilot.get("lr"), pilot.get("seed")) != \
                (rec["attempt_id"], arm, lr, seed):
            problems.append("metadata pilot block does not match the attempt")
        if (meta.get("git") or {}).get("available") is not False:
            problems.append("metadata git record is not the disabled record")
    if cfg is not None:
        expected_exp = copy.deepcopy(cond["arms"][arm]["config"])
        expected_exp["reproducibility"]["seed"] = seed
        diffs = diff_paths(expected_exp, cfg.get("experiment"))
        if diffs:
            problems.append(f"config.json experiment differs from the frozen arm config at {diffs[:5]}")
        if cfg.get("training") != {**cond["protocol"]["training"], "lr": lr}:
            problems.append(f"config.json training {cfg.get('training')} differs from the protocol")
        if cfg.get("model") != expected_exp["model"]["params"]:  # the trainer's architecture record
            problems.append(f"config.json model {cfg.get('model')} differs from the frozen architecture")
        if (cfg.get("pilot") or {}).get("attempt_id") != rec["attempt_id"]:
            problems.append("config.json pilot block does not match the attempt")

    # history: contiguous epochs 0..n-1, all fields, cumulative steps n·steps_per_epoch
    try:
        history = read_jsonl(run_dir / "history.jsonl")
        for h in history:
            missing = [f for f in HISTORY_FIELDS if f not in h]
            if missing:
                raise ValueError(f"row {h.get('epoch')} lacks {missing}")
            for f in ("running_train_nll", "train_nll", "dev_nll", "seconds"):
                num(h[f])
    except (OSError, ValueError, TypeError) as exc:
        return problems + [f"history.jsonl unreadable: {exc}"], []
    n = len(history)
    if [h["epoch"] for h in history] != list(range(n)):
        problems.append(f"history epochs are {[h['epoch'] for h in history]}, expected 0..{n - 1}")
    if status == "completed" and n != MAX_EPOCHS:
        problems.append(f"completed run has {n} history rows, expected {MAX_EPOCHS}")
    if status == "mechanically_unstable" and n >= MAX_EPOCHS:
        problems.append("mechanically unstable run has a complete history")
    per_epoch = history[0]["steps"] if history else 0
    if any(h["steps"] != (h["epoch"] + 1) * per_epoch for h in history):
        problems.append("history step counts are not cumulative multiples of one epoch")
    total_steps = n * per_epoch

    # step diagnostics: one record per scheduled step, each with gradients and updates
    diag = cond["diagnostics"]
    steps_path = run_dir / "diagnostics" / "steps.jsonl"
    if diag["enabled"] and status == "completed":
        try:
            recs = read_jsonl(steps_path)
            expected = len(range(0, total_steps, diag["every_n_steps"]))
            if len(recs) != expected:
                problems.append(f"steps.jsonl has {len(recs)} records, expected {expected}")
            if diag["gradients"] and any("grad_global_norm" not in r for r in recs):
                problems.append("steps.jsonl records without gradient diagnostics")
            if diag["parameters"] and any("updates" not in r for r in recs):
                problems.append("steps.jsonl records without update diagnostics")
        except (OSError, ValueError) as exc:
            problems.append(f"steps.jsonl unreadable: {exc}")

    # checkpoints: loadable, right kind, consistent with the history, finite weights
    dev = [num(h["dev_nll"]) for h in history]
    best_e = min(range(n), key=lambda i: (dev[i], i)) if n else None

    def ckpt(kind, filename):
        path = run_dir / filename
        try:
            payload = load_checkpoint(path, map_location="cpu", expected_kind=kind)
        except Exception as exc:  # noqa: BLE001  (CheckpointError and anything torch raises)
            problems.append(f"{filename}: {type(exc).__name__}: {str(exc)[:160]}")
            return None
        weights = [t for t in payload["model_state"].values() if t.is_floating_point()]
        if status == "completed" and not all(bool(torch.isfinite(t).all()) for t in weights):
            problems.append(f"{filename}: non-finite weights")
        return payload

    if status == "completed" or n >= 1:
        best = ckpt("best", "model_best.pt")
        if best is not None and best_e is not None and (
                best["epoch"] != best_e + 1 or (best.get("extra") or {}).get("best_epoch") != best_e
                or best["best_metric"] != dev[best_e]):
                problems.append(f"model_best.pt (epoch {best['epoch']}, metric {best['best_metric']}) does not match "
                                f"the history's best epoch {best_e + 1} ({dev[best_e]})")
        resume = ckpt("resume", "checkpoint_last.pt")
        if resume is not None and (resume["epoch"] != n or resume["step"] != total_steps):
            problems.append(f"checkpoint_last.pt at epoch {resume['epoch']} step {resume['step']}, "
                            f"expected {n} / {total_steps}")
    final_path = run_dir / "model_final.pt"
    if status == "completed":
        final = ckpt("final", "model_final.pt")
        if final is not None and (final["epoch"] != MAX_EPOCHS or final["step"] != total_steps):
            problems.append(f"model_final.pt at epoch {final['epoch']} step {final['step']}")
    elif final_path.exists():
        problems.append("a mechanically unstable run has model_final.pt")
    return problems, history


# ================================================================== per-run evidence
def rising_epochs(curve: list[float]) -> list[int]:
    """1-based epochs e >= 2 with curve[e] > curve[e-1]."""
    return [e + 1 for e in range(1, len(curve)) if curve[e] > curve[e - 1]]


def gradient_evidence(steps: list[dict]) -> dict:
    """Global gradient norm from the step diagnostics: per epoch first (S4), whole run secondary.

    Within-run ratios (max / median) are scale-free. Raw norms are not comparable between
    BatchNorm and non-BatchNorm arms: with BatchNorm the loss is invariant to the scale of the
    weights feeding it, so their gradients shrink as those weights grow.
    """
    by_epoch = defaultdict(list)
    for r in steps:
        if "grad_global_norm" in r:
            by_epoch[r["epoch"] + 1].append((r["step"], r["grad_global_norm"]))
    if not by_epoch:
        return {"available": False}
    per_epoch = {}
    for e, vals in sorted(by_epoch.items()):
        norms = [g for _, g in vals]
        med = statistics.median(norms)
        s_max, g_max = max(vals, key=lambda v: v[1])
        per_epoch[e] = {"records": len(vals), "median": med, "max": g_max, "max_step": s_max,
                        "max_over_median": g_max / med if med > 0 else None}
    ratios = [(v["max_over_median"], e) for e, v in per_epoch.items() if v["max_over_median"] is not None]
    worst = max(ratios) if ratios else (None, None)
    allv = [g for vals in by_epoch.values() for _, g in vals]
    whole_med = statistics.median(allv)
    return {"available": True, "per_epoch": per_epoch,
            "largest_within_epoch_max_over_median": worst[0], "at_epoch": worst[1],
            "secondary_whole_run": {"median": whole_med, "max": max(allv),
                                    "max_over_median": max(allv) / whole_med if whole_med > 0 else None},
            "note": "raw norms are not comparable between BN and non-BN arms; compare ratios within a run"}


def batchnorm_epoch10(history_row: dict) -> dict | None:
    """``ema_vs_inference`` at epoch 10: arm B is flat, arm D has one entry per level."""
    bn = history_row.get("batchnorm")
    if not bn:
        return None
    if "ema_vs_inference" in bn:
        return {"layer": bn["ema_vs_inference"]}
    return {level: v["ema_vs_inference"] for level, v in bn.items() if isinstance(v, dict) and "ema_vs_inference" in v}


def run_evidence(rec: dict, history: list[dict]) -> dict:
    run_dir = REPO / rec["run_dir"]
    dev = [num(h["dev_nll"]) for h in history]
    train = [num(h["train_nll"]) for h in history]
    running = [num(h["running_train_nll"]) for h in history]
    out = {"curves": {"dev_nll": dev, "train_nll": train, "running_train_nll": running,
                      "trainer_epoch_seconds": [num(h["seconds"]) for h in history]}}
    nonfinite = [i + 1 for i, v in enumerate(dev) if not finite(v)] + \
                [i + 1 for i, v in enumerate(train + running) if not finite(v)]
    out["numerical_status"] = rec["status"]
    if rec["status"] == "completed" and nonfinite:
        out["numerical_status"] = "mechanically_unstable"
        out["reclassified"] = f"recorded as completed, but non-finite NLL in epochs {sorted(set(nonfinite))}"
    if dev:
        best = min(range(len(dev)), key=lambda i: (dev[i], i))
        out["best_dev_nll"], out["best_epoch"] = dev[best], best + 1
    if len(dev) == MAX_EPOCHS:
        out["dev_nll_epoch10"], out["train_nll_epoch10"] = dev[-1], train[-1]
        out["bn_ema_vs_inference_epoch10"] = batchnorm_epoch10(history[-1])
    dev_up, run_up = rising_epochs(dev), rising_epochs(running)
    steps_path = run_dir / "diagnostics" / "steps.jsonl"
    out["observations"] = {
        "dev_nll_rose_in_epochs": dev_up,
        "running_train_nll_rose_in_epochs": run_up,
        "epoch10_minus_best_dev": (dev[-1] - out["best_dev_nll"]) if len(dev) == MAX_EPOCHS else None,
        "gradients": gradient_evidence(read_jsonl(steps_path) if steps_path.is_file() else []),
    }
    out["flags"] = [f for f, on in (
        ("dev_nll_rose", bool(dev_up)), ("repeated_dev_deterioration", len(dev_up) >= 2),
        ("running_train_nll_rose", bool(run_up)),
        ("epoch10_not_best", len(dev) == MAX_EPOCHS and dev[-1] > out["best_dev_nll"])) if on]
    timing = read_json(run_dir / "timing.json") if (run_dir / "timing.json").is_file() else {}
    result = read_json(ATTEMPTS_DIR / f"{rec['attempt_id']}.json")
    meta = read_json(run_dir / "metadata.json")
    epochs = [e for e in timing.get("epochs", []) if "error" not in e]

    def mean(key):
        vals = [e[key] for e in epochs if e.get(key) is not None]
        return statistics.fmean(vals) if vals else None

    out["compute"] = {
        "parameters": meta["pilot"]["parameters"],
        "training_tokens": meta["pilot"]["train_windows"] * len(history),
        "setup_seconds": timing.get("setup_seconds"), "training_call_seconds": timing.get("training_call_seconds"),
        "trainer_epoch_seconds_mean": statistics.fmean(out["curves"]["trainer_epoch_seconds"]) if history else None,
        **{f"{k}_mean": mean(k) for k in (
            "measured_sgd_phase_seconds", "measured_epoch_end_seconds", "estimated_step_diagnostic_overhead_seconds",
            "measured_tokens_per_second_including_diagnostics", "estimated_tokens_per_second_without_diagnostics",
            "idealized_steady_state_tokens_per_second")},
        "peak_process_rss_mb": result["peak_rss_bytes"] / 2**20 if "peak_rss_bytes" in result else None,
        "model_device": result.get("model_device"),
    }
    return out


# ================================================================== keys, candidates, arms
def compact(rec: dict) -> dict:
    keep = ("attempt_id", "status", "phase", "round", "retry_of", "retry_reason", "started_at", "ended_at",
            "run_dir", "failure", "returncode", "note")
    return {k: rec.get(k) for k in keep if rec.get(k) is not None}


def evaluate_key(recs: list[dict], frozen: dict | None) -> dict:
    """One (arm, lr, seed): every attempt, and the state of its final result."""
    out = {"attempts": [compact(r) for r in recs], "latest_status": recs[-1]["status"]}
    finals = [r for r in recs if r["status"] in FINAL_STATUSES]
    if len(finals) > 1:
        out.update(state="integrity_error", problems=["more than one final attempt"])
    elif not finals:
        out.update(state="unresolved" if recs[-1]["status"] != "launched" else "unfinished")
    elif finals[0] is not recs[-1]:
        out.update(state="integrity_error", problems=["an attempt was made after the final one"])
    elif frozen is None:
        out.update(state="integrity_error", problems=["no frozen conditions to validate against"])
    else:
        problems, history = validate_run(finals[0], frozen)
        out["final_attempt"] = finals[0]["attempt_id"]
        if problems:
            out.update(state="integrity_error", problems=problems)
        else:
            out.update(state="final", **run_evidence(finals[0], history))
    return out


def summarise_candidate(seed_runs: dict[int, dict | None], open_: bool) -> dict:
    states = {s: (r["state"] if r else "not_run") for s, r in seed_runs.items()}
    if any(v != "final" for v in states.values()):
        category = "incomplete"
    elif any(r["numerical_status"] == "mechanically_unstable" for r in seed_runs.values()):
        category = "mechanically_unstable"
    elif any(r["flags"] for r in seed_runs.values()):
        category = "flagged_for_review"
    else:
        category = "no_flags_raised"
    out = {"category": category, "seed_states": states}
    if open_ and category not in ("incomplete", "mechanically_unstable"):  # means only in selection mode
        e10 = [seed_runs[s]["dev_nll_epoch10"] for s in SEEDS]
        out["dev_nll_epoch10_mean"] = statistics.fmean(e10)
        out["dev_nll_epoch10_sample_sd"] = statistics.stdev(e10)
    return out


def select(arm: str, candidates: dict[str, dict], rounds_used: int) -> dict:
    """Provisional selection and the boundary decision (only called in selection mode)."""
    evaluated = sorted(float(lr) for lr in candidates)
    eligible = {lr: c for lr, c in candidates.items() if c["category"] != "mechanically_unstable"}
    if not eligible:
        return {"decision": "inconclusive", "detail": "every candidate is mechanically unstable; "
                "a revised grid must be proposed for approval"}

    def best(pool):
        return min(pool, key=lambda lr: (pool[lr]["dev_nll_epoch10_mean"], float(lr))) if pool else None

    chosen = best(eligible)
    unflagged = best({lr: c for lr, c in eligible.items() if c["category"] == "no_flags_raised"})
    out = {"provisional_lr": float(chosen), "provisional_category": eligible[chosen]["category"],
           "lowest_unflagged_lr": float(unflagged) if unflagged is not None else None,
           "evaluated_lrs": evaluated, "rounds_used": rounds_used}
    c = float(chosen)
    if evaluated[0] < c < evaluated[-1]:
        return {**out, "decision": "interior", "detail": "provisional selection is interior to the evaluated grid; "
                "flagged candidates must be reviewed before the selection is accepted"}
    direction = "up" if c == evaluated[-1] else "down"
    nxt = (next((v for v in EXTENSION_UP if v > evaluated[-1]), None) if direction == "up"
           else next((v for v in EXTENSION_DOWN if v < evaluated[0]), None))
    if rounds_used >= MAX_ROUNDS or nxt is None:
        return {**out, "decision": "inconclusive", "detail": f"still at the {direction} boundary after "
                f"{rounds_used} extension round(s); report all results and propose a revised experiment"}
    return {**out, "decision": "extension_required", "direction": direction, "next_lr": nxt,
            "next_round": rounds_used + 1,
            "command": f"python training/wavenet_study/lr_pilot.py run --threads <frozen> --arms {arm} "
                       f"--lrs {lr_tag(nxt)} --round {rounds_used + 1}",
            "detail": f"provisional selection {lr_tag(c)} is at the {direction} boundary of the evaluated grid"}


def evaluate_arm(arm: str, keys: dict[tuple, dict], recs_by_key: dict[tuple, list], blocking: list[str]) -> dict:
    # extension rounds recorded for this arm: exactly one LR per round, rounds 1..r in order (M2)
    rounds: dict[str, int] = {}
    for (a, lr, _s), recs in recs_by_key.items():
        if a == arm:
            rounds[lr] = max(rounds.get(lr, 0), max(int(r.get("round") or 0) for r in recs))
    ext = {lr: r for lr, r in rounds.items() if r >= 1}
    problems = []
    if sorted(ext.values()) != list(range(1, len(ext) + 1)) or len(ext) > MAX_ROUNDS:
        problems.append(f"extension rounds out of order or more than one LR per round: {ext}")
    if any(lr_tag(g) in ext for g in INITIAL_GRID) or any(float(lr) not in EXTENSION_UP + EXTENSION_DOWN
                                                         for lr in ext):
        problems.append(f"round labels do not match the protocol LRs: {ext}")
    required = [lr_tag(g) for g in INITIAL_GRID] + sorted(ext, key=lambda lr: ext[lr])
    seed_runs = {lr: {s: keys.get((arm, lr, s)) for s in SEEDS} for lr in required}
    complete = all(r is not None and r["state"] == "final" for runs in seed_runs.values() for r in runs.values())
    open_ = complete and not problems and not blocking
    candidates = {lr: summarise_candidate(seed_runs[lr], open_) for lr in required}
    if open_:
        selection = select(arm, candidates, len(ext))
    else:
        reasons = list(blocking) + problems
        missing = [f"lr {lr} seed {s}: {r['state'] if r else 'not run'}" for lr, runs in seed_runs.items()
                   for s, r in runs.items() if r is None or r["state"] != "final"]
        if missing:
            reasons.append(f"{len(missing)} of {len(required) * len(SEEDS)} required runs lack a final, "
                           f"valid result")
        selection = {"decision": "withheld", "mode": "status_only", "reasons": reasons, "missing": missing}
    return {"label": ARMS[arm]["label"], "rounds": rounds, "required_lrs": required, "candidates": candidates,
            "selection": selection, "problems": problems,
            "runs": {f"{lr}/{s}": r for lr, runs in seed_runs.items() for s, r in runs.items() if r is not None}}


def evaluate_pilot() -> dict:
    """The whole pilot state. Pure reading; the caller holds the pilot lock."""
    frozen = load_frozen()
    blocking: list[str] = []
    current = source_checksums()
    if frozen is None:
        blocking.append("no conditions.json: the pilot has not been started")
    else:
        drift = diff_paths(frozen["conditions"]["checksums"], current["files"])
        if drift:
            blocking.append(f"files changed since the freeze (amendment required): {drift}")
    by_id = attempts_by_id()
    unfinished = [aid for aid, r in by_id.items() if r["status"] == "launched"]
    if unfinished:
        blocking.append(f"attempts without a finished record (run `lr_pilot.py run` to register them): {unfinished}")
    orphans = unrecorded_run_dirs()
    if orphans:
        blocking.append(f"run directories without an attempt record: {orphans}")
    recs_by_key = attempts_by_key()
    keys = {k: evaluate_key(recs, frozen) for k, recs in recs_by_key.items()}
    unknown = sorted({k[0] for k in keys} - set(ARMS))
    if unknown:
        blocking.append(f"attempts for unknown arms {unknown}")
    arms = {arm: evaluate_arm(arm, keys, recs_by_key, blocking) for arm in ARMS}
    not_completed = [{"key": f"{a}/lr{lr}/s{s}", **att} for (a, lr, s), k in sorted(keys.items())
                     for att in k["attempts"] if att["status"] != "completed"]
    return {"protocol": frozen["conditions"]["protocol"] if frozen else None,
            "conditions_sha256": frozen["conditions_sha256"] if frozen else None,
            "summary_script_sha256": current["files"]["training/wavenet_study/summarize_lr_pilot.py"],
            "blocking": blocking, "attempts_not_completed": not_completed,
            "keys_with_multiple_attempts": {f"{a}/lr{lr}/s{s}": [x["attempt_id"] + ":" + x["status"]
                                                                 for x in k["attempts"]]
                                            for (a, lr, s), k in sorted(keys.items()) if len(k["attempts"]) > 1},
            "arms": arms}


# ================================================================== report
def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def markdown(s: dict) -> str:
    L = ["# WaveNet study — learning-rate pilot summary", "",
         (f"Protocol: wavenet_spec.md v0.4 §16.4. Seeds {list(SEEDS)}; {MAX_EPOCHS} epochs; constant LR; selection "
          "metric = mean full-dev NLL at epoch 10, computed only when an arm's required grid is complete and valid. "
          "Test data never used. Epochs are 1-based."),
         f"Frozen conditions: `{(s['conditions_sha256'] or 'none')[:16]}`; summary script `{s['summary_script_sha256'][:16]}`.",
         "", "## Blocking conditions", ""]
    L += [f"- {b}" for b in s["blocking"]] or ["- none"]
    L += ["", "## Attempts that did not complete (every one, including retried ones)", ""]
    if s["attempts_not_completed"]:
        L += ["| run | attempt | status | phase | failure | retry of / reason |", "|---|---|---|---|---|---|"]
        for a in s["attempts_not_completed"]:
            f = a.get("failure") or {}
            where = f"epoch {f.get('epoch')}, step {f.get('step')}: " if f.get("epoch") else ""
            L.append(f"| {a['key']} | {a['attempt_id']} | {a['status']} | {a.get('phase', '—')} | "
                     f"{f.get('type', '')} {where}{str(f.get('message', ''))[:160]} | "
                     f"{a.get('retry_of', '—')} {a.get('retry_reason', '')} |")
    else:
        L.append("none")
    if s["keys_with_multiple_attempts"]:
        L += ["", "Runs with several attempts: " + "; ".join(f"{k}: {' → '.join(v)}"
                                                             for k, v in s["keys_with_multiple_attempts"].items())]
    for arm, a in s["arms"].items():
        L += ["", f"## Arm {arm}: {a['label']} ({ARMS[arm]['params']:,} parameters)", "",
              ("| LR | seed | attempts | state | epoch-10 dev NLL | best dev (epoch) | flags | "
              "largest within-epoch grad max/median (epoch) | problems |"), "|---|---|---|---|---|---|---|---|---|"]
        for key, r in a["runs"].items():
            lr, seed = key.split("/")
            g = (r.get("observations") or {}).get("gradients") or {}
            grad = (f"{fmt(g.get('largest_within_epoch_max_over_median'), 1)} ({g.get('at_epoch')})"
                    if g.get("available") else "—")
            state = r["state"] + (f" ({r['numerical_status']})" if r.get("numerical_status") else "")
            L.append(f"| {lr} | {seed} | {len(r['attempts'])} | {state} | {fmt(r.get('dev_nll_epoch10'))} | "
                     f"{fmt(r.get('best_dev_nll'))} ({r.get('best_epoch', '—')}) | "
                     f"{','.join(r.get('flags', [])) or '—'} | {grad} | {'; '.join(r.get('problems', [])) or '—'} |")
        sel = a["selection"]
        if sel["decision"] == "withheld":
            L += ["", "**Selection withheld (status-only mode).** " + " ".join(f"{x}." for x in sel["reasons"])]
            continue
        L += ["", "| LR | category | mean ± sample sd of epoch-10 dev NLL |", "|---|---|---|"]
        for lr, c in sorted(a["candidates"].items(), key=lambda kv: float(kv[0])):
            L.append(f"| {lr} | {c['category']} | {fmt(c.get('dev_nll_epoch10_mean'))} ± "
                     f"{fmt(c.get('dev_nll_epoch10_sample_sd'))} |")
        L += ["", f"**Decision: {sel['decision']}.** {sel.get('detail', '')}"]
        if "provisional_lr" in sel:
            u = sel["lowest_unflagged_lr"]
            L.append(f"Provisional LR {lr_tag(sel['provisional_lr'])} ({sel['provisional_category']}); "
                     f"lowest unflagged LR: {lr_tag(u) if u is not None else '—'}.")
        if sel.get("command"):
            L.append(f"Next, after review: `{sel['command']}`")
    L += ["", "## Compute (means over final runs; operational, not efficiency evidence)", "",
          ("| arm | LR | params | training tokens | set-up s | trainer epoch s | measured SGD-phase s | est. diag "
          "overhead s | est. tok/s without diag | idealized tok/s | measured epoch-end s | process RSS MB |"),
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, a in s["arms"].items():
        by_lr = defaultdict(list)
        for key, r in a["runs"].items():
            if "compute" in r:
                by_lr[key.split("/")[0]].append(r["compute"])
        for lr, comp in sorted(by_lr.items(), key=lambda kv: float(kv[0])):
            def avg(k, comp=comp):
                v = [x[k] for x in comp if x.get(k) is not None]
                return statistics.fmean(v) if v else None
            L.append(f"| {arm} | {lr} | {comp[0]['parameters']:,} | {fmt(avg('training_tokens'), 0)} | "
                     f"{fmt(avg('setup_seconds'), 1)} | {fmt(avg('trainer_epoch_seconds_mean'), 2)} | "
                     f"{fmt(avg('measured_sgd_phase_seconds_mean'), 2)} | "
                     f"{fmt(avg('estimated_step_diagnostic_overhead_seconds_mean'), 3)} | "
                     f"{fmt(avg('estimated_tokens_per_second_without_diagnostics_mean'), 0)} | "
                     f"{fmt(avg('idealized_steady_state_tokens_per_second_mean'), 0)} | "
                     f"{fmt(avg('measured_epoch_end_seconds_mean'), 2)} | {fmt(avg('peak_process_rss_mb'), 0)} |")
    L += ["", ("Timing definitions are stored in every run's `timing.json`. Epoch-end time includes the trainers' "
          "arm-specific epoch-end diagnostics. Process RSS is dominated by the PyTorch import, not the model."),
          ("Gradient ratios are within-run (scale-free); raw gradient norms are not comparable between BatchNorm and "
          "non-BatchNorm arms. Flags are optimisation observations for review, not evidence of instability."), ""]
    return "\n".join(L)


def write_outputs(s: dict) -> None:
    (PILOT_ROOT / "summary.json").write_text(json.dumps(s, indent=2, sort_keys=True, default=str) + "\n")
    (PILOT_ROOT / "summary.md").write_text(markdown(s))
    with (PILOT_ROOT / "epochs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "lr", "seed", "attempt_id", "epoch", "running_train_nll", "train_nll", "dev_nll",
                    "trainer_epoch_seconds", "state"])
        for arm, a in s["arms"].items():
            for key, r in a["runs"].items():
                lr, seed = key.split("/")
                c = r.get("curves")
                for e, d in enumerate(c["dev_nll"] if c else [], 1):
                    w.writerow([arm, lr, seed, r.get("final_attempt"), e, c["running_train_nll"][e - 1],
                                c["train_nll"][e - 1], d, c["trainer_epoch_seconds"][e - 1], r["state"]])


def main() -> None:
    if not PILOT_ROOT.is_dir():
        raise SystemExit(f"no pilot results at {repo_relative(PILOT_ROOT)}")
    with pilot_lock():
        s = evaluate_pilot()
        write_outputs(s)
    print(f"wrote summary.json, summary.md, epochs.csv in {repo_relative(PILOT_ROOT)}")
    for b in s["blocking"]:
        print(f"  BLOCKING: {b}")
    if s["attempts_not_completed"]:
        print(f"  {len(s['attempts_not_completed'])} attempt(s) did not complete; see summary.md")
    for arm, a in s["arms"].items():
        sel = a["selection"]
        print(f"  arm {arm}: {sel['decision']}" + (f" — {sel.get('detail', '')}" if sel["decision"] != "withheld"
                                                   else f" ({len(sel['missing'])} required runs not final/valid)"))


if __name__ == "__main__":
    main()
