"""Independent, read-only verification of Model 6 run records (plan §5.2): hash alignment across history,
checkpoints and diagnostic points; selection (strict <, earliest tie, k ≥ 1); step-record counts;
diagnostic schema and raw files; metadata status versus the training outcome. Returns findings; it
never repairs anything. Milestone MS8.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def verify_run(run_dir: str | Path) -> list[str]:
    from utils.checkpoint import load_checkpoint

    run_dir = Path(run_dir)
    findings: list[str] = []
    try:
        history = [json.loads(x) for x in (run_dir / "history.jsonl").read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"history unreadable: {exc}"]
    meta = json.loads((run_dir / "metadata.json").read_text())
    if [r["completed_epochs"] for r in history] != list(range(len(history))):
        findings.append("history epochs are not 0 … K")
    sel = [r for r in history if r["completed_epochs"] >= 1]
    if sel:
        best = sel[0]
        for r in sel[1:]:
            if r["dev_nll"] < best["dev_nll"]:
                best = r
        ck = run_dir / "model_best.pt"
        if not ck.exists():
            findings.append("model_best.pt missing")
        else:
            c = load_checkpoint(ck)
            if c["epoch"] != best["completed_epochs"]:
                findings.append(f"best checkpoint epoch {c['epoch']} != selected {best['completed_epochs']}")
            if c["extra"]["param_sha256"] != best["param_sha256"]:
                findings.append("best checkpoint hash differs from its history row")
    for kind, name in (("resume", "checkpoint_last.pt"), ("final", "model_final.pt")):
        p = run_dir / name
        if p.exists():
            c = load_checkpoint(p)
            row = next((r for r in history if r["completed_epochs"] == c["epoch"]), None)
            if row is None or row["param_sha256"] != c["extra"]["param_sha256"]:
                findings.append(f"{name} hash does not match history epoch {c['epoch']}")
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    steps_path = run_dir / "diagnostics" / "steps.jsonl"
    if steps_path.exists() and sel:
        steps = [json.loads(x) for x in steps_path.read_text().splitlines()]
        if [s["step"] for s in steps] != list(range(1, len(steps) + 1)):
            findings.append("step records are not 1 … σ")
        per_epoch = {}
        for s in steps:
            per_epoch[s["epoch"]] = per_epoch.get(s["epoch"], 0) + 1
        if len(set(per_epoch.values())) > 1:
            findings.append(f"unequal steps per epoch: {per_epoch}")
        if not all(math.isfinite(s["grad_norm"]) for s in steps):
            findings.append("non-finite gradient norm recorded")
    outcome = meta.get("outcome")
    if meta.get("status") == "completed" and outcome not in ("completed", "failed_diverged"):
        findings.append(f"completed run with outcome {outcome!r}")
    if outcome == "failed_diverged" and not (history and history[-1].get("diverged")):
        findings.append("failed_diverged without a diverged history row")
    diag = run_dir / "research_diagnostics"
    if diag.exists():
        pts = [json.loads(x) for x in (diag / "points.jsonl").read_text().splitlines()]
        if {p["schema"] for p in pts} - {"m6_diagnostics/3"}:
            findings.append("diagnostic records with another schema")
        for p in pts:
            if not (diag / "raw" / f"point_{p['point_id']:04d}.npz").exists():
                findings.append(f"raw file of point {p['point_id']} missing")
            if p["kind"] == "epoch_end":
                row = next((r for r in history if r["completed_epochs"] == p["completed_epochs"]), None)
                if row is None or row["param_sha256"] != p["param_sha256_before"]:
                    findings.append(f"point {p['point_id']} hash does not match history epoch {p['completed_epochs']}")
            if p["param_sha256_before"] != p["param_sha256_after"]:
                findings.append(f"point {p['point_id']} changed the parameters")
        invalid = [p["point_id"] for p in pts if not p["valid"]]
        if invalid and meta.get("diagnostics_status") != "incomplete":
            findings.append(f"invalid points {invalid} but diagnostics_status {meta.get('diagnostics_status')!r}")
        if sorted(meta.get("invalid_diagnostic_points") or []) != sorted(invalid):
            findings.append(f"metadata invalid_diagnostic_points {meta.get('invalid_diagnostic_points')!r} "
                            f"!= invalid points recorded {invalid}")
        o3 = diag / "o3.json"
        if o3.exists() and sorted(json.loads(o3.read_text()).get("invalid_points", [])) != sorted(invalid):
            findings.append("o3.json invalid_points differ from the invalid points recorded")
    del cfg
    return findings


def verify_study(root: str | Path) -> dict[str, Any]:
    from training.rnn_study.attempts import AttemptStore
    from training.rnn_study.common import StudyPaths

    paths = StudyPaths(Path(root))
    store = AttemptStore(paths)
    report = {}
    for aid, rec in store.attempts().items():
        run_dir = paths.root / rec["run_dir"]
        findings = verify_run(run_dir) if (run_dir / "history.jsonl").exists() else ["no history"]
        meta = json.loads((run_dir / "metadata.json").read_text()) if (run_dir / "metadata.json").exists() else {}
        if rec["status"] in ("completed", "failed_diverged", "failed_mechanical") and meta.get("outcome") != rec["status"]:
            findings.append(f"manifest status {rec['status']} != metadata outcome {meta.get('outcome')!r}")
        report[aid] = findings
    return report


def main(argv=None) -> int:
    import argparse

    from training.rnn_study.common import DEFAULT_STUDY_ROOT, install_git_guard

    install_git_guard()
    ap = argparse.ArgumentParser(description="Verify Model 6 study run records (read-only)")
    ap.add_argument("--root", default=str(DEFAULT_STUDY_ROOT))
    args = ap.parse_args(argv)
    report = verify_study(args.root)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 1 if any(report.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
