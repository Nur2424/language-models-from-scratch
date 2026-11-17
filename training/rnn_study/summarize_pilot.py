"""Pilot summaries: the predeclared learning-rate rule (training spec §7.3) and the hybrid horizon rule
(§7.4), applied mechanically to recorded pilot histories. Read-only. The output states what the rules
give; the decision (and the validity-only human review, with the threshold-free pilot flags of §12.3)
is recorded separately. Milestone MS8.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EPSILON_FLOOR = 1e-3
#: Precedent boundary extensions (training spec §6): at most two rounds each way.
EXTENSION_DOWN = (0.01, 0.003)
EXTENSION_UP = (3.0, 10.0)


def _se(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan


PILOT_SEEDS = (100, 101, 102)


def select_learning_rate(outcomes: Mapping[float, Mapping[int, Mapping[str, Any]]], *, horizon: int = 30,
                         seeds: Sequence[int] = PILOT_SEEDS) -> dict[str, Any]:
    """``outcomes[η][seed] = {"status": …, "history": [rows]}``. Eligible: every seed ``completed``
    (Category A or D anywhere → ineligible). Primary: the eligible η with the lowest mean dev NLL at
    epoch ``horizon``. Boundary: if that η is the smallest or largest *evaluated* η, extension is
    indicated (an ineligible point counts as evaluated). Near-tie: after the grid is final, the
    smallest eligible η with D̄_η ≤ D̄_η̂ + max(1e-3, SE_η̂). Every η must hold a counted attempt for
    every seed in ``seeds`` (100 – 102); otherwise nothing is selected (the grid is not yet evaluated)."""
    grid = sorted(outcomes)
    missing = {eta: sorted(set(seeds) - set(outcomes[eta])) for eta in grid if set(outcomes[eta]) != set(seeds)}
    if missing:
        return {"evaluated": grid, "eligible": [], "selected": None, "primary": None,
                "inconclusive": f"not every η has exactly the seeds {list(seeds)}: missing or extra {missing}"}
    means, ses, eligible = {}, {}, []
    for eta in grid:
        seeds = outcomes[eta]
        if all(s["status"] == "completed" for s in seeds.values()):
            vals = [next(r["dev_nll"] for r in s["history"] if r["completed_epochs"] == horizon) for s in seeds.values()]
            means[eta], ses[eta] = statistics.fmean(vals), _se(vals)
            eligible.append(eta)
    if not eligible:
        return {"eligible": [], "selected": None, "inconclusive": "no eligible candidate"}
    best = min(eligible, key=lambda e: (means[e], e))
    boundary = None
    if best == grid[0]:
        boundary = "lower"
    elif best == grid[-1]:
        boundary = "upper"
    eps = max(EPSILON_FLOOR, ses[best]) if not math.isnan(ses[best]) else EPSILON_FLOOR
    near = min(e for e in eligible if means[e] <= means[best] + eps)
    nxt = None
    if boundary == "lower":
        nxt = next((e for e in EXTENSION_DOWN if e < grid[0]), None)
    elif boundary == "upper":
        nxt = next((e for e in EXTENSION_UP if e > grid[-1]), None)
    return {"evaluated": grid, "eligible": eligible, "mean_dev_nll": means, "se": ses, "primary": best,
            "at_boundary": boundary, "extension_indicated": nxt if boundary else None,
            "inconclusive": "still at a boundary after two extension rounds" if boundary and nxt is None else None,
            "near_tie_epsilon": eps, "selected_if_grid_final": near}


def plateau_epoch(histories: Sequence[Sequence[Mapping[str, Any]]], *, last: int) -> dict[str, Any]:
    """e* (``wavenet_spec.md`` §16.5 Steps 2 – 3 as adopted by training spec §7.4): D̄(e), SE(e) = SD/√n,
    ε(e) = max(1e‑3, SE(e)), e* = earliest e ≤ ``last`` with D̄(e) ≤ min D̄ + ε(e)."""
    by_epoch = {}
    for e in range(1, last + 1):
        vals = [next(r["dev_nll"] for r in h if r["completed_epochs"] == e) for h in histories]
        by_epoch[e] = (statistics.fmean(vals), _se(vals))
    best = min(m for m, _ in by_epoch.values())
    e_star = next(e for e, (m, se) in by_epoch.items() if m <= best + max(EPSILON_FLOOR, se if not math.isnan(se) else 0.0))
    eps_from_se = by_epoch[e_star][1] > EPSILON_FLOOR
    return {"e_star": e_star, "epsilon_from_se": eps_from_se, "requires_review": eps_from_se,
            "mean_dev_nll": {e: m for e, (m, _) in by_epoch.items()}}


def horizon_rule(e_star_30: int, e_star_60: int | None = None, x_failed: bool = False) -> dict[str, Any]:
    """§7.4: e*₃₀ < 30 → E = 30; else extension X; X failed → no budget; e*₆₀ < 60 → E = max(30, e*₆₀);
    e*₆₀ = 60 → inconclusive."""
    if e_star_30 < 30:
        return {"E": 30, "extension_needed": False}
    if x_failed:
        return {"E": None, "inconclusive": "extension run failed (Category A or D): human decision"}
    if e_star_60 is None:
        return {"E": None, "extension_needed": True}
    if e_star_60 < 60:
        return {"E": max(30, e_star_60), "extension_needed": False}
    return {"E": None, "inconclusive": "e*₆₀ = 60: human decision"}


def pilot_flags(history: Sequence[Mapping[str, Any]], *, horizon: int = 30) -> dict[str, Any]:
    """Threshold-free pilot flags (training spec §12.3)."""
    rows = sorted(history, key=lambda r: r["completed_epochs"])
    dev_rises = [r["completed_epochs"] for p, r in zip(rows, rows[1:]) if r["completed_epochs"] >= 2 and r["dev_nll"] > p["dev_nll"]]
    run_rises = [r["completed_epochs"] for p, r in zip(rows, rows[1:])
                 if p.get("running_train_nll") is not None and r["running_train_nll"] > p["running_train_nll"]]
    final = next((r for r in rows if r["completed_epochs"] == horizon), None)
    best = min((r["dev_nll"] for r in rows if r["completed_epochs"] >= 1), default=None)
    return {"dev_nll_rose": dev_rises, "dev_nll_rose_repeated": len(dev_rises) >= 2, "running_train_nll_rose": run_rises,
            "final_above_best": final is not None and best is not None and final["dev_nll"] > best}


def load_pilot_outcomes(root: str | Path) -> dict[float, dict[int, dict[str, Any]]]:
    """Counted pilot attempts with their histories, from the study manifest."""
    from training.rnn_study.attempts import AttemptStore
    from training.rnn_study.common import StudyPaths

    paths = StudyPaths(Path(root))
    store = AttemptStore(paths)
    out: dict[float, dict[int, dict[str, Any]]] = {}
    for rec in store.attempts().values():
        counted = store.counted(rec["key"]) if rec["step"] == "L6_pilot" else None
        if counted is None or counted["attempt_id"] != rec["attempt_id"]:
            continue  # only the one counted attempt per (η, seed)
        hist = [json.loads(x) for x in (paths.root / rec["run_dir"] / "history.jsonl").read_text().splitlines()]
        out.setdefault(float(rec["arm"]), {})[rec["seed"]] = {"status": rec["status"], "history": hist}
    return out


def main(argv=None) -> int:
    import argparse

    from training.rnn_study.common import DEFAULT_STUDY_ROOT, install_git_guard

    install_git_guard()
    ap = argparse.ArgumentParser(description="Pilot P summary (training spec §7.3, §7.4)")
    ap.add_argument("--root", default=str(DEFAULT_STUDY_ROOT))
    args = ap.parse_args(argv)
    outcomes = load_pilot_outcomes(args.root)
    summary = {"learning_rate": select_learning_rate(outcomes),
               "flags": {f"{eta}:{seed}": pilot_flags(s["history"]) for eta, seeds in outcomes.items()
                         for seed, s in seeds.items()}}
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
