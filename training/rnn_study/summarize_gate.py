"""Initialisation-gate summary: observed minus predicted per position, and the per-(Δ, s) decomposition,
each next to the across-seed spread, from ``L5_gate/initial_statistics.json`` (arch spec §7.2 item 7).
Read-only. No thresholds, no fitted slopes, no decision. Milestone MS8.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    table = {row["t"]: row for row in report["mean_field"]}
    spread_pos = {(r["t"], r["stratum"]): r for r in report["across_seed_spread"]["positions"]}
    spread_pair = {(r["delta"], r["start"]): r for r in report["across_seed_spread"]["pairs"]}
    seeds = report["per_seed"]
    positions = {}
    for seed in seeds:
        for p in seed["positions"]:
            positions.setdefault((p["t"], p["stratum"]), []).append(p)
    pos_rows = []
    for (t, s), ps in sorted(positions.items()):
        mean = lambda k: sum(p[k] for p in ps) / len(ps)  # noqa: E731
        pred = table[t]
        pos_rows.append({"t": t, "stratum": s, "seeds": len(ps),
                         "pre_var_minus_q": mean("pre_var") - pred["q"],
                         "saturation_minus_pred": mean("saturation") - pred["saturation"],
                         "mean_dtanh_minus_pred": mean("mean_dtanh") - pred["mean_dtanh"],
                         "spread": {k: spread_pos[(t, s)].get(k) for k in ("pre_var", "saturation", "mean_dtanh")}})
    pairs = {}
    for seed in seeds:
        for p in seed["pairs_by_delta_start"]:
            pairs.setdefault((p["delta"], p["start"]), []).append(p)
    pair_rows = []
    for (d, s), ps in sorted(pairs.items()):
        mean = lambda k: sum(p[k] for p in ps) / len(ps)  # noqa: E731
        pair_rows.append({"delta": d, "start": s, "obs_minus_mf": mean("obs_minus_mf"),
                          "obs_minus_hat": mean("obs_minus_hat"), "hat_minus_mf": mean("hat_minus_mf"),
                          "spread": {k: spread_pair[(d, s)].get(k) for k in ("obs_minus_mf", "obs_minus_hat", "hat_minus_mf")}})
    return {"seeds": report["seeds"], "fixed_values": report["fixed_values"], "positions": pos_rows,
            "pairs": pair_rows, "W_orthogonality_defect": [s["W_orthogonality_defect"] for s in seeds],
            "U_var": [s["U_var"] for s in seeds],
            "note": "Descriptive only (arch §7.2): any change of g or s² needs a separate recorded decision."}


def main(argv=None) -> int:
    import argparse

    from training.rnn_study.common import DEFAULT_STUDY_ROOT, install_git_guard

    install_git_guard()
    ap = argparse.ArgumentParser(description="Initialisation-gate summary")
    ap.add_argument("--root", default=str(DEFAULT_STUDY_ROOT))
    args = ap.parse_args(argv)
    report = json.loads((Path(args.root) / "L5_gate" / "initial_statistics.json").read_text())
    print(json.dumps(summarize(report), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
