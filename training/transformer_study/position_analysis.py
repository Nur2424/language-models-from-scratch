"""T-RQ1b with name-clustered standard errors (study plan §6, §7).

``dev_diagnostics.py`` stores dev NLL sums by position for each run, which gives seed-level standard
errors only. The plan asks for standard errors clustered by name, which need every target's NLL.
This script recomputes them from the same ``model_best.pt`` files, on the dev split only:

* per target, the paired difference of NLL between two arms, averaged over seeds 0–9;
* by group of position i (characters already read): i ≤ 7 (D's window holds the whole prefix and BOS),
  i = 8 (D no longer sees BOS), i ≥ 9 (D loses characters), and all targets;
* mean, name-clustered SE (the RNN study's ``_clustered``), seed SE, and the count of seeds in which
  the first arm is lower.

Usage (from the repository root)::

    python3 training/transformer_study/position_analysis.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import training.transformer_study.dev_diagnostics as dd  # noqa: E402
import training.transformer_study.transformer_runs as tr  # noqa: E402
import training.transformer_study.wavenet_d_arm as wd  # noqa: E402
from models.recurrent.rnn_diagnostics import _clustered  # noqa: E402
from utils.data import load_names, split_names  # noqa: E402

PAIRS = (("gpt2_style", "wavenet_d"), ("gpt2_style", "rnn"), ("gpt2_style", "transformer_2017"),
         ("rnn", "wavenet_d"), ("transformer_2017", "wavenet_d"))
OUT = dd.OUT / "position_paired.json"


def groups(positions: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"i<=7": positions <= 7, "i=8": positions == 8, "i>=9": positions >= 9,
            "all": torch.ones_like(positions, dtype=torch.bool)}


def paired_by_group(nll_a: torch.Tensor, nll_b: torch.Tensor, names: torch.Tensor,
                    positions: torch.Tensor) -> dict[str, dict[str, Any]]:
    """``nll_a``, ``nll_b``: [seeds, targets] float64; ``names``, ``positions``: [targets] int64."""
    delta = (nll_a - nll_b).mean(0)
    seeds = nll_a.shape[0]
    out = {}
    for label, sel in groups(positions).items():
        by_name: dict[int, list[float]] = {}
        for n, d in zip(names[sel].tolist(), delta[sel].tolist()):
            by_name.setdefault(n, []).append(d)
        c = _clustered(by_name)
        per_seed = [float((nll_a[s] - nll_b[s])[sel].mean()) for s in range(seeds)]
        m = math.fsum(per_seed) / seeds
        se_seed = math.sqrt(math.fsum((v - m) ** 2 for v in per_seed) / (seeds - 1) / seeds) if seeds > 1 else None
        out[label] = {"targets": c["targets"], "names": c["names"], "mean": c["mean"],
                      "se_name_clustered": c["se"], "se_seed": se_seed,
                      "seeds_first_lower": sum(v < 0 for v in per_seed)}
    return out


def per_target_nll(arm: str, seed: int, data: tr.SequenceData, dev_windows: wd.NameWindows) -> torch.Tensor:
    """Every dev target's NLL in row-major order (name, then position), float64."""
    model = dd.load_model(arm, seed, data.vocab)
    with torch.no_grad():
        if arm == "wavenet_d":
            w = dev_windows.windows
            return torch.cat([F.cross_entropy(model(w.contexts[s:s + 4096]), w.targets[s:s + 4096], reduction="none")
                              for s in range(0, dev_windows.num_targets, 4096)]).double()
        dev = data.dev
        full = torch.cat([F.cross_entropy(model(dev.inputs[s:s + 1024]).transpose(1, 2),
                                          dev.targets[s:s + 1024].clamp(min=0), reduction="none")
                          for s in range(0, dev.inputs.shape[0], 1024)])
        return full[dev.mask].double()


def main() -> int:
    wd.set_threads()
    data = tr.load_study_data()
    splits = split_names(load_names(), fractions=wd.SPLIT_FRACTIONS, seed=wd.SPLIT_SEED,
                         duplicate_policy=wd.DUPLICATE_POLICY)
    dev_windows = wd.build_name_windows(splits.dev, data.vocab)
    names, positions = data.dev.mask.nonzero(as_tuple=True)
    # the two dev representations must list the same targets in the same order
    assert torch.equal(dev_windows.windows.name_index, names)
    assert torch.equal(dev_windows.positions(), positions)
    assert torch.equal(dev_windows.windows.targets, data.dev.targets[names, positions])
    arms = sorted({a for pair in PAIRS for a in pair})
    nll = {a: torch.stack([per_target_nll(a, s, data, dev_windows) for s in dd.SEEDS]) for a in arms}
    out: dict[str, Any] = {"dev_nll": {a: float(nll[a].mean()) for a in arms}, "pairs": {}}
    for a, b in PAIRS:
        res = paired_by_group(nll[a], nll[b], names, positions)
        out["pairs"][f"{a} - {b}"] = res
        print(f"{a} - {b}")
        for label, r in res.items():
            print(f"  {label:5s} targets {r['targets']:5d}  diff {r['mean']:+.4f}  "
                  f"SE(names) {r['se_name_clustered']:.4f}  SE(seeds) {r['se_seed']:.4f}  "
                  f"first lower in {r['seeds_first_lower']}/10")
    wd._write_json(OUT, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
