"""Loaders for the figures. They read only published result files (never checkpoints or local-only
records), so anyone with a clone of the repository can rebuild every figure.

Sources:
    results/experiments/wavenet_study/final/analysis/final/per_seed.csv       WaveNet study, per seed
    results/experiments/wavenet_study/final/analysis/final/context_test.csv   WaveNet study, context strata
    results/experiments/transformer_study/test_evaluation/result.json         Transformer study, per seed
    results/experiments/transformer_study/dev_diagnostics/position_by_index.json   position analysis
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "experiments"
WAVENET_PER_SEED = RESULTS / "wavenet_study/final/analysis/final/per_seed.csv"
WAVENET_CONTEXT = RESULTS / "wavenet_study/final/analysis/final/context_test.csv"
TRANSFORMER_RESULT = RESULTS / "transformer_study/test_evaluation/result.json"
POSITION_BY_INDEX = RESULTS / "transformer_study/dev_diagnostics/position_by_index.json"
SEEDS = tuple(range(10))

# short names used in the figures (full names in figures/README.md and the captions)
WAVENET_ARM = {"A": "MLP", "B": "MLP+BN", "C": "WaveNet", "D": "WaveNet+BN"}
TRANSFORMER_ARM = {"gpt2_style": "Transformer (GPT-2)", "transformer_2017": "Transformer (2017)",
                   "rnn": "RNN", "wavenet_d": "WaveNet+BN"}


@dataclass(frozen=True)
class Paired:
    """A paired difference over seeds: first minus second."""
    per_seed: np.ndarray
    level: float = 0.95

    @property
    def mean(self) -> float:
        return float(self.per_seed.mean())

    @property
    def interval(self) -> tuple[float, float]:
        d = self.per_seed
        se = d.std(ddof=1) / math.sqrt(len(d))
        q = stats.t.ppf(0.5 + self.level / 2, len(d) - 1)
        return self.mean - q * se, self.mean + q * se


def wavenet_test_nll() -> dict[str, dict[int, float]]:
    """Test NLL per seed of the WaveNet study's arms A–D (window protocol)."""
    out: dict[str, dict[int, float]] = {}
    for row in csv.DictReader(WAVENET_PER_SEED.open()):
        out.setdefault(row["arm"], {})[int(row["seed"])] = float(row["test_nll"])
    return out


def transformer_test_nll() -> dict[str, dict[int, float]]:
    """Test NLL per seed of the Transformer study's four arms (name-batch protocol)."""
    per_seed = json.loads(TRANSFORMER_RESULT.read_text())["per_seed"]
    return {arm: {int(s): v["test_nll"] for s, v in seeds.items()} for arm, seeds in per_seed.items()}


def paired(a: dict[int, float], b: dict[int, float], level: float = 0.95) -> Paired:
    return Paired(np.array([a[s] - b[s] for s in SEEDS]), level)


def wavenet_context_frequency() -> dict[tuple[str, str], dict[str, float]]:
    """WaveNet study, test set, by frequency of the 8-character context in training (exploratory)."""
    out = {}
    for row in csv.DictReader(WAVENET_CONTEXT.open()):
        if row["stratification"] == "frequency":
            out[(row["category"], row["comparison"])] = {
                "mean": float(row["mean_difference"]), "low": float(row["ci95_low"]),
                "high": float(row["ci95_high"]), "targets": int(row["targets"])}
    return out


def position_by_index(pair: str) -> dict[str, dict]:
    """Transformer study, dev set: the paired difference by position, name-clustered SEs."""
    return json.loads(POSITION_BY_INDEX.read_text())["pairs"][pair]
