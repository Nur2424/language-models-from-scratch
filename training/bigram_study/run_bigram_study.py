"""Bigram study: the statistical and the neural bigram on the standard split.

Two deterministic models condition on the previous character only:

* the statistical bigram (``models/statistical/bigram.py``), estimated in closed
  form from transition counts, unsmoothed (alpha = 0) or with additive smoothing;
* the neural bigram (``models/feedforward/neural_bigram.py``), a zero-initialised
  27 x 27 logit table trained by full-batch gradient descent.

What this script measures
-------------------------
1. Statistical bigram, for each alpha in ALPHAS: train and dev NLL, and the number
   of dev transitions the model gives probability 0 (which makes the dev NLL +inf).
2. Neural bigram, at geometric step counts up to MAX_STEPS (lr = 25, no weight decay):
   train and dev NLL, and how far it is from the counting solution:
   - excess train NLL over the unsmoothed counting model (the training-loss infimum);
   - the largest |p - q| over observed transitions (q = counting probabilities);
   - the probability mass still placed on transitions never seen in training,
     weighted by how often each context occurs;
   - P(EOS | BOS), a transition that never occurs, which should shrink like 1/t.
   This is a reduced version of experiment E1 in ``neural_bigram_spec.md`` section 13.4.
3. Ten samples from each model with a fixed generator.

Nothing is selected or tuned on the results. The test split is never used.
Both models are deterministic, so there are no seeds to repeat.

Output: ``results/experiments/bigram_study/results.json`` (overwritten on each run).

Run from the repository root:

    python3 training/bigram_study/run_bigram_study.py
    python3 training/bigram_study/run_bigram_study.py --max-steps 1000   # quick check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from models.feedforward.neural_bigram import (  # noqa: E402
    NeuralBigramLanguageModel,
    make_bigram_pairs,
    train_full_batch,
)
from models.statistical.bigram import BigramConfig, BigramLanguageModel  # noqa: E402
from utils.config import default_config_paths, load_config  # noqa: E402
from utils.data import check_vocab_coverage, dataset_fingerprint, load_names, split_names  # noqa: E402
from utils.io import atomic_write_json  # noqa: E402
from utils.metrics import NLLAccumulator  # noqa: E402
from utils.reproducibility import derive_seed, make_generator  # noqa: E402
from utils.run_manager import environment_info, utc_now  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

ALPHAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
LR = 25.0  # inside the sufficient step-size bound eta < 26.9 (neural_bigram_spec.md section 4.5)
CHECKPOINTS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)
NUM_SAMPLES, MAX_LENGTH, SAMPLE_SEED = 10, 30, 0
OUT_PATH = REPO_ROOT / "results" / "experiments" / "bigram_study" / "results.json"


def load_data():
    """Standard split from configs/base.yaml; vocabulary from the training names only."""
    cfg = load_config(default_config_paths("bigram"), model_config_cls=BigramConfig)
    names = load_names(cfg.data.path)
    splits = split_names(names, fractions=cfg.data.split_fractions, seed=cfg.data.split_seed,
                         duplicate_policy=cfg.data.duplicate_policy)
    splits.assert_disjoint()
    vocab = Vocabulary.from_names(splits.train)
    check_vocab_coverage(vocab, splits)
    return cfg, splits, vocab


def samples(model) -> list[dict]:
    """NUM_SAMPLES names from a fresh generator with a fixed seed."""
    g = make_generator(derive_seed(SAMPLE_SEED, "generation"))
    out = model.generate(NUM_SAMPLES, max_length=MAX_LENGTH, generator=g)
    return [{"text": s.text, "finished": s.finished} for s in out]


def statistical_bigram(splits, vocab) -> tuple[list[dict], BigramLanguageModel]:
    rows = []
    mle = None
    for alpha in ALPHAS:
        model = BigramLanguageModel(vocab, alpha=alpha).fit(splits.train)
        train, dev = model.evaluate(splits.train), model.evaluate(splits.dev)
        rows.append({
            "alpha": alpha,
            "train_nll": train.mean_nll,
            "dev_nll": dev.mean_nll,  # "inf" in the JSON when a dev transition has probability 0
            "dev_zero_probability_events": dev.zero_probability_events,
        })
        print(f"  alpha {alpha:>5}: train {train.mean_nll:.4f}  dev {dev.mean_nll:.4f}  "
              f"(zero-probability dev events: {dev.zero_probability_events})")
        if alpha == 0.0:
            mle = model
    return rows, mle


def neural_bigram(splits, vocab, mle: BigramLanguageModel, max_steps: int) -> tuple[list[dict], NeuralBigramLanguageModel]:
    x, y = make_bigram_pairs(splits.train, vocab)
    xd, yd = make_bigram_pairs(splits.dev, vocab)

    counts = mle.counts().cpu().to(torch.float64)            # C [V, V]
    q = mle.probabilities().cpu().to(torch.float64)          # counting probabilities
    context_weight = counts.sum(1) / counts.sum()            # N_i / M
    observed = counts > 0
    unseen = (~observed) & (counts.sum(1, keepdim=True) > 0)  # unseen target in an observed context
    mle_train_nll = mle.evaluate(splits.train).mean_nll

    model = NeuralBigramLanguageModel(vocab)

    def nll(inputs, targets) -> float:
        acc = NLLAccumulator()
        with torch.no_grad():
            acc.update(model(inputs), targets)
        return acc.mean_nll

    rows, done = [], 0
    t0 = time.perf_counter()
    for step in (c for c in CHECKPOINTS if c <= max_steps):
        if step > done:
            train_full_batch(model, x, y, lr=LR, steps=step - done)
            done = step
        with torch.no_grad():
            p = torch.softmax(model.transition_logits.detach().to(torch.float64), dim=1)
        train_nll = nll(x, y)
        row = {
            "step": step,
            "train_nll": train_nll,
            "dev_nll": nll(xd, yd),
            "excess_train_nll": train_nll - mle_train_nll,
            "max_abs_p_minus_q_observed": float((p - q)[observed].abs().max()),
            "unseen_target_mass": float((context_weight[:, None] * p * unseen).sum()),
            "p_eos_given_bos": float(p[0, 0]),
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(f"  step {step:>7}: train {row['train_nll']:.4f}  dev {row['dev_nll']:.4f}  "
              f"excess {row['excess_train_nll']:.2e}  unseen mass {row['unseen_target_mass']:.2e}  "
              f"({row['seconds']:.0f} s)")
    return rows, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=CHECKPOINTS[-1],
                        help=f"last neural-bigram step count to reach (default {CHECKPOINTS[-1]})")
    args = parser.parse_args()

    cfg, splits, vocab = load_data()
    print(f"Split sizes (names): {splits.sizes()}; test split not used.")

    print("Statistical bigram")
    stat_rows, mle = statistical_bigram(splits, vocab)
    print(f"Neural bigram (full-batch gradient descent, lr = {LR})")
    neural_rows, neural = neural_bigram(splits, vocab, mle, args.max_steps)

    finite = [r for r in stat_rows if r["dev_zero_probability_events"] == 0]
    result = {
        "description": "Statistical and neural bigram on the standard split; descriptive, nothing selected; "
                       "test split not used.",
        "created_at": utc_now(),
        "script": "training/bigram_study/run_bigram_study.py",
        "environment": {**environment_info(), "torch_num_threads": torch.get_num_threads()},
        "dataset": dataset_fingerprint(cfg.data.path),
        "split": splits.fingerprint(),
        "vocabulary_size": vocab.num_classes,
        "statistical_bigram": {
            "alphas": stat_rows,
            "lowest_dev_nll_alpha": min(finite, key=lambda r: r["dev_nll"])["alpha"],
        },
        "neural_bigram": {
            "lr": LR,
            "weight_decay": 0.0,
            "initialisation": "zeros",
            "counting_model_train_nll": mle.evaluate(splits.train).mean_nll,
            "checkpoints": neural_rows,
            "lowest_dev_nll_step": min(neural_rows, key=lambda r: r["dev_nll"])["step"],
        },
        "samples": {
            "generator_seed": f"derive_seed({SAMPLE_SEED}, 'generation')",
            "statistical_bigram_alpha_0": samples(mle),
            "neural_bigram_final_step": samples(neural),
        },
    }
    atomic_write_json(OUT_PATH, result)
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
