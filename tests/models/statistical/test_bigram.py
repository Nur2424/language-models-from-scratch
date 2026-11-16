"""Tests for models/statistical/bigram.py.

Hand-computed reference used throughout ("tiny" fixture)
--------------------------------------------------------
names = ["a", "ab"], vocabulary chars (a, b) → ids  . = 0, a = 1, b = 2, V = 3

transitions:  "a"  → . → a → .            "ab" → . → a → b → .

counts C (row = previous, column = next):      MLE  P = C / N             alpha = 1: (C + 1) / (N + 3)
         .  a  b    N                                .    a    b                .     a     b
    .  [ 0, 2, 0 ]  2                           .  [ 0,   1,   0  ]        . [ 1/5,  3/5,  1/5 ]
    a  [ 1, 0, 1 ]  2                           a  [ 1/2, 0,   1/2]        a [ 2/5,  1/5,  2/5 ]
    b  [ 1, 0, 0 ]  1                           b  [ 1,   0,   0  ]        b [ 2/4,  1/4,  1/4 ]

Training NLL (MLE): -(log 1 + log ½ + log 1 + log ½ + log 1) = 2 ln 2 over M = 5 tokens.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from models.statistical.bigram import (
    BigramConfig,
    BigramEvaluation,
    BigramLanguageModel,
    GeneratedName,
    IncompatibleStateError,
    NotFittedError,
    UndefinedDistributionError,
    check_fitted_counts,
    check_flow_conservation,
    count_transitions,
    unreachable_from_bos,
)
from utils.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from utils.config import ConfigError, default_config_paths, load_config
from utils.data import load_names, split_names
from utils.device import is_mps_available
from utils.generation import next_token_probs
from utils.metrics import token_nll
from utils.paths import DEFAULT_NAMES_PATH
from utils.reproducibility import make_generator
from utils.vocabulary import Vocabulary, VocabularyError

LN2 = math.log(2.0)

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    pytest.param("mps", id="mps", marks=pytest.mark.skipif(not is_mps_available(), reason="MPS unavailable")),
]


@pytest.fixture
def tiny_names() -> list[str]:
    return ["a", "ab"]


@pytest.fixture
def tiny_vocab(tiny_names) -> Vocabulary:
    return Vocabulary.from_names(tiny_names)


@pytest.fixture
def tiny_mle(tiny_vocab, tiny_names) -> BigramLanguageModel:
    return BigramLanguageModel(tiny_vocab).fit(tiny_names)


@pytest.fixture
def small_names() -> list[str]:
    return ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "liam", "noah", "oliver",
            "elijah", "lucas", "mason", "logan", "anna", "anna", "eva"]


# ============================================================ 8.1 construction
class TestConstruction:
    def test_valid_construction_and_defaults(self, tiny_vocab):
        m = BigramLanguageModel(tiny_vocab)
        assert m.alpha == 0.0 and m.num_classes == 3 and m.device == torch.device("cpu")
        assert not m.is_fitted and "not fitted" in repr(m)
        assert BigramLanguageModel(tiny_vocab, alpha=1).alpha == 1.0          # int accepted, stored as float
        assert BigramLanguageModel.from_config(tiny_vocab, BigramConfig(alpha=0.5)).alpha == 0.5

    @pytest.mark.parametrize("alpha", [-1e-9, -1.0, float("inf"), float("nan"), True, "0.1", None])
    def test_invalid_alpha(self, tiny_vocab, alpha):
        with pytest.raises(ValueError, match="alpha"):
            BigramLanguageModel(tiny_vocab, alpha=alpha)
        with pytest.raises(ValueError, match="alpha"):
            BigramConfig(alpha=alpha)

    def test_invalid_vocabulary_and_device(self, tiny_vocab):
        with pytest.raises(TypeError, match="Vocabulary"):
            BigramLanguageModel({".": 0, "a": 1})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="BigramConfig"):
            BigramLanguageModel.from_config(tiny_vocab, {"alpha": 0.0})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="invalid device"):
            BigramLanguageModel(tiny_vocab, device="gpu")

    def test_operations_before_fit_fail_clearly(self, tiny_vocab, tmp_path):
        m = BigramLanguageModel(tiny_vocab)
        calls = [
            m.counts, m.probabilities, m.log_probabilities, m.defined_contexts, m.state_dict,
            lambda: m.transition_probability(".", "a"),
            lambda: m.evaluate(["a"]),
            lambda: m.sample_next(torch.tensor([0])),
            lambda: m.generate(1, max_length=5),
            lambda: m.save(tmp_path / "m.pt"),
        ]
        for call in calls:
            with pytest.raises(NotFittedError, match="not fitted"):
                call()

    @pytest.mark.parametrize(
        "names, exc, match",
        [
            ([], ValueError, "empty"),
            ("anna", TypeError, "single string"),
            (["a", ""], ValueError, "empty name"),
            (["a", 3], TypeError, "must be a string"),
            (["az"], VocabularyError, "unknown character 'z'"),
            (["a.b"], VocabularyError, "boundary"),
        ],
    )
    def test_fit_rejects_invalid_data(self, tiny_vocab, names, exc, match):
        with pytest.raises(exc, match=match):
            BigramLanguageModel(tiny_vocab).fit(names)

    def test_config_file_integration(self):
        cfg = load_config(default_config_paths("bigram"), model_config_cls=BigramConfig)
        assert cfg.model.name == "bigram" and cfg.model.params == BigramConfig(alpha=0.0)
        cfg = load_config(default_config_paths("bigram"), overrides=["model.params.alpha=0.5"],
                          model_config_cls=BigramConfig)
        assert cfg.model.params.alpha == 0.5
        with pytest.raises(ConfigError, match="alpha must be >= 0"):
            load_config(default_config_paths("bigram"), overrides=["model.params.alpha=-1"], model_config_cls=BigramConfig)
        with pytest.raises(ConfigError, match="did you mean 'alpha'"):
            load_config(default_config_paths("bigram"), overrides=["model.params.alpah=1"], model_config_cls=BigramConfig)


# ========================================================= 8.2 transition counts
class TestCounts:
    def test_hand_computed_counts(self, tiny_mle):
        expected = torch.tensor([[0, 2, 0],
                                 [1, 0, 1],
                                 [1, 0, 0]])
        c = tiny_mle.counts()
        assert c.dtype == torch.int64
        assert torch.equal(c, expected)          # the four named entries AND every other entry is zero
        assert int(c.sum()) == sum(len(n) + 1 for n in ["a", "ab"])

    def test_anna_sequence(self):
        v = Vocabulary.from_names(["anna"])     # . = 0, a = 1, n = 2
        c = count_transitions(["anna"], v)
        # . → a → n → n → a → .
        expected = torch.zeros(3, 3, dtype=torch.long)
        for i, j in [(0, 1), (1, 2), (2, 2), (2, 1), (1, 0)]:
            expected[i, j] += 1
        assert torch.equal(c, expected)

    def test_counts_use_shared_vocabulary_ids(self, small_names):
        v = Vocabulary.from_names(small_names)
        c = count_transitions(small_names, v)
        stoi = v.stoi
        manual = sum(1 for n in small_names for p, q in zip("." + n, n + ".") if (p, q) == ("a", "."))
        assert int(c[stoi["a"], stoi["."]]) == manual
        assert int(c[0].sum()) == len(small_names)          # one BOS transition per name (duplicates count)

    def test_deterministic_and_order_independent(self, small_names):
        v = Vocabulary.from_names(small_names)
        shuffled = list(small_names)
        random.Random(0).shuffle(shuffled)
        assert torch.equal(count_transitions(small_names, v), count_transitions(shuffled, v))

    def test_flow_conservation(self, small_names):
        v = Vocabulary.from_names(small_names)
        c = count_transitions(small_names, v)
        check_flow_conservation(c)
        broken = c.clone()
        broken[1, 2] += 1
        with pytest.raises(ValueError, match="flow conservation"):
            check_flow_conservation(broken)

    def test_flow_conservation_is_necessary_but_not_sufficient(self):
        v = Vocabulary(chars=("a", "b", "c", "d"))          # . = 0, a = 1, b = 2, c = 3, d = 4
        c = count_transitions(["ab"], v)
        c[3, 4] += 1                                         # c → d → c: balanced, but detached from BOS
        c[4, 3] += 1
        check_flow_conservation(c)                           # passes ...
        assert unreachable_from_bos(c) == [3, 4]
        with pytest.raises(ValueError, match=r"token id\(s\) \[3, 4\].*not reachable from BOS"):
            check_fitted_counts(c)                           # ... but the full check rejects it

    def test_fitted_counts_always_pass_full_check(self):
        rng = random.Random(0)
        v = Vocabulary(chars=tuple("abcdef"))                # includes characters some draws never use
        for _ in range(200):
            names = ["".join(rng.choice("abcd") for _ in range(rng.randint(1, 6))) for _ in range(rng.randint(1, 8))]
            check_fitted_counts(count_transitions(names, v))

    def test_full_check_rejects_empty_name_and_empty_matrix(self, tiny_vocab, tiny_names):
        c = count_transitions(tiny_names, tiny_vocab)
        c[0, 0] += 1                                         # BOS → EOS: row 0 and column 0 both +1
        check_flow_conservation(c)
        with pytest.raises(ValueError, match="BOS → EOS"):
            check_fitted_counts(c)
        with pytest.raises(ValueError, match="no BOS transitions"):
            check_fitted_counts(torch.zeros(3, 3, dtype=torch.long))

    def test_refit_replaces_state(self, tiny_vocab):
        m = BigramLanguageModel(tiny_vocab).fit(["a"])
        m.fit(["ab"])
        assert int(m.counts()[0].sum()) == 1 and int(m.counts()[1, 2]) == 1

    def test_accessors_return_copies(self, tiny_mle):
        c = tiny_mle.counts()
        c.zero_()
        p = tiny_mle.probabilities()
        p.fill_(7)
        assert int(tiny_mle.counts().sum()) == 5 and tiny_mle.transition_probability(".", "a") == 1.0


# ================================================================ 8.3 MLE
class TestMaximumLikelihood:
    def test_hand_computed_probabilities(self, tiny_mle):
        expected = torch.tensor([[0.0, 1.0, 0.0],
                                 [0.5, 0.0, 0.5],
                                 [1.0, 0.0, 0.0]])
        assert torch.allclose(tiny_mle.probabilities(), expected, atol=0, rtol=0)
        assert tiny_mle.transition_probability(".", "a") == 1.0
        assert tiny_mle.transition_probability("a", ".") == 0.5
        assert tiny_mle.transition_probability("a", "b") == 0.5
        assert tiny_mle.transition_probability("b", ".") == 1.0
        assert tiny_mle.transition_probability("b", "b") == 0.0

    def test_zero_probabilities_are_exact_and_log_is_neg_inf(self, tiny_mle):
        p, lp = tiny_mle.probabilities(), tiny_mle.log_probabilities()
        zero = p == 0
        assert int(zero.sum()) == 5                                   # no hidden smoothing
        assert torch.isneginf(lp[zero]).all() and torch.isfinite(lp[~zero]).all()
        assert not torch.isnan(lp).any()
        assert torch.allclose(lp[~zero].exp(), p[~zero])
        assert p.dtype == lp.dtype == torch.float32

    def test_rows_sum_to_one(self, small_names):
        m = BigramLanguageModel(Vocabulary.from_names(small_names)).fit(small_names)
        assert torch.allclose(m.probabilities().sum(1), torch.ones(m.num_classes), atol=1e-6)
        assert m.defined_contexts().all()

    def test_invalid_transition_symbol(self, tiny_mle):
        with pytest.raises(VocabularyError, match="not a vocabulary token"):
            tiny_mle.transition_probability("a", "z")


# ============================================================ 8.4 smoothing
class TestSmoothing:
    def test_hand_computed_alpha_one(self, tiny_vocab, tiny_names):
        m = BigramLanguageModel(tiny_vocab, alpha=1.0).fit(tiny_names)
        expected = torch.tensor([[1 / 5, 3 / 5, 1 / 5],
                                 [2 / 5, 1 / 5, 2 / 5],
                                 [2 / 4, 1 / 4, 1 / 4]])
        assert torch.allclose(m.probabilities(), expected, atol=1e-7)
        assert torch.allclose(m.probabilities().sum(1), torch.ones(3), atol=1e-6)
        assert torch.equal(m.counts(), BigramLanguageModel(tiny_vocab).fit(tiny_names).counts())  # counts unsmoothed

    @pytest.mark.parametrize("alpha", [0.01, 0.5, 3.0])
    def test_formula_matches_definition(self, small_names, alpha):
        v = Vocabulary.from_names(small_names)
        m = BigramLanguageModel(v, alpha=alpha).fit(small_names)
        C = m.counts().double()
        N = C.sum(1, keepdim=True)
        reference = (C + alpha) / (N + alpha * v.num_classes)
        assert torch.allclose(m.probabilities().double(), reference, atol=1e-7)
        assert (m.probabilities() > 0).all()
        assert torch.allclose(m.probabilities().sum(1), torch.ones(v.num_classes), atol=1e-5)

    def test_row_without_observed_transitions(self):
        v = Vocabulary(chars=("a", "b", "c"))      # 'c' never appears in training
        mle = BigramLanguageModel(v).fit(["ab"])
        p = mle.probabilities()
        c_id = v.stoi["c"]
        assert torch.equal(p[c_id], torch.zeros(4))                  # undefined row: no fabricated mass
        assert mle.defined_contexts().tolist() == [True, True, True, False]
        defined = mle.defined_contexts()
        assert torch.allclose(p[defined].sum(1), torch.ones(3))
        smooth = BigramLanguageModel(v, alpha=2.0).fit(["ab"])
        # N_c = 0 ⇒ P_α(j|c) = α / (α V) = 1/V: smoothing turns an empty row into the uniform distribution
        assert torch.allclose(smooth.probabilities()[c_id], torch.full((4,), 0.25))
        assert smooth.defined_contexts().all()

    def test_alpha_that_underflows_float32_is_rejected(self, tiny_vocab, tiny_names):
        with pytest.raises(ValueError, match="too small to represent in float32"):
            BigramLanguageModel(tiny_vocab, alpha=1e-46).fit(tiny_names)


# ============================================================== 8.5 likelihood
class TestLikelihood:
    def test_training_nll_hand_computed(self, tiny_mle, tiny_names):
        r = tiny_mle.evaluate(tiny_names)
        assert isinstance(r, BigramEvaluation)
        assert r.num_tokens == 5 and r.num_names == 2 and r.zero_probability_events == 0 and r.is_finite
        assert r.total_nll == pytest.approx(2 * LN2, rel=1e-6)  # float32 log-probabilities
        assert r.mean_nll == pytest.approx(2 * LN2 / 5, rel=1e-6)  # float32 log-probabilities
        assert r.perplexity == pytest.approx(2 ** (2 / 5), rel=1e-6)  # float32 log-probabilities
        assert r.bits_per_token == pytest.approx(2 / 5, rel=1e-6)  # float32 log-probabilities

    def test_token_weighted_not_mean_of_names(self, tiny_mle):
        r = tiny_mle.evaluate(["a", "ab"])
        per_name = [tiny_mle.evaluate([n]) for n in ["a", "ab"]]
        assert per_name[0].mean_nll == pytest.approx(LN2 / 2)       # 2 tokens
        assert per_name[1].mean_nll == pytest.approx(LN2 / 3)       # 3 tokens
        mean_of_means = (LN2 / 2 + LN2 / 3) / 2                       # 0.2888 — the wrong aggregate
        assert r.mean_nll == pytest.approx(2 * LN2 / 5)               # 0.2773 — the right one
        assert abs(r.mean_nll - mean_of_means) > 0.01
        assert r.total_nll == pytest.approx(sum(x.total_nll for x in per_name))
        assert r.num_tokens == sum(x.num_tokens for x in per_name)

    def test_eos_transition_is_included(self, tiny_mle):
        r = tiny_mle.evaluate(["a"])
        # . → a has P = 1 (contributes 0); a → . has P = ½. Without EOS the NLL would be 0 over 1 token.
        assert r.num_tokens == 2
        assert r.total_nll == pytest.approx(LN2)

    def test_zero_probability_events_give_inf_not_nan(self, tiny_mle):
        r = tiny_mle.evaluate(["b"])            # . → b unseen (P = 0), b → . has P = 1
        assert r.zero_probability_events == 1 and r.num_tokens == 2 and not r.is_finite
        assert r.total_nll == math.inf and r.mean_nll == math.inf and r.perplexity == math.inf
        assert not math.isnan(r.total_nll)
        r2 = tiny_mle.evaluate(["bb", "a"])     # . → b (0), b → b (0), b → . (1), . → a (1), a → . (½)
        assert r2.zero_probability_events == 2 and r2.num_tokens == 5   # denominator keeps the zero events
        assert r2.total_nll == math.inf

    def test_unseen_context_counts_as_zero_probability(self):
        v = Vocabulary(chars=("a", "b", "c"))
        m = BigramLanguageModel(v).fit(["ab"])
        r = m.evaluate(["c"])                    # . → c (P = 0), c → . (undefined row)
        assert r.zero_probability_events == 2 and r.total_nll == math.inf

    def test_smoothing_gives_finite_nll(self, tiny_vocab, tiny_names):
        m = BigramLanguageModel(tiny_vocab, alpha=1.0).fit(tiny_names)
        r = m.evaluate(["b"])                    # P(b|.) = 1/5, P(.|b) = 2/4
        assert r.is_finite and r.zero_probability_events == 0
        assert r.total_nll == pytest.approx(math.log(10.0), rel=1e-6)
        assert r.perplexity == pytest.approx(math.sqrt(10.0), rel=1e-6)

    def test_matches_brute_force_and_shared_token_nll(self, small_names):
        v = Vocabulary.from_names(small_names)
        train, held_out = small_names[:12], ["olivia", "lucas", "anna", "mia", "ava"]
        m = BigramLanguageModel(v, alpha=0.3).fit(train)
        brute = -sum(math.log(m.transition_probability(p, q)) for n in held_out for p, q in zip("." + n, n + "."))
        r = m.evaluate(held_out)
        assert r.total_nll == pytest.approx(brute, rel=1e-6)
        # Independent route: shared token_nll on log-probabilities used as logits (log-softmax(log p) = log p).
        prev = torch.tensor([v.stoi[p] for n in held_out for p in "." + n])
        nxt = torch.tensor([v.stoi[q] for n in held_out for q in n + "."])
        shared = token_nll(m.log_probabilities()[prev], nxt, reduction="sum").item()
        assert r.total_nll == pytest.approx(shared, rel=1e-5)

    def test_evaluation_does_not_change_state(self, tiny_mle):
        before = (tiny_mle.counts(), tiny_mle.probabilities())
        tiny_mle.evaluate(["ab", "b", "a"])
        assert torch.equal(tiny_mle.counts(), before[0]) and torch.equal(tiny_mle.probabilities(), before[1])

    def test_unknown_characters_are_rejected_not_scored(self, tiny_mle):
        with pytest.raises(VocabularyError, match="unknown character 'z'"):
            tiny_mle.evaluate(["az"])
        with pytest.raises(ValueError, match="empty"):
            tiny_mle.evaluate([])

    def test_fit_uses_only_supplied_names(self, small_names):
        v = Vocabulary.from_names(small_names)
        train, held_out = small_names[:10], small_names[10:]
        a = BigramLanguageModel(v).fit(train)
        b = BigramLanguageModel(v).fit(train)
        b.evaluate(held_out)
        assert torch.equal(a.counts(), b.counts())
        assert int(a.counts()[0].sum()) == len(train)


# ============================================================== 8.6 generation
class TestGeneration:
    def test_deterministic_model_generates_the_training_name(self, tiny_vocab):
        m = BigramLanguageModel(tiny_vocab).fit(["ab"])
        out = m.generate(5, max_length=10, generator=make_generator(0))
        assert out == [GeneratedName("ab", True)] * 5

    def test_max_length_handling(self, tiny_vocab):
        m = BigramLanguageModel(tiny_vocab).fit(["ab"])
        # max_length = 1: draws "a", then "b" would exceed the limit → stop, truncated
        assert m.generate(1, max_length=1, generator=make_generator(0)) == [GeneratedName("a", False)]
        # max_length = 2: "a", "b", then EOS on the extra draw → complete name of exactly the limit
        assert m.generate(1, max_length=2, generator=make_generator(0)) == [GeneratedName("ab", True)]

    def test_eos_stopping_and_valid_characters(self, small_names):
        v = Vocabulary.from_names(small_names)
        m = BigramLanguageModel(v, alpha=1.0).fit(small_names)
        out = m.generate(300, max_length=8, generator=make_generator(1))
        assert len(out) == 300
        allowed = set(v.chars)
        for s in out:
            assert set(s.text) <= allowed and "." not in s.text
            assert len(s.text) <= 8
            if not s.finished:
                assert len(s.text) == 8
        assert any(s.finished for s in out) and any(not s.finished for s in out)
        assert any(s.text == "" for s in out)   # . → . has P = α/(N₀+αV) > 0: the empty name is a legal sample

    def test_controlled_generator_is_reproducible(self, small_names):
        v = Vocabulary.from_names(small_names)
        m = BigramLanguageModel(v).fit(small_names)
        a = m.generate(50, max_length=20, generator=make_generator(123))
        b = m.generate(50, max_length=20, generator=make_generator(123))
        assert a == b

    def test_mle_generation_only_uses_observed_transitions(self, small_names):
        v = Vocabulary.from_names(small_names)
        m = BigramLanguageModel(v).fit(small_names)
        C = m.counts()
        for s in m.generate(200, max_length=30, generator=make_generator(7)):
            seq = "." + s.text + ("." if s.finished else "")
            for p, q in zip(seq, seq[1:]):
                assert C[v.stoi[p], v.stoi[q]] > 0, (s, p, q)
        assert m.generate(1, max_length=30, generator=make_generator(7))[0].text != ""   # P(.|.) = 0 under MLE

    def test_sampling_follows_learned_distribution(self, tiny_mle):
        # Context 'a': P(.|a) = P(b|a) = ½. A fixed seed makes this assertion deterministic.
        draws = tiny_mle.sample_next(torch.full((20000,), 1), generator=make_generator(0))
        freq_b = (draws == 2).float().mean().item()
        assert set(draws.unique().tolist()) == {0, 2} and abs(freq_b - 0.5) < 0.02
        assert (tiny_mle.sample_next(torch.zeros(100, dtype=torch.long), generator=make_generator(0)) == 1).all()
        # The distribution handed to the shared sampler is exactly the learned one.
        assert torch.allclose(next_token_probs(tiny_mle.log_probabilities()), tiny_mle.probabilities(), atol=1e-7)

    def test_greedy_decoding(self):
        names = ["ab", "ab", "ac"]
        m = BigramLanguageModel(Vocabulary.from_names(names)).fit(names)
        assert m.generate(3, max_length=5, temperature=0.0) == [GeneratedName("ab", True)] * 3

    def test_undefined_row_raises_clearly(self):
        v = Vocabulary(chars=("a", "b", "c"))
        m = BigramLanguageModel(v).fit(["ab"])
        with pytest.raises(UndefinedDistributionError, match=r"\['c'\].*undefined"):
            m.sample_next(torch.tensor([1, v.stoi["c"]]), generator=make_generator(0))
        # From BOS the undefined row is unreachable (flow conservation), so generation still works.
        assert all(s == GeneratedName("ab", True) for s in m.generate(10, max_length=5, generator=make_generator(0)))
        smooth = BigramLanguageModel(v, alpha=1.0).fit(["ab"])
        assert smooth.sample_next(torch.tensor([v.stoi["c"]]), generator=make_generator(0)).shape == (1,)

    def test_generation_does_not_mutate_state(self, small_names):
        m = BigramLanguageModel(Vocabulary.from_names(small_names), alpha=0.5).fit(small_names)
        before = (m.counts(), m.probabilities(), m.log_probabilities())
        m.generate(100, max_length=10, generator=make_generator(3))
        after = (m.counts(), m.probabilities(), m.log_probabilities())
        assert all(torch.equal(x, y) for x, y in zip(before, after))

    @pytest.mark.parametrize(
        "kwargs, exc",
        [
            (dict(num_samples=0, max_length=5), ValueError),
            (dict(num_samples=2, max_length=0), ValueError),
            (dict(num_samples=True, max_length=5), ValueError),
            (dict(num_samples=1, max_length=5, temperature=-1.0), ValueError),
            (dict(num_samples=1, max_length=5, top_p=2.0), ValueError),
        ],
    )
    def test_invalid_generation_arguments(self, tiny_mle, kwargs, exc):
        with pytest.raises(exc):
            tiny_mle.generate(**kwargs)

    def test_invalid_contexts(self, tiny_mle):
        for bad in (torch.tensor([0.0]), torch.tensor([[0]]), torch.tensor([3]), torch.tensor([-1])):
            with pytest.raises(ValueError):
                tiny_mle.sample_next(bad)


# ============================================================= 8.7 persistence
class TestPersistence:
    def test_round_trip(self, small_names, tmp_path):
        v = Vocabulary.from_names(small_names)
        m = BigramLanguageModel(v, alpha=0.25).fit(small_names[:12])
        path = m.save(tmp_path / "model_final.pt")
        ckpt = load_checkpoint(path)                                  # weights_only=True
        assert ckpt["kind"] == "final" and ckpt["model_class"] == "BigramLanguageModel"
        assert ckpt["extra"] == {"num_training_names": 12, "num_training_transitions": int(m.counts().sum())}
        loaded = BigramLanguageModel.load(path, expected_vocab=v)
        assert loaded.vocab == v and loaded.vocab.chars == v.chars and loaded.alpha == 0.25
        assert torch.equal(loaded.counts(), m.counts())
        assert torch.equal(loaded.probabilities(), m.probabilities())
        held_out = small_names[12:]
        assert loaded.evaluate(held_out) == m.evaluate(held_out)
        assert loaded.generate(20, max_length=10, generator=make_generator(5)) == \
            m.generate(20, max_length=10, generator=make_generator(5))

    def test_mle_round_trip_preserves_zero_probability_behaviour(self, tiny_mle, tmp_path):
        loaded = BigramLanguageModel.load(tiny_mle.save(tmp_path / "m.pt"))
        assert loaded.alpha == 0.0 and loaded.evaluate(["b"]) == tiny_mle.evaluate(["b"])

    def test_full_experiment_config_can_be_embedded(self, tiny_mle, tmp_path):
        p = tiny_mle.save(tmp_path / "m.pt", config={"model": {"name": "bigram", "params": {"alpha": 0.0}}, "x": 1})
        assert load_checkpoint(p)["config"]["x"] == 1

    def test_incompatible_vocabulary_rejected(self, tiny_mle, tmp_path):
        path = tiny_mle.save(tmp_path / "m.pt")
        other = Vocabulary(chars=("a", "c"))
        with pytest.raises(IncompatibleStateError, match="vocabulary mismatch"):
            BigramLanguageModel.load(path, expected_vocab=other)
        reordered_target = BigramLanguageModel(Vocabulary(chars=("a", "b", "c")))
        with pytest.raises(IncompatibleStateError, match="vocabulary mismatch"):
            reordered_target.load_state_dict(tiny_mle.state_dict())

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda s: s.update(alpha=1.0), "alpha mismatch"),
            (lambda s: s.update(format="bigram/v0"), "unsupported bigram state format"),
            (lambda s: s.update(counts=s["counts"].float()), "int64"),
            (lambda s: s.update(counts=s["counts"][:2]), r"\[3, 3\]"),
            (lambda s: s.update(counts=-s["counts"]), "non-negative"),
            (lambda s: s["counts"].__setitem__((1, 2), 5), "flow conservation"),
            (lambda s: s.update(counts=torch.zeros(3, 3, dtype=torch.long)), "no BOS"),
            (lambda s: s.update(extra=1), "must have keys"),
            (lambda s: s.pop("alpha"), "must have keys"),
            (lambda s: s.update(vocab={"chars": "ba", "boundary": "."}), "invalid"),
        ],
    )
    def test_corrupted_state_rejected(self, tiny_mle, tiny_vocab, mutate, match):
        state = tiny_mle.state_dict()
        mutate(state)
        with pytest.raises(IncompatibleStateError, match=match):
            BigramLanguageModel(tiny_vocab).load_state_dict(state)

    def test_flow_conserving_disconnected_cycle_rejected(self):
        v = Vocabulary(chars=("a", "b", "c", "d"))
        state = BigramLanguageModel(v).fit(["ab"]).state_dict()
        state["counts"][3, 4] += 1                           # c → d
        state["counts"][4, 3] += 1                           # d → c
        check_flow_conservation(state["counts"])             # the old check alone would accept this
        with pytest.raises(IncompatibleStateError, match="not reachable from BOS"):
            BigramLanguageModel(v).load_state_dict(state)

    def test_positive_bos_to_eos_count_rejected(self, tiny_mle, tiny_vocab):
        state = tiny_mle.state_dict()
        state["counts"][0, 0] += 1                           # an "empty name": still flow-conserving
        check_flow_conservation(state["counts"])
        with pytest.raises(IncompatibleStateError, match="BOS → EOS"):
            BigramLanguageModel(tiny_vocab).load_state_dict(state)

    def test_unused_vocabulary_tokens_load_successfully(self, tmp_path):
        v = Vocabulary(chars=("a", "b", "c", "d"))           # c and d never occur in training
        m = BigramLanguageModel(v).fit(["ab", "ba"])
        c_id, d_id = v.stoi["c"], v.stoi["d"]
        assert int(m.counts()[[c_id, d_id]].sum()) == 0 and int(m.counts()[:, [c_id, d_id]].sum()) == 0
        restored = BigramLanguageModel(v)
        restored.load_state_dict(m.state_dict())
        assert torch.equal(restored.counts(), m.counts())
        loaded = BigramLanguageModel.load(m.save(tmp_path / "m.pt"), expected_vocab=v)
        assert torch.equal(loaded.counts(), m.counts())
        assert loaded.defined_contexts().tolist() == [True, True, True, False, False]

    def test_state_dict_does_not_alias_fitted_counts(self, tiny_mle):
        state = tiny_mle.state_dict()
        state["counts"].zero_()
        assert int(tiny_mle.counts().sum()) == 5

    def test_non_strict_load_not_supported(self, tiny_mle, tiny_vocab):
        with pytest.raises(ValueError, match="strict=True"):
            BigramLanguageModel(tiny_vocab).load_state_dict(tiny_mle.state_dict(), strict=False)

    def test_wrong_checkpoint_rejected(self, tmp_path):
        p = save_checkpoint(tmp_path / "linear.pt", kind="final", model=torch.nn.Linear(2, 2))
        with pytest.raises(IncompatibleStateError, match="not a BigramLanguageModel"):
            BigramLanguageModel.load(p)
        with pytest.raises(CheckpointError, match="not found"):
            BigramLanguageModel.load(tmp_path / "missing.pt")


# ================================================================= 8.8 devices
@pytest.mark.parametrize("device", DEVICES)
class TestDevices:
    def test_fit_evaluate_generate_on_device(self, device, small_names):
        v = Vocabulary.from_names(small_names)
        cpu = BigramLanguageModel(v, alpha=0.5).fit(small_names)
        m = BigramLanguageModel(v, alpha=0.5, device=device).fit(small_names)
        assert m.counts().device.type == device and m.probabilities().device.type == device
        assert m.counts().dtype == torch.int64 and m.probabilities().dtype == torch.float32
        assert torch.equal(m.counts().cpu(), cpu.counts())
        assert torch.allclose(m.probabilities().cpu(), cpu.probabilities(), atol=1e-6)
        r, r_cpu = m.evaluate(small_names), cpu.evaluate(small_names)
        assert r.num_tokens == r_cpu.num_tokens and r.total_nll == pytest.approx(r_cpu.total_nll, rel=1e-5)
        out = m.generate(20, max_length=10, generator=make_generator(0))
        assert out == m.generate(20, max_length=10, generator=make_generator(0))
        assert all(set(s.text) <= set(v.chars) for s in out)
        assert m.sample_next(torch.tensor([0, 1])).device.type == device

    def test_mle_zero_probabilities_on_device(self, device, tiny_vocab, tiny_names):
        m = BigramLanguageModel(tiny_vocab, device=device).fit(tiny_names)
        r = m.evaluate(["b"])
        assert r.zero_probability_events == 1 and r.total_nll == math.inf

    def test_move_and_load_to_device(self, device, tiny_mle, tmp_path):
        moved = BigramLanguageModel.load(tiny_mle.save(tmp_path / "m.pt"), device=device)
        assert moved.counts().device.type == device
        assert moved.to("cpu").counts().device.type == "cpu"
        assert moved.evaluate(["a", "ab"]) == tiny_mle.evaluate(["a", "ab"])


# ========================================================== real-data checks
@pytest.mark.skipif(not DEFAULT_NAMES_PATH.is_file(), reason="names dataset not present")
def test_real_dataset_invariants():
    names = load_names()
    splits = split_names(names, seed=42)
    vocab = Vocabulary.from_names(splits.train)
    m = BigramLanguageModel(vocab).fit(splits.train)
    C = m.counts()
    assert vocab.num_classes == 27 and C.shape == (27, 27)
    assert int(C.sum()) == sum(len(n) + 1 for n in splits.train)
    assert int(C[0].sum()) == len(splits.train)
    check_fitted_counts(C)
    assert m.defined_contexts().all()          # vocabulary from train ⇒ every context observed
    dev = m.evaluate(splits.dev)
    assert dev.num_tokens == sum(len(n) + 1 for n in splits.dev)
    smoothed = BigramLanguageModel(vocab, alpha=1.0).fit(splits.train).evaluate(splits.dev)
    assert smoothed.is_finite and 1.0 < smoothed.mean_nll < math.log(27)
