"""Statistical character-level bigram language model.

Model
-----
A bigram model assumes the next character depends only on the previous one:

    P(x_1 .. x_L, EOS) = P(x_1 | BOS) · Π_{t=2..L} P(x_t | x_{t-1}) · P(EOS | x_L)

With the shared vocabulary, BOS and EOS are the same token ``.`` (id 0), so the
name ``anna`` is scored as the transitions ``. → a → n → n → a → .``.

Estimation (closed form, no gradients)
--------------------------------------
Let ``V`` be the number of output classes (27 for the names data) and

    C_ij = #(i → j) in the training names                      counts,  [V, V] int64
    N_i  = Σ_j C_ij                                            context totals, [V]

Maximum likelihood (``alpha = 0``, the default)::

    P(j | i) = C_ij / N_i

Additive (Laplace / Lidstone) smoothing (``alpha > 0``)::

    P_α(j | i) = (C_ij + α) / (N_i + α V)

Only ``alpha = 0`` is a maximum-likelihood estimate: for each context ``i``, the
row ``C_i· / N_i`` is the categorical distribution that maximises the likelihood
of the observed training transitions out of ``i``. ``alpha > 0`` is *not* a
maximum-likelihood estimate. It adds ``α`` pseudo-counts to every cell of the
count matrix and then normalises each row, which moves mass from observed toward
unobserved transitions. The effect is strong when ``α V`` is large relative to
``N_i`` and vanishes as ``N_i`` grows. (Equivalently, it is the posterior mean of
each row under a symmetric Dirichlet(α) prior.) Smoothing is never applied
implicitly: ``alpha`` is stored with the fitted state, and the stored counts are
always the raw, unsmoothed ``C``.

Zero-probability events
-----------------------
With ``alpha = 0`` a transition never seen in training has probability exactly 0,
so a held-out name containing it has likelihood 0 and NLL ``+inf``. This model
does **not** clamp probabilities to an epsilon, because that would silently
change the model. Instead :meth:`BigramLanguageModel.evaluate` reports

* ``zero_probability_events`` — how many evaluated transitions had P = 0, and
* ``total_nll = mean_nll = +inf`` whenever that count is positive,

and computes the finite part only over events with P > 0, so ``0 · log 0`` is
never formed and no NaN can appear. Any ``alpha > 0`` that is representable in
float32 makes every probability positive, so the NLL is finite.

A context row with ``N_i = 0`` (a vocabulary character never seen in training)
has no MLE distribution (0/0). Under ``alpha = 0`` such a row is stored as all
zeros and is **undefined**:

* evaluating a transition out of it counts as a zero-probability event;
* sampling from it raises :class:`UndefinedDistributionError` instead of falling
  back to an arbitrary distribution.

When the vocabulary is built from the training names (the project convention)
every row has ``N_i > 0``. Each character occurrence is followed by exactly one
transition, so such rows cannot exist. Generation from BOS also never reaches an
undefined row, because ``C_ij > 0`` implies ``N_j ≥ 1`` (flow conservation, see
:func:`check_flow_conservation`).

Device and dtype
----------------
Counting runs on CPU, because the inputs are Python strings. The resulting
``[V, V]`` counts are moved once to the model's device, where probabilities,
log-probabilities and sampling are computed. Counts are int64. Probabilities
are float32 on every device, so CPU, CUDA and MPS share one dtype (MPS has no
float64). The final NLL reduction runs in float64 on CPU over a ``[V, V]``
matrix, which is one tiny, explicit transfer per evaluation.

Scope
-----
No ``nn.Module``, no parameters, no autograd, no optimizer. The model exposes
the PyTorch ``state_dict`` / ``load_state_dict`` protocol only so that it can
be persisted with the shared :mod:`utils.checkpoint` primitives.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils.checkpoint import load_checkpoint, restore_model, save_checkpoint
from utils.data import encode_names
from utils.device import resolve_device
from utils.generation import sample_next_token
from utils.metrics import bits_per_token, perplexity
from utils.vocabulary import Vocabulary, VocabularyError

MODEL_NAME = "bigram"
STATE_FORMAT = "bigram/v1"
_PROB_DTYPE = torch.float32
_ROW_SUM_ATOL = 1e-5  # float32 rows of ≤ 27 terms sum to 1 within ~1e-7; 1e-5 leaves margin


class NotFittedError(RuntimeError):
    """Raised when an operation needs fitted state but :meth:`fit` has not run."""


class UndefinedDistributionError(RuntimeError):
    """Raised when sampling from a context whose conditional distribution is undefined."""


class IncompatibleStateError(ValueError):
    """Raised when saved state does not match this model (vocabulary, alpha, format, shape)."""


# =================================================================== config
@dataclass(frozen=True)
class BigramConfig:
    """Model-specific configuration (``model.params`` in ``configs/models/bigram.yaml``).

    alpha:
        Additive smoothing pseudo-count added to every transition. ``0.0`` (the
        default) is the unsmoothed maximum-likelihood estimate. Must be finite and ≥ 0.
    """

    alpha: float = 0.0

    def __post_init__(self) -> None:
        validate_alpha(self.alpha)


def validate_alpha(alpha: float) -> float:
    """Return ``alpha`` as a float if it is a finite number ≥ 0, otherwise raise ``ValueError``."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be a real number, got {alpha!r}")
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0 (0 = maximum likelihood), got {alpha}")
    return alpha


# ================================================================ counting
def _validate_names(names: Sequence[str], what: str) -> None:
    if isinstance(names, str):
        raise TypeError(f"{what} must be a sequence of names, not a single string {names!r}")
    if len(names) == 0:
        raise ValueError(f"{what} is empty")
    for pos, name in enumerate(names):
        if not isinstance(name, str):
            raise TypeError(f"{what}[{pos}] must be a string, got {type(name).__name__}")
        if name == "":
            raise ValueError(f"{what}[{pos}] is an empty name; every name needs at least one character")


def count_transitions(names: Sequence[str], vocab: Vocabulary) -> torch.Tensor:
    """Count bigram transitions ``i → j`` in ``names``, framed with BOS/EOS.

    Each name of length ``L`` contributes ``L + 1`` transitions: ``BOS → x_1``,
    ``x_{t-1} → x_t`` and ``x_L → EOS``.

    Returns
    -------
    torch.Tensor
        ``[V, V]`` int64 tensor on CPU with ``V = vocab.num_classes``. Row ``i`` is
        the previous token, column ``j`` the next token; row 0 holds BOS
        transitions and column 0 holds EOS transitions.

    Raises
    ------
    TypeError, ValueError
        Empty input, empty names or non-string items.
    VocabularyError
        A character that is not in ``vocab``. The vocabulary is never extended.
    """
    _validate_names(names, "names")
    V = vocab.num_classes
    encoded = encode_names(names, vocab, add_bos=True, add_eos=True)  # [[0, x1, ..., xL, 0], ...]
    prev_ids: list[int] = []
    next_ids: list[int] = []
    for ids in encoded:
        prev_ids.extend(ids[:-1])
        next_ids.extend(ids[1:])
    prev = torch.tensor(prev_ids, dtype=torch.long)
    nxt = torch.tensor(next_ids, dtype=torch.long)
    # Each pair (i, j) maps to the flat index i·V + j. A histogram over the V² cells is the count matrix.
    counts = torch.bincount(prev * V + nxt, minlength=V * V).reshape(V, V)
    return counts


def check_flow_conservation(counts: torch.Tensor) -> None:
    """Raise ``ValueError`` if ``counts`` violates flow conservation.

    This is a **necessary** condition for counts produced by :func:`count_transitions`,
    not a sufficient one. Every occurrence of an ordinary character ``j`` has exactly
    one incoming and one outgoing transition, so column sums equal row sums for
    ``j ≠ 0``. Every name has one BOS and one EOS transition, so
    ``Σ_j C_0j = Σ_i C_i0``. Together, the two marginals are identical vectors.

    A matrix can conserve flow and still be impossible, for example with a cycle
    ``c → d → c`` that never connects to BOS, or with a positive ``BOS → EOS``
    count (an empty name). :func:`check_fitted_counts` adds the conditions that
    rule these out.
    """
    incoming = counts.sum(dim=0)  # [V]: transitions into j (EOS for j = 0)
    outgoing = counts.sum(dim=1)  # [V]: transitions out of i (BOS for i = 0)
    if not torch.equal(incoming, outgoing):
        bad = torch.nonzero(incoming != outgoing).flatten().tolist()
        raise ValueError(f"count matrix violates flow conservation at token id(s) {bad}")


def unreachable_from_bos(counts: torch.Tensor) -> list[int]:
    """Ids of tokens that have transitions but cannot be reached from BOS.

    Treats ``counts`` as a directed graph with an edge ``i → j`` whenever
    ``C_ij > 0`` and runs a breadth-first search from token 0. A token is *used*
    if it has any incoming or outgoing count. Used tokens that the search does
    not reach are returned in ascending order. Unused vocabulary tokens (an
    all-zero row and column) are not reported.
    """
    edges = counts > 0  # [V, V] bool
    used = edges.any(dim=0) | edges.any(dim=1)  # [V]
    reached = [False] * counts.shape[0]
    reached[0] = True
    frontier = [0]
    while frontier:
        i = frontier.pop()
        for j in torch.nonzero(edges[i]).flatten().tolist():
            if not reached[j]:
                reached[j] = True
                frontier.append(j)
    return [k for k in range(counts.shape[0]) if bool(used[k]) and not reached[k]]


def check_fitted_counts(counts: torch.Tensor) -> None:
    """Raise ``ValueError`` unless a non-negative integer ``counts`` is exactly what
    :func:`count_transitions` could produce from some non-empty list of non-empty names.

    Conditions checked:

    1. ``Σ_j C_0j > 0``: at least one name (one BOS transition per name).
    2. ``C_00 = 0``: no ``BOS → EOS`` transition, because empty names are rejected.
    3. Flow conservation (:func:`check_flow_conservation`).
    4. Every used token is reachable from BOS along positive-count transitions
       (:func:`unreachable_from_bos`). Unused vocabulary tokens are allowed.

    These are also sufficient. By 3 every token has in-degree equal to out-degree,
    and by 4 all used tokens lie in one connected component that contains BOS. A
    balanced directed multigraph whose edges form one connected component has an
    Eulerian circuit (a closed walk using every edge exactly once). Cutting that
    circuit at each visit to token 0 splits it into ``Σ_j C_0j`` closed walks
    ``0 → x_1 → … → x_L → 0``. By 2 each walk has ``L ≥ 1``, so each one is a valid
    name, and counting those names reproduces ``C`` exactly.
    """
    if int(counts[0].sum()) == 0:
        raise ValueError("counts contain no BOS transitions (fitted on zero names)")
    if int(counts[0, 0]) != 0:
        raise ValueError(
            f"counts contain {int(counts[0, 0])} BOS → EOS transition(s), i.e. empty names, which fit() rejects"
        )
    check_flow_conservation(counts)
    unreachable = unreachable_from_bos(counts)
    if unreachable:
        raise ValueError(
            f"token id(s) {unreachable} have transitions but are not reachable from BOS (token 0); "
            "no sequence of names produces such counts"
        )


# ================================================================== results
@dataclass(frozen=True)
class BigramEvaluation:
    """Token-weighted likelihood of a collection of names.

    total_nll:
        ``-Σ_m log P(x_m | x_{m-1})`` over all evaluated transitions, in nats.
        ``+inf`` if any transition has probability 0.
    num_tokens:
        Number of predicted tokens ``M = Σ_names (len(name) + 1)``. EOS is included.
        Zero-probability events are included too, never dropped from the denominator.
    num_names:
        Number of names evaluated.
    zero_probability_events:
        Number of evaluated transitions whose model probability is exactly 0.
    """

    total_nll: float
    num_tokens: int
    num_names: int
    zero_probability_events: int

    @property
    def mean_nll(self) -> float:
        """``total_nll / num_tokens`` in nats per token (``+inf`` if any event had P = 0)."""
        return self.total_nll / self.num_tokens

    @property
    def perplexity(self) -> float:
        return perplexity(self.mean_nll)

    @property
    def bits_per_token(self) -> float:
        return bits_per_token(self.mean_nll)

    @property
    def is_finite(self) -> bool:
        return self.zero_probability_events == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_nll": self.total_nll,
            "num_tokens": self.num_tokens,
            "num_names": self.num_names,
            "zero_probability_events": self.zero_probability_events,
            "mean_nll": self.mean_nll,
            "perplexity": self.perplexity,
            "bits_per_token": self.bits_per_token,
        }


@dataclass(frozen=True)
class GeneratedName:
    """One generated sample.

    text:
        Characters between BOS and EOS (or up to the length limit). Never contains ``.``.
    finished:
        ``True`` if EOS was sampled. ``False`` if generation stopped at ``max_length``,
        in which case ``text`` is a truncated, incomplete sample.
    """

    text: str
    finished: bool


# ==================================================================== model
class BigramLanguageModel:
    """Count-based bigram model over the shared character vocabulary.

    Parameters
    ----------
    vocab:
        Shared :class:`~utils.vocabulary.Vocabulary`. Its ids index the count
        matrix rows and columns. Build it from the training split.
    alpha:
        Additive smoothing (≥ 0). ``0.0`` gives the maximum-likelihood estimate.
    device:
        Where fitted tensors live, e.g. ``"cpu"``, ``"cuda"``, ``"mps"`` or
        ``"auto"``. Resolved with :func:`utils.device.resolve_device`, so an
        explicit accelerator that is unavailable raises instead of falling back.

    State
    -----
    The only learned state is the count matrix. Probabilities and
    log-probabilities are derived from it (and ``alpha``) and cached. They are
    never saved separately, so they can never disagree with the counts.
    """

    def __init__(self, vocab: Vocabulary, *, alpha: float = 0.0, device: str | torch.device = "cpu") -> None:
        if not isinstance(vocab, Vocabulary):
            raise TypeError(f"vocab must be a utils.vocabulary.Vocabulary, got {type(vocab).__name__}")
        self.vocab = vocab
        self.alpha = validate_alpha(alpha)
        self.device = resolve_device(str(device))
        self._counts: torch.Tensor | None = None  # [V, V] int64
        self._probs: torch.Tensor | None = None  # [V, V] float32
        self._log_probs: torch.Tensor | None = None  # [V, V] float32, -inf where P = 0

    @classmethod
    def from_config(
        cls, vocab: Vocabulary, config: BigramConfig, *, device: str | torch.device = "cpu"
    ) -> BigramLanguageModel:
        """Build from a validated :class:`BigramConfig` (e.g. ``cfg.model.params``)."""
        if not isinstance(config, BigramConfig):
            raise TypeError(f"config must be a BigramConfig, got {type(config).__name__}")
        return cls(vocab, alpha=config.alpha, device=device)

    # ----------------------------------------------------------- properties
    @property
    def num_classes(self) -> int:
        """``V``: output vocabulary size (boundary token plus characters)."""
        return self.vocab.num_classes

    @property
    def is_fitted(self) -> bool:
        return self._counts is not None

    @property
    def config(self) -> BigramConfig:
        return BigramConfig(alpha=self.alpha)

    def _require_fitted(self) -> None:
        if self._counts is None:
            raise NotFittedError("BigramLanguageModel is not fitted; call fit(training_names) or load() first")

    # ------------------------------------------------------------------ fit
    def fit(self, training_names: Sequence[str]) -> BigramLanguageModel:
        """Estimate transition probabilities from training names only.

        Replaces any previously fitted state. The result is deterministic: it
        depends only on the multiset of names, ``vocab`` and ``alpha``. Returns
        ``self`` so calls can be chained.

        Raises ``ValueError``/``TypeError`` for empty or malformed input, and
        :class:`~utils.vocabulary.VocabularyError` for characters outside the
        vocabulary.
        """
        counts = count_transitions(training_names, self.vocab)
        self._set_counts(counts)
        return self

    def _set_counts(self, counts: torch.Tensor) -> None:
        counts = counts.to(self.device)
        probs = self._estimate_probabilities(counts)
        self._counts = counts
        self._probs = probs
        self._log_probs = torch.log(probs)  # log(0) = -inf exactly; no clamping

    def _estimate_probabilities(self, counts: torch.Tensor) -> torch.Tensor:
        """Map ``[V, V]`` counts to ``[V, V]`` conditional probabilities and validate them."""
        V = self.num_classes
        c = counts.to(_PROB_DTYPE)
        row_totals = c.sum(dim=1, keepdim=True)  # N_i, [V, 1]
        if self.alpha > 0:
            probs = (c + self.alpha) / (row_totals + self.alpha * V)
        else:
            defined = row_totals > 0
            # clamp_min(1) only guards the 0/0 branch that torch.where discards; it never changes a probability.
            probs = torch.where(defined, c / row_totals.clamp_min(1.0), torch.zeros_like(c))
        self._validate_probabilities(probs, counts)
        return probs

    def _validate_probabilities(self, probs: torch.Tensor, counts: torch.Tensor) -> None:
        if not bool(torch.isfinite(probs).all()):
            raise ValueError("estimated probabilities contain NaN or inf")
        if bool((probs < 0).any()):
            raise ValueError("estimated probabilities contain negative values")
        if self.alpha > 0 and bool((probs == 0).any()):
            raise ValueError(
                f"alpha={self.alpha!r} is too small to represent in float32: some smoothed probabilities "
                "underflow to 0. Use alpha=0 for maximum likelihood or a larger alpha."
            )
        defined_rows = counts.sum(dim=1) > 0 if self.alpha == 0 else torch.ones(
            self.num_classes, dtype=torch.bool, device=probs.device
        )
        row_sums = probs.sum(dim=1)
        if not bool(torch.allclose(row_sums[defined_rows], torch.ones_like(row_sums[defined_rows]), atol=_ROW_SUM_ATOL)):
            raise ValueError("conditional probability rows do not sum to 1")

    # -------------------------------------------------------------- access
    def counts(self) -> torch.Tensor:
        """Copy of the transition counts ``C``: ``[V, V]`` int64 on ``self.device``."""
        self._require_fitted()
        return self._counts.clone()

    def probabilities(self) -> torch.Tensor:
        """Copy of ``P(j | i)``: ``[V, V]`` float32. Row ``i`` is the context.

        Rows sum to 1, except that under ``alpha = 0`` a context never seen in
        training is an all-zero row (an undefined distribution, see module docs).
        """
        self._require_fitted()
        return self._probs.clone()

    def log_probabilities(self) -> torch.Tensor:
        """Copy of ``log P(j | i)``: ``[V, V]`` float32 with ``-inf`` where P = 0."""
        self._require_fitted()
        return self._log_probs.clone()

    def defined_contexts(self) -> torch.Tensor:
        """``[V]`` bool: rows that are proper probability distributions."""
        self._require_fitted()
        if self.alpha > 0:
            return torch.ones(self.num_classes, dtype=torch.bool, device=self.device)
        return self._counts.sum(dim=1) > 0

    def transition_probability(self, prev: str, next_: str) -> float:
        """``P(next_ | prev)`` for single characters. Use ``"."`` for BOS (as ``prev``) and EOS (as ``next_``)."""
        self._require_fitted()
        i, j = self._token_id(prev), self._token_id(next_)
        return float(self._probs[i, j])

    def _token_id(self, symbol: str) -> int:
        stoi = self.vocab.stoi
        if symbol not in stoi:
            raise VocabularyError(f"{symbol!r} is not a vocabulary token (use {self.vocab.boundary!r} for BOS/EOS)")
        return stoi[symbol]

    # ------------------------------------------------------------ evaluate
    def evaluate(self, names: Sequence[str]) -> BigramEvaluation:
        """Token-weighted NLL of ``names``. Does not modify the fitted state.

        The held-out transitions are counted into ``E_ij`` with the same routine
        as :meth:`fit`. Then

            total_nll = - Σ_{i,j : E_ij > 0} E_ij · log P(j | i)
            num_tokens = Σ_ij E_ij = Σ_names (len + 1)

        Cells with ``E_ij > 0`` and ``P(j | i) = 0`` are zero-probability events.
        If there are any, ``total_nll`` is ``+inf``. The sum runs only over cells
        with ``E_ij > 0`` and ``P > 0``, so ``0 · (-inf)`` is never evaluated.

        Raises :class:`~utils.vocabulary.VocabularyError` for characters outside
        the vocabulary. Unknown characters are not zero-probability events,
        because they have no id at all. Check splits with
        :func:`utils.data.check_vocab_coverage` first.
        """
        self._require_fitted()
        eval_counts = count_transitions(names, self.vocab).to(torch.float64)  # E, [V, V] on CPU
        log_probs = self._log_probs.detach().cpu().to(torch.float64)  # [V, V]; one small transfer
        observed = eval_counts > 0
        zero_prob = observed & torch.isneginf(log_probs)
        finite = observed & ~zero_prob
        num_zero = int(eval_counts[zero_prob].sum().item())
        finite_nll = float(-(eval_counts[finite] * log_probs[finite]).sum().item())
        return BigramEvaluation(
            total_nll=math.inf if num_zero > 0 else finite_nll,
            num_tokens=int(eval_counts.sum().item()),
            num_names=len(names),
            zero_probability_events=num_zero,
        )

    # ------------------------------------------------------------ generate
    def sample_next(
        self,
        contexts: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Sample one next token for each context id.

        Parameters
        ----------
        contexts:
            ``[B]`` int64 previous-token ids in ``[0, V)``.
        generator:
            Explicit ``torch.Generator`` for reproducible draws. A CPU generator
            from :func:`utils.reproducibility.make_generator` is the project
            convention. ``None`` uses the global torch RNG.
        temperature, top_k, top_p:
            Passed to :func:`utils.generation.sample_next_token`. With the defaults
            the draw follows ``P(· | context)`` exactly. Other values sample from a
            *modified* distribution.

        Returns
        -------
        torch.Tensor
            ``[B]`` int64 next-token ids on ``self.device``.

        Raises
        ------
        UndefinedDistributionError
            If a context's conditional distribution is undefined (``alpha = 0`` and
            the context never occurred in training).
        """
        self._require_fitted()
        if contexts.ndim != 1 or contexts.dtype != torch.long:
            raise ValueError(f"contexts must be a 1-D int64 tensor, got shape {tuple(contexts.shape)} {contexts.dtype}")
        contexts = contexts.to(self.device)
        if bool(((contexts < 0) | (contexts >= self.num_classes)).any()):
            raise ValueError(f"context ids must lie in [0, {self.num_classes})")
        undefined = ~self.defined_contexts()[contexts]
        if bool(undefined.any()):
            ids = sorted(set(contexts[undefined].tolist()))
            symbols = [self.vocab.itos[i] for i in ids]
            raise UndefinedDistributionError(
                f"context(s) {symbols} have no observed outgoing transitions; with alpha=0 their "
                "conditional distribution is undefined (0/0). Refit with alpha > 0 or a vocabulary "
                "built from the training names."
            )
        # log P is a valid logit vector: softmax(log p) = p. Zero-probability tokens stay at -inf.
        logits = self._log_probs[contexts]  # [B, V]
        return sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator)

    def generate(
        self,
        num_samples: int,
        *,
        max_length: int,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[GeneratedName]:
        """Autoregressively sample ``num_samples`` names.

        Every sample starts from BOS (``.``) and repeatedly draws
        ``x_t ~ P(· | x_{t-1})`` until EOS (``.``) is drawn. Samples are drawn in
        parallel: at each step, one draw is made for every unfinished sample.

        ``max_length`` is the maximum number of ordinary characters. A sample
        that has ``max_length`` characters gets one more draw. If that draw is
        EOS, the sample is complete. Otherwise it is returned truncated with
        ``finished=False``. So at most ``max_length + 1`` draws are made per sample.

        Sampling options are as in :meth:`sample_next`. The fitted state is never
        modified. Returns a list of :class:`GeneratedName` in sample order.
        """
        self._require_fitted()
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError(f"num_samples must be a positive int, got {num_samples!r}")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError(f"max_length must be a positive int, got {max_length!r}")
        eos = self.vocab.eos_id
        prev = torch.full((num_samples,), self.vocab.bos_id, dtype=torch.long, device=self.device)
        chars: list[list[int]] = [[] for _ in range(num_samples)]
        finished = [False] * num_samples
        active = list(range(num_samples))  # indices of unfinished samples, in sample order
        while active:
            nxt = self.sample_next(
                prev[active], generator=generator, temperature=temperature, top_k=top_k, top_p=top_p
            ).tolist()
            still_active = []
            for sample_idx, token in zip(active, nxt):
                if token == eos:
                    finished[sample_idx] = True
                elif len(chars[sample_idx]) == max_length:
                    pass  # limit reached and EOS not drawn: stop, truncated
                else:
                    chars[sample_idx].append(token)
                    still_active.append(sample_idx)
            if still_active:
                prev[still_active] = torch.tensor([chars[k][-1] for k in still_active], device=self.device)
            active = still_active
        return [
            GeneratedName(text=self.vocab.decode(ids, strip_bos=False, stop_at_eos=False), finished=done)
            for ids, done in zip(chars, finished)
        ]

    # --------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        """Learned state as plain data plus one CPU tensor (safe for ``weights_only`` loading).

        Keys are ``format``, ``vocab`` (``Vocabulary.to_dict()``), ``alpha`` and
        ``counts`` (``[V, V]`` int64 on CPU). Probabilities are derived on load.
        """
        self._require_fitted()
        return {
            "format": STATE_FORMAT,
            "vocab": self.vocab.to_dict(),
            "alpha": self.alpha,
            # clone(): on CPU, .cpu() returns the same storage; editing the state must not edit the model.
            "counts": self._counts.detach().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        """Restore state saved by :meth:`state_dict`, validating compatibility.

        The saved vocabulary (characters and ordering) and ``alpha`` must equal
        this model's. The counts must be a non-negative ``[V, V]`` int64 matrix
        that some non-empty list of non-empty names could produce (see
        :func:`check_fitted_counts`): at least one BOS transition, no BOS → EOS
        transition, flow conservation, and every used token reachable from BOS.
        Unused vocabulary tokens are allowed. Only ``strict=True`` is supported,
        because a bigram state has no optional parts.
        """
        if not strict:
            raise ValueError("BigramLanguageModel.load_state_dict supports only strict=True")
        expected = {"format", "vocab", "alpha", "counts"}
        if not isinstance(state, Mapping) or set(state) != expected:
            got = sorted(state) if isinstance(state, Mapping) else type(state).__name__
            raise IncompatibleStateError(f"bigram state must have keys {sorted(expected)}, got {got}")
        if state["format"] != STATE_FORMAT:
            raise IncompatibleStateError(f"unsupported bigram state format {state['format']!r} (expected {STATE_FORMAT!r})")
        try:
            saved_vocab = Vocabulary.from_dict(state["vocab"])
        except VocabularyError as exc:
            raise IncompatibleStateError(f"saved vocabulary is invalid: {exc}") from exc
        if saved_vocab != self.vocab:
            raise IncompatibleStateError(
                "vocabulary mismatch: saved "
                f"{saved_vocab.boundary!r}+{''.join(saved_vocab.chars)!r}, model has "
                f"{self.vocab.boundary!r}+{''.join(self.vocab.chars)!r}"
            )
        if validate_alpha(state["alpha"]) != self.alpha:
            raise IncompatibleStateError(f"alpha mismatch: saved {state['alpha']!r}, model has {self.alpha!r}")
        counts = state["counts"]
        V = self.num_classes
        if not isinstance(counts, torch.Tensor) or counts.shape != (V, V):
            shape = tuple(counts.shape) if isinstance(counts, torch.Tensor) else type(counts).__name__
            raise IncompatibleStateError(f"counts must be a [{V}, {V}] tensor, got {shape}")
        if counts.dtype != torch.int64:
            raise IncompatibleStateError(f"counts must be int64, got {counts.dtype}")
        if bool((counts < 0).any()):
            raise IncompatibleStateError("counts must be non-negative")
        try:
            check_fitted_counts(counts.cpu())
        except ValueError as exc:
            raise IncompatibleStateError(str(exc)) from exc
        self._set_counts(counts.clone())

    def save(self, path: str | Path, *, config: Mapping[str, Any] | None = None) -> Path:
        """Save the fitted model as a ``"final"`` checkpoint via :mod:`utils.checkpoint`.

        A fitted count model has no training trajectory, so it is neither a
        "best" (selected by a dev metric) nor a "resume" checkpoint. ``config``
        may be the fully resolved experiment config (``config_to_dict(cfg)``).
        By default only this model's ``BigramConfig`` is stored.
        """
        self._require_fitted()
        return save_checkpoint(  # type: ignore[arg-type]
            path,
            kind="final",
            model=self,  # duck-typed: the checkpoint utility only needs state_dict()
            config=dict(config) if config is not None else {"model": {"name": MODEL_NAME, "params": {"alpha": self.alpha}}},
            vocab=self.vocab.to_dict(),
            extra={
                "num_training_names": int(self._counts[0].sum()),
                "num_training_transitions": int(self._counts.sum()),
            },
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        expected_vocab: Vocabulary | None = None,
    ) -> BigramLanguageModel:
        """Load a model saved with :meth:`save`.

        The vocabulary and ``alpha`` come from the checkpoint. If
        ``expected_vocab`` is given (e.g. rebuilt from the current training
        split), it must match exactly, otherwise :class:`IncompatibleStateError`
        is raised. Uses ``torch.load(weights_only=True)``.
        """
        ckpt = load_checkpoint(path, map_location="cpu", expected_kind="final")
        if ckpt.get("model_class") != cls.__name__:
            raise IncompatibleStateError(f"{path} holds a {ckpt.get('model_class')!r}, not a {cls.__name__}")
        state = ckpt["model_state"]
        if not isinstance(state, Mapping) or "vocab" not in state or "alpha" not in state:
            raise IncompatibleStateError(f"{path} does not contain a bigram state")
        vocab = Vocabulary.from_dict(state["vocab"])
        if ckpt.get("vocab") is not None and Vocabulary.from_dict(ckpt["vocab"]) != vocab:
            raise IncompatibleStateError(f"{path}: checkpoint vocabulary metadata disagrees with the model state")
        if expected_vocab is not None and expected_vocab != vocab:
            raise IncompatibleStateError(
                f"vocabulary mismatch: checkpoint has {''.join(vocab.chars)!r}, "
                f"expected {''.join(expected_vocab.chars)!r}"
            )
        model = cls(vocab, alpha=state["alpha"], device=device)
        restore_model(model, ckpt, strict=True)  # type: ignore[arg-type]  # calls model.load_state_dict
        return model

    # ---------------------------------------------------------------- misc
    def to(self, device: str | torch.device) -> BigramLanguageModel:
        """Move fitted tensors to ``device`` (resolved with ``resolve_device``) and return ``self``."""
        self.device = resolve_device(str(device))
        if self._counts is not None:
            self._counts = self._counts.to(self.device)
            self._probs = self._probs.to(self.device)
            self._log_probs = self._log_probs.to(self.device)
        return self

    def __repr__(self) -> str:
        status = f"fitted on {int(self._counts[0].sum())} names" if self._counts is not None else "not fitted"
        return f"BigramLanguageModel(V={self.num_classes}, alpha={self.alpha}, device={self.device}, {status})"
