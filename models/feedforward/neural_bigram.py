"""Neural bigram: bias-free softmax regression on the previous token.

Specification: ``models/feedforward/neural_bigram_spec.md`` (v0.6). Section
numbers below (§) refer to it.

Architecture (§3, §4.1)
-----------------------
One trainable table ``W ∈ ℝ^{V×V}`` (``transition_logits``), initialised to
zero. Row ``i`` holds the next-token logits after context ``i``::

    z = W[x_{t-1}]                          logits, [*S, V]
    P(x_t = j | x_{t-1} = i) = softmax(W_i)_j

This equals a bias-free linear layer applied to ``onehot(x_{t-1})``. There is
no bias, no embedding dimension and no hidden layer. The parameterisation is
unfactorized (one free logit per (context, next-token) cell). No parameter is
shared between contexts, so evidence about context ``i`` changes row ``i``
only. Within a row, softmax couples the targets.

Learning dynamics (§4.2, §4.5)
------------------------------
With ``M`` training pairs, counts ``C``, ``N_i = Σ_j C_ij`` and ``q_i = C_i / N_i``::

    ∂L/∂W_ij = (N_i / M) (p_ij − q_ij)

* Observed context, unseen target (``C_ij = 0``): the gradient is
  ``(N_i/M) p_ij > 0``, so gradient descent actively pushes that logit down
  (with no weight decay). The probability tends to 0 but never reaches it.
* Unseen context (``N_i = 0``): the gradient is zero and the row stays at its
  zero initialisation, i.e. the uniform distribution.
* Gradient rows sum to 0 and the table starts at 0, so every row of ``W``
  stays zero-mean throughout training (also with coupled weight decay).
  Logits remain zero-mean (centred) within each context row; the
  log-probabilities ``W_ij − logsumexp(W_i)`` do not.

Training procedure (§4.6), separate from the architecture
---------------------------------------------------------
:func:`train_full_batch` runs full-batch SGD (no momentum) on

    J(W) = L_data(W) + (λ/2)·‖W‖²_F,     λ = weight_decay (default 0)

using SGD's coupled weight decay, so each step is ``W ← W − η(∇L_data + λW)``.
It reports **data loss** ``L_data`` only, never ``J``. With ``λ > 0``,
``L_data`` may increase even though ``J`` decreases.

Device policy (§11)
-------------------
Parameters are created on CPU; move the model with ``model.to(device)``.
``forward`` and :func:`train_full_batch` never move data. Inputs on a
different device raise ``ValueError``.

This module is independent of ``models.statistical``. Only the tests use the
statistical bigram, as an oracle.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from utils.checkpoint import load_checkpoint, restore_model
from utils.data import make_next_token_sequences
from utils.generation import sample_next_token
from utils.metrics import token_nll
from utils.vocabulary import Vocabulary, VocabularyError

MODEL_NAME = "neural_bigram"


class IncompatibleCheckpointError(ValueError):
    """Raised when a checkpoint cannot be loaded into this model (class, vocabulary or shape)."""


# =================================================================== config
@dataclass(frozen=True)
class NeuralBigramConfig:
    """Architecture configuration (``model.params`` in ``configs/models/neural_bigram.yaml``).

    It has no fields. The architecture has no hyperparameters: ``V`` comes from
    the vocabulary. The class exists so that the config loader rejects any key
    under ``model.params`` as unknown.
    """


# ================================================================== results
@dataclass(frozen=True)
class NeuralBigramSample:
    """One generated sample (model-specific; same semantics as the statistical bigram's).

    text:
        Characters between BOS and EOS, or up to ``max_length``. Never contains ``.``.
    finished:
        ``True`` iff EOS was drawn. ``False`` means the sample was truncated at ``max_length``.
    """

    text: str
    finished: bool


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned by :func:`train_full_batch`.

    Both values are the **data loss** ``L_data`` (mean cross-entropy over the
    training pairs, nats/token), **not** the optimised objective
    ``J = L_data + (λ/2)‖W‖²``. With ``weight_decay > 0``, ``final_loss`` may
    exceed ``initial_loss`` even though ``J`` decreased. Compute ``J`` from
    ``final_loss`` and the model's parameters when needed.
    """

    initial_loss: float  # L_data(W_0), before the first update
    final_loss: float  # L_data(W_steps), after the last update


# ===================================================================== model
class NeuralBigramLanguageModel(nn.Module):
    """Bias-free softmax-regression bigram over the shared character vocabulary.

    Parameters
    ----------
    vocab:
        Shared :class:`~utils.vocabulary.Vocabulary`. ``V = vocab.num_classes``
        output classes. PAD (``vocab.pad_id == V``) is input-only and must never
        reach the model.

    Attributes
    ----------
    transition_logits:
        ``nn.Parameter`` of shape ``[V, V]``, float32, created on CPU and initialised
        to zeros. Row ``i`` holds the next-token logits after context ``i``.
    """

    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__()
        if not isinstance(vocab, Vocabulary):
            raise TypeError(f"vocab must be a utils.vocabulary.Vocabulary, got {type(vocab).__name__}")
        self.vocab = vocab
        V = vocab.num_classes
        # Zero initialisation (§7): every row starts as the uniform distribution and the initial
        # loss is exactly log V. No RNG is involved, and row sums start at 0, which keeps them at 0.
        self.transition_logits = nn.Parameter(torch.zeros(V, V, dtype=torch.float32))

    @classmethod
    def from_config(cls, vocab: Vocabulary, config: NeuralBigramConfig) -> NeuralBigramLanguageModel:
        """Build from the (empty) validated architecture config."""
        if not isinstance(config, NeuralBigramConfig):
            raise TypeError(f"config must be a NeuralBigramConfig, got {type(config).__name__}")
        return cls(vocab)

    @property
    def num_classes(self) -> int:
        """``V``: number of output classes (boundary token plus characters)."""
        return self.vocab.num_classes

    @property
    def device(self) -> torch.device:
        return self.transition_logits.device

    def extra_repr(self) -> str:
        return f"V={self.num_classes}, params={self.num_classes ** 2}, bias=False"

    # ------------------------------------------------------------ forward
    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        """Next-token logits for each previous-token id.

        Parameters
        ----------
        contexts:
            int64 tensor of any shape ``[*S]`` with ids in ``[0, V)``, on the
            model's device. PAD and out-of-range ids raise ``ValueError``.

        Returns
        -------
        torch.Tensor
            ``[*S, V]`` float32 **logits** (not probabilities): ``W[contexts]``.
            Gradients flow only into the selected rows.
        """
        _check_token_ids(contexts, num_classes=self.num_classes, pad_id=self.vocab.pad_id,
                         device=self.device, what="contexts")
        return self.transition_logits[contexts]

    @torch.no_grad()
    def transition_probabilities(self) -> torch.Tensor:
        """``softmax(W, dim=1)``: ``[V, V]`` float32. Row ``i`` is ``P(· | i)``. No gradient."""
        return torch.softmax(self.transition_logits, dim=1)

    @torch.no_grad()
    def transition_log_probabilities(self) -> torch.Tensor:
        """``log_softmax(W, dim=1)``: ``[V, V]`` float32. No gradient."""
        return torch.log_softmax(self.transition_logits, dim=1)

    # ---------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        *,
        max_length: int,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[NeuralBigramSample]:
        """Ancestral sampling ``x_t ~ softmax(W[x_{t-1}])``, starting from BOS (§15).

        Unfinished samples are drawn in parallel, one draw per active sample per
        step. A sample stops when EOS is drawn. ``max_length`` is the maximum
        number of ordinary characters. A sample that reaches it gets one more
        draw: EOS there gives a complete name (``finished=True``); anything else
        returns the sample truncated (``finished=False``). So there are at most
        ``max_length + 1`` draws per sample.

        Sampling uses :func:`utils.generation.sample_next_token`. With the
        default ``temperature``/``top_k``/``top_p`` it follows the model
        distribution exactly; ``temperature=0`` is greedy. Pass an explicit
        ``generator`` for reproducibility (the project convention is a CPU
        generator from ``make_generator(derive_seed(seed, "generation"))``).
        Parameters and ``self.training`` are not modified.
        """
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError(f"num_samples must be a positive int, got {num_samples!r}")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError(f"max_length must be a positive int, got {max_length!r}")
        eos = self.vocab.eos_id
        prev = torch.full((num_samples,), self.vocab.bos_id, dtype=torch.long, device=self.device)
        chars: list[list[int]] = [[] for _ in range(num_samples)]
        finished = [False] * num_samples
        active = list(range(num_samples))
        while active:
            logits = self(prev[active])  # [A, V]
            drawn = sample_next_token(
                logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
            ).tolist()
            still_active = []
            for k, token in zip(active, drawn):
                if token == eos:
                    finished[k] = True
                elif len(chars[k]) == max_length:
                    pass  # limit reached and EOS not drawn: stop, truncated
                else:
                    chars[k].append(token)
                    still_active.append(k)
            if still_active:
                prev[still_active] = torch.tensor([chars[k][-1] for k in still_active], device=self.device)
            active = still_active
        return [
            NeuralBigramSample(text=self.vocab.decode(ids, strip_bos=False, stop_at_eos=False), finished=done)
            for ids, done in zip(chars, finished)
        ]

    # ---------------------------------------------------------- checkpoint
    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_vocab: Vocabulary | None = None,
        map_location: str | torch.device = "cpu",
    ) -> NeuralBigramLanguageModel:
        """Load model weights from a shared checkpoint of any kind (§14).

        Uses ``torch.load(weights_only=True)``. Requires the checkpoint's
        ``vocab`` metadata (saved with ``save_checkpoint(vocab=vocab.to_dict())``)
        and rejects a different model class, a vocabulary that differs from
        ``expected_vocab``, or a weight shape that does not match the
        vocabulary. Only weights are restored. Optimiser state, step and RNG
        are the caller's job (``utils.checkpoint.restore_training_state``).
        The model is returned on ``map_location``.
        """
        ckpt = load_checkpoint(path, map_location="cpu")
        if ckpt.get("model_class") != cls.__name__:
            raise IncompatibleCheckpointError(f"{path} holds a {ckpt.get('model_class')!r}, not a {cls.__name__}")
        if ckpt.get("vocab") is None:
            raise IncompatibleCheckpointError(
                f"{path} has no vocabulary metadata; save with save_checkpoint(..., vocab=vocab.to_dict())"
            )
        try:
            vocab = Vocabulary.from_dict(ckpt["vocab"])
        except VocabularyError as exc:
            raise IncompatibleCheckpointError(f"{path}: invalid vocabulary metadata: {exc}") from exc
        if expected_vocab is not None and expected_vocab != vocab:
            raise IncompatibleCheckpointError(
                f"vocabulary mismatch: checkpoint has {vocab.boundary!r}+{''.join(vocab.chars)!r}, expected "
                f"{expected_vocab.boundary!r}+{''.join(expected_vocab.chars)!r}"
            )
        model = cls(vocab)
        try:
            restore_model(model, ckpt, strict=True)
        except RuntimeError as exc:  # e.g. size mismatch between saved weights and the vocabulary
            raise IncompatibleCheckpointError(f"{path}: weights do not match the vocabulary: {exc}") from exc
        return model.to(map_location)


# ================================================================ helpers
def _check_token_ids(
    ids: torch.Tensor, *, num_classes: int, pad_id: int, device: torch.device, what: str
) -> None:
    """Validate an id tensor: type, int64, device, then the range ``[0, V)``.

    The range check costs one device-to-host synchronisation (accepted, D8).
    """
    if not isinstance(ids, torch.Tensor):
        raise TypeError(f"{what} must be a torch.Tensor, got {type(ids).__name__}")
    if ids.dtype != torch.int64:
        raise TypeError(f"{what} must be int64 token ids, got {ids.dtype}")
    if ids.device != device:
        raise ValueError(f"{what} are on {ids.device} but the model is on {device}; move them explicitly")
    if ids.numel() == 0:
        return
    out_of_range = (ids < 0) | (ids >= num_classes)
    if bool(out_of_range.any()):
        if bool((ids == pad_id).any()):
            raise ValueError(f"{what} contain the PAD id {pad_id}: padding must not reach the model")
        bad = sorted(set(ids[out_of_range].tolist()))[:5]
        raise ValueError(f"{what} contain ids {bad} outside [0, {num_classes})")


def make_bigram_pairs(names: Sequence[str], vocab: Vocabulary) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat ``(previous, next)`` training pairs for ``names`` (§4.6, D3).

    Built from the shared :func:`utils.data.make_next_token_sequences` by
    keeping only valid positions (``inputs[mask]``, ``targets[mask]``), so
    neither PAD nor ``IGNORE_INDEX`` can appear. Each name of length ``L``
    gives ``L + 1`` pairs (BOS → x₁, …, x_L → EOS), in name order.

    Returns
    -------
    (contexts, targets):
        Two ``[M]`` int64 CPU tensors, with ``M = Σ (len(name) + 1)``.

    Raises ``TypeError``/``ValueError`` for an empty list, empty names or a
    single string, and ``VocabularyCoverageError`` for unknown characters.
    """
    if isinstance(names, str):
        raise TypeError(f"names must be a sequence of names, not a single string {names!r}")
    if len(names) == 0:
        raise ValueError("names is empty")
    for pos, name in enumerate(names):
        if not isinstance(name, str):
            raise TypeError(f"names[{pos}] must be a string, got {type(name).__name__}")
        if name == "":
            raise ValueError(f"names[{pos}] is an empty name; every name needs at least one character")
    seq = make_next_token_sequences(names, vocab)
    return seq.inputs[seq.mask], seq.targets[seq.mask]


def train_full_batch(
    model: NeuralBigramLanguageModel,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    *,
    lr: float,
    steps: int,
    weight_decay: float = 0.0,
) -> TrainingResult:
    """Reference training procedure: full-batch SGD on all pairs (§4.6).

    Each step::

        optimizer.zero_grad(set_to_none=True)
        L_data = token_nll(model(contexts), targets, reduction="mean")
        L_data.backward()
        optimizer.step()        # W ← W − lr·(∇L_data + weight_decay·W)

    with ``torch.optim.SGD(lr=lr, momentum=0, weight_decay=weight_decay)``.
    SGD's weight decay is coupled, so the optimised objective is
    ``J = L_data + (weight_decay/2)·‖W‖²``. The procedure is deterministic: no
    data RNG and no initialisation RNG. Plain SGD keeps no state between
    steps, so ``k`` calls with ``steps=1`` equal one call with ``steps=k``.

    Parameters
    ----------
    model:
        Trained in place.
    contexts, targets:
        ``[M]`` int64 pairs (``M ≥ 1``) on the model's device, from :func:`make_bigram_pairs`.
    lr:
        Step size ``η`` (finite, > 0). ``η < 2/(ℓ + λ)`` with ``ℓ ≤ max_i N_i/(2M)``
        guarantees ``J`` decreases monotonically (a sufficient condition only).
    steps:
        Number of full-batch updates (int ≥ 1).
    weight_decay:
        Coupled L2 coefficient ``λ`` (finite, ≥ 0). Default 0: no regularisation.

    Returns
    -------
    TrainingResult
        ``L_data`` before the first update and after the last update. **Data loss
        only**, never ``J``.
    """
    if not isinstance(model, NeuralBigramLanguageModel):
        raise TypeError(f"model must be a NeuralBigramLanguageModel, got {type(model).__name__}")
    _check_positive_finite(lr, "lr")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"steps must be an int >= 1, got {steps!r}")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) \
            or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(f"weight_decay must be a finite number >= 0, got {weight_decay!r}")
    for name, t in (("contexts", contexts), ("targets", targets)):
        if not isinstance(t, torch.Tensor) or t.ndim != 1:
            raise ValueError(f"{name} must be a 1-D tensor of pairs, got {type(t).__name__} "
                             f"{tuple(t.shape) if isinstance(t, torch.Tensor) else ''}")
    if contexts.shape != targets.shape:
        raise ValueError(f"contexts {tuple(contexts.shape)} and targets {tuple(targets.shape)} differ in shape")
    if contexts.numel() == 0:
        raise ValueError("cannot train on zero pairs")
    # contexts are validated by every forward call; targets once here (IGNORE_INDEX is not allowed in pairs)
    _check_token_ids(targets, num_classes=model.num_classes, pad_id=model.vocab.pad_id,
                     device=model.device, what="targets")

    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.0, weight_decay=float(weight_decay))
    initial_loss: float | None = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        data_loss = token_nll(model(contexts), targets, reduction="mean")
        if initial_loss is None:
            initial_loss = float(data_loss.detach())
        data_loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(token_nll(model(contexts), targets, reduction="mean"))
    return TrainingResult(initial_loss=initial_loss, final_loss=final_loss)


def _check_positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
