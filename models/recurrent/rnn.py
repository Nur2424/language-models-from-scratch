"""Model 6: single-layer vanilla (Elman) RNN character language model.

Specification: ``models/recurrent/rnn_spec.md`` v0.2 (approved, D-44), as integrated by
``models/recurrent/rnn_integration_spec.md`` v0.2 (approved, D-48). Implementation milestone
MS2 of ``models/recurrent/rnn_implementation_plan.md`` v0.2 (approved, D-49).

Architecture and model behaviour only (UMIS §12): training lives in ``rnn_training.py`` and
research diagnostics in ``rnn_diagnostics.py``.

Mathematics (§4.1), per name, h_0 = 0, t = 1 … T::

    a_t = U[x̂_t] + W h_{t−1} + b
    h_t = tanh(a_t)        where position t is valid (m_t = 1)
    h_t = h_{t−1}          where position t is padding (m_t = 0): the state is frozen
    z_t = V h_t + c        (logits; padded positions give the frozen-state logits, §4.1)

x̂_t is the input id at valid positions and 0 (a safe in-range id) at padded positions; PAD
(id 27) is never looked up. In code states are row vectors: ``a = U[x̂] + h @ Wᵀ + b`` and
``z = h @ Vᵀ + c``. Full backpropagation through time is PyTorch autograd through this explicit
loop; no recurrent module, fused kernel or packed-sequence helper is used (§6.2).

The recurrence equation exists only in :meth:`RNNLanguageModel._step` and the output
projection only in :meth:`RNNLanguageModel._logits` (§6.3, M6-IMP2). ``forward``, ``generate``
and the research diagnostics all go through them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn

from utils.checkpoint import load_checkpoint, restore_model
from utils.generation import sample_next_token
from utils.reproducibility import make_generator
from utils.vocabulary import Vocabulary, VocabularyError

MODEL_NAME = "rnn"


class IncompatibleCheckpointError(ValueError):
    """Raised when a checkpoint cannot be loaded into an :class:`RNNLanguageModel` (§14).

    Model-local on purpose (precedent ``mlp.py``)."""


# =================================================================== configuration (§9.1)
def _check_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return value


def _check_positive_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class RNNConfig:
    """Architecture configuration (``model.params`` in ``configs/models/rnn.yaml``; §9.1).

    hidden_dim:
        H, the hidden size (approved: 128, M6-ARC7).
    recurrent_gain:
        g in W = g·Q (M6-INI1). Provisional value 1.0 (D-41); non-default values only under
        an explicit decision.
    input_init_variance:
        s² in U ~ N(0, s²) (M6-INI2). Provisional value 0.5 (D-41).
    """

    hidden_dim: int = 128
    recurrent_gain: float = 1.0
    input_init_variance: float = 0.5

    def __post_init__(self) -> None:
        _check_positive_int(self.hidden_dim, "hidden_dim")
        object.__setattr__(self, "recurrent_gain", _check_positive_real(self.recurrent_gain, "recurrent_gain"))
        object.__setattr__(self, "input_init_variance",
                           _check_positive_real(self.input_init_variance, "input_init_variance"))


# =================================================================== records
@dataclass(frozen=True)
class RNNSample:
    """One generated sample (§15; same semantics as ``MLPSample``, duplicated on purpose).

    text:
        Characters between BOS and EOS, or up to ``max_length``. Never contains ``.``.
    finished:
        ``True`` iff EOS was drawn; ``False`` means the sample was truncated at ``max_length``.
    """

    text: str
    finished: bool


class RecurrenceOutput(NamedTuple):
    """Result of :meth:`RNNLanguageModel._recurrence` (§6.3).

    states:
        [B, T, H] hidden states h_t; frozen at h_{n_β} after a row's last valid step.
    valid:
        [B, T] bool validity m (``inputs != PAD``).
    preactivations:
        [B, T, H] pre-activations a_t if requested, else ``None``. At padded positions a_t is
        computed from the safe id 0 and has no meaning (§5).
    """

    states: torch.Tensor
    valid: torch.Tensor
    preactivations: torch.Tensor | None


# =================================================================== initialisation (§7.1)
def _haar_orthogonal(hidden_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Haar-distributed orthogonal Q ∈ ℝ^{H×H} in float64 on CPU (§7.1 steps 1 – 3).

    Z_ij ~ N(0, 1) (float64, from ``generator``); (Q, R) = QR(Z) (reduced); Q ← Q·diag(σ) with
    σ_i = sign(R_ii) and σ_i = +1 where R_ii = 0 (Mezzadri 2007).
    """
    z = torch.randn(hidden_dim, hidden_dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(z, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


# =================================================================== model
class RNNLanguageModel(nn.Module):
    """Single-layer Elman RNN over characters (§3 – §8).

    Parameters
    ----------
    vocab:
        Shared vocabulary. K = ``vocab.num_classes`` (27) sizes U and V; PAD (``vocab.pad_id``,
        27) is input-only and never looked up.
    config:
        Validated :class:`RNNConfig` (defaults: H = 128, g = 1.0, s² = 0.5).
    generator:
        CPU ``torch.Generator`` used for every initialisation draw; by convention
        ``make_generator(derive_seed(seed, "init"))``. Required. Nothing else is random.

    Learned state: five raw parameters, no buffers (§4.3, §14)::

        input_lookup      U  [K, H]
        recurrent_weight  W  [H, H]
        hidden_bias       b  [H]
        output_weight     V  [K, H]
        output_bias       c  [K]

    Parameters are created in float32 on CPU; ``model.to(torch.float64)`` gives a float64
    model and the forward path follows the parameter dtype (§5, §10).
    """

    def __init__(
        self,
        vocab: Vocabulary,
        config: RNNConfig = RNNConfig(),  # noqa: B008  (frozen, immutable dataclass: a shared default is safe)
        *,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if not isinstance(vocab, Vocabulary):
            raise TypeError(f"vocab must be a utils.vocabulary.Vocabulary, got {type(vocab).__name__}")
        if not isinstance(config, RNNConfig):
            raise TypeError(f"config must be an RNNConfig, got {type(config).__name__}")
        if not isinstance(generator, torch.Generator):
            raise TypeError(f"generator must be a torch.Generator, got {type(generator).__name__}")
        if generator.device.type != "cpu":
            raise ValueError(f"generator must be a CPU generator (initialisation runs on CPU), got {generator.device}")
        self.vocab = vocab
        self.config = config
        K, H = vocab.num_classes, config.hidden_dim

        # §7.1, in the fixed draw order: W first, then U; all draws float64 on CPU, then cast.
        q = _haar_orthogonal(H, generator)                                       # steps 1 – 3
        w = (config.recurrent_gain * q).to(torch.float32)                         # step 4
        u = (math.sqrt(config.input_init_variance)
             * torch.randn(K, H, generator=generator, dtype=torch.float64)).to(torch.float32)  # step 5
        # Raw parameters (M6-ARC9); no PyTorch default initialisation is ever run (§6.2).
        self.input_lookup = nn.Parameter(u)                                       # U
        self.recurrent_weight = nn.Parameter(w)                                   # W
        self.hidden_bias = nn.Parameter(torch.zeros(H, dtype=torch.float32))      # b  (step 6)
        self.output_weight = nn.Parameter(torch.zeros(K, H, dtype=torch.float32))  # V (step 6)
        self.output_bias = nn.Parameter(torch.zeros(K, dtype=torch.float32))      # c  (step 6)

    # ------------------------------------------------------------ properties
    @property
    def num_classes(self) -> int:
        return self.vocab.num_classes

    @property
    def hidden_dim(self) -> int:
        return self.config.hidden_dim

    @property
    def device(self) -> torch.device:
        return self.recurrent_weight.device

    def extra_repr(self) -> str:
        c = self.config
        return f"K={self.num_classes}, H={c.hidden_dim}, g={c.recurrent_gain}, s2={c.input_init_variance}"

    # ------------------------------------------------------------ core (§6.3, §8)
    def _step(self, h_prev: torch.Tensor, x_hat_t: torch.Tensor,
              m_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step for a batch: the only place the recurrence equation exists.

        h_prev [B, H]; x_hat_t [B] long ids in 0 … K−1 (safe id at padded rows);
        m_t [B] bool validity. Returns (h_t, a_t) with
        a_t = U[x̂_t] + h_prev Wᵀ + b, h_t = tanh(a_t) where m_t, else h_prev (state frozen).
        """
        a_t = self.input_lookup[x_hat_t] + h_prev @ self.recurrent_weight.T + self.hidden_bias
        h_t = torch.where(m_t.unsqueeze(1), torch.tanh(a_t), h_prev)
        return h_t, a_t

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        """z = h Vᵀ + c for any leading shape ([B, H] or [B, T, H]): the only output projection."""
        return h @ self.output_weight.T + self.output_bias

    def _recurrence(self, inputs: torch.Tensor, *, return_preactivations: bool = False) -> RecurrenceOutput:
        """Validate ``inputs`` (§5.1), derive m and x̂, and run ``_step`` over t = 1 … T (§8).

        Pre-activations are collected only when requested (never on the training path).
        With gradients enabled the graph through every step is kept; nothing is detached.
        """
        self._validate_inputs(inputs)
        valid = inputs != self.vocab.pad_id                                    # m      [B, T]
        x_hat = torch.where(valid, inputs, torch.zeros_like(inputs))           # x̂     [B, T]
        B, T = inputs.shape
        h = torch.zeros(B, self.hidden_dim, dtype=self.recurrent_weight.dtype, device=self.device)  # h_0 = 0
        states: list[torch.Tensor] = []
        pre: list[torch.Tensor] | None = [] if return_preactivations else None
        for t in range(T):
            h, a = self._step(h, x_hat[:, t], valid[:, t])
            states.append(h)
            if pre is not None:
                pre.append(a)
        return RecurrenceOutput(
            states=torch.stack(states, dim=1),
            valid=valid,
            preactivations=torch.stack(pre, dim=1) if pre is not None else None,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Next-token logits for right-padded name batches (§5, §8).

        Parameters
        ----------
        inputs:
            [B, T] ``torch.long`` ids on the model's device: column 0 is BOS (0); valid ids
            0 … 26; padded positions exactly PAD (27), right-contiguous. ``B = 0`` is allowed.

        Returns
        -------
        torch.Tensor
            [B, T, K] logits in the parameter dtype. Only positions with ``inputs != PAD`` are
            meaningful; padded positions hold the frozen-state logits V h_{n_β} + c (§4.1).
        """
        return self._logits(self._recurrence(inputs).states)

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """§5.1 (M6-IMP6): raise before any computation."""
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"inputs must be a torch.Tensor, got {type(inputs).__name__}")
        if inputs.dtype != torch.long:
            raise ValueError(f"inputs must have dtype torch.long, got {inputs.dtype}")
        if inputs.ndim != 2:
            raise ValueError(f"inputs must be 2-D [B, T], got shape {tuple(inputs.shape)}")
        if inputs.shape[1] == 0:
            raise ValueError("inputs has T = 0 columns: column 0 must hold BOS")
        if inputs.device != self.device:
            raise ValueError(f"inputs are on {inputs.device} but the model is on {self.device} "
                             "(no implicit transfer)")
        if inputs.shape[0] == 0:
            return
        pad = self.vocab.pad_id
        bad = (inputs < 0) | (inputs > pad)
        if bool(bad.any()):
            row = int(bad.any(dim=1).nonzero()[0])
            raise ValueError(f"row {row}: ids must lie in 0 … {pad} (PAD = {pad})")
        not_bos = inputs[:, 0] != self.vocab.bos_id
        if bool(not_bos.any()):
            row = int(not_bos.nonzero()[0])
            raise ValueError(f"row {row}: column 0 must be BOS ({self.vocab.bos_id}), got {int(inputs[row, 0])}")
        # Right-contiguous padding: the valid positions of each row must be exactly its first
        # n_β columns, n_β = number of non-PAD ids (device-agnostic: comparisons only).
        valid = inputs != pad
        n = valid.sum(dim=1, keepdim=True)
        positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
        broken = valid != (positions < n)
        if bool(broken.any()):
            row = int(broken.any(dim=1).nonzero()[0])
            raise ValueError(f"row {row}: padding is not right-contiguous (a PAD is followed by a non-PAD id)")

    # ---------------------------------------------------------- generation (§15)
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
    ) -> list[RNNSample]:
        """Ancestral sampling from h_0 = 0 and BOS (§15).

        Per active row: h ← ``_step``(h, BOS); then repeatedly logits = ``_logits``(h), token =
        :func:`utils.generation.sample_next_token`(logits, …). EOS finishes the row
        (``finished=True``). A row that already has ``max_length`` characters gets this one
        further draw: EOS finishes it, any other token is discarded and the row stops
        truncated (``finished=False``). Otherwise the character is appended and
        h ← ``_step``(h, token). At most ``max_length + 1`` draws and ``_step`` calls per row.

        Finished rows leave the active set: their state is not advanced and the sampler is never
        called for them again. ``temperature=0`` is greedy; with ``generator=None`` PyTorch's
        default generator is consumed. Runs under ``torch.no_grad()``; parameters and
        ``self.training`` are not modified.
        """
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError(f"num_samples must be a positive int, got {num_samples!r}")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError(f"max_length must be a positive int, got {max_length!r}")
        eos, bos = self.vocab.eos_id, self.vocab.bos_id
        dev, dtype = self.device, self.recurrent_weight.dtype

        chars: list[list[int]] = [[] for _ in range(num_samples)]
        finished = [False] * num_samples
        active = list(range(num_samples))  # sample indices, aligned with the rows of h
        all_valid = torch.ones(num_samples, dtype=torch.bool, device=dev)
        h = torch.zeros(num_samples, self.hidden_dim, dtype=dtype, device=dev)
        h, _ = self._step(h, torch.full((num_samples,), bos, dtype=torch.long, device=dev), all_valid)
        while active:
            drawn = sample_next_token(
                self._logits(h), temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
            ).tolist()
            keep_rows: list[int] = []
            for row, (k, token) in enumerate(zip(active, drawn)):
                if token == eos:
                    finished[k] = True
                elif len(chars[k]) == max_length:
                    pass  # limit reached and EOS not drawn: stop, truncated (token discarded)
                else:
                    chars[k].append(token)
                    keep_rows.append(row)
            active = [active[row] for row in keep_rows]
            if active:
                rows = torch.tensor(keep_rows, dtype=torch.long, device=dev)
                tokens = torch.tensor([chars[k][-1] for k in active], dtype=torch.long, device=dev)
                h, _ = self._step(h[rows], tokens, all_valid[: len(active)])
        return [
            RNNSample(text=self.vocab.decode(ids, strip_bos=False, stop_at_eos=False), finished=done)
            for ids, done in zip(chars, finished)
        ]

    # ---------------------------------------------------------- checkpoint (§14)
    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_vocab: Vocabulary | None = None,
        map_location: str | torch.device = "cpu",
    ) -> RNNLanguageModel:
        """Load weights from a checkpoint of any kind (§14), with ``weights_only=True``.

        Requires ``model_class == "RNNLanguageModel"``, the checkpoint's ``vocab`` metadata and
        ``config["model"]`` (the :class:`RNNConfig` as plain data). Raises
        :class:`IncompatibleCheckpointError` for a different model class, missing or invalid
        vocabulary metadata, a vocabulary differing from ``expected_vocab``, a missing or invalid
        model configuration, or weights not matching it. Only weights are restored.
        """
        ckpt = load_checkpoint(path, map_location="cpu")
        if ckpt.get("model_class") != cls.__name__:
            raise IncompatibleCheckpointError(f"{path} holds a {ckpt.get('model_class')!r}, not a {cls.__name__}")
        if ckpt.get("vocab") is None:
            raise IncompatibleCheckpointError(
                f"{path} has no vocabulary metadata; save with save_checkpoint(..., vocab=vocab.to_dict())")
        try:
            vocab = Vocabulary.from_dict(ckpt["vocab"])
        except (VocabularyError, KeyError, TypeError) as exc:
            raise IncompatibleCheckpointError(f"{path}: invalid vocabulary metadata: {exc}") from exc
        if expected_vocab is not None and expected_vocab != vocab:
            raise IncompatibleCheckpointError(
                f"vocabulary mismatch: checkpoint has {vocab.boundary!r}+{''.join(vocab.chars)!r}, expected "
                f"{expected_vocab.boundary!r}+{''.join(expected_vocab.chars)!r}")
        model_cfg = (ckpt.get("config") or {}).get("model")
        if not isinstance(model_cfg, Mapping):
            raise IncompatibleCheckpointError(f"{path} has no config['model'] (the RNNConfig used to build it)")
        try:
            config = RNNConfig(**model_cfg)
        except (TypeError, ValueError) as exc:
            raise IncompatibleCheckpointError(f"{path}: invalid model config {dict(model_cfg)!r}: {exc}") from exc
        # The weights are overwritten by the checkpoint; this generator only satisfies the constructor.
        model = cls(vocab, config, generator=make_generator(0))
        try:
            restore_model(model, ckpt, strict=True)
        except RuntimeError as exc:  # e.g. a size mismatch between the weights and the config
            raise IncompatibleCheckpointError(f"{path}: weights do not match the model config: {exc}") from exc
        return model.to(map_location)
