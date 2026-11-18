"""Transformer language models for Transformer Part I (specification: ``transformer_spec.md``).

Two decoder-only, character-level, autoregressive models share every computation in this file:

* ``transformer_2017`` (Model 01): the 2017 Transformer's decoder stack without cross-attention:
  √d-scaled embeddings, sinusoidal positions, post-LayerNorm, ReLU, no attention biases.
* ``gpt2_style`` (Model 02): GPT-2's block: learned positions, pre-LayerNorm with a final
  LayerNorm, GELU (tanh approximation), biases throughout, residual projections scaled at init.

Both tie the output to the token embedding (no output bias), use H heads of size d / H, a causal
mask and a context of at most 16 positions (BOS plus 15 characters). Attention, the mask,
LayerNorm, GELU and the sinusoidal table are each written once, below, and used by both models.
No packaged attention, Transformer, LayerNorm or GELU module is used (spec §6).

Training lives outside the model (``training/transformer_study/``).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from utils.checkpoint import load_checkpoint, restore_model
from utils.generation import sample_next_token
from utils.reproducibility import make_generator
from utils.vocabulary import Vocabulary, VocabularyError

VARIANTS = ("transformer_2017", "gpt2_style")
MAX_LEN = 16
FFN_MULTIPLIER = 4
LN_EPS = {"transformer_2017": 1e-6, "gpt2_style": 1e-5}
GPT2_STD, GPT2_POS_STD = 0.02, 0.01
SINUSOID_BASE = 10_000.0


class IncompatibleCheckpointError(ValueError):
    """A checkpoint does not belong to this model class, variant, shape or vocabulary."""


# =================================================================== configuration (§9)
@dataclass(frozen=True)
class TransformerConfig:
    """Architecture only. Everything not listed is fixed per variant (spec §3)."""

    variant: str = "gpt2_style"
    d_model: int = 30
    num_layers: int = 2
    num_heads: int = 3
    max_len: int = MAX_LEN

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        for name in ("d_model", "num_layers", "num_heads", "max_len"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.d_model % self.num_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        if self.variant == "transformer_2017" and self.d_model % 2:
            raise ValueError("the sinusoidal encoding needs an even d_model")
        if self.max_len > MAX_LEN:
            raise ValueError(f"max_len must be <= {MAX_LEN}, got {self.max_len}")

    @property
    def d_k(self) -> int:
        return self.d_model // self.num_heads

    @property
    def d_ff(self) -> int:
        return FFN_MULTIPLIER * self.d_model

    @property
    def pre_ln(self) -> bool:
        return self.variant == "gpt2_style"

    @property
    def ln_eps(self) -> float:
        return LN_EPS[self.variant]


def parameter_count(config: TransformerConfig, num_classes: int = 27) -> int:
    """Closed form of spec §4.3."""
    d, L = config.d_model, config.num_layers
    if config.variant == "transformer_2017":
        return L * (12 * d * d + 9 * d) + num_classes * d
    return L * (12 * d * d + 13 * d) + num_classes * d + config.max_len * d + 2 * d


# =================================================================== shared computations (§4.2)
def causal_mask(T: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """[T, T] bool, True where the key is in the future (s > t): those scores become −∞."""
    return torch.ones(T, T, dtype=torch.bool, device=device).triu(1)


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm over the last dimension (one position's features), biased variance."""
    mu = x.mean(dim=-1, keepdim=True)
    var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta


def gelu_tanh(z: torch.Tensor) -> torch.Tensor:
    """GELU, tanh approximation, as in GPT-2's released code."""
    return 0.5 * z * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z.pow(3))))


def sinusoidal_table(max_len: int, d: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[max_len, d]: PE(p, 2i) = sin(ω_i p), PE(p, 2i+1) = cos(ω_i p), ω_i = 10000^(−2i/d)."""
    p = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)
    omega = SINUSOID_BASE ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    pe = torch.zeros(max_len, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(p * omega)
    pe[:, 1::2] = torch.cos(p * omega)
    return pe.to(dtype)


def causal_self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
                          mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-head scaled dot-product attention with a causal mask.

    q, k, v: [B, T, d] (already projected). Returns the concatenated heads [B, T, d] and the
    attention weights [B, H, T, T] (rows: queries, columns: keys).
    """
    B, T, d = q.shape
    d_k = d // num_heads

    def split(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, T, num_heads, d_k).transpose(1, 2)            # [B, H, T, d_k]

    qh, kh, vh = split(q), split(k), split(v)
    scores = qh @ kh.transpose(-2, -1) / math.sqrt(d_k)                 # [B, H, T, T]
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    heads = weights @ vh                                                # [B, H, T, d_k]
    return heads.transpose(1, 2).reshape(B, T, d), weights


# =================================================================== modules
def _xavier_uniform(fan_out: int, fan_in: int, g: torch.Generator) -> torch.Tensor:
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return torch.empty(fan_out, fan_in, dtype=torch.float64).uniform_(-bound, bound, generator=g).float()


def _normal(shape: tuple[int, ...], std: float, g: torch.Generator) -> torch.Tensor:
    return (std * torch.randn(*shape, generator=g, dtype=torch.float64)).float()


class LayerNormParams(nn.Module):
    """γ and β of one LayerNorm; the computation is :func:`layer_norm`."""

    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(x, self.gamma, self.beta, self.eps)


class TransformerBlock(nn.Module):
    """One block: causal multi-head attention and FFN, post-LN (2017) or pre-LN (GPT-2)."""

    def __init__(self, config: TransformerConfig, g: torch.Generator) -> None:
        super().__init__()
        d, d_ff, L = config.d_model, config.d_ff, config.num_layers
        self.config = config
        gpt2 = config.variant == "gpt2_style"
        # Weights are stored [out, in] and applied as x @ Wᵀ. Fixed draw order: Q, K, V, O, W1, W2.
        if gpt2:
            resid_std = GPT2_STD / math.sqrt(2 * L)
            w = [_normal((d, d), GPT2_STD, g) for _ in range(3)] + [_normal((d, d), resid_std, g),
                                                                    _normal((d_ff, d), GPT2_STD, g),
                                                                    _normal((d, d_ff), resid_std, g)]
        else:
            w = [_xavier_uniform(d, d, g) for _ in range(4)] + [_xavier_uniform(d_ff, d, g),
                                                                _xavier_uniform(d, d_ff, g)]
        self.w_q, self.w_k, self.w_v, self.w_o, self.w_1, self.w_2 = (nn.Parameter(t) for t in w)
        if gpt2:
            self.b_q, self.b_k, self.b_v, self.b_o = (nn.Parameter(torch.zeros(d)) for _ in range(4))
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = None
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.b_2 = nn.Parameter(torch.zeros(d))
        self.ln_1 = LayerNormParams(d, config.ln_eps)
        self.ln_2 = LayerNormParams(d, config.ln_eps)

    @staticmethod
    def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        y = x @ w.transpose(0, 1)
        return y if b is None else y + b

    def attention(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._linear(x, self.w_q, self.b_q)
        k = self._linear(x, self.w_k, self.b_k)
        v = self._linear(x, self.w_v, self.b_v)
        heads, weights = causal_self_attention(q, k, v, self.config.num_heads, mask)
        return self._linear(heads, self.w_o, self.b_o), weights

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        h = self._linear(x, self.w_1, self.b_1)
        h = gelu_tanh(h) if self.config.variant == "gpt2_style" else torch.relu(h)
        return self._linear(h, self.w_2, self.b_2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, record: dict[str, list] | None = None) -> torch.Tensor:
        if record is not None:
            record["residual_before_attention"].append(x)
        if self.config.pre_ln:
            a_in = self.ln_1(x)
            a, w = self.attention(a_in, mask)
            u = x + a
            if record is not None:
                record["ln_1_input"].append(x)
                record["residual_before_ffn"].append(u)
                record["ln_2_input"].append(u)
            out = u + self.ffn(self.ln_2(u))
        else:
            a, w = self.attention(x, mask)
            if record is not None:
                record["ln_1_input"].append(x + a)
            u = self.ln_1(x + a)
            f = self.ffn(u)
            if record is not None:
                record["residual_before_ffn"].append(u)
                record["ln_2_input"].append(u + f)
            out = self.ln_2(u + f)
        if record is not None:
            record["attention"].append(w)
        return out


# =================================================================== the model
@dataclass(frozen=True)
class TransformerSample:
    """One generated sample: ``text`` between BOS and EOS (or up to ``max_length``); ``finished``
    is ``True`` iff EOS was drawn."""

    text: str
    finished: bool


class TransformerLanguageModel(nn.Module):
    """Models 01 and 02 (spec §3–§8). ``forward(inputs, positions=None)`` → logits [B, T, 27]."""

    def __init__(self, vocab: Vocabulary, config: TransformerConfig = TransformerConfig(),  # noqa: B008
                 *, generator: torch.Generator) -> None:
        super().__init__()
        if not isinstance(vocab, Vocabulary):
            raise TypeError(f"vocab must be a utils.vocabulary.Vocabulary, got {type(vocab).__name__}")
        if not isinstance(config, TransformerConfig):
            raise TypeError(f"config must be a TransformerConfig, got {type(config).__name__}")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise TypeError("generator must be a CPU torch.Generator")
        self.vocab, self.config = vocab, config
        K, d = vocab.num_classes, config.d_model
        gpt2 = config.variant == "gpt2_style"
        # Fixed draw order (§7): token embedding, positions (Model 02), then each block in order.
        std_e = GPT2_STD if gpt2 else d ** -0.5
        self.token_embedding = nn.Parameter(_normal((K, d), std_e, generator))       # E [K, d], tied
        if gpt2:
            self.position_embedding = nn.Parameter(_normal((config.max_len, d), GPT2_POS_STD, generator))
            self.register_buffer("sinusoids", None, persistent=False)
        else:
            self.position_embedding = None
            self.register_buffer("sinusoids", sinusoidal_table(config.max_len, d, torch.float64), persistent=False)
        self.blocks = nn.ModuleList(TransformerBlock(config, generator) for _ in range(config.num_layers))
        self.final_ln = LayerNormParams(d, config.ln_eps) if gpt2 else None

    # ------------------------------------------------------------ properties
    @property
    def num_classes(self) -> int:
        return self.vocab.num_classes

    @property
    def device(self) -> torch.device:
        return self.token_embedding.device

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        c = self.config
        return f"variant={c.variant}, d={c.d_model}, L={c.num_layers}, H={c.num_heads}, params={self.num_parameters}"

    # ------------------------------------------------------------ validation (§5)
    def _validate(self, inputs: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or inputs.dtype != torch.long or inputs.dim() != 2:
            raise ValueError("inputs must be a [B, T] int64 tensor")
        T, pad = inputs.shape[1], self.vocab.pad_id
        if not 1 <= T <= self.config.max_len:
            raise ValueError(f"T must be in 1 … {self.config.max_len}, got {T}")
        if inputs.numel() and (inputs.min() < 0 or inputs.max() > pad):
            raise ValueError(f"input ids must be in 0 … {pad} (PAD = {pad})")
        valid = inputs != pad
        if inputs.numel() and bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("PAD must be right-contiguous: no valid id after a PAD")
        if positions is not None:
            if not isinstance(positions, torch.Tensor) or positions.dtype != torch.long \
                    or positions.shape != inputs.shape:
                raise ValueError("positions must be an int64 tensor with the shape of inputs")
            if positions.numel() and (positions.min() < 0 or positions.max() >= self.config.max_len):
                raise ValueError(f"positions must be in 0 … {self.config.max_len - 1}")
            both = valid[:, 1:] & valid[:, :-1]
            if bool(((positions[:, 1:] < positions[:, :-1]) & both).any()):
                raise ValueError("positions must be non-decreasing along valid rows")
        return valid

    # ------------------------------------------------------------ forward (§8)
    def _embed(self, inputs: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        valid = self._validate(inputs, positions)
        x_hat = torch.where(valid, inputs, torch.zeros_like(inputs))          # PAD → 0, never looked up
        if positions is None:
            positions = torch.arange(inputs.shape[1], device=inputs.device).expand_as(inputs)
        tok = self.token_embedding[x_hat]                                    # [B, T, d]
        if self.config.variant == "transformer_2017":
            return math.sqrt(self.config.d_model) * tok + self.sinusoids[positions].to(tok.dtype)
        return tok + self.position_embedding[positions]

    def _run(self, inputs: torch.Tensor, positions: torch.Tensor | None,
             record: dict[str, list] | None) -> torch.Tensor:
        h = self._embed(inputs, positions)
        mask = causal_mask(inputs.shape[1], h.device)
        for block in self.blocks:
            h = block(h, mask, record)
        if self.final_ln is not None:
            if record is not None:
                record["final_ln_input"] = h
            h = self.final_ln(h)
        return h @ self.token_embedding.transpose(0, 1)                      # tied output, no bias

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        return self._run(inputs, positions, None)

    def forward_with_internals(self, inputs: torch.Tensor, positions: torch.Tensor | None = None
                               ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Logits plus, per layer: attention weights [B, H, T, T], the residual stream before each
        sublayer and every LayerNorm input [B, T, d]. For diagnostics and tests only."""
        record: dict[str, Any] = {k: [] for k in ("attention", "residual_before_attention", "residual_before_ffn",
                                                  "ln_1_input", "ln_2_input")}
        return self._run(inputs, positions, record), record

    # ------------------------------------------------------------ generation (§15)
    @torch.no_grad()
    def generate(self, num_samples: int, *, max_length: int = MAX_LEN - 1, generator: torch.Generator | None = None,
                 temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None
                 ) -> list[TransformerSample]:
        """Ancestral sampling from BOS, same semantics as the RNN: EOS finishes a sample; a sample
        that already has ``max_length`` characters gets one further draw (EOS finishes it,
        anything else stops it unfinished). The whole prefix is re-read at every step."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError(f"num_samples must be a positive int, got {num_samples!r}")
        if not 1 <= max_length <= self.config.max_len - 1:
            raise ValueError(f"max_length must be in 1 … {self.config.max_len - 1}")
        bos, eos = self.vocab.bos_id, self.vocab.eos_id
        chars: list[list[int]] = [[] for _ in range(num_samples)]
        finished = [False] * num_samples
        active = list(range(num_samples))
        was_training = self.training
        self.eval()
        try:
            while active:
                prefix = torch.tensor([[bos, *chars[k]] for k in active], dtype=torch.long, device=self.device)
                drawn = sample_next_token(self(prefix)[:, -1, :], temperature=temperature, top_k=top_k,
                                          top_p=top_p, generator=generator).tolist()
                still = []
                for k, token in zip(active, drawn):
                    if token == eos:
                        finished[k] = True
                    elif len(chars[k]) < max_length:
                        chars[k].append(token)
                        still.append(k)
                active = still
        finally:
            self.train(was_training)
        return [TransformerSample(self.vocab.decode(ids, strip_bos=False, stop_at_eos=False), done)
                for ids, done in zip(chars, finished)]

    # ------------------------------------------------------------ checkpoint (§14)
    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, expected_vocab: Vocabulary | None = None,
                        expected_variant: str | None = None, map_location: str | torch.device = "cpu"
                        ) -> TransformerLanguageModel:
        """Rebuild from a checkpoint written with ``config={"model": model.config_dict()}`` and the
        vocabulary; refuses a different class, variant, vocabulary or shape."""
        ckpt = load_checkpoint(path, map_location="cpu")
        if ckpt.get("model_class") != cls.__name__:
            raise IncompatibleCheckpointError(f"{path} holds a {ckpt.get('model_class')!r}, not a {cls.__name__}")
        try:
            vocab = Vocabulary.from_dict(ckpt["vocab"])
        except (VocabularyError, KeyError, TypeError) as exc:
            raise IncompatibleCheckpointError(f"{path}: invalid or missing vocabulary: {exc}") from exc
        if expected_vocab is not None and expected_vocab != vocab:
            raise IncompatibleCheckpointError("vocabulary mismatch")
        model_cfg = (ckpt.get("config") or {}).get("model")
        if not isinstance(model_cfg, Mapping):
            raise IncompatibleCheckpointError(f"{path} has no config['model']")
        try:
            config = TransformerConfig(**model_cfg)
        except (TypeError, ValueError) as exc:
            raise IncompatibleCheckpointError(f"{path}: invalid model config: {exc}") from exc
        if expected_variant is not None and config.variant != expected_variant:
            raise IncompatibleCheckpointError(f"variant mismatch: {config.variant!r} != {expected_variant!r}")
        model = cls(vocab, config, generator=make_generator(0))  # weights are overwritten below
        try:
            restore_model(model, ckpt, strict=True)
        except RuntimeError as exc:
            raise IncompatibleCheckpointError(f"{path}: weights do not match the config: {exc}") from exc
        return model.to(map_location)
