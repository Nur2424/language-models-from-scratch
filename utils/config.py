"""Typed, validated experiment configuration built from YAML.

Resolution order
----------------
1. ``configs/base.yaml``           shared defaults for every experiment
2. ``configs/models/<name>.yaml``  model choice + model-specific parameters
3. overrides ``["data.split_seed=7", "model.params.hidden_size=128"]``

Later sources are deep-merged over earlier ones (mappings merge, every other
value — including lists — is replaced). The merged mapping is then converted
into frozen dataclasses with strict validation:

* unknown keys raise :class:`ConfigError` naming the full path and suggesting
  close matches (``data.split_sed — did you mean 'split_seed'?``);
* values are type-checked (``bool`` is never accepted as ``int``, ``int`` is
  accepted for ``float``); ``Literal`` fields list their allowed values;
* range checks live in each section's ``__post_init__``;
* duplicate keys inside one YAML file are an error (PyYAML would silently keep
  the last one).

Model-specific parameters
-------------------------
Architectures do not share a schema. ``model.params`` is kept as a plain dict
unless a dataclass is passed as ``model_config_cls``; each model module will
define its own dataclass (e.g. ``LSTMConfig``) and the entry point passes it
explicitly, so there is no hidden global registry::

    cfg = load_config(default_config_paths("lstm"), model_config_cls=LSTMConfig)

Training-loop settings (optimiser, schedule, batch size, steps) will be added
as a new section by the training-pipeline task.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import re
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

from utils.data import DUPLICATE_POLICIES, validate_fractions
from utils.device import validate_device_spec
from utils.diagnostics.schedule import DiagnosticsSchedule
from utils.generation import SamplingError, validate_sampling_params
from utils.io import atomic_write_json, to_jsonable
from utils.paths import CONFIGS_DIR, resolve_path
from utils.reproducibility import validate_seed

T = TypeVar("T")
_MODEL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types or invalid values (message includes the key path)."""


# ================================================================ sections
@dataclass(frozen=True)
class RunConfig:
    """Identification of the experiment (not hyperparameters)."""

    name: str | None = None
    results_root: str = "results"
    notes: str = ""


@dataclass(frozen=True)
class DataConfig:
    path: str = "data/names.txt"
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 42
    duplicate_policy: Literal["group", "dedupe", "none"] = "group"

    def __post_init__(self) -> None:
        validate_fractions(self.split_fractions)
        validate_seed(self.split_seed)
        if self.duplicate_policy not in DUPLICATE_POLICIES:  # pragma: no cover - Literal check runs first
            raise ValueError(f"duplicate_policy must be one of {DUPLICATE_POLICIES}")


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int = 0
    deterministic: bool = False
    warn_only: bool = True

    def __post_init__(self) -> None:
        validate_seed(self.seed)


@dataclass(frozen=True)
class DeviceConfig:
    device: str = "auto"
    dtype: Literal["float32", "float64", "float16", "bfloat16"] = "float32"

    def __post_init__(self) -> None:
        validate_device_spec(self.device)


@dataclass(frozen=True)
class LoggingConfig:
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console: bool = True
    jsonl: bool = True


@dataclass(frozen=True)
class CheckpointConfig:
    save_best: bool = True
    save_final: bool = True
    resume_every_n_steps: int | None = None

    def __post_init__(self) -> None:
        if self.resume_every_n_steps is not None and self.resume_every_n_steps < 1:
            raise ValueError("resume_every_n_steps must be >= 1 or null")


@dataclass(frozen=True)
class GenerationConfig:
    """Neutral sampling defaults; architecture-specific limits live with each model."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    num_samples: int = 20
    max_new_tokens: int = 32

    def __post_init__(self) -> None:
        try:
            validate_sampling_params(self.temperature, self.top_k, self.top_p)
        except SamplingError as exc:
            raise ValueError(str(exc)) from None
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")


@dataclass(frozen=True)
class ModelSpec:
    """Which model to build and its architecture-specific parameters."""

    name: str
    params: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _MODEL_NAME_RE.match(self.name):
            raise ValueError(f"model.name must match {_MODEL_NAME_RE.pattern}, got {self.name!r}")
        if not (isinstance(self.params, Mapping) or dataclasses.is_dataclass(self.params)):
            raise ValueError("model.params must be a mapping or a model-config dataclass")


@dataclass(frozen=True)
class ExperimentConfig:
    """The fully resolved configuration of one experiment."""

    model: ModelSpec
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    diagnostics: DiagnosticsSchedule = field(default_factory=DiagnosticsSchedule)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


# ============================================================ type building
def _fmt(path: str) -> str:
    return path or "<root>"


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


def _build(tp: Any, value: Any, path: str) -> Any:
    """Convert plain data ``value`` into an instance of type annotation ``tp``."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp):
        return _build_dataclass(tp, value, path)
    if origin is typing.Union or origin is types.UnionType:
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{_fmt(path)}: null is not allowed (expected {_type_name(tp)})")
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _build(arg, value, path)
            except ConfigError:
                continue
        raise ConfigError(f"{_fmt(path)}: {value!r} does not match {_type_name(tp)}")
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{_fmt(path)}: {value!r} is not one of {list(args)}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{_fmt(path)}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{_fmt(path)}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{_fmt(path)}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{_fmt(path)}: expected a string, got {value!r}")
        return value
    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{_fmt(path)}: expected a list, got {value!r}")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ConfigError(f"{_fmt(path)}: expected {len(args)} items, got {len(value)}")
            return tuple(_build(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
        item = args[0] if args else Any
        built = [_build(item, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(built) if origin is tuple else built
    if origin in (dict, Mapping) or tp in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{_fmt(path)}: expected a mapping, got {value!r}")
        vt = args[1] if len(args) == 2 else Any
        return {str(k): _build(vt, v, f"{path}.{k}") for k, v in value.items()}
    raise ConfigError(f"{_fmt(path)}: unsupported annotation {_type_name(tp)} in config schema")


def _build_dataclass(cls: type[T], value: Any, path: str) -> T:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{_fmt(path)}: expected a mapping for {cls.__name__}, got {value!r}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = [k for k in value if k not in fields]
    if unknown:
        msgs = []
        for k in unknown:
            close = difflib.get_close_matches(str(k), list(fields), n=1)
            hint = f" — did you mean {close[0]!r}?" if close else ""
            msgs.append(f"{path + '.' if path else ''}{k}{hint}")
        raise ConfigError(f"unknown config key(s): {', '.join(msgs)} (allowed in {_fmt(path)}: {sorted(fields)})")
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        sub = f"{path}.{name}" if path else name
        if name in value:
            kwargs[name] = _build(hints[name], value[name], sub)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{sub}: required key is missing")
    try:
        return cls(**kwargs)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{_fmt(path)}: {exc}") from exc


def build_config(data: Mapping[str, Any], *, model_config_cls: type | None = None) -> ExperimentConfig:
    """Validate a merged mapping and build an :class:`ExperimentConfig`.

    If ``model_config_cls`` (a dataclass) is given, ``model.params`` is built
    as that class with the same strict key/type checking.
    """
    if model_config_cls is not None and not dataclasses.is_dataclass(model_config_cls):
        raise TypeError("model_config_cls must be a dataclass type")
    cfg = _build_dataclass(ExperimentConfig, data, "")
    if model_config_cls is not None:
        params = _build_dataclass(model_config_cls, cfg.model.params, "model.params")
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, params=params))
    return cfg


# ============================================================== YAML + merge
class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    out: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in out:
            raise ConfigError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        out[key] = loader.construct_object(value_node, deep=deep)
    return out


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping (empty file → ``{}``). Errors name the file."""
    file = resolve_path(path)
    if not file.is_file():
        raise ConfigError(f"config file not found: {file}")
    try:
        data = yaml.load(file.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except ConfigError as exc:
        raise ConfigError(f"{file}: {exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{file}: invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{file}: top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: mappings merge recursively, all other values are replaced."""
    out: dict[str, Any] = {k: (deep_merge(v, {}) if isinstance(v, Mapping) else v) for k, v in base.items()}
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deep_merge(v, {}) if isinstance(v, Mapping) else v
    return out


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse ``"a.b.c=value"``; the value is parsed as YAML (``7``, ``true``, ``null``, ``[1, 2]``)."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} must look like key.path=value")
    key, raw = text.split("=", 1)
    keys = key.strip().split(".")
    if not all(k and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k) for k in keys):
        raise ConfigError(f"override {text!r} has an invalid key path {key!r}")
    try:
        value = yaml.load(raw, Loader=_StrictLoader) if raw.strip() else ""
    except yaml.YAMLError as exc:
        raise ConfigError(f"override {text!r}: invalid value ({exc})") from exc
    return keys, value


def apply_overrides(data: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides to a mapping (new keys are validated later)."""
    out = deep_merge(data, {})
    for text in overrides:
        keys, value = parse_override(text)
        node: dict[str, Any] = out
        for i, k in enumerate(keys[:-1]):
            nxt = node.setdefault(k, {})
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {text!r}: {'.'.join(keys[: i + 1])} is not a mapping")
            node = nxt
        node[keys[-1]] = value
    return out


def default_config_paths(model_name: str) -> list[Path]:
    """``[configs/base.yaml, configs/models/<model_name>.yaml]`` (both must exist)."""
    paths = [CONFIGS_DIR / "base.yaml", CONFIGS_DIR / "models" / f"{model_name}.yaml"]
    for p in paths:
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
    return paths


def load_config(
    paths: str | Path | Sequence[str | Path],
    *,
    overrides: Sequence[str] = (),
    model_config_cls: type | None = None,
) -> ExperimentConfig:
    """Load, merge (in order), override and validate configuration files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ConfigError("at least one config file is required")
    merged: dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, load_yaml(p))
    merged = apply_overrides(merged, overrides)
    return build_config(merged, model_config_cls=model_config_cls)


# ============================================================ serialisation
def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Plain, JSON-ready dict of the resolved config (tuples become lists).

    ``build_config(config_to_dict(cfg), model_config_cls=...)`` reproduces ``cfg``.
    """
    return to_jsonable(cfg)


def config_hash(cfg: ExperimentConfig) -> str:
    """SHA-256 of the canonical JSON form; equal configs give equal hashes."""
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_resolved_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    """Atomically write the resolved config as JSON (e.g. a run's ``config.json``)."""
    return atomic_write_json(path, config_to_dict(cfg))
