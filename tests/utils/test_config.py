import json
from dataclasses import dataclass
from typing import Literal

import pytest

from utils.config import (
    ConfigError,
    ExperimentConfig,
    apply_overrides,
    build_config,
    config_hash,
    config_to_dict,
    deep_merge,
    load_config,
    load_yaml,
    save_resolved_config,
)
from utils.paths import CONFIGS_DIR


@dataclass(frozen=True)
class ToyModelConfig:
    hidden_size: int
    cell: Literal["lstm", "gru"] = "lstm"
    dropout: float = 0.0

    def __post_init__(self):
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")


@pytest.fixture
def model_yaml(tmp_path):
    p = tmp_path / "toy.yaml"
    p.write_text("model:\n  name: toy\n  params:\n    hidden_size: 8\n", encoding="utf-8")
    return p


BASE = CONFIGS_DIR / "base.yaml"


def test_base_plus_model_resolves(model_yaml):
    cfg = load_config([BASE, model_yaml])
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model.name == "toy" and cfg.model.params == {"hidden_size": 8}
    assert cfg.data.split_fractions == (0.8, 0.1, 0.1) and cfg.data.duplicate_policy == "group"
    assert cfg.diagnostics.enabled is False


def test_typed_model_params(model_yaml):
    cfg = load_config([BASE, model_yaml], model_config_cls=ToyModelConfig)
    assert cfg.model.params == ToyModelConfig(hidden_size=8)
    with pytest.raises(ConfigError, match=r"model.params.hidden_sise — did you mean 'hidden_size'"):
        load_config([BASE, model_yaml], overrides=["model.params.hidden_sise=3"], model_config_cls=ToyModelConfig)
    with pytest.raises(ConfigError, match="hidden_size must be >= 1"):
        load_config([BASE, model_yaml], overrides=["model.params.hidden_size=0"], model_config_cls=ToyModelConfig)
    with pytest.raises(ConfigError, match="not one of"):
        load_config([BASE, model_yaml], overrides=["model.params.cell=rnn"], model_config_cls=ToyModelConfig)


def test_overrides_parse_yaml_values(model_yaml):
    cfg = load_config(
        [BASE, model_yaml],
        overrides=["data.split_seed=7", "generation.top_k=5", "generation.top_p=null", "device.device=cpu",
                   "data.split_fractions=[0.7, 0.15, 0.15]"],
    )
    assert cfg.data.split_seed == 7 and cfg.generation.top_k == 5 and cfg.generation.top_p is None
    assert cfg.data.split_fractions == (0.7, 0.15, 0.15)


@pytest.mark.parametrize(
    "override, message",
    [
        ("data.split_sed=1", r"data.split_sed — did you mean 'split_seed'"),
        ("trainig.lr=1", r"unknown config key\(s\): trainig"),
        ("reproducibility.seed=true", "expected an integer"),
        ("reproducibility.seed=abc", "expected an integer"),
        ("reproducibility.seed=-1", "seed must be in"),
        ("data.split_fractions=[0.5, 0.5]", "expected 3 items"),
        ("data.split_fractions=[0.5, 0.4, 0.4]", "sum to 1"),
        ("data.duplicate_policy=drop", "not one of"),
        ("device.device=gpu", "invalid device"),
        ("generation.temperature=-1", "temperature"),
        ("generation.top_p=1.5", "top_p"),
        ("diagnostics.every_n_steps=0", "every_n_steps"),
        ("logging.level=LOUD", "not one of"),
        ("model.name=Bad-Name", "model.name must match"),
        ("data=5", "expected a mapping"),
    ],
)
def test_validation_errors(model_yaml, override, message):
    with pytest.raises(ConfigError, match=message):
        load_config([BASE, model_yaml], overrides=[override])


def test_model_section_is_required():
    with pytest.raises(ConfigError, match="model: required key is missing"):
        load_config(BASE)


def test_override_syntax_errors():
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no_equals"])
    with pytest.raises(ConfigError, match="not a mapping"):
        apply_overrides({"a": 1}, ["a.b=2"])
    with pytest.raises(ConfigError, match="invalid key path"):
        apply_overrides({}, ["a..b=2"])


def test_deep_merge_semantics():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    out = deep_merge(base, {"a": {"y": [3]}, "c": 2})
    assert out == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}  # inputs untouched


def test_yaml_duplicate_keys_and_bad_top_level(tmp_path):
    dup = tmp_path / "dup.yaml"
    dup.write_text("data:\n  split_seed: 1\n  split_seed: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key 'split_seed'"):
        load_yaml(dup)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_yaml(lst)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml(empty) == {}
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "nope.yaml")


def test_serialisation_round_trip_and_hash(tmp_path, model_yaml):
    cfg = load_config([BASE, model_yaml], model_config_cls=ToyModelConfig)
    path = save_resolved_config(cfg, tmp_path / "config.json")
    data = json.loads(path.read_text())
    assert data["model"]["params"] == {"hidden_size": 8, "cell": "lstm", "dropout": 0.0}
    rebuilt = build_config(data, model_config_cls=ToyModelConfig)
    assert rebuilt == cfg and config_hash(rebuilt) == config_hash(cfg)
    other = load_config([BASE, model_yaml], overrides=["data.split_seed=1"], model_config_cls=ToyModelConfig)
    assert config_hash(other) != config_hash(cfg)
    assert config_to_dict(cfg)["data"]["split_fractions"] == [0.8, 0.1, 0.1]


def test_template_and_base_files_are_valid():
    cfg = load_config([BASE, CONFIGS_DIR / "models" / "_template.yaml"])
    assert cfg.model.name == "example_model" and cfg.model.params == {}
