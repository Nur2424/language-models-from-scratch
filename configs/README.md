# Configuration

Configs are YAML files turned into frozen, validated dataclasses by `utils/config.py`.

## Resolution order

1. `configs/base.yaml` — shared infrastructure defaults
2. `configs/models/<name>.yaml` — model choice and model-specific `params`
3. overrides — `"section.key=value"` strings (values parsed as YAML)

Mappings are merged recursively; every other value (including lists) is replaced.

```python
from utils.config import load_config, default_config_paths, save_resolved_config

from models.feedforward.mlp import MLPConfig

cfg = load_config(
    default_config_paths("mlp"),
    overrides=["data.split_seed=7", "device.device=cpu"],
    model_config_cls=MLPConfig,        # optional: typed model params
)
save_resolved_config(cfg, run.config_path)   # the run records exactly what it used
```

## Validation rules

| Situation | Behaviour |
|---|---|
| Unknown key (typo) | `ConfigError: unknown config key(s): data.split_sed — did you mean 'split_seed'?` |
| Wrong type | `ConfigError: reproducibility.seed: expected an integer, got 'abc'` (`true` is never an int) |
| Out of range | `ConfigError: data: fractions must sum to 1 ...` |
| Duplicate key in a YAML file | error with line number |
| Required key missing | `ConfigError: model: required key is missing` |
| `model.params` without a dataclass | kept as an untyped dict — typos there are only caught once the model's dataclass is passed |

`base.yaml` alone is not a complete experiment: a `model` section is required.

## Sections

`run`, `data`, `reproducibility`, `device`, `logging`, `checkpoint`, `diagnostics`,
`generation`, `model`. See comments in `base.yaml`.

There is no shared `training` section. Training settings (learning rate, batch size,
number of epochs) differ in kind between models, so each model defines its own
training configuration class next to its training procedure: `MLPTrainingConfig`,
`BatchNormMLPTrainingConfig`, `WaveNetTrainingConfig` and `RNNTrainingConfig`.
