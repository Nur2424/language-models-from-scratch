# Model configs

One YAML file per implemented model, named after the model (`mlp.yaml`, `rnn.yaml`, ...).
Each file is added together with its model; `_template.yaml` is the starting point.

These files hold **architecture** settings only (`model.params`). Training settings
live in each model's own training configuration class (see `configs/README.md`).

Each model module defines a frozen dataclass for its parameters. For the MLP:

```python
@dataclass(frozen=True)
class MLPConfig:
    context_length: int = 3
    embedding_dim: int = 10
    hidden_dim: int = 200

    def __post_init__(self) -> None:
        # each field must be an int >= 1
        ...
```

and the entry point loads it explicitly:

```python
from models.feedforward.mlp import MLPConfig

cfg = load_config(default_config_paths("mlp"), model_config_cls=MLPConfig)
cfg.model.params.hidden_dim     # typed, validated
```

Supported field types: `bool`, `int`, `float`, `str`, `Literal[...]`, `X | None`,
`tuple[...]`, `list[X]`, `dict[str, X]` and nested dataclasses.
