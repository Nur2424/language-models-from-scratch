# `utils/` — shared infrastructure

Architecture-independent building blocks for every model in this repository.
Nothing here trains, plots or compares models; it provides the primitives a
training pipeline and model code will call.

Requires Python ≥ 3.10 and the packages in `requirements.txt`. Run the tests with
`pip install -r requirements-dev.txt && pytest tests/utils`.

## Module map

| Module | Responsibility |
|---|---|
| `paths` | Repository-root–relative paths (no dependence on the working directory) |
| `io` | Atomic JSON / `torch.save`, JSONL append/read, strict JSON conversion |
| `config` | YAML → frozen, validated dataclasses; merge, overrides, hashing, saving |
| `vocabulary` | Character vocabulary; BOS/EOS `.`=0, input-only `pad_id`, `IGNORE_INDEX=-100` |
| `data` | Load/validate names, leakage-safe splits, fingerprints, coverage, padded sequences, fixed-length context windows |
| `batching` | Seeded, shuffled, without-replacement epoch ordering of example indices (`EpochSampler`) |
| `reproducibility` | Seeding, derived independent streams, determinism switch, RNG capture/restore |
| `device` | Explicit CPU/CUDA/MPS selection, dtype checks, device metadata |
| `statistics` | Stable scalar and tensor summaries (Welford / Chan merges) |
| `metrics` | Token NLL with explicit reduction, exact accumulation, perplexity, parameter counts |
| `logging_utils` | Console + `run.log` + `events.jsonl`, run ids, no duplicate handlers |
| `checkpoint` | `best` / `final` / `resume` checkpoints, safe loading |
| `run_manager` | Run folders (by default `results/runs/<model>/run_NNN/`), metadata, status, history |
| `generation` | Temperature / top-k / top-p sampling with explicit generators |
| `diagnostics/` | Gradient, activation and *measured* parameter-update summaries; schedule |

## Context windows and mini-batches

```python
from utils.data import make_context_windows
from utils.batching import EpochSampler
from utils.reproducibility import derive_seed

w = make_context_windows(splits.train, vocab, context_length=3)   # contexts [M, 3], targets [M], name_index [M]
# "emma": [.,.,.]→e  [.,.,e]→m  [.,e,m]→m  [e,m,m]→a  [m,m,a]→.   (repeated BOS, never crosses names, no PAD)

sampler = EpochSampler(w.num_examples, batch_size=32, seed=derive_seed(seed, "data"))
for epoch in range(start_epoch, num_epochs):            # order of epoch e = f(seed, e, N): resume needs only e
    for idx in sampler.batches(epoch):                  # [b] int64 CPU indices; final partial batch kept
        xb, yb = w.contexts[idx].to(device), w.targets[idx].to(device)
        ...
```

Report epoch losses example-weighted with `NLLAccumulator` (never the mean of batch means).
How the final partial batch enters the gradient is a model-specific training decision; see `batching.py`.

## Token conventions

```
name "emma", vocabulary of 26 letters → 27 output classes

inputs   .  e  m  m  a  PAD      ids  0  5 13 13  1  27   (pad_id = num_classes = 27, input-only)
targets  e  m  m  a  .  -100     ids  5 13 13  1  0 -100  (IGNORE_INDEX, skipped by the loss)
```

`.` (id 0) is both BOS and EOS, so every model predicts over the same 27 classes and
NLLs are comparable. Padding never reuses id 0, so a padded position cannot be
mistaken for a real boundary. Padded positions must never affect a valid output:
the fixed-window models never receive padding at all, and the RNN never looks up
the PAD id and freezes its state at padded steps.

## Usage sketches

```python
from utils.config import load_config, default_config_paths, config_to_dict
from utils.data import load_names, split_names, check_vocab_coverage, dataset_fingerprint, make_next_token_sequences
from utils.vocabulary import Vocabulary
from utils.reproducibility import seed_everything, derive_seed, make_generator, capture_rng_state
from utils.device import resolve_device
from models.feedforward.mlp import MLPConfig

cfg = load_config(default_config_paths("mlp"), model_config_cls=MLPConfig)
names = load_names(cfg.data.path)
splits = split_names(names, fractions=cfg.data.split_fractions, seed=cfg.data.split_seed,
                     duplicate_policy=cfg.data.duplicate_policy)
splits.assert_disjoint()
vocab = Vocabulary.from_names(splits.train)
check_vocab_coverage(vocab, splits)                        # raises on unseen dev/test characters
train = make_next_token_sequences(splits.train, vocab)     # or a model-specific adapter

seed_everything(cfg.reproducibility.seed)
g_init = make_generator(derive_seed(cfg.reproducibility.seed, "init"))
g_data = make_generator(derive_seed(cfg.reproducibility.seed, "data"))
g_gen = make_generator(derive_seed(cfg.reproducibility.seed, "generation"))
device = resolve_device(cfg.device.device)                 # explicit requests never fall back
```

```python
from utils.metrics import token_nll, NLLAccumulator

loss = token_nll(logits, targets, reduction="mean")        # training loss (mean over valid tokens)

acc = NLLAccumulator()                                     # evaluation: exact per-token NLL
with torch.no_grad():
    for xb, yb in batches:
        acc.update(model(xb), yb)
acc.mean_nll, acc.perplexity, acc.bits_per_token
```

```python
from utils.run_manager import RunManager
from utils.logging_utils import setup_logger, log_event
from utils.checkpoint import save_checkpoint

run = RunManager.create("mlp", config=config_to_dict(cfg),
                        dataset_fingerprint={**dataset_fingerprint(cfg.data.path), "splits": splits.fingerprint()})
log = setup_logger(run_id=run.run_id, log_dir=run.logs_dir)
with run.track():                                          # running → completed / failed / interrupted
    ...
    log_event(log, "eval", step=step, dev_nll=dev_nll)
    run.append_history({"step": step, "dev_nll": dev_nll})
    save_checkpoint(run.checkpoint_path("resume"), kind="resume", model=model, optimizer=opt,
                    step=step, rng_state=capture_rng_state({"data": g_data}), config=config_to_dict(cfg))
```

```python
from utils.generation import sample_next_token
nxt = sample_next_token(logits[:, -1], temperature=0.8, top_k=10, generator=g_gen)
```

```python
from utils.diagnostics.gradients import gradient_summaries, global_grad_norm
from utils.diagnostics.parameters import ParameterSnapshot
from utils.diagnostics.activations import ActivationMonitor, tanh_saturation

collect = cfg.diagnostics.should_collect(step, "parameters")
snap = ParameterSnapshot.capture(model) if collect else None   # after backward, before step
optimizer.step()
if snap is not None:
    updates = snap.update_stats(model)                     # real ‖Δθ‖, also correct for AdamW

with ActivationMonitor(model, module_types=(nn.Tanh,), saturation_by_type={nn.Tanh: tanh_saturation()}) as mon:
    model(xb)
mon.summaries()
```

## Reproducibility limits

CPU runs are reproducible given identical seeds, library versions and thread
counts. CUDA needs `reproducibility.deterministic: true` (and
`CUBLAS_WORKSPACE_CONFIG`, set automatically before CUDA initialises). MPS has
no determinism guarantee. Results are never expected to match bit-for-bit
across devices. `nn.Dropout` always uses the global torch RNG. Details are in
`reproducibility.py`.
