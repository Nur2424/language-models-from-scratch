"""Final controlled WaveNet experiment — runner (F3).

Design: ``PROJECT_RESEARCH_RECORD.md`` D-22 (learning rate 0.03 for every arm), D-31 and its
addendum (ten paired seeds 0–9, test NLL primary, dev-NLL checkpoint selection), D-32 (common
budget 26 epochs); F0 decisions 1–10 of the implementation design. Arms A–D, plain SGD, batch 32,
``EpochSampler`` without replacement, CPU float32, intra-op threads 1, inter-op threads 8,
diagnostics option B, full per-forward finiteness monitor, best checkpoint by dev NLL (earliest
epoch on a tie, as implemented by the accepted trainers).

Commands (``python training/wavenet_study/final_experiment.py <command> …``)::

    status                                              status only (no learning metrics)
    run --threads 1 --stage validation                  seed 0, all four arms
    review-validation --reviewer NAME --decision approved|rejected --notes TEXT
    run --threads 1 --stage completion                  seeds 1–9, all four arms (after approval)
    retry --threads 1 --arm X --seed S --reason TEXT --authorized-by NAME   restart from scratch
    exclude-seed --seed S --decision D-xx --reason TEXT --authorized-by NAME
    category-b-resolve --event-id ID --outcome O --reviewer NAME --rationale TEXT --evidence TEXT
    verify-records                                      read-only record verification
    derive-schemas --output PATH                        regenerate final_record_schemas.json
    one …                                               internal: one attempt (child process)

Execution model (adapted from the accepted epoch-budget pilot runner, copied, never imported for
execution): one child process per attempt, sequential, under the exclusive lock; the child trains
with the accepted trainer through ``study_common.execute_run`` (finiteness monitor, state hashes,
category A/B/C classification). There is no reference prefix check (no G5/G6): the final seeds
have no earlier reference runs. Integrity gates are recomputed independently by
``verify_final_runs.py``.

Rules enforced here:
  * every condition is frozen in ``conditions.json`` at the first validation run; any blocking
    difference refuses every later command;
  * the completion stage needs every validation run completed **and** an approved, status-only
    human review of exactly those validation attempts;
  * a failed attempt is restarted from scratch only by an explicit retry with a written reason
    and ``--authorized-by`` (always required in the final experiment);
  * category A (mechanical instability) halts the whole experiment (``HALTED``); nothing is
    excluded automatically. A human may exclude a seed by a documented decision; an exclusion
    always applies to all four arms of that seed and is recorded in a hash-chained log;
  * once the protected test freeze (``test_freeze/freeze.json``, F4) exists, training is closed:
    no run, retry, review or exclusion is accepted any more;
  * the test split is never turned into windows here; its names are read only by the
    split-disjointness and vocabulary-coverage checks inside the accepted data utilities.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import string
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sc = importlib.import_module("training.wavenet_study.study_common")  # after the repository is on sys.path

# ============================================================ approved conditions
PROTOCOL_VERSION = "final-1.0"
ARMS: dict[str, dict[str, Any]] = {  # identical to the LR and epoch-budget pilots' arm table (D-09)
    "A": {"label": "Flat-8 MLP, no BatchNorm", "config": "mlp", "params": 21_897,
          "overrides": ["model.params.context_length=8"]},
    "B": {"label": "Flat-8 MLP, BatchNorm", "config": "batchnorm_mlp", "params": 22_097,
          "overrides": ["model.params.context_length=8"]},
    "C": {"label": "Hierarchical WaveNet, no BatchNorm", "config": "wavenet", "params": 21_989, "overrides": []},
    "D": {"label": "Hierarchical WaveNet, BatchNorm", "config": "wavenet_batchnorm", "params": 22_397,
          "overrides": []},
}
LEARNING_RATES = {"A": 0.03, "B": 0.03, "C": 0.03, "D": 0.03}  # D-22
SEEDS = tuple(range(10))                                        # D-31
EPOCHS = 26                                                     # D-32
DIAGNOSTICS = {"enabled": True, "every_n_steps": 100, "gradients": True, "parameters": True,
               "activations": False}                            # option B
INTRA_OP_THREADS = 1
INTER_OP_THREADS = 8
EXPECTED_TRAINING = {"batch_size": 32, "lr_decay_epoch": None, "lr_decay_factor": 0.1, "eval_batch_size": 4096}
STAGES = ("validation", "completion")
DECISIONS = {"learning_rate": "D-22", "design": "D-31 and addendum", "budget": "D-32",
             "implementation": "F0 decisions 1-10"}
RESULTS_ROOT = REPO / "results" / "experiments" / "wavenet_study" / "final"
RECORD_SCHEMAS = "training/wavenet_study/final_record_schemas.json"
RECORD_SCHEMAS_FORMAT = "wavenet_study.final_record_schemas/1"
FINITENESS_POLICY = {
    "parameters": "every parameter, immediately after every optimizer.step() (global post-step hook)",
    "module_outputs": "every leaf module output in every forward, training and evaluation",
    "intervention": "none (read-only)", "cost": "measured per epoch in timing.json",
}
RECORDED_DOCUMENTS = ("models/hierarchical/wavenet_spec.md", "models/feedforward/mlp_spec.md",
                      "models/feedforward/batchnorm_mlp_spec.md", "UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md",
                      "PROJECT_RESEARCH_RECORD.md")
TRAINING_CODE = ("training/wavenet_study/study_common.py", "training/wavenet_study/final_experiment.py",
                 RECORD_SCHEMAS)
KEY_FIELDS = ("arm", "seed")
METADATA_BLOCK = "final"
FREEZE_DIR = "test_freeze"
FREEZE_FILE = "freeze.json"
REVIEW_LOG = "validation_review.jsonl"
EXCLUSION_LOG = "exclusions.jsonl"
REVIEW_DECISIONS = ("approved", "rejected")


# ============================================================ specification
@dataclass(frozen=True)
class FinalSpec:
    """Every setting of the final experiment, frozen into ``conditions.json`` at the first run.

    The command line always uses :func:`production_spec`. Synthetic tests construct their own
    spec (temporary root, synthetic names, few epochs and seeds); its different root and frozen
    conditions keep it apart from the real experiment.
    """

    root: str
    repo: str
    arms: dict[str, dict[str, Any]]
    learning_rates: dict[str, float]
    seeds: list[int]
    epochs: int
    diagnostics: dict[str, Any]
    intra_op_threads: int
    inter_op_threads: int
    blocking_files: list[str]
    recorded_files: list[str]
    record_schemas: str = RECORD_SCHEMAS
    data_path: str | None = None
    expected_training: dict[str, Any] = field(default_factory=lambda: dict(EXPECTED_TRAINING))
    decisions: dict[str, str] = field(default_factory=lambda: dict(DECISIONS))
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if set(self.learning_rates) != set(self.arms) or not set(self.arms) <= set(ARMS):
            raise ValueError("learning rates must be given for exactly the spec's arms, all of them known arms")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) \
                or any(isinstance(s, bool) or not isinstance(s, int) for s in self.seeds):
            raise ValueError(f"seeds must be distinct ints, got {self.seeds!r}")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError(f"epochs must be a positive int, got {self.epochs!r}")

    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> FinalSpec:
        return cls(**dict(data))

    @property
    def paths(self) -> sc.PilotPaths:
        return sc.PilotPaths(Path(self.root))

    @property
    def validation_seed(self) -> int:
        return self.seeds[0]


def production_spec() -> FinalSpec:
    """The approved final experiment."""
    blocking = ["models/feedforward/mlp.py", "models/feedforward/batchnorm_mlp.py", "models/hierarchical/wavenet.py",
                *sc.python_files_under(REPO, "utils"), "configs/base.yaml",
                *(f"configs/models/{ARMS[a]['config']}.yaml" for a in ARMS), "data/names.txt", *TRAINING_CODE]
    return FinalSpec(root=str(RESULTS_ROOT), repo=str(REPO), arms=ARMS, learning_rates=LEARNING_RATES,
                     seeds=list(SEEDS), epochs=EPOCHS, diagnostics=DIAGNOSTICS, intra_op_threads=INTRA_OP_THREADS,
                     inter_op_threads=INTER_OP_THREADS, blocking_files=blocking,
                     recorded_files=list(RECORDED_DOCUMENTS))


def freeze_path(root: str | Path) -> Path:
    return Path(root) / FREEZE_DIR / FREEZE_FILE


def require_training_open(spec: FinalSpec) -> None:
    """Training closure: nothing that trains, retries, reviews or excludes after the test freeze."""
    if freeze_path(spec.root).exists():
        raise sc.Blocked("training is closed: the protected test freeze exists (test_freeze/freeze.json)")


# ============================================================ attempt store, exclusions, reviews
class FinalAttemptStore(sc.AttemptStore):
    """The accepted attempt store with one change: an explicitly excluded seed no longer halts.

    Category A halts the experiment (``sweep_state()['halted']``). A documented exclusion of the
    affected seed (all four arms) is the human decision that lifts that halt for this seed only;
    every other rule of the accepted store is unchanged.
    """

    def __init__(self, paths: sc.PilotPaths, *, excluded_seeds: Sequence[int] = (), lock: sc.PilotLock | None = None):
        super().__init__(paths, key_fields=KEY_FIELDS, lock=lock, run_attempt_id_path=(METADATA_BLOCK, "attempt_id"),
                         attempt_id=lambda k, n: f"{k['arm']}-s{k['seed']}-a{n}")
        self.excluded_seeds = frozenset(excluded_seeds)

    def sweep_state(self) -> dict[str, list[str]]:
        state = super().sweep_state()
        latest = {recs[-1]["attempt_id"]: recs[-1] for recs in self.by_key().values()}
        state["halted"] = [aid for aid in state["halted"] if latest[aid]["seed"] not in self.excluded_seeds]
        return state


def _chain(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return sc.HashChainedLog(path).read()[0]
    except sc.LedgerError as exc:
        raise sc.Blocked(f"{path.name} failed verification: {exc}") from None


def _cross_checked_chain(root: str | Path, name: str, note_event: str, head_field: str) -> list[dict[str, Any]]:
    """A hash-chained log whose heads are also recorded in ``invocations.jsonl``.

    The chain alone cannot reveal an edit of its last line; every append is therefore followed by
    an invocation note carrying the new head. Every recorded head must be a head of the current
    log and the current head must be the last recorded one (no uncovered tail).
    """
    path = Path(root) / name
    rows = _chain(path)
    heads = [sc.GENESIS] + ([sc._sha256_text(line) for line in path.read_text(encoding="utf-8").split("\n")[:-1]]
                            if path.exists() else [])
    inv = sc.PilotPaths(Path(root)).invocations
    recorded = [r[head_field] for r in _chain(inv) if r.get("event") == note_event]
    if any(h not in heads for h in recorded) or (recorded[-1] if recorded else sc.GENESIS) != heads[-1]:
        raise sc.Blocked(f"{name} does not match the heads recorded in invocations.jsonl (edited or truncated)")
    return rows


def read_exclusions(root: str | Path) -> list[dict[str, Any]]:
    """Verified exclusion records (hash chain and recorded heads checked), in order."""
    rows = _cross_checked_chain(root, EXCLUSION_LOG, "seed_excluded", "exclusions_head")
    seen: set[int] = set()
    for n, r in enumerate(rows, start=1):
        if not isinstance(r.get("seed"), int) or r["seed"] in seen or not all(
                isinstance(r.get(k), str) and r[k].strip() for k in ("decision", "reason", "authorized_by")):
            raise sc.Blocked(f"{EXCLUSION_LOG} record {n} is malformed or repeats a seed")
        seen.add(r["seed"])
    return rows


def excluded_seeds(root: str | Path) -> list[int]:
    return [r["seed"] for r in read_exclusions(root)]


def read_reviews(root: str | Path) -> list[dict[str, Any]]:
    """Verified validation reviews (hash chain and recorded heads checked), in order."""
    return _cross_checked_chain(root, REVIEW_LOG, "validation_review", "review_head")


def make_store(spec: FinalSpec, lock: sc.PilotLock | None = None) -> FinalAttemptStore:
    return FinalAttemptStore(spec.paths, excluded_seeds=excluded_seeds(spec.root), lock=lock)


def latest_attempts(store: sc.AttemptStore) -> dict[tuple, dict[str, Any]]:
    return {key: recs[-1] for key, recs in store.by_key().items()}


# ============================================================ threads and conditions
def prepare_threads(spec: FinalSpec) -> None:
    """Intra-op threads as frozen; inter-op threads must already equal the frozen value."""
    import torch

    torch.set_num_threads(spec.intra_op_threads)
    if torch.get_num_threads() != spec.intra_op_threads:
        raise sc.Blocked(f"requested {spec.intra_op_threads} intra-op threads, torch uses {torch.get_num_threads()}")
    if torch.get_num_interop_threads() != spec.inter_op_threads:
        raise sc.Blocked(f"inter-op threads are {torch.get_num_interop_threads()}, the experiment requires "
                         f"{spec.inter_op_threads}")


def arm_config(spec: FinalSpec, arm: str, seed: int | None):
    """The resolved configuration of one arm (copied from the pilot; ``seed=None``: without it)."""
    from models.feedforward import batchnorm_mlp, mlp
    from models.hierarchical import wavenet
    from utils.config import default_config_paths, load_config

    cfg_cls = {"mlp": mlp.MLPConfig, "batchnorm_mlp": batchnorm_mlp.BatchNormMLPConfig,
               "wavenet": wavenet.WaveNetConfig, "wavenet_batchnorm": wavenet.WaveNetConfig}[spec.arms[arm]["config"]]
    overrides = [*spec.arms[arm]["overrides"], "device.device=cpu",
                 *(f"diagnostics.{k}={str(v).lower()}" for k, v in spec.diagnostics.items())]
    if spec.data_path is not None:
        overrides.append(f"data.path={json.dumps(spec.data_path)}")
    if seed is not None:
        overrides.append(f"reproducibility.seed={seed}")
    return load_config(default_config_paths(spec.arms[arm]["config"]), overrides=overrides, model_config_cls=cfg_cls)


def config_without_seed(cfg: Any) -> dict[str, Any]:
    from utils.config import config_to_dict

    d = config_to_dict(cfg)
    d["reproducibility"] = {k: v for k, v in d["reproducibility"].items() if k != "seed"}
    return d


def _resolve(spec: FinalSpec, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(spec.repo) / p


def load_record_schemas(spec: FinalSpec) -> dict[str, Any]:
    """The schema document (format checked); the arms must be exactly the spec's arms."""
    path = _resolve(spec, spec.record_schemas)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise sc.Blocked(f"record schemas {path} unreadable: {exc}") from None
    if doc.get("format") != RECORD_SCHEMAS_FORMAT or not set(spec.arms) <= set(doc.get("arms", {})):
        raise sc.Blocked(f"record schemas {path}: wrong format or arms missing")
    return doc


def schemas_for(doc: Mapping[str, Any], arm: str) -> sc.RecordSchemas:
    arm_doc = doc["arms"][arm]
    return sc.RecordSchemas(frozenset(tuple(p) for p in arm_doc["history"]),
                            frozenset(tuple(p) for p in arm_doc["steps"]))


def current_conditions(spec: FinalSpec) -> dict[str, Any]:
    """Everything that must stay identical for the whole experiment."""
    schemas = load_record_schemas(spec)
    return {
        "spec": spec.to_json(),
        "protocol": {"version": spec.protocol_version, "decisions": spec.decisions, "seeds": spec.seeds,
                     "validation_seed": spec.validation_seed, "epochs": spec.epochs, "stages": list(STAGES),
                     "selection": "best dev NLL, earliest epoch on a tie (strict improvement)",
                     "test_split": "never evaluated during training; protected one-time evaluation after the freeze"},
        "learning_rates": {a: {"lr": lr, "decision": spec.decisions["learning_rate"]}
                           for a, lr in spec.learning_rates.items()},
        "arms": {a: {"label": s["label"], "expected_parameters": s["params"],
                     "config": config_without_seed(arm_config(spec, a, None))} for a, s in spec.arms.items()},
        "expected_training": spec.expected_training,
        "diagnostics": spec.diagnostics,
        "finiteness_policy": FINITENESS_POLICY,
        "record_schemas": {"path": spec.record_schemas, "sha256": sc.sha256_json(schemas), "document": schemas},
        "environment": sc.environment_record(),
        "checksums": sc.checksum_record(spec.repo, blocking=spec.blocking_files, recorded=spec.recorded_files),
    }


# ============================================================ one arm (copied from the accepted pilot)
def build_arm(spec: FinalSpec, arm: str, seed: int, epochs: int, phase: list[str]) -> dict[str, Any]:
    """Configuration, data, model and trainer of one arm.

    Copied line by line from ``epoch_budget_pilot.build_arm`` (F0 decision 4); the parity tests
    compare both on synthetic data. ``phase[0]`` names the set-up step.
    """
    import torch

    from models.feedforward import batchnorm_mlp, mlp
    from models.hierarchical import wavenet
    from utils.data import check_vocab_coverage, dataset_fingerprint, load_names, make_context_windows, split_names
    from utils.device import resolve_device
    from utils.metrics import count_parameters
    from utils.reproducibility import derive_seed, make_generator, seed_everything
    from utils.vocabulary import Vocabulary

    phase[0] = "setup:config"
    module, model_cls, train_cls, train_fn = {
        "mlp": (mlp, mlp.MLPLanguageModel, mlp.MLPTrainingConfig, mlp.train_mlp),
        "batchnorm_mlp": (batchnorm_mlp, batchnorm_mlp.BatchNormMLPLanguageModel,
                          batchnorm_mlp.BatchNormMLPTrainingConfig, batchnorm_mlp.train_batchnorm_mlp),
        "wavenet": (wavenet, wavenet.WaveNetLanguageModel, wavenet.WaveNetTrainingConfig, wavenet.train_wavenet),
        "wavenet_batchnorm": (wavenet, wavenet.WaveNetLanguageModel, wavenet.WaveNetTrainingConfig,
                              wavenet.train_wavenet),
    }[spec.arms[arm]["config"]]
    cfg = arm_config(spec, arm, seed)
    if cfg.device.device != "cpu" or cfg.device.dtype != "float32":
        raise ValueError(f"expected device cpu / float32, config has {cfg.device.device} / {cfg.device.dtype}")
    ck = cfg.checkpoint
    if not (ck.save_best and ck.save_final and ck.resume_every_n_steps is None):
        raise ValueError(f"checkpoint config must match the trainers' fixed behaviour, got {ck}")

    phase[0] = "setup:data"
    names = load_names(cfg.data.path)
    splits = split_names(names, fractions=cfg.data.split_fractions, seed=cfg.data.split_seed,
                         duplicate_policy=cfg.data.duplicate_policy)
    splits.assert_disjoint()
    vocab = Vocabulary.from_names(splits.train)
    check_vocab_coverage(vocab, splits)
    k = cfg.model.params.context_length
    train = make_context_windows(splits.train, vocab, context_length=k)
    dev = make_context_windows(splits.dev, vocab, context_length=k)
    # The test split is never turned into windows here (only evaluate_final_test.py does, after the freeze).

    phase[0] = "setup:model"
    seed_everything(seed)  # global RNG hygiene only; init and data order use explicit derived streams
    device = resolve_device(cfg.device.device)
    model = model_cls(vocab, cfg.model.params, generator=make_generator(derive_seed(seed, "init")))
    params = list(model.parameters())
    if any(p.device != device or p.dtype != torch.float32 for p in params):
        raise RuntimeError(f"model parameters are not all on {device} / float32")
    n_params = count_parameters(model).total
    if k != 8 or n_params != spec.arms[arm]["params"]:
        raise RuntimeError(f"arm {arm}: expected k=8 and {spec.arms[arm]['params']} parameters, got k={k}, {n_params}")
    model_name = getattr(model, "model_name", None) or module.MODEL_NAME
    if model_name != cfg.model.name:
        raise RuntimeError(f"arm {arm}: model name {model_name!r} != config {cfg.model.name!r}")
    training = train_cls(lr=spec.learning_rates[arm], max_epochs=epochs)  # every other setting: trainer defaults
    expected = {**spec.expected_training, "max_epochs": epochs}
    actual = {f: getattr(training, f) for f in expected}
    if actual != expected:
        raise RuntimeError(f"training defaults changed: expected {expected}, got {actual}")
    fingerprint = {**dataset_fingerprint(cfg.data.path), "splits": splits.fingerprint()}  # hashes only
    normalization = "batchnorm" if spec.arms[arm]["config"] in ("batchnorm_mlp", "wavenet_batchnorm") else "none"
    return {"module": module, "cfg": cfg, "model": model, "model_name": model_name, "n_params": n_params,
            "training": training, "train_fn": train_fn, "train": train, "dev": dev, "fingerprint": fingerprint,
            "normalization": normalization, "device": str(params[0].device), "dtype": str(params[0].dtype),
            "vocab": vocab}


# ============================================================ record schemas (derived from accepted code)
def synthetic_schema_names(seed: int = 20261112, n: int = 400) -> list[str]:
    """Deterministic synthetic names covering all 26 letters (so the approved parameter counts hold)."""
    rng = random.Random(seed)
    while True:
        names, seen = [], set()
        while len(names) < n:
            name = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
            if name not in seen:
                seen.add(name)
                names.append(name)
        if set("".join(names)) == set(string.ascii_lowercase):
            return names
        n += 1


def derive_record_schemas(spec: FinalSpec, *, epochs: int = 2, every_n_steps: int = 5,
                          names: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the accepted trainers briefly on synthetic names and record each arm's leaf-path schemas.

    Uses :func:`build_arm` with the spec's configurations (production widths), a synthetic names
    file and a shortened step-diagnostic interval (the interval changes how many step records
    exist, never their fields). No project data and no pilot result is read.
    """
    import torch

    from utils.config import config_to_dict
    from utils.run_manager import RunManager

    sc.disable_git()
    names = list(names or synthetic_schema_names())
    out: dict[str, Any] = {"format": RECORD_SCHEMAS_FORMAT, "arms": {}, "derivation": {
        "method": "accepted trainers on deterministic synthetic names (final_experiment.derive_record_schemas)",
        "epochs": epochs, "every_n_steps": every_n_steps, "names_sha256": sc.sha256_json(names),
        "trainer_sha256": {p: sc.sha256_file(Path(spec.repo) / p) for p in (
            "models/feedforward/mlp.py", "models/feedforward/batchnorm_mlp.py", "models/hierarchical/wavenet.py")}}}
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "names.txt"
        data.write_text("\n".join(names) + "\n", encoding="utf-8")
        local = FinalSpec(**{**spec.to_json(), "root": str(Path(tmp) / "root"), "data_path": str(data),
                             "diagnostics": {**spec.diagnostics, "every_n_steps": every_n_steps}})
        torch.set_num_threads(1)
        for arm in local.arms:
            b = build_arm(local, arm, 0, epochs, ["setup"])
            experiment = config_to_dict(b["cfg"])
            run = RunManager.create(b["model_name"], config=b["module"].run_config(experiment, b["model"].config,
                                                                                    b["training"]),
                                    results_root=Path(tmp) / "runs")
            with run.track():
                b["train_fn"](b["model"], b["train"], b["dev"], training=b["training"], seed=0, run=run,
                              diagnostics=b["cfg"].diagnostics, config=experiment)
            history = sc.read_jsonl_strict(Path(run.run_dir) / "history.jsonl")
            steps = sc.read_jsonl_strict(Path(run.run_dir).joinpath(*sc.STEP_RECORDS_FILE))
            out["arms"][arm] = {key: [list(p) for p in sorted(sc.record_schema(rows), key=sc.path_sort_key)]
                                for key, rows in (("history", history), ("steps", steps))}
    return out


# ============================================================ child: one attempt
def _launched_head(spec: FinalSpec, attempt_id: str) -> str:
    entries, head = sc.HashChainedLog(spec.paths.manifest).read()
    if not entries or entries[-1].get("event") != "launched" or entries[-1].get("attempt_id") != attempt_id:
        raise sc.Blocked(f"the last manifest row is not the launch of {attempt_id}")
    return head


def _failure(exc: BaseException, phase: str) -> dict[str, Any]:
    import traceback

    return {"type": type(exc).__name__, "message": str(exc)[:2000], "phase": phase,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]}


def child_main(root: str, attempt_id: str, arm: str, seed: int) -> int:
    """Run one launched attempt; write its result file once; return the exit code."""
    sc.disable_git()
    paths = sc.PilotPaths(Path(root))
    key = {"arm": arm, "seed": seed}
    frozen = sc.load_frozen(paths)
    if frozen is None:
        raise sc.Blocked("no frozen conditions")
    spec = FinalSpec.from_json(frozen["conditions"]["spec"])
    if Path(spec.root) != paths.root:
        raise sc.Blocked("the frozen spec belongs to another experiment root")
    store = make_store(spec)
    rec = store.verify_launched(attempt_id, key)
    head = _launched_head(spec, attempt_id)
    base = {"attempt_id": attempt_id, **key, "stage": rec.get("stage"), "retry_of": rec.get("retry_of"),
            "manifest_head": head, "conditions_sha256": frozen["conditions_sha256"], "started_at": sc._utc_now()}
    phase = ["setup:consistency"]
    try:
        prepare_threads(spec)
        require_training_open(spec)
        sc.verify_or_freeze(paths, current_conditions(spec), may_create=False, lock=None)
        schemas = schemas_for(frozen["conditions"]["record_schemas"]["document"], arm)
        b = build_arm(spec, arm, seed, spec.epochs, phase)
        phase[0] = "setup:run_directory"
        from utils.config import config_to_dict
        from utils.run_manager import RunManager

        experiment = config_to_dict(b["cfg"])
        block = {"study": "wavenet_study", "stage": "final_experiment", "experiment_stage": rec.get("stage"),
                 "protocol": spec.protocol_version, "attempt_id": attempt_id, **key, "epochs": spec.epochs,
                 "lr": spec.learning_rates[arm], "arm_label": spec.arms[arm]["label"], "parameters": b["n_params"],
                 "train_windows": b["train"].num_examples, "dev_windows": b["dev"].num_examples,
                 "model_device": b["device"], "model_dtype": b["dtype"]}
        run = RunManager.create(b["model_name"],
                                config={**b["module"].run_config(experiment, b["model"].config, b["training"]),
                                        METADATA_BLOCK: block},
                                results_root=paths.root, dataset_fingerprint=b["fingerprint"],
                                extra_metadata={METADATA_BLOCK: block,
                                                "conditions_sha256": frozen["conditions_sha256"]})
    except KeyboardInterrupt as exc:
        return _write_child_result(paths, {**base, "status": "interrupted", "phase": phase[0], "run_dir": None,
                                           "failure": _failure(exc, phase[0])})
    except (Exception, SystemExit) as exc:  # noqa: BLE001  (set-up failure: a software error, never evidence)
        return _write_child_result(paths, {**base, "status": "error", "phase": phase[0], "run_dir": None,
                                           "failure": _failure(exc, phase[0])})
    prepared = sc.PreparedRun(b["model"], b["train"], b["dev"], b["training"], b["train_fn"], run,
                              b["cfg"].diagnostics, experiment)
    context = sc.RunContext(attempt_id, arm, seed, spec.epochs, b["normalization"],
                            {"stage": rec.get("stage"), "experiment": "final"})

    def progress(info: Mapping[str, Any]) -> None:  # status-only: progress and time, no learning metrics
        print(f"  [{attempt_id}] epoch {info['epoch']}/{info['cap']} done ({info['seconds']:.1f}s)", flush=True)

    result = sc.execute_run(prepared, context, schemas=schemas, ledger=sc.CategoryBLedger(paths.root),
                            prefix_checks=(), on_epoch=progress)
    return _write_child_result(paths, {**base, **result})


def _write_child_result(paths: sc.PilotPaths, result: dict[str, Any]) -> int:
    sc.write_attempt_result(paths, result["attempt_id"], {**result, "ended_at": sc._utc_now()})
    return {"completed": 0, "mechanically_unstable": 0, "integrity_blocked": 0, "interrupted": 130}.get(
        result["status"], 1)


# ============================================================ parent: invocations
def console(line: str) -> None:
    print(line, flush=True)


def _launch_child(spec: FinalSpec, session: sc.InvocationSession, aid: str, key: Mapping[str, Any],
                  out: Callable[[str], None]) -> dict[str, Any]:
    """Run one launched attempt in a child process and close it in the manifest."""
    paths = spec.paths
    session.note("attempt_launched", attempt_id=aid, **key)
    cmd = [sys.executable, str(Path(__file__).resolve()), "one", "--root", str(paths.root), "--attempt-id", aid,
           "--arm", key["arm"], "--seed", str(key["seed"])]
    env = {**os.environ, "OMP_NUM_THREADS": str(spec.intra_op_threads), "MKL_NUM_THREADS": str(spec.intra_op_threads)}
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = paths.logs_dir / f"{aid}.log"
    t0, interrupted = time.perf_counter(), False
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=spec.repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        try:
            for line in proc.stdout:
                out(line.rstrip("\n"))
                log.write(line)
            returncode = proc.wait()
        except KeyboardInterrupt:
            interrupted = True
            returncode = proc.wait()
        finally:
            proc.stdout.close()
    rec = session.store.finish(aid, returncode=returncode, interrupted=interrupted,
                               parent_wall_seconds=time.perf_counter() - t0, log=sc._repo_relative(log_file))
    session.note("attempt_finished", attempt_id=aid, status=rec["status"])
    sc.verify_record_heads(paths, session.store)
    out(f"  -> {aid}: {rec['status']}" + (f" ({rec.get('phase')})" if rec["status"] != "completed" else ""))
    return rec


STOP_CODES = {"mechanically_unstable": 4, "integrity_blocked": 5, "error": 3, "crashed": 3, "interrupted": 130}
STOP_MESSAGES = {
    "mechanically_unstable": "experiment HALTED after mechanical instability; human review required (no automatic "
                             "exclusion)",
    "integrity_blocked": "integrity_blocked; human review and an authorised retry are required",
    "error": "stopped after a software / infrastructure error; inspect the attempt log",
    "crashed": "stopped: the attempt ended without a result; inspect it",
    "interrupted": "interrupted; inspect the attempt, then retry it explicitly",
}


def validation_keys(spec: FinalSpec) -> list[dict[str, Any]]:
    return [{"arm": a, "seed": spec.validation_seed} for a in spec.arms]


def completion_keys(spec: FinalSpec, excluded: Sequence[int] = ()) -> list[dict[str, Any]]:
    return [{"arm": a, "seed": s} for s in spec.seeds[1:] if s not in excluded for a in spec.arms]


def _validation_attempts(store: sc.AttemptStore, spec: FinalSpec) -> dict[str, dict[str, Any]] | None:
    """Latest validation attempts, if every one is completed (else ``None``)."""
    latest = latest_attempts(store)
    out = {}
    for k in validation_keys(spec):
        rec = latest.get((k["arm"], k["seed"]))
        if rec is None or rec["status"] != "completed":
            return None
        out[f"{k['arm']}/{k['seed']}"] = {"attempt_id": rec["attempt_id"], "result_sha256": rec.get("result_sha256")}
    return out


def validation_approved(spec: FinalSpec, store: sc.AttemptStore) -> bool:
    """The last review approves exactly the current completed validation attempts."""
    current = _validation_attempts(store, spec)
    reviews = read_reviews(spec.root)
    return current is not None and bool(reviews) and reviews[-1].get("decision") == "approved" \
        and reviews[-1].get("validation_attempts") == current


def _stage_keys(spec: FinalSpec, store: sc.AttemptStore, stage: str) -> list[dict[str, Any]]:
    if stage == "validation":
        if spec.validation_seed in excluded_seeds(spec.root):
            raise sc.Blocked("the validation seed is excluded; the validation stage cannot run")
        return validation_keys(spec)
    if stage != "completion":
        raise sc.Blocked(f"unknown stage {stage!r}; stages: {', '.join(STAGES)}")
    if not validation_approved(spec, store):
        raise sc.Blocked("the completion stage needs every validation run completed and an approved human review "
                         "of exactly those attempts (review-validation)")
    return completion_keys(spec, excluded_seeds(spec.root))


def run_stage(spec: FinalSpec, stage: str, *, out: Callable[[str], None] = console) -> int:
    """``run --stage …``: launch the stage's missing attempts sequentially; stop on any non-completed one."""
    sc.disable_git()
    prepare_threads(spec)
    require_training_open(spec)
    paths = spec.paths
    with sc.PilotLock(paths, f"run {stage}") as lock:
        require_training_open(spec)
        frozen, nonblocking = sc.verify_or_freeze(paths, current_conditions(spec), may_create=stage == "validation",
                                                  lock=lock)
        store = make_store(spec, lock)
        session = sc.InvocationSession(paths, lock, store, "run", stage=stage, argv=sys.argv,
                                       recorded_document_changes=[list(p) for p in nonblocking])
        code = 0
        try:
            registered = store.check_state()
            if registered:
                session.note("registered", attempts=registered)
            store.require_runnable()
            jobs = store.pending(_stage_keys(spec, store, stage))
            out(f"{stage}: {len(jobs)} run(s) to do; results in {paths.root}")
            for i, key in enumerate(jobs, 1):
                out(f"[{i}/{len(jobs)}] arm {key['arm']} seed {key['seed']}")
                aid = store.launch(key, stage=stage, conditions_sha256=frozen["conditions_sha256"])
                rec = _launch_child(spec, session, aid, key, out)
                if rec["status"] != "completed":
                    out(STOP_MESSAGES[rec["status"]])
                    code = STOP_CODES[rec["status"]]
                    break
        finally:
            session.end(exit_code=code)
    return code


def retry(spec: FinalSpec, arm: str, seed: int, *, reason: str, authorized_by: str | None,
          out: Callable[[str], None] = console) -> int:
    """``retry``: one explicitly authorised restart from scratch (a new attempt, a new run directory)."""
    sc.disable_git()
    prepare_threads(spec)
    require_training_open(spec)
    if not isinstance(authorized_by, str) or not authorized_by.strip():
        raise sc.Blocked("every retry in the final experiment requires --authorized-by")
    if seed in excluded_seeds(spec.root):
        raise sc.Blocked(f"seed {seed} is excluded; its runs cannot be retried")
    paths = spec.paths
    key = {"arm": arm, "seed": seed}
    with sc.PilotLock(paths, "retry") as lock:
        require_training_open(spec)
        frozen, _ = sc.verify_or_freeze(paths, current_conditions(spec), may_create=False, lock=lock)
        store = make_store(spec, lock)
        session = sc.InvocationSession(paths, lock, store, "retry", **key, reason=reason, authorized_by=authorized_by,
                                       argv=sys.argv)
        code = 0
        try:
            registered = store.check_state()
            if registered:
                session.note("registered", attempts=registered)
            token = store.authorize_retry(key, reason=reason, authorized_by=authorized_by,
                                          conditions_sha256=frozen["conditions_sha256"])
            aid = store.launch(key, retry=token, stage="retry", conditions_sha256=frozen["conditions_sha256"])
            rec = _launch_child(spec, session, aid, key, out)
            if rec["status"] != "completed":
                out(STOP_MESSAGES[rec["status"]])
                code = STOP_CODES[rec["status"]]
        finally:
            session.end(exit_code=code)
    return code


def _nonempty(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise sc.Blocked(f"{what} must be a non-empty text")
    return value.strip()


def exclude_seed(spec: FinalSpec, seed: int, *, decision: str, reason: str, authorized_by: str) -> str:
    """Record a documented human exclusion of one seed for all four arms; returns the log head.

    Only a seed with a failed latest attempt (any non-completed status) can be excluded, so an
    exclusion is always failure-driven, never outcome-driven. Nothing is excluded automatically.
    """
    sc.disable_git()
    require_training_open(spec)
    decision, reason, authorized_by = (_nonempty(decision, "--decision"), _nonempty(reason, "--reason"),
                                       _nonempty(authorized_by, "--authorized-by"))
    paths = spec.paths
    if seed not in spec.seeds:
        raise sc.Blocked(f"seed {seed} is not a seed of the experiment")
    with sc.PilotLock(paths, "exclude-seed") as lock:
        require_training_open(spec)
        frozen, _ = sc.verify_or_freeze(paths, current_conditions(spec), may_create=False, lock=lock)
        if seed in excluded_seeds(spec.root):
            raise sc.Blocked(f"seed {seed} is already excluded")
        store = make_store(spec, lock)
        session = sc.InvocationSession(paths, lock, store, "exclude-seed", seed=seed, decision=decision,
                                       authorized_by=authorized_by)
        try:
            store.check_state()
            latest = {a: rec for (a, s), rec in latest_attempts(store).items() if s == seed}
            failed = {a: {"attempt_id": r["attempt_id"], "status": r["status"]} for a, r in latest.items()
                      if r["status"] != "completed"}
            if not failed:
                raise sc.Blocked(f"seed {seed} has no failed attempt; exclusions are failure-driven only")
            head = sc.HashChainedLog(paths.root / EXCLUSION_LOG).append(sc._plain({
                "seed": seed, "arms": list(spec.arms), "decision": decision, "reason": reason,
                "authorized_by": authorized_by, "failed_attempts": failed, "at": sc._utc_now(),
                "conditions_sha256": frozen["conditions_sha256"], "manifest_head": sc.manifest_head(paths)}))
            session.note("seed_excluded", seed=seed, decision=decision, exclusions_head=head)
        finally:
            session.end()
    return head


# ============================================================ status and validation review
def status(spec: FinalSpec) -> dict[str, Any]:
    """Status only: attempt statuses, experiment state, exclusions, review — no learning metric."""
    paths = spec.paths
    frozen = sc.load_frozen(paths)
    if frozen is None:
        return {"state": "NOT_STARTED"}
    store = make_store(spec)
    latest = latest_attempts(store)
    excl = excluded_seeds(spec.root)
    sweep = store.sweep_state()
    runs = {f"{a}/{s}": latest[(a, s)]["status"] if (a, s) in latest else "pending"
            for s in spec.seeds for a in spec.arms}
    needed = [f"{a}/{s}" for s in spec.seeds if s not in excl for a in spec.arms]
    if freeze_path(spec.root).exists():
        state = "CLOSED"
    elif sweep["halted"]:
        state = "HALTED"
    elif sweep["integrity_blocked"] or sweep["unresolved"] or sweep["launched"]:
        state = "BLOCKED"
    elif all(runs[k] == "completed" for k in needed):
        state = "COMPLETE"
    elif not validation_approved(spec, store):
        state = "AWAITING_VALIDATION_REVIEW" if _validation_attempts(store, spec) else "VALIDATION"
    else:
        state = "COMPLETION"
    return {"state": state, "runs": runs, "excluded_seeds": excl, "halted": sweep["halted"],
            "integrity_blocked": sweep["integrity_blocked"], "unresolved": sweep["unresolved"] + sweep["launched"],
            "validation_review": (read_reviews(spec.root) or [{}])[-1].get("decision"),
            "conditions_sha256": frozen["conditions_sha256"]}


def review_validation(spec: FinalSpec, *, reviewer: str, decision: str, notes: str,
                      verifier: Callable[[str | Path], Mapping[str, Any]] | None = None) -> str:
    """Record the human, status-only review of the validation stage; returns the review-log head.

    The review covers exactly the current completed validation attempts; the independent verifier
    must pass for each of them before an approval is accepted. What the reviewer is shown is
    status only (statuses and gate outcomes), never a learning metric.
    """
    sc.disable_git()
    require_training_open(spec)
    reviewer, notes = _nonempty(reviewer, "--reviewer"), _nonempty(notes, "--notes")
    if decision not in REVIEW_DECISIONS:
        raise sc.Blocked(f"--decision must be one of {REVIEW_DECISIONS}")
    paths = spec.paths
    with sc.PilotLock(paths, "review-validation") as lock:
        require_training_open(spec)
        frozen, _ = sc.verify_or_freeze(paths, current_conditions(spec), may_create=False, lock=lock)
        store = make_store(spec, lock)
        session = sc.InvocationSession(paths, lock, store, "review-validation", reviewer=reviewer, decision=decision)
        try:
            store.check_state()
            attempts = _validation_attempts(store, spec)
            if attempts is None:
                raise sc.Blocked("every validation run must be completed before the review")
            if verifier is None:
                verifier = importlib.import_module("training.wavenet_study.verify_final_runs").verify_validation
            check = verifier(paths.root)
            if decision == "approved" and check.get("status") != "pass":
                raise sc.Blocked(f"the independent verifier does not pass the validation runs: {check.get('problems')}")
            head = sc.HashChainedLog(paths.root / REVIEW_LOG).append(sc._plain({
                "reviewer": reviewer, "decision": decision, "notes": notes, "validation_attempts": attempts,
                "verification": check, "conditions_sha256": frozen["conditions_sha256"],
                "manifest_head": sc.manifest_head(paths), "at": sc._utc_now()}))
            session.note("validation_review", decision=decision, review_head=head)
        finally:
            session.end()
    return head


def resolve_category_b(spec: FinalSpec, *, event_id: str, outcome: str, reviewer: str, rationale: str,
                       evidence: str) -> str:
    """``category-b-resolve``: append one human resolution of a category B event (copied from the pilot)."""
    sc.disable_git()
    paths = spec.paths
    if not paths.conditions.is_file():
        raise sc.Blocked("the experiment has not been started")
    with sc.PilotLock(paths, "category-b-resolve") as lock:
        store = make_store(spec, lock)
        session = sc.InvocationSession(paths, lock, store, "category-b-resolve", event_id=event_id, outcome=outcome,
                                       reviewer=reviewer)
        try:
            head = sc.CategoryBLedger(paths.root).resolve({"event_id": event_id, "outcome": outcome,
                                                           "reviewer": reviewer, "timestamp": sc._utc_now(),
                                                           "rationale": rationale, "evidence": evidence})
            session.note("category_b_resolution", event_id=event_id, outcome=outcome, resolutions_head=head)
        finally:
            session.end()
    return head


def verify_records(spec: FinalSpec) -> dict[str, Any]:
    """``verify-records`` (read-only): manifest, invocations, attempt results and category B ledger."""
    store = make_store(spec)
    report = sc.verify_record_heads(spec.paths, store)
    report["attempts"] = {aid: r["status"] for aid, r in store.attempts().items()}
    report["orphans"] = store.orphans()
    report["category_b"] = sc.CategoryBLedger(spec.paths.root).heads()
    report["exclusions"] = len(read_exclusions(spec.root))
    report["validation_reviews"] = len(read_reviews(spec.root))
    return report


# ============================================================ command line
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, required=True)
    r.add_argument("--stage", choices=STAGES, required=True)
    v = sub.add_parser("review-validation")
    v.add_argument("--reviewer", required=True)
    v.add_argument("--decision", choices=REVIEW_DECISIONS, required=True)
    v.add_argument("--notes", required=True)
    t = sub.add_parser("retry")
    t.add_argument("--threads", type=int, required=True)
    t.add_argument("--arm", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--reason", required=True)
    t.add_argument("--authorized-by", required=True)
    e = sub.add_parser("exclude-seed")
    e.add_argument("--seed", type=int, required=True)
    for name in ("--decision", "--reason", "--authorized-by"):
        e.add_argument(name, required=True)
    c = sub.add_parser("category-b-resolve")
    for name in ("--event-id", "--reviewer", "--rationale", "--evidence"):
        c.add_argument(name, required=True)
    c.add_argument("--outcome", choices=sc.OUTCOMES, required=True)
    sub.add_parser("verify-records")
    d = sub.add_parser("derive-schemas")
    d.add_argument("--output", required=True)
    o = sub.add_parser("one")
    o.add_argument("--root", required=True)
    o.add_argument("--attempt-id", required=True)
    o.add_argument("--arm", required=True)
    o.add_argument("--seed", type=int, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sc.disable_git()
    if args.command == "one":
        return child_main(args.root, args.attempt_id, args.arm, args.seed)
    spec = production_spec()
    if getattr(args, "threads", spec.intra_op_threads) != spec.intra_op_threads:
        raise sc.Blocked(f"--threads must be {spec.intra_op_threads}")
    if args.command == "status":
        print(json.dumps(status(spec), indent=2))
        return 0
    if args.command == "run":
        return run_stage(spec, args.stage)
    if args.command == "retry":
        return retry(spec, args.arm, args.seed, reason=args.reason, authorized_by=args.authorized_by)
    if args.command == "review-validation":
        print(review_validation(spec, reviewer=args.reviewer, decision=args.decision, notes=args.notes))
        return 0
    if args.command == "exclude-seed":
        print(exclude_seed(spec, args.seed, decision=args.decision, reason=args.reason,
                           authorized_by=args.authorized_by))
        return 0
    if args.command == "category-b-resolve":
        print(resolve_category_b(spec, event_id=args.event_id, outcome=args.outcome, reviewer=args.reviewer,
                                 rationale=args.rationale, evidence=args.evidence))
        return 0
    if args.command == "derive-schemas":
        doc = derive_record_schemas(spec)
        Path(args.output).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    print(json.dumps(verify_records(spec), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
