"""Common epoch-budget pilot runner — Stage 4.

Canonical protocol: ``training/wavenet_study/EPOCH_BUDGET_PILOT_PROTOCOL.md`` v1.1 (D-23, D-26);
scientific rule: ``models/hierarchical/wavenet_spec.md`` v0.5 §16.5. This module orchestrates
the runs only. It computes no learning-curve statistic, no selected epoch and no budget: those
belong to ``summarize_epoch_budget_pilot.py`` (Stage 5), which also recomputes every integrity
gate independently. An extension to 60 epochs therefore needs both a human authorisation
(``--authorized-by``) and the summary script's decision that the §16.5 rule requires it; the
command line takes that decision from ``summarize_epoch_budget_pilot.extension_decision``,
which refuses unless every selection prerequisite and gate holds (protocol §7.1, DEC-8).

Commands (``python training/wavenet_study/epoch_budget_pilot.py <command> …``)::

    run --threads 1 --stage validation [--authorized-by NAME]      seed 100, all arms, cap 30
    run --threads 1 --stage completion [--authorized-by NAME]      seeds 101–102, all arms, cap 30
    run --threads 1 --stage extension --arms X … --authorized-by NAME   cap 60 (§16.5 decision)
    retry --threads 1 --arm X --cap N --seed S --reason TEXT [--authorized-by NAME]
    category-b-resolve --event-id ID --outcome O --reviewer NAME --rationale TEXT --evidence TEXT
    verify-references                                              read-only
    verify-records                                                 read-only
    one …                                                          internal: one attempt

Execution model (adapted from the accepted LR pilot, copied, never imported; AM-1): one child
process per attempt, sequential, under the exclusive lock; the child runs the accepted trainer
through ``study_common.execute_run`` (finiteness monitor, state hashes, category A/B/C,
runner-side G5/G6). Git is disabled in every process (D-15). The test split is never turned
into windows and never evaluated; its names are read only by the split-disjointness and
vocabulary-coverage checks and hashed by ``Splits.fingerprint()`` (DEC-11 with approval A1).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sc = importlib.import_module("training.wavenet_study.study_common")  # after the repository is on sys.path

# ============================================================ protocol constants (protocol §2, §5)
PROTOCOL_VERSION = "1.1"
ARMS: dict[str, dict[str, Any]] = {  # identical to the LR pilot's arm table (D-09)
    "A": {"label": "Flat-8 MLP, no BatchNorm", "config": "mlp", "params": 21_897,
          "overrides": ["model.params.context_length=8"]},
    "B": {"label": "Flat-8 MLP, BatchNorm", "config": "batchnorm_mlp", "params": 22_097,
          "overrides": ["model.params.context_length=8"]},
    "C": {"label": "Hierarchical WaveNet, no BatchNorm", "config": "wavenet", "params": 21_989, "overrides": []},
    "D": {"label": "Hierarchical WaveNet, BatchNorm", "config": "wavenet_batchnorm", "params": 22_397,
          "overrides": []},
}
LEARNING_RATES = {"A": 0.03, "B": 0.03, "C": 0.03, "D": 0.03}  # D-22
SEEDS = (100, 101, 102)
CAPS = (30, 60)
G5_EPOCHS = 10
DIAGNOSTICS = {"enabled": True, "every_n_steps": 100, "gradients": True, "parameters": True,
               "activations": False}  # option B (DEC-4)
INTRA_OP_THREADS = 1  # D-24
INTER_OP_THREADS = 8  # the LR pilot's recorded value (DEC-3, clarification 7)
EXPECTED_TRAINING = {"batch_size": 32, "lr_decay_epoch": None, "lr_decay_factor": 0.1, "eval_batch_size": 4096}
STAGES = ("validation", "completion", "extension")
RESULTS_ROOT = REPO / "results" / "experiments" / "wavenet_study" / "epoch_budget_pilot"
LR_PILOT_ROOT = REPO / "results" / "experiments" / "wavenet_study" / "lr_pilot"
LR_PILOT_SUMMARY_SHA256 = "5e52cc5bb8795b57daf640e4178724f1bf02186d392be83d1ac76be9c9eeccad"  # D-22
LR_PILOT_REFERENCE_RUNS = {  # the LR-0.03 runs of the closed LR pilot (record §8.5.1); verified on use
    f"{arm}/{seed}": f"runs/{model}/run_00{i}"
    for arm, model in (("A", "mlp"), ("B", "batchnorm_mlp"), ("C", "wavenet"), ("D", "wavenet_batchnorm"))
    for i, seed in ((1, 100), (2, 101), (3, 102))
}
REFERENCE_FILES = ("history.jsonl", "diagnostics/steps.jsonl", "model_final.pt")
FINITENESS_POLICY = {
    "parameters": "every parameter, immediately after every optimizer.step() (global post-step hook)",
    "module_outputs": "every leaf module output in every forward, training and evaluation",
    "intervention": "none (read-only)", "cost": "measured per epoch in timing.json",
}
RECORDED_DOCUMENTS = ("models/hierarchical/wavenet_spec.md", "models/feedforward/mlp_spec.md",
                      "models/feedforward/batchnorm_mlp_spec.md", "UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md",
                      "training/wavenet_study/EPOCH_BUDGET_PILOT_PROTOCOL.md", "PROJECT_RESEARCH_RECORD.md")
PILOT_CODE = ("training/wavenet_study/study_common.py", "training/wavenet_study/epoch_budget_pilot.py",
              "training/wavenet_study/summarize_epoch_budget_pilot.py")
KEY_FIELDS = ("arm", "cap", "seed")


# ============================================================ pilot specification
@dataclass(frozen=True)
class PilotSpec:
    """Every setting of one pilot. Frozen into ``conditions.json`` (as ``spec``) at the first run.

    The command-line interface always uses :func:`production_spec`. Synthetic tests construct
    their own spec (temporary root, synthetic data and references, small caps); its different
    root and frozen conditions keep it apart from the real pilot.
    """

    root: str
    repo: str
    arms: dict[str, dict[str, Any]]
    learning_rates: dict[str, float]
    seeds: list[int]
    caps: list[int]
    g5_epochs: int
    diagnostics: dict[str, Any]
    intra_op_threads: int
    inter_op_threads: int
    reference_root: str
    reference_runs: dict[str, str]
    reference_summary: str
    reference_summary_sha256: str
    blocking_files: list[str]
    recorded_files: list[str]
    data_path: str | None = None
    expected_training: dict[str, Any] = field(default_factory=lambda: dict(EXPECTED_TRAINING))
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if len(self.caps) != 2 or not 0 < self.g5_epochs <= self.caps[0] < self.caps[1]:
            raise ValueError(f"need 0 < g5_epochs <= caps[0] < caps[1], got {self.g5_epochs}, {self.caps}")
        if set(self.learning_rates) != set(self.arms) or not self.seeds:
            raise ValueError("learning rates must be given for exactly the spec's arms, and seeds must be non-empty")
        missing = [f"{a}/{s}" for a in self.arms for s in self.seeds if f"{a}/{s}" not in self.reference_runs]
        if missing:
            raise ValueError(f"reference runs missing for {missing}")

    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PilotSpec:
        return cls(**dict(data))

    @property
    def paths(self) -> sc.PilotPaths:
        return sc.PilotPaths(Path(self.root))


def production_spec() -> PilotSpec:
    """The approved pilot (protocol v1.1)."""
    blocking = ["models/feedforward/mlp.py", "models/feedforward/batchnorm_mlp.py", "models/hierarchical/wavenet.py",
                *sc.python_files_under(REPO, "utils"), "configs/base.yaml",
                *(f"configs/models/{ARMS[a]['config']}.yaml" for a in ARMS), "data/names.txt", *PILOT_CODE]
    return PilotSpec(root=str(RESULTS_ROOT), repo=str(REPO), arms=ARMS, learning_rates=LEARNING_RATES,
                     seeds=list(SEEDS), caps=list(CAPS), g5_epochs=G5_EPOCHS, diagnostics=DIAGNOSTICS,
                     intra_op_threads=INTRA_OP_THREADS, inter_op_threads=INTER_OP_THREADS,
                     reference_root=str(LR_PILOT_ROOT), reference_runs=LR_PILOT_REFERENCE_RUNS,
                     reference_summary="summary.json", reference_summary_sha256=LR_PILOT_SUMMARY_SHA256,
                     blocking_files=blocking, recorded_files=list(RECORDED_DOCUMENTS))


def make_store(spec: PilotSpec, lock: sc.PilotLock | None = None) -> sc.AttemptStore:
    return sc.AttemptStore(spec.paths, key_fields=KEY_FIELDS, lock=lock,
                           attempt_id=lambda k, n: f"{k['arm']}-c{k['cap']}-s{k['seed']}-a{n}")


# ============================================================ threads, references, conditions
def prepare_threads(spec: PilotSpec) -> None:
    """Intra-op threads as frozen; inter-op threads must already equal the frozen value."""
    import torch

    torch.set_num_threads(spec.intra_op_threads)
    if torch.get_num_threads() != spec.intra_op_threads:
        raise sc.Blocked(f"requested {spec.intra_op_threads} intra-op threads, torch uses {torch.get_num_threads()}")
    if torch.get_num_interop_threads() != spec.inter_op_threads:
        raise sc.Blocked(f"inter-op threads are {torch.get_num_interop_threads()}, the pilot requires "
                         f"{spec.inter_op_threads}")


def _reference_dir(spec: PilotSpec, arm: str, seed: int) -> Path:
    return Path(spec.reference_root) / spec.reference_runs[f"{arm}/{seed}"]


def reference_record(spec: PilotSpec) -> dict[str, Any]:
    """Verify and hash the closed LR pilot's reference files (read only; protocol §10.1, §17)."""
    root = Path(spec.reference_root)
    summary = root / spec.reference_summary
    if not summary.is_file() or sc.sha256_file(summary) != spec.reference_summary_sha256:
        raise sc.Blocked(f"LR-pilot summary {summary} does not have the SHA-256 recorded by D-22")
    out: dict[str, Any] = {"summary": {"path": sc._repo_relative(summary), "sha256": spec.reference_summary_sha256},
                           "runs": {}}
    for arm in spec.arms:
        for seed in spec.seeds:
            d = _reference_dir(spec, arm, seed)
            try:
                res = json.loads((d / "pilot_result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise sc.Blocked(f"reference run {d}: unreadable pilot_result.json ({exc})") from None
            expected = {"arm": arm, "lr": spec.learning_rates[arm], "seed": seed, "status": "completed"}
            if {k: res.get(k) for k in expected} != expected:
                raise sc.Blocked(f"reference run {d} is not the completed LR-{spec.learning_rates[arm]} run of "
                                 f"arm {arm}, seed {seed}")
            files = {}
            for rel in REFERENCE_FILES:
                if not (d / rel).is_file():
                    raise sc.Blocked(f"reference file missing: {d / rel}")
                files[rel] = sc.sha256_file(d / rel)
            out["runs"][f"{arm}/{seed}"] = {"run_dir": sc._repo_relative(d), "files": files}
    return out


def arm_config(spec: PilotSpec, arm: str, seed: int | None):
    """The resolved experiment configuration of one arm (as in the LR pilot; ``seed=None``: without it)."""
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


def _config_without_seed(cfg: Any) -> dict[str, Any]:
    from utils.config import config_to_dict

    d = config_to_dict(cfg)
    d["reproducibility"] = {k: v for k, v in d["reproducibility"].items() if k != "seed"}
    return d


def current_conditions(spec: PilotSpec) -> dict[str, Any]:
    """Everything that must stay identical for the whole pilot (protocol §10.1)."""
    return {
        "spec": spec.to_json(),
        "protocol": {"version": spec.protocol_version, "seeds": spec.seeds, "caps": spec.caps,
                     "g5_epochs": spec.g5_epochs, "g6_epochs": spec.caps[0], "stages": list(STAGES),
                     "epsilon": "max(1e-3, SE)", "epsilon_labels": ["SE-derived", "floor-derived", "floor (SE equal)"]},
        "learning_rates": {a: {"lr": lr, "decision": "D-22", "lr_pilot_summary_sha256": spec.reference_summary_sha256}
                           for a, lr in spec.learning_rates.items()},
        "references": reference_record(spec),
        "arms": {a: {"label": s["label"], "expected_parameters": s["params"],
                     "config": _config_without_seed(arm_config(spec, a, None))} for a, s in spec.arms.items()},
        "expected_training": spec.expected_training,
        "diagnostics": spec.diagnostics,
        "finiteness_policy": FINITENESS_POLICY,
        "environment": sc.environment_record(),
        "checksums": sc.checksum_record(spec.repo, blocking=spec.blocking_files, recorded=spec.recorded_files),
    }


# ============================================================ one arm (set-up through the accepted code only)
def build_arm(spec: PilotSpec, arm: str, seed: int, cap: int, phase: list[str]) -> dict[str, Any]:
    """Configuration, data, model and trainer of one arm, adapted line by line from the LR pilot's
    ``build_arm``. ``phase[0]`` names the set-up step, so a failure says where it happened."""
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
    # The test split is never turned into windows and never evaluated (protocol §4, DEC-11).

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
    training = train_cls(lr=spec.learning_rates[arm], max_epochs=cap)  # every other setting: trainer defaults
    expected = {**spec.expected_training, "max_epochs": cap}
    actual = {f: getattr(training, f) for f in expected}
    if actual != expected:
        raise RuntimeError(f"training defaults changed: expected {expected}, got {actual}")
    fingerprint = {**dataset_fingerprint(cfg.data.path), "splits": splits.fingerprint()}  # A1: hashes only
    normalization = "batchnorm" if spec.arms[arm]["config"] in ("batchnorm_mlp", "wavenet_batchnorm") else "none"
    return {"module": module, "cfg": cfg, "model": model, "model_name": model_name, "n_params": n_params,
            "training": training, "train_fn": train_fn, "train": train, "dev": dev, "fingerprint": fingerprint,
            "normalization": normalization, "device": str(params[0].device), "dtype": str(params[0].dtype)}


# ============================================================ references for G5 / G6 and schemas
def _verified_reference_file(spec: PilotSpec, frozen: Mapping[str, Any], arm: str, seed: int, rel: str) -> Path:
    path = _reference_dir(spec, arm, seed) / rel
    if sc.sha256_file(path) != frozen["conditions"]["references"]["runs"][f"{arm}/{seed}"]["files"][rel]:
        raise sc.Blocked(f"reference file {path} differs from its frozen SHA-256")
    return path


def _run_path(run_dir: str) -> Path:
    p = Path(run_dir)
    return p if p.is_absolute() else REPO / p


def prefix_checks_for(spec: PilotSpec, frozen: Mapping[str, Any], store: sc.AttemptStore, arm: str, cap: int,
                      seed: int) -> tuple[list[sc.PrefixCheck], sc.RecordSchemas]:
    """Runner-side G5 (and G6 for extension runs) references and the arm's frozen record schemas."""
    history = sc.read_jsonl_strict(_verified_reference_file(spec, frozen, arm, seed, "history.jsonl"))
    steps = sc.read_jsonl_strict(_verified_reference_file(spec, frozen, arm, seed, "diagnostics/steps.jsonl"))
    ref_hash = sc.checkpoint_state_hash(_verified_reference_file(spec, frozen, arm, seed, "model_final.pt"))
    schemas = sc.RecordSchemas(sc.record_schema(history), sc.record_schema(steps))
    checks = [sc.PrefixCheck("G5", spec.g5_epochs, tuple(history), tuple(steps), ref_hash,
                             f"LR-pilot run {spec.reference_runs[f'{arm}/{seed}']} (model_final.pt state)")]
    if cap == spec.caps[1]:
        recs = store.by_key().get((arm, spec.caps[0], seed), [])
        if not recs or recs[-1]["status"] != "completed" or not recs[-1].get("run_dir"):
            raise sc.Blocked(f"G6 reference missing: no completed cap-{spec.caps[0]} run for arm {arm}, seed {seed}")
        d = _run_path(recs[-1]["run_dir"])
        hashes = sc.read_jsonl_strict(d / sc.STATE_HASH_FILE)
        if len(hashes) != spec.caps[0] or hashes[-1]["epoch"] != spec.caps[0] - 1:
            raise sc.Blocked(f"G6 reference {d}: state-hash records incomplete")
        if hashes[-1]["sha256"] != sc.checkpoint_state_hash(d / "model_final.pt"):
            raise sc.Blocked(f"G6 reference {d}: epoch-{spec.caps[0]} hash differs from model_final.pt")
        checks.append(sc.PrefixCheck("G6", spec.caps[0], tuple(sc.read_jsonl_strict(d / "history.jsonl")),
                                     tuple(sc.read_jsonl_strict(d / "diagnostics" / "steps.jsonl")),
                                     hashes[-1]["sha256"], f"cap-{spec.caps[0]} run {recs[-1]['run_dir']}"))
    return checks, schemas


# ============================================================ child: one attempt
def _launched_head(spec: PilotSpec, attempt_id: str) -> str:
    """The manifest head right after this attempt's ``launched`` row, which must be the last row."""
    entries, head = sc.HashChainedLog(spec.paths.manifest).read()
    if not entries or entries[-1].get("event") != "launched" or entries[-1].get("attempt_id") != attempt_id:
        raise sc.Blocked(f"the last manifest row is not the launch of {attempt_id}")
    return head


def _failure(exc: BaseException, phase: str) -> dict[str, Any]:
    import traceback

    return {"type": type(exc).__name__, "message": str(exc)[:2000], "phase": phase,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]}


def child_main(root: str, attempt_id: str, arm: str, cap: int, seed: int) -> int:
    """Run one launched attempt; write its result file once; return the exit code."""
    sc.disable_git()
    paths = sc.PilotPaths(Path(root))
    key = {"arm": arm, "cap": cap, "seed": seed}
    frozen = sc.load_frozen(paths)
    if frozen is None:
        raise sc.Blocked("no frozen conditions")
    spec = PilotSpec.from_json(frozen["conditions"]["spec"])
    if Path(spec.root) != paths.root:
        raise sc.Blocked("the frozen spec belongs to another pilot root")
    store = make_store(spec)
    rec = store.verify_launched(attempt_id, key)  # refuses unless launched by an active invocation
    head = _launched_head(spec, attempt_id)
    base = {"attempt_id": attempt_id, **key, "stage": rec.get("stage"), "retry_of": rec.get("retry_of"),
            "manifest_head": head, "conditions_sha256": frozen["conditions_sha256"], "started_at": sc._utc_now()}
    phase = ["setup:consistency"]
    try:
        prepare_threads(spec)
        sc.verify_or_freeze(paths, current_conditions(spec), may_create=False, lock=None)
        phase[0] = "setup:references"
        checks, schemas = prefix_checks_for(spec, frozen, store, arm, cap, seed)
        b = build_arm(spec, arm, seed, cap, phase)
        phase[0] = "setup:run_directory"
        from utils.config import config_to_dict
        from utils.run_manager import RunManager

        experiment = config_to_dict(b["cfg"])
        pilot = {"study": "wavenet_study", "stage": "epoch_budget_pilot", "pilot_stage": rec.get("stage"),
                 "spec": "models/hierarchical/wavenet_spec.md v0.5 §16.5; protocol v" + spec.protocol_version,
                 "attempt_id": attempt_id, **key, "lr": spec.learning_rates[arm], "arm_label": spec.arms[arm]["label"],
                 "parameters": b["n_params"], "train_windows": b["train"].num_examples,
                 "dev_windows": b["dev"].num_examples, "model_device": b["device"], "model_dtype": b["dtype"]}
        run = RunManager.create(b["model_name"],
                                config={**b["module"].run_config(experiment, b["model"].config, b["training"]),
                                        "pilot": pilot},
                                results_root=paths.root, dataset_fingerprint=b["fingerprint"],
                                extra_metadata={"pilot": pilot, "conditions_sha256": frozen["conditions_sha256"]})
    except KeyboardInterrupt as exc:
        return _write_child_result(paths, {**base, "status": "interrupted", "phase": phase[0], "run_dir": None,
                                           "failure": _failure(exc, phase[0])})
    except (Exception, SystemExit) as exc:  # noqa: BLE001  (set-up failure: a software error, never evidence)
        return _write_child_result(paths, {**base, "status": "error", "phase": phase[0], "run_dir": None,
                                           "failure": _failure(exc, phase[0])})
    prepared = sc.PreparedRun(b["model"], b["train"], b["dev"], b["training"], b["train_fn"], run,
                              b["cfg"].diagnostics, experiment)
    context = sc.RunContext(attempt_id, arm, seed, cap, b["normalization"],
                            {"stage": rec.get("stage"), "pilot": "epoch_budget_pilot"})

    def progress(info: Mapping[str, Any]) -> None:  # status-only: progress and time, no learning metrics
        print(f"  [{attempt_id}] epoch {info['epoch']}/{info['cap']} done ({info['seconds']:.1f}s)", flush=True)

    result = sc.execute_run(prepared, context, schemas=schemas, ledger=sc.CategoryBLedger(paths.root),
                            prefix_checks=checks, on_epoch=progress)
    run_dir = Path(run.run_dir)
    gates = []
    if result["status"] == "completed":  # informational; the summary recomputes every gate independently
        history = sc.read_jsonl_strict(run_dir / "history.jsonl")
        gates = [g.to_dict() for g in (sc.gate_g1(history, cap), sc.gate_g4(run_dir, cap=cap), sc.gate_g7(run_dir))]
    return _write_child_result(paths, {**base, **result, "runner_gates": gates})


def _write_child_result(paths: sc.PilotPaths, result: dict[str, Any]) -> int:
    sc.write_attempt_result(paths, result["attempt_id"], {**result, "ended_at": sc._utc_now()})
    return {"completed": 0, "mechanically_unstable": 0, "integrity_blocked": 0, "interrupted": 130}.get(
        result["status"], 1)


# ============================================================ parent: invocations
def console(line: str) -> None:
    """Default progress output: flushed per line, so it is visible at once even when stdout is a
    pipe (``| tee``, a log collector, a parent process); block buffering would hold it until exit."""
    print(line, flush=True)


def _launch_child(spec: PilotSpec, session: sc.InvocationSession, aid: str, key: Mapping[str, Any],
                  out: Callable[[str], None]) -> dict[str, Any]:
    """Run one launched attempt in a child process and close it in the manifest."""
    paths = spec.paths
    session.note("attempt_launched", attempt_id=aid, **key)
    cmd = [sys.executable, str(Path(__file__).resolve()), "one", "--root", str(paths.root), "--attempt-id", aid,
           "--arm", key["arm"], "--cap", str(key["cap"]), "--seed", str(key["seed"])]
    env = {**os.environ, "OMP_NUM_THREADS": str(spec.intra_op_threads), "MKL_NUM_THREADS": str(spec.intra_op_threads)}
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = paths.logs_dir / f"{aid}.log"
    t0, interrupted = time.perf_counter(), False
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
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
    "mechanically_unstable": "pilot HALTED after mechanical instability (protocol §9.4); human review required",
    "integrity_blocked": "sweep integrity_blocked (protocol §10.5); human review and an authorised retry required",
    "error": "sweep stopped after a software / infrastructure error; inspect the attempt log",
    "crashed": "sweep stopped: the attempt ended without a result; inspect it",
    "interrupted": "sweep interrupted; inspect the attempt, then retry it explicitly",
}


def _stage_keys(spec: PilotSpec, store: sc.AttemptStore, stage: str, arms: Sequence[str] | None,
                authorized_by: str | None,
                extension_decision: Callable[[PilotSpec, Sequence[str]], Mapping[str, bool]] | None) -> list[dict]:
    """The keys a stage covers, after its prerequisites (protocol §5.2, §7.1; DEC-8)."""
    by_key = store.by_key()

    def completed(keys: Sequence[tuple]) -> bool:
        return all(by_key.get(k) and by_key[k][-1]["status"] == "completed" for k in keys)

    initial, extended = spec.caps
    if stage == "validation":
        if arms:
            raise sc.Blocked("the validation stage always covers every arm")
        return [{"arm": a, "cap": initial, "seed": spec.seeds[0]} for a in spec.arms]
    if stage == "completion":
        if arms:
            raise sc.Blocked("the completion stage always covers every arm")
        if not completed([(a, initial, spec.seeds[0]) for a in spec.arms]):
            raise sc.Blocked("the completion stage needs every validation run completed (and reviewed)")
        return [{"arm": a, "cap": initial, "seed": s} for a in spec.arms for s in spec.seeds[1:]]
    if stage != "extension":
        raise sc.Blocked(f"unknown stage {stage!r}; stages: {', '.join(STAGES)}")
    if not isinstance(authorized_by, str) or not authorized_by.strip():
        raise sc.Blocked("an extension runs only after human authorisation: --authorized-by is required (§7.1)")
    arms = list(dict.fromkeys(arms or []))
    if not arms or any(a not in spec.arms for a in arms):
        raise sc.Blocked(f"--arms must name arms of the pilot ({', '.join(spec.arms)})")
    if not completed([(a, initial, s) for a in spec.arms for s in spec.seeds]):
        raise sc.Blocked(f"extensions need every cap-{initial} run of every arm completed")
    if extension_decision is None:
        raise sc.Blocked("extension decisions come only from the summary script's selection mode (Stage 5), which "
                         "is not available; no extension can be run")
    # The decision is computed only now: after authorisation and after every cap-30 run is completed.
    # It refuses (sc.Blocked) unless every selection prerequisite and integrity gate holds. A True
    # decision says only that the §16.5 rule requires the extension; it never replaces --authorized-by.
    try:
        decision = extension_decision(spec, arms)
    except sc.Blocked:
        raise
    except Exception as exc:  # noqa: BLE001  (any failure of the decision refuses the extension)
        raise sc.Blocked(f"the extension decision failed ({type(exc).__name__}: {exc}); no extension is run") from None
    if not isinstance(decision, Mapping) or any(not isinstance(decision.get(a), bool) for a in arms):
        raise sc.Blocked(f"invalid extension decision {decision!r}: expected true/false for each of {arms}")
    refused = [a for a in arms if decision[a] is not True]
    if refused:
        raise sc.Blocked(f"the §16.5 rule does not require an extension for arm(s) {refused}")
    return [{"arm": a, "cap": extended, "seed": s} for a in arms for s in spec.seeds]


def run_stage(spec: PilotSpec, stage: str, *, arms: Sequence[str] | None = None, authorized_by: str | None = None,
              extension_decision: Callable[[PilotSpec, Sequence[str]], Mapping[str, bool]] | None = None,
              out: Callable[[str], None] = console) -> int:
    """``run --stage …``: launch the stage's missing attempts sequentially; stop on any non-completed one."""
    sc.disable_git()
    prepare_threads(spec)
    paths = spec.paths
    with sc.PilotLock(paths, f"run {stage}") as lock:
        frozen, nonblocking = sc.verify_or_freeze(paths, current_conditions(spec), may_create=stage == "validation",
                                                  lock=lock)
        store = make_store(spec, lock)
        session = sc.InvocationSession(paths, lock, store, "run", stage=stage, arms=list(arms or []),
                                       authorized_by=authorized_by, argv=sys.argv,
                                       recorded_document_changes=[list(p) for p in nonblocking])
        code = 0
        try:
            registered = store.check_state()
            if registered:
                session.note("registered", attempts=registered)
            store.require_runnable()
            jobs = store.pending(_stage_keys(spec, store, stage, arms, authorized_by, extension_decision))
            out(f"{stage}: {len(jobs)} run(s) to do; results in {paths.root}")
            for i, key in enumerate(jobs, 1):
                out(f"[{i}/{len(jobs)}] arm {key['arm']} cap {key['cap']} seed {key['seed']}")
                aid = store.launch(key, stage=stage, authorized_by=authorized_by,
                                   conditions_sha256=frozen["conditions_sha256"])
                rec = _launch_child(spec, session, aid, key, out)
                if rec["status"] != "completed":
                    out(STOP_MESSAGES[rec["status"]])
                    code = STOP_CODES[rec["status"]]
                    break
        finally:
            session.end(exit_code=code)
    return code


def retry(spec: PilotSpec, arm: str, cap: int, seed: int, *, reason: str, authorized_by: str | None = None,
          out: Callable[[str], None] = console) -> int:
    """``retry``: one explicit, reasoned re-run from scratch (protocol §10.5, §12.3; approval A4)."""
    sc.disable_git()
    prepare_threads(spec)
    paths = spec.paths
    key = {"arm": arm, "cap": cap, "seed": seed}
    with sc.PilotLock(paths, "retry") as lock:
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


def resolve_category_b(spec: PilotSpec, *, event_id: str, outcome: str, reviewer: str, rationale: str,
                       evidence: str) -> str:
    """``category-b-resolve``: append one human resolution (protocol §9.7). No run is started.

    The frozen conditions are deliberately not required to match: a ``corrected`` resolution
    is typically recorded after the corrected code was changed (a new execution still needs
    separate authorisation and matching conditions).
    """
    sc.disable_git()
    paths = spec.paths
    if not paths.conditions.is_file():
        raise sc.Blocked("the pilot has not been started")
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


def verify_records(spec: PilotSpec) -> dict[str, Any]:
    """``verify-records`` (read-only): manifest, invocations, attempt results and category B ledger."""
    store = make_store(spec)
    report = sc.verify_record_heads(spec.paths, store)
    report["attempts"] = {aid: r["status"] for aid, r in store.attempts().items()}
    report["orphans"] = store.orphans()
    report["category_b"] = sc.CategoryBLedger(spec.paths.root).heads()
    return report


# ============================================================ command line
SUMMARY_MODULE = "training.wavenet_study.summarize_epoch_budget_pilot"


def _load_extension_decision() -> Callable[[PilotSpec, Sequence[str]], Mapping[str, bool]]:
    """The summary script's §16.5 extension decision, for ``run --stage extension`` only.

    Imported here, not at module level: the summary script imports this module, so a module-level
    import would be circular. Loading the function authorises nothing and computes nothing;
    :func:`_stage_keys` still requires ``--authorized-by`` and every completed cap-30 run first.
    A failure to load refuses the extension before anything is locked or written.
    """
    try:
        fn = importlib.import_module(SUMMARY_MODULE).extension_decision
    except (Exception, sc.Blocked) as exc:  # noqa: BLE001  (Blocked is a SystemExit)
        raise sc.Blocked(f"the summary script's extension decision could not be loaded ({type(exc).__name__}: "
                         f"{exc}); no extension can be run") from None
    if not callable(fn):
        raise sc.Blocked(f"{SUMMARY_MODULE}.extension_decision is not callable; no extension can be run")
    return fn


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, required=True)
    r.add_argument("--stage", choices=STAGES, required=True)
    r.add_argument("--arms", nargs="+", default=None)
    r.add_argument("--authorized-by", default=None)
    t = sub.add_parser("retry")
    t.add_argument("--threads", type=int, required=True)
    t.add_argument("--arm", required=True)
    t.add_argument("--cap", type=int, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--reason", required=True)
    t.add_argument("--authorized-by", default=None)
    c = sub.add_parser("category-b-resolve")
    for name in ("--event-id", "--reviewer", "--rationale", "--evidence"):
        c.add_argument(name, required=True)
    c.add_argument("--outcome", choices=sc.OUTCOMES, required=True)
    sub.add_parser("verify-references")
    sub.add_parser("verify-records")
    o = sub.add_parser("one")
    o.add_argument("--root", required=True)
    o.add_argument("--attempt-id", required=True)
    o.add_argument("--arm", required=True)
    o.add_argument("--cap", type=int, required=True)
    o.add_argument("--seed", type=int, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sc.disable_git()
    if args.command == "one":
        return child_main(args.root, args.attempt_id, args.arm, args.cap, args.seed)
    spec = production_spec()
    if getattr(args, "threads", spec.intra_op_threads) != spec.intra_op_threads:
        raise sc.Blocked(f"--threads must be {spec.intra_op_threads} (D-24)")
    if args.command == "run":
        decision = _load_extension_decision() if args.stage == "extension" else None
        return run_stage(spec, args.stage, arms=args.arms, authorized_by=args.authorized_by,
                         extension_decision=decision)
    if args.command == "retry":
        return retry(spec, args.arm, args.cap, args.seed, reason=args.reason, authorized_by=args.authorized_by)
    if args.command == "category-b-resolve":
        print(resolve_category_b(spec, event_id=args.event_id, outcome=args.outcome, reviewer=args.reviewer,
                                 rationale=args.rationale, evidence=args.evidence))
        return 0
    if args.command == "verify-references":
        print(json.dumps(reference_record(spec), indent=2))
        return 0
    print(json.dumps(verify_records(spec), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
