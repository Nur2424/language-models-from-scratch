"""Stage 4: the multi-run epoch-budget pilot runner, on synthetic data and synthetic references only.

Everything runs under pytest's temporary folder: a synthetic names file (all 26 letters, so the
accepted configurations give the approved parameter counts), synthetic "LR-pilot" reference runs
produced with the accepted trainers exactly as the LR pilot built them, and a test ``PilotSpec``
with small caps (3 → 6 epochs, G5 at 2). The runner starts real child processes
(``epoch_budget_pilot.py one …``). Faults are injected only from the test side, through a
``sitecustomize`` module placed on ``PYTHONPATH`` for the child processes. The real pilot folder
is never touched (checked at the end).
"""

from __future__ import annotations

import json
import os
import random
import signal
import string
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch

from training.wavenet_study import epoch_budget_pilot as ebp
from training.wavenet_study import study_common as sc

REPO = ebp.REPO
CAPS, G5 = [3, 6], 2
EVERY = 7  # step-diagnostic interval of the test spec (option B uses 100; small data needs more records)


# ============================================================ synthetic world
def _names() -> list[str]:
    from utils.data import make_context_windows, split_names
    from utils.vocabulary import Vocabulary

    rng = random.Random(1234)
    n = 260
    while True:
        names, seen = [], set()
        while len(names) < n:
            name = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 7)))
            if name not in seen:
                seen.add(name)
                names.append(name)
        splits = split_names(names, fractions=(0.8, 0.1, 0.1), seed=42, duplicate_policy="group")
        vocab = Vocabulary.from_names(splits.train)
        windows = make_context_windows(splits.train, vocab, context_length=8).num_examples
        letters = set("".join(splits.train))
        if letters == set(string.ascii_lowercase) and windows % 32 != 1 \
                and set("".join(splits.dev + splits.test)) <= letters:
            return names
        n += 1


def lr_style_reference(arm: str, seed: int, *, data: Path, root: Path, epochs: int) -> Path:
    """A reference run built exactly like the LR pilot's ``build_arm`` + child (test-side copy)."""
    from models.feedforward import batchnorm_mlp, mlp
    from models.hierarchical import wavenet
    from utils.config import config_to_dict, default_config_paths, load_config
    from utils.data import load_names, make_context_windows, split_names
    from utils.reproducibility import derive_seed, make_generator, seed_everything
    from utils.run_manager import RunManager
    from utils.vocabulary import Vocabulary

    spec_arm = ebp.ARMS[arm]
    module, model_cls, cfg_cls, train_cls, train_fn = {
        "A": (mlp, mlp.MLPLanguageModel, mlp.MLPConfig, mlp.MLPTrainingConfig, mlp.train_mlp),
        "B": (batchnorm_mlp, batchnorm_mlp.BatchNormMLPLanguageModel, batchnorm_mlp.BatchNormMLPConfig,
              batchnorm_mlp.BatchNormMLPTrainingConfig, batchnorm_mlp.train_batchnorm_mlp),
        "C": (wavenet, wavenet.WaveNetLanguageModel, wavenet.WaveNetConfig, wavenet.WaveNetTrainingConfig,
              wavenet.train_wavenet),
        "D": (wavenet, wavenet.WaveNetLanguageModel, wavenet.WaveNetConfig, wavenet.WaveNetTrainingConfig,
              wavenet.train_wavenet),
    }[arm]
    diag = {**ebp.DIAGNOSTICS, "every_n_steps": EVERY}
    overrides = [*spec_arm["overrides"], "device.device=cpu", *(f"diagnostics.{k}={str(v).lower()}" for k, v in
                                                                diag.items()),
                 f"data.path={json.dumps(str(data))}", f"reproducibility.seed={seed}"]
    cfg = load_config(default_config_paths(spec_arm["config"]), overrides=overrides, model_config_cls=cfg_cls)
    splits = split_names(load_names(cfg.data.path), fractions=cfg.data.split_fractions, seed=cfg.data.split_seed,
                         duplicate_policy=cfg.data.duplicate_policy)
    vocab = Vocabulary.from_names(splits.train)
    train = make_context_windows(splits.train, vocab, context_length=8)
    dev = make_context_windows(splits.dev, vocab, context_length=8)
    torch.set_num_threads(1)
    seed_everything(seed)
    model = model_cls(vocab, cfg.model.params, generator=make_generator(derive_seed(seed, "init")))
    training = train_cls(lr=0.03, max_epochs=epochs)
    experiment = config_to_dict(cfg)
    name = getattr(model, "model_name", None) or module.MODEL_NAME
    run = RunManager.create(name, config=module.run_config(experiment, model.config, training), results_root=root)
    with run.track():
        train_fn(model, train, dev, training=training, seed=seed, run=run, diagnostics=cfg.diagnostics,
                 config=experiment)
    (run.run_dir / "pilot_result.json").write_text(json.dumps({"arm": arm, "lr": 0.03, "seed": seed,
                                                                "status": "completed"}))
    return Path(run.run_dir)


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("ebp_world")
    data = base / "names.txt"
    data.write_text("\n".join(_names()) + "\n")
    ref_root = base / "lr_pilot_reference"
    runs = {}
    for arm in "ABCD":
        for seed in ebp.SEEDS:
            d = lr_style_reference(arm, seed, data=data, root=ref_root, epochs=G5)
            runs[f"{arm}/{seed}"] = d.relative_to(ref_root).as_posix()
    (ref_root / "summary.json").write_text(json.dumps({"synthetic": True}))
    doc = base / "protocol_doc.md"
    doc.write_text("recorded, non-blocking document\n")
    return {"base": base, "data": data, "ref_root": ref_root, "runs": runs, "doc": doc}


def make_spec(world, root: Path, *, arms="ABCD", **over) -> ebp.PilotSpec:
    blocking = ["models/feedforward/mlp.py", "models/feedforward/batchnorm_mlp.py", "models/hierarchical/wavenet.py",
                *sc.python_files_under(REPO, "utils"), "configs/base.yaml",
                *(f"configs/models/{ebp.ARMS[a]['config']}.yaml" for a in ebp.ARMS), str(world["data"]),
                "training/wavenet_study/study_common.py", "training/wavenet_study/epoch_budget_pilot.py"]
    fields = {"root": str(root), "repo": str(REPO), "arms": {a: ebp.ARMS[a] for a in arms},
              "learning_rates": {a: 0.03 for a in arms}, "seeds": list(ebp.SEEDS), "caps": CAPS, "g5_epochs": G5,
              "diagnostics": {**ebp.DIAGNOSTICS, "every_n_steps": EVERY}, "intra_op_threads": 1,
              "inter_op_threads": torch.get_num_interop_threads(), "reference_root": str(world["ref_root"]),
              "reference_runs": world["runs"], "reference_summary": "summary.json",
              "reference_summary_sha256": sc.sha256_file(world["ref_root"] / "summary.json"),
              "blocking_files": blocking, "recorded_files": [str(world["doc"])], "data_path": str(world["data"])}
    fields.update(over)
    return ebp.PilotSpec(**fields)


def quiet(lines):
    return lambda line: lines.append(line)


def results(spec):
    store = ebp.make_store(spec)
    return {aid: store.read_result(aid, rec["key"]) for aid, rec in store.attempts().items()}


# ============================================================ fault injection (child processes only)
SITECUSTOMIZE = textwrap.dedent("""
    import os, sys
    mode, target = os.environ.get("EBP_FAULT"), os.environ.get("EBP_FAULT_ATTEMPT", "")
    if mode and target in sys.argv:
        step_at = int(os.environ.get("EBP_FAULT_STEP", "5"))
        if mode == "category_b":
            sys.path.insert(0, os.environ["EBP_REPO"])
            from utils.diagnostics import parameters
            original = parameters.ParameterSnapshot.update_stats
            calls = [0]
            def update_stats(self, model, **kw):
                out = original(self, model, **kw)
                calls[0] += 1
                if calls[0] == step_at:
                    name = sorted(out["per_parameter"])[0]
                    out["per_parameter"][name]["relative_update"] = None
                return out
            parameters.ParameterSnapshot.update_stats = update_stats
        else:
            import torch
            from torch.optim.optimizer import register_optimizer_step_post_hook
            count = [0]
            def fault(optimizer, args, kwargs):
                count[0] += 1
                if count[0] != step_at:
                    return
                if mode == "crash":
                    os._exit(9)
                if mode == "nan":
                    with torch.no_grad():
                        optimizer.param_groups[0]["params"][0].view(-1)[0] = float("nan")
                if mode == "error":
                    raise RuntimeError("injected software failure")
                if mode == "interrupt":
                    raise KeyboardInterrupt
                if mode == "block":  # tell the test, then block inside the attempt until killed
                    import socket
                    conn = socket.create_connection(("127.0.0.1", int(os.environ["EBP_FAULT_PORT"])))
                    conn.sendall(b"blocked\\n")
                    conn.recv(1)  # never answered: returns only when the process is killed
            register_optimizer_step_post_hook(fault)  # registered before the monitor's hook
""")


@pytest.fixture()
def inject(tmp_path, monkeypatch):
    d = tmp_path / "fault_site"
    d.mkdir()
    (d / "sitecustomize.py").write_text(SITECUSTOMIZE)

    def set_fault(mode, attempt, step=5):
        monkeypatch.setenv("PYTHONPATH", str(d))
        monkeypatch.setenv("EBP_FAULT", mode)
        monkeypatch.setenv("EBP_FAULT_ATTEMPT", attempt)
        monkeypatch.setenv("EBP_FAULT_STEP", str(step))
        monkeypatch.setenv("EBP_REPO", str(REPO))

    def clear():
        monkeypatch.delenv("EBP_FAULT", raising=False)

    set_fault.clear = clear
    return set_fault


# ============================================================ end-to-end flow (four arms)
@pytest.fixture(scope="session")
def e2e(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("ebp_e2e") / "pilot"
    spec = make_spec(world, root)
    lines: list[str] = []
    refusals = {}
    ref_hashes = {p: sc.sha256_file(p) for p in world["ref_root"].rglob("*") if p.is_file()}
    t0 = time.time()
    codes = {"validation": ebp.run_stage(spec, "validation", out=quiet(lines))}
    with pytest.raises(sc.Blocked) as info:  # extension before completion
        ebp.run_stage(spec, "extension", arms=["C"], authorized_by="Nur", extension_decision=lambda s, a: {"C": True},
                      out=quiet(lines))
    refusals["extension_early"] = info.value.message
    codes["completion"] = ebp.run_stage(spec, "completion", out=quiet(lines))
    for name, kwargs in {
        "no_authorized_by": {"extension_decision": lambda s, a: {"C": True}},
        "no_decision": {"authorized_by": "Nur"},
        "rule_says_no": {"authorized_by": "Nur", "extension_decision": lambda s, a: {"C": False}},
        "unknown_arm": {"authorized_by": "Nur", "arms": ["Z"], "extension_decision": lambda s, a: {"Z": True}},
    }.items():
        with pytest.raises(sc.Blocked) as info:
            ebp.run_stage(spec, "extension", **{"arms": ["C"], **kwargs}, out=quiet(lines))
        refusals[name] = info.value.message
    codes["extension"] = ebp.run_stage(spec, "extension", arms=["C"], authorized_by="Nur",
                                       extension_decision=lambda s, a: {"C": True}, out=quiet(lines))
    codes["repeat_validation"] = ebp.run_stage(spec, "validation", out=quiet(lines))
    return {"spec": spec, "codes": codes, "lines": lines, "refusals": refusals, "seconds": time.time() - t0,
            "ref_hashes": ref_hashes}


def test_all_stages_complete_and_every_attempt_passes_the_runner_side_gates(e2e):
    spec = e2e["spec"]
    assert e2e["codes"] == {"validation": 0, "completion": 0, "extension": 0, "repeat_validation": 0}
    attempts = ebp.make_store(spec).attempts()
    assert len(attempts) == 4 + 8 + 3 and all(r["status"] == "completed" for r in attempts.values())
    for aid, res in results(spec).items():
        gates = {i["gate"]: i["status"] for i in res["integrity"]}
        assert gates["G5"] == "pass", aid
        assert gates.get("G6") == ("pass" if res["cap"] == CAPS[1] else None), aid
        assert [g["status"] for g in res["runner_gates"]] == ["pass", "pass", "pass"], aid  # G1, G4, G7
        assert res["category_a"] is None and res["category_b_events"] == []
        assert res["conditions_sha256"] == sc.load_frozen(spec.paths)["conditions_sha256"]
    assert sorted(k for k in attempts if "-c6-" in k) == ["C-c6-s100-a1", "C-c6-s101-a1", "C-c6-s102-a1"]
    assert any("0 run(s) to do" in line for line in e2e["lines"])  # a repeated stage never re-runs


def test_stage_prerequisites_and_extension_authorisation_are_enforced(e2e):
    r = e2e["refusals"]
    assert "every cap-3 run" in r["extension_early"]
    assert "--authorized-by" in r["no_authorized_by"]
    assert "Stage 5" in r["no_decision"]
    assert "does not require an extension" in r["rule_says_no"]
    assert "--arms" in r["unknown_arm"]


def test_records_cross_verify_and_every_invocation_ended(e2e):
    spec = e2e["spec"]
    report = ebp.verify_records(spec)
    assert report["unrecorded_tail"] == 0 and report["orphans"] == []
    assert report["category_b"] == {"events": sc.GENESIS, "resolutions": sc.GENESIS}
    invocations = sc.HashChainedLog(spec.paths.invocations).read()[0]
    starts = [r["invocation_id"] for r in invocations if r["event"] == "start"]
    ends = [r["invocation_id"] for r in invocations if r["event"] == "end"]
    assert starts == ends and len(starts) == 9  # 4 stages run, 5 refused extension attempts; all ended
    assert invocations[-1]["manifest_head"] == sc.manifest_head(spec.paths)


def test_run_directories_disable_git_and_record_their_attempt(e2e):
    spec = e2e["spec"]
    store = ebp.make_store(spec)
    for aid, rec in store.attempts().items():
        meta = json.loads((ebp._run_path(rec["run_dir"]) / "metadata.json").read_text())
        assert meta["git"] == sc.GIT_DISABLED and meta["pilot"]["attempt_id"] == aid
        assert meta["status"] == "completed"


def test_references_are_only_read(e2e, world):
    now = {p: sc.sha256_file(p) for p in world["ref_root"].rglob("*") if p.is_file()}
    assert now == e2e["ref_hashes"]


def test_progress_output_contains_no_learning_metrics(e2e):
    for line in e2e["lines"]:
        assert "nll" not in line.lower() and "loss" not in line.lower()


# ============================================================ frozen conditions
@pytest.fixture()
def one_arm(world, tmp_path):
    def build(arms="A", **over):
        return make_spec(world, tmp_path / f"pilot_{arms}", arms=arms, **over)
    return build


def test_blocking_condition_changes_refuse_every_command(one_arm, world):
    spec = one_arm()
    assert ebp.run_stage(spec, "validation", out=quiet([])) == 0
    original = world["data"].read_text()
    try:
        world["data"].write_text(original + "zzzz\n")  # the dataset changed
        with pytest.raises(sc.Blocked, match="names.txt"):
            ebp.run_stage(spec, "completion", out=quiet([]))
    finally:
        world["data"].write_text(original)
    changed = ebp.PilotSpec(**{**spec.to_json(), "learning_rates": {"A": 0.1}})
    with pytest.raises(sc.Blocked, match="LR-0.1"):  # refused already by the reference check
        ebp.run_stage(changed, "completion", out=quiet([]))
    other_schedule = ebp.PilotSpec(**{**spec.to_json(), "diagnostics": {**spec.diagnostics, "every_n_steps": 5}})
    with pytest.raises(sc.Blocked, match='"diagnostics"'):
        ebp.run_stage(other_schedule, "completion", out=quiet([]))
    with pytest.raises(sc.Blocked, match="threads"):
        ebp.run_stage(ebp.PilotSpec(**{**spec.to_json(), "inter_op_threads": 99}), "completion", out=quiet([]))


def test_recorded_document_changes_are_reported_not_blocking(one_arm, world):
    spec = one_arm()
    assert ebp.run_stage(spec, "validation", out=quiet([])) == 0
    original = world["doc"].read_text()
    try:
        world["doc"].write_text(original + "amended\n")
        assert ebp.run_stage(spec, "completion", out=quiet([])) == 0
    finally:
        world["doc"].write_text(original)
    starts = [r for r in sc.HashChainedLog(spec.paths.invocations).read()[0] if r["event"] == "start"]
    assert starts[-1]["recorded_document_changes"] == [["checksums", "recorded", str(world["doc"])]]


def test_references_must_match_d22_and_must_not_change(one_arm, world, tmp_path):
    with pytest.raises(sc.Blocked, match="D-22"):
        ebp.run_stage(one_arm(reference_summary_sha256="0" * 64), "validation", out=quiet([]))
    bad_runs = {**world["runs"], "A/100": world["runs"]["B/100"]}  # the wrong arm's run
    with pytest.raises(sc.Blocked, match="not the completed"):
        ebp.run_stage(one_arm(reference_runs=bad_runs), "validation", out=quiet([]))
    assert not (tmp_path / "pilot_A" / "conditions.json").exists()


def test_completion_before_validation_and_wrong_arguments_are_refused(one_arm):
    spec = one_arm()
    with pytest.raises(sc.Blocked, match="not been started"):
        ebp.run_stage(spec, "completion", out=quiet([]))
    with pytest.raises(sc.Blocked, match="every arm"):
        ebp.run_stage(spec, "validation", arms=["A"], out=quiet([]))
    with pytest.raises(sc.Blocked, match="unknown stage"):
        ebp.run_stage(spec, "final", out=quiet([]))


# ============================================================ category A / B, integrity, recovery
def test_mechanical_instability_halts_the_pilot(one_arm, inject):
    spec = one_arm()
    inject("nan", "A-c3-s100-a1")
    assert ebp.run_stage(spec, "validation", out=quiet([])) == ebp.STOP_CODES["mechanically_unstable"]
    res = results(spec)["A-c3-s100-a1"]
    assert res["status"] == "mechanically_unstable" and res["category_a"]["code"] == "A3"
    inject.clear()
    for call in (lambda: ebp.run_stage(spec, "completion", out=quiet([])),
                 lambda: ebp.retry(spec, "A", 3, 100, reason="try again", authorized_by="Nur", out=quiet([]))):
        with pytest.raises(sc.Blocked, match="halted|final"):
            call()


def test_crashed_child_is_recorded_and_needs_an_explicit_reasoned_retry(one_arm, inject):
    spec = one_arm()
    inject("crash", "A-c3-s100-a1")
    assert ebp.run_stage(spec, "validation", out=quiet([])) == ebp.STOP_CODES["crashed"]
    rec = ebp.make_store(spec).attempts()["A-c3-s100-a1"]
    assert rec["status"] == "crashed" and rec["returncode"] == 9 and rec["run_dir"]
    inject.clear()
    with pytest.raises(sc.Blocked, match="unresolved"):
        ebp.run_stage(spec, "validation", out=quiet([]))
    with pytest.raises(sc.Blocked, match="non-empty written reason"):
        ebp.retry(spec, "A", 3, 100, reason="  ", out=quiet([]))
    assert ebp.retry(spec, "A", 3, 100, reason="crash injected by a test", out=quiet([])) == 0
    recs = ebp.make_store(spec).by_key()[("A", 3, 100)]
    assert [(r["attempt_id"], r["status"]) for r in recs] == [("A-c3-s100-a1", "crashed"),
                                                              ("A-c3-s100-a2", "completed")]
    assert recs[1]["retry_of"] == "A-c3-s100-a1" and recs[1]["retry_reason"] == "crash injected by a test"
    with pytest.raises(sc.Blocked, match="final"):
        ebp.retry(spec, "A", 3, 100, reason="again", out=quiet([]))


@pytest.mark.parametrize("mode,status", [("error", "error"), ("interrupt", "interrupted")])
def test_software_error_and_interruption_stop_the_sweep(one_arm, inject, mode, status):
    spec = one_arm()
    inject(mode, "A-c3-s100-a1")
    assert ebp.run_stage(spec, "validation", out=quiet([])) == ebp.STOP_CODES[status]
    res = results(spec)["A-c3-s100-a1"]
    assert res["status"] == status and res["category_a"] is None
    inject.clear()
    assert ebp.retry(spec, "A", 3, 100, reason="fault injected by a test", out=quiet([])) == 0


def test_category_b_is_recorded_training_continues_and_resolution_is_audited(one_arm, inject):
    import math

    spec = one_arm()
    steps_per_epoch = math.ceil(ebp.build_arm(spec, "A", 100, 3, ["x"])["train"].num_examples / 32)
    step = math.ceil(G5 * steps_per_epoch / EVERY) * EVERY  # first diagnostic step after the G5 epochs
    inject("category_b", "A-c3-s100-a1", step=step // EVERY + 1)  # inside G5 it would (rightly) fail G5
    assert ebp.run_stage(spec, "validation", out=quiet([])) == 0
    res = results(spec)["A-c3-s100-a1"]
    assert res["status"] == "completed" and len(res["category_b_events"]) == 1
    ledger = sc.CategoryBLedger(spec.paths.root)
    event = ledger.load()["events"][0]
    assert event["status"] == "unresolved" and event["global_step"] == step and event["epoch_1based"] == 3
    assert event["field"][-1] == "relative_update"
    assert {i["gate"]: i["status"] for i in res["integrity"]} == {"G5": "pass"}
    assert ledger.blocking_events() == res["category_b_events"]
    inject.clear()
    with pytest.raises(sc.LedgerError):
        ebp.resolve_category_b(spec, event_id=event["event_id"], outcome="accepted_as_expected", reviewer="",
                               rationale="r", evidence="e")
    ebp.resolve_category_b(spec, event_id=event["event_id"], outcome="rejected", reviewer="Nur",
                           rationale="injected by a test", evidence="test_epoch_budget_pilot.py")
    assert ledger.status()[event["event_id"]]["status"] == "rejected" and ledger.blocking_events()
    notes = [r for r in sc.HashChainedLog(spec.paths.invocations).read()[0] if r["event"] == "category_b_resolution"]
    assert notes[0]["event_id"] == event["event_id"]
    assert ebp.verify_records(spec)["unrecorded_tail"] == 0


def test_g5_mismatch_is_integrity_blocked_and_needs_authorised_retry(world, tmp_path):
    ref_root = tmp_path / "perturbed_reference"
    import shutil

    shutil.copytree(world["ref_root"], ref_root)
    hist = ref_root / world["runs"]["A/100"] / "history.jsonl"
    rows = [json.loads(line) for line in hist.read_text().splitlines()]
    rows[1]["dev_nll"] = float.fromhex(rows[1]["dev_nll"].hex()) + 1e-12  # a different number at epoch 2
    hist.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = make_spec(world, tmp_path / "pilot", arms="A", reference_root=str(ref_root))
    assert ebp.run_stage(spec, "validation", out=quiet([])) == ebp.STOP_CODES["integrity_blocked"]
    res = results(spec)["A-c3-s100-a1"]
    assert res["status"] == "integrity_blocked" and res["integrity"][0]["first_mismatch"]["path"] == ["dev_nll"]
    with pytest.raises(sc.Blocked, match="integrity_blocked"):
        ebp.run_stage(spec, "validation", out=quiet([]))
    with pytest.raises(sc.Blocked, match="--authorized-by"):
        ebp.retry(spec, "A", 3, 100, reason="reviewed", out=quiet([]))
    assert ebp.retry(spec, "A", 3, 100, reason="reviewed", authorized_by="Nur",
                     out=quiet([])) == ebp.STOP_CODES["integrity_blocked"]  # a fresh run, same reference
    recs = ebp.make_store(spec).by_key()[("A", 3, 100)]
    assert [r["status"] for r in recs] == ["integrity_blocked", "integrity_blocked"]
    assert recs[1]["retry_authorized_by"] == "Nur"


# ============================================================ tampering with the records
@pytest.fixture()
def done(one_arm):
    spec = one_arm()
    assert ebp.run_stage(spec, "validation", out=quiet([])) == 0
    return spec


def _rewrite_last_line(path, mutate):
    lines = path.read_text().splitlines()
    obj = json.loads(lines[-1])
    mutate(obj)
    path.write_text("\n".join([*lines[:-1], sc.canonical_json(obj)]) + "\n")


def test_edit_of_the_last_manifest_line_is_detected(done):
    _rewrite_last_line(done.paths.manifest, lambda o: o.update(status="error"))
    sc.HashChainedLog(done.paths.manifest).read()  # the chain alone still verifies
    with pytest.raises(sc.Blocked, match="not a head of the current manifest"):
        ebp.verify_records(done)
    with pytest.raises(sc.Blocked, match="not a head of the current manifest"):
        ebp.run_stage(done, "completion", out=quiet([]))


def test_truncated_manifest_is_detected(done):
    lines = done.paths.manifest.read_text().splitlines()
    done.paths.manifest.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(sc.Blocked, match="not a head of the current manifest"):
        ebp.verify_records(done)


def test_lines_appended_after_a_completed_invocation_are_detected(done):
    sc.HashChainedLog(done.paths.manifest).append({"event": "launched", "attempt_id": "A-c3-s101-a1", "arm": "A",
                                                   "cap": 3, "seed": 101})
    with pytest.raises(sc.Blocked, match="not covered by any invocation record"):
        ebp.verify_records(done)


def test_edited_attempt_result_and_edited_invocation_record_are_detected(done):
    path = done.paths.attempt_result("A-c3-s100-a1")
    original = path.read_text()
    path.write_text(original.replace('"completed"', '"error"', 1))
    with pytest.raises(sc.Blocked, match="changed after the attempt was closed"):
        ebp.verify_records(done)
    path.write_text(original)
    _rewrite_last_line(done.paths.invocations, lambda o: o.update(manifest_head="1" * 64))
    with pytest.raises(sc.Blocked, match="not a head of the current manifest"):
        ebp.verify_records(done)


def _await_blocked_child(server, parent, deadline_s=180):
    """Wait (event-driven, with a deadline) until the child reports that it is blocked mid-attempt."""
    server.settimeout(1.0)
    deadline = time.monotonic() + deadline_s
    while True:
        try:
            conn, _ = server.accept()
            break
        except TimeoutError:
            assert parent.poll() is None, "the runner exited before the attempt blocked"
            assert time.monotonic() < deadline, "the attempt never reached the blocking point"
    conn.settimeout(deadline_s)
    assert conn.recv(16) == b"blocked\n"
    return conn


def test_killed_runner_is_recovered_by_the_next_invocation(one_arm, tmp_path, inject, monkeypatch):
    """Kill the runner and its child while the attempt is provably active (deterministic).

    The child blocks inside its first optimizer step of epoch 2 (after epoch 1's history row and
    progress line) and reports this over a localhost socket; only then are runner and child
    killed with SIGKILL. No sleep and no output timing decide when the kill happens.
    """
    import math
    import queue
    import socket
    import threading

    spec = one_arm()
    steps_per_epoch = math.ceil(ebp.build_arm(spec, "A", 100, 3, ["x"])["train"].num_examples / 32)
    server = socket.create_server(("127.0.0.1", 0))
    inject("block", "A-c3-s100-a1", step=steps_per_epoch + 1)
    monkeypatch.setenv("EBP_FAULT_PORT", str(server.getsockname()[1]))
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec.to_json()))
    script = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(REPO)!r})
        from training.wavenet_study import epoch_budget_pilot as ebp
        spec = ebp.PilotSpec.from_json(json.load(open({str(spec_file)!r})))
        ebp.run_stage(spec, "validation")
    """)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}  # as on a normal terminal / Mac
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, start_new_session=True, env=env)
    lines: queue.Queue = queue.Queue()
    reader = threading.Thread(target=lambda: [lines.put(line) for line in parent.stdout], daemon=True)
    reader.start()
    conn = None
    try:
        conn = _await_blocked_child(server, parent)
        # The attempt is genuinely active: launched in the manifest, epoch 1 written, runner alive.
        assert parent.poll() is None
        store = ebp.make_store(spec)
        rec = store.attempts()["A-c3-s100-a1"]
        assert rec["status"] == "launched"
        run_dirs = store.run_dirs_by_attempt()["A-c3-s100-a1"]
        assert len(run_dirs) == 1
        assert len(sc.read_jsonl_strict(ebp._run_path(run_dirs[0]) / "history.jsonl")) == 1
        # Regression (buffering): the runner's progress is visible while it is still running.
        seen = []
        while not any("epoch 1/3 done" in x for x in seen):
            seen.append(lines.get(timeout=60))
        os.killpg(parent.pid, signal.SIGKILL)
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
        parent.wait()
        reader.join(timeout=60)
        parent.stdout.close()
        if conn is not None:
            conn.close()
        server.close()
    assert parent.returncode == -signal.SIGKILL
    inject.clear()
    store = ebp.make_store(spec)
    assert store.attempts()["A-c3-s100-a1"]["status"] == "launched"  # nobody closed it
    assert not spec.paths.attempt_result("A-c3-s100-a1").exists()
    # the manifest still ends with this attempt's launch, but no invocation holds the lock any more:
    orphan_child = subprocess.run([sys.executable, str(REPO / "training/wavenet_study/epoch_budget_pilot.py"), "one",
                                   "--root", str(spec.paths.root), "--attempt-id", "A-c3-s100-a1", "--arm", "A",
                                   "--cap", "3", "--seed", "100"], capture_output=True, text=True, cwd=REPO,
                                  check=False)
    assert orphan_child.returncode != 0 and "not registered as launched" in orphan_child.stderr
    assert not spec.paths.attempt_result("A-c3-s100-a1").exists()
    report = ebp.verify_records(spec)  # the dead invocation never ended: its records still verify
    assert report["unrecorded_tail"] == 0
    with pytest.raises(sc.Blocked, match="unresolved"):
        ebp.run_stage(spec, "validation", out=quiet([]))
    rec = store.attempts()["A-c3-s100-a1"]  # registered by the refused invocation
    assert rec["status"] == "crashed" and rec["events"] == ["launched", "registered"]
    assert rec["run_dir"] == run_dirs[0]  # the partial run directory is attributed, not orphaned
    assert ebp.retry(spec, "A", 3, 100, reason="runner killed by a test", out=quiet([])) == 0
    assert ebp.verify_records(spec)["orphans"] == []
    recs = ebp.make_store(spec).by_key()[("A", 3, 100)]
    assert [r["status"] for r in recs] == ["crashed", "completed"]


def test_a_child_started_without_an_active_invocation_refuses_and_writes_nothing(done):
    proc = subprocess.run([sys.executable, str(REPO / "training/wavenet_study/epoch_budget_pilot.py"), "one",
                           "--root", str(done.paths.root), "--attempt-id", "A-c3-s100-a1", "--arm", "A", "--cap", "3",
                           "--seed", "100"], capture_output=True, text=True, cwd=REPO, check=False)
    assert proc.returncode != 0 and "BLOCKED" in proc.stderr + proc.stdout
    assert len(list(done.paths.attempts_dir.glob("*.json"))) == 1


# ============================================================ test split, Git, command line
def test_the_test_split_is_never_turned_into_windows(world, tmp_path, monkeypatch):
    from utils import data
    from utils.data import load_names, split_names

    spec = make_spec(world, tmp_path / "p", arms="ABCD")
    splits = split_names(load_names(str(world["data"])), fractions=(0.8, 0.1, 0.1), seed=42, duplicate_policy="group")
    test_names = set(splits.test)
    seen = []
    original = data.make_context_windows

    def spy(names, *a, **k):
        seen.append(list(names))
        return original(names, *a, **k)

    monkeypatch.setattr(data, "make_context_windows", spy)
    for arm in "ABCD":
        b = ebp.build_arm(spec, arm, 100, 3, ["x"])
        assert b["train"].num_examples > 0 and b["dev"].num_examples > 0
    assert len(seen) == 8 and all(not (set(names) & test_names) for names in seen)
    source = (REPO / "training/wavenet_study/epoch_budget_pilot.py").read_text()
    assert ".test" not in source and "splits.test" not in source and '["test"]' not in source


def test_production_git_guard_refuses_git_before_any_process_starts(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "ran"
    (bin_dir / "git").write_text(f'#!/bin/sh\necho ran >> "{marker}"\n')
    (bin_dir / "git").chmod(0o755)
    script = textwrap.dedent(f"""
        import os, subprocess, sys
        sys.path.insert(0, {str(REPO)!r})
        from training.wavenet_study import study_common as sc
        sc.disable_git()
        from utils import run_manager
        assert run_manager.git_info() == sc.GIT_DISABLED
        refused = 0
        for attempt in (lambda: subprocess.run(["git", "status"]), lambda: subprocess.run("git log", shell=True),
                        lambda: os.system("git status"), lambda: os.posix_spawnp("git", ["git"], dict(os.environ)),
                        lambda: subprocess.check_output(["sh", "-c", "git status"]),
                        lambda: os.execvp("git", ["git"])):
            try:
                attempt()
            except PermissionError:
                refused += 1
        print(refused)
    """)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    script_file = tmp_path / "guard_check.py"  # a file: the test package's own guard refuses "-c 'git …'"
    script_file.write_text(script)
    out = subprocess.run([sys.executable, str(script_file)], capture_output=True, text=True, env=env, check=False)
    assert out.stdout.strip() == "6", out.stderr
    assert not marker.exists()


def test_command_line_refuses_other_thread_counts_without_touching_anything():
    with pytest.raises(sc.Blocked, match="--threads must be 1"):
        ebp.main(["run", "--threads", "2", "--stage", "validation"])
    with pytest.raises(SystemExit):
        ebp.main(["retry", "--threads", "1", "--arm", "A", "--cap", "30", "--seed", "100"])  # --reason required


def test_the_real_pilot_folder_was_never_created():
    # Guard for the synthetic suite before the real pilot existed. Once the real pilot has been started,
    # the folder exists legitimately and this guard no longer applies (its contents are never read here).
    if (ebp.RESULTS_ROOT / "conditions.json").is_file():
        pytest.skip("the real epoch-budget pilot has been started; the folder exists legitimately")
    assert not ebp.RESULTS_ROOT.exists()


def test_attempt_result_with_a_wrong_manifest_head_is_detected(tmp_path):
    spec_paths = sc.PilotPaths(tmp_path / "p")
    with sc.PilotLock(spec_paths, "test") as lock:
        store = sc.AttemptStore(spec_paths, key_fields=ebp.KEY_FIELDS, lock=lock,
                                attempt_id=lambda k, n: f"{k['arm']}-c{k['cap']}-s{k['seed']}-a{n}")
        session = sc.InvocationSession(spec_paths, lock, store, "test")
        aid = store.launch({"arm": "A", "cap": 3, "seed": 100})
        session.note("attempt_launched", attempt_id=aid)
        right = sc.manifest_head(spec_paths)
        sc.write_attempt_result(spec_paths, aid, {"attempt_id": aid, "arm": "A", "cap": 3, "seed": 100,
                                                  "status": "completed", "manifest_head": sc.GENESIS})
        with pytest.raises(sc.Blocked, match="not the head after its launched row"):
            sc.verify_record_heads(spec_paths, store)
        spec_paths.attempt_result(aid).write_text(json.dumps({"attempt_id": aid, "arm": "A", "cap": 3, "seed": 100,
                                                              "status": "completed", "manifest_head": right}))
        assert sc.verify_record_heads(spec_paths, store)["unrecorded_tail"] == 0
        session.end()


def test_completion_needs_completed_validation_runs_even_with_frozen_conditions(one_arm):
    spec = one_arm()
    with sc.PilotLock(spec.paths, "test") as lock:  # conditions frozen, no attempt yet
        sc.verify_or_freeze(spec.paths, ebp.current_conditions(spec), may_create=True, lock=lock)
    with pytest.raises(sc.Blocked, match="every validation run completed"):
        ebp.run_stage(spec, "completion", out=quiet([]))
    assert ebp.make_store(spec).attempts() == {}


def test_wrong_inter_op_thread_count_is_refused_before_anything_is_written(one_arm):
    spec = one_arm(inter_op_threads=torch.get_num_interop_threads() + 1)
    with pytest.raises(sc.Blocked, match="inter-op threads"):
        ebp.run_stage(spec, "validation", out=quiet([]))
    assert not spec.paths.root.exists()


# ============================================================ command-line extension stage (summary decision)
# `run --stage extension` takes its decision from the Stage 5 summary script. The decision is
# computed only after --authorized-by and every completed cap-3 run have been checked; it refuses
# unless every selection prerequisite and integrity gate holds; a True decision never replaces the
# human authorisation. Synthetic pilot: arms A and B, caps 3 → 6.
def _summary():
    import importlib

    return importlib.import_module(ebp.SUMMARY_MODULE)


def _cli_extension(*arms, authorized_by="Nur"):
    argv = ["run", "--threads", "1", "--stage", "extension", "--arms", *arms]
    return ebp.main(argv + (["--authorized-by", authorized_by] if authorized_by is not None else []))


def _launch_state(spec):
    """What a refused command must leave unchanged: every attempt and the manifest."""
    return sorted(ebp.make_store(spec).attempts()), sc.manifest_head(spec.paths)


@pytest.fixture(scope="module")
def cli_pilot(world, tmp_path_factory):
    spec = make_spec(world, tmp_path_factory.mktemp("cli_extension") / "pilot", arms="AB")
    assert ebp.run_stage(spec, "validation", out=quiet([])) == 0
    assert ebp.run_stage(spec, "completion", out=quiet([])) == 0
    decision = _summary().extension_decision(spec, list(spec.arms))
    needed = [a for a, v in decision.items() if v]
    not_needed = [a for a, v in decision.items() if not v]
    assert needed and not_needed, f"the synthetic pilot must contain both cases, got {decision}"
    return {"spec": spec, "needed": needed[0], "not_needed": not_needed[0]}


@pytest.fixture()
def cli(cli_pilot, monkeypatch):
    """The real command line, with production_spec() pointed at the synthetic pilot."""
    monkeypatch.setattr(ebp, "production_spec", lambda: cli_pilot["spec"])
    return cli_pilot


def _spy_decision(monkeypatch, result=None):
    calls = []
    real = _summary().extension_decision

    def spy(spec, arms):
        calls.append(list(arms))
        return real(spec, arms) if result is None else result(arms)

    monkeypatch.setattr(_summary(), "extension_decision", spy)
    return calls


def test_cli_extension_runs_only_when_authorised_and_required(cli):
    spec, arm = cli["spec"], cli["needed"]
    assert _cli_extension(arm) == 0
    store = ebp.make_store(spec)
    extended = [r for r in store.attempts().values() if r["key"][1] == CAPS[1]]
    assert sorted(r["key"] for r in extended) == [(arm, CAPS[1], s) for s in ebp.SEEDS]
    assert all(r["status"] == "completed" and r["stage"] == "extension" and r["authorized_by"] == "Nur"
               for r in extended)
    report = _summary().evaluate_pilot(spec.root, selection=True)  # independent recomputation, G6 included
    assert report["integrity_problems"] == [] and report["blocking"] == []
    assert all(e["gates"]["G6"]["status"] == "pass" for e in report["attempts"] if e["cap"] == CAPS[1])
    assert report["arms"][arm]["extension_used"] is True


def test_cli_extension_refused_when_the_rule_does_not_require_it(cli, monkeypatch):
    calls, before = _spy_decision(monkeypatch), _launch_state(cli["spec"])
    with pytest.raises(sc.Blocked, match="does not require an extension"):
        _cli_extension(cli["not_needed"])
    assert calls == [[cli["not_needed"]]] and _launch_state(cli["spec"]) == before


@pytest.mark.parametrize("authorized_by", [None, "", "   "])
def test_cli_extension_without_authorisation_never_computes_the_decision(cli, monkeypatch, authorized_by):
    calls = _spy_decision(monkeypatch, result=lambda arms: {a: True for a in arms})  # "required" by the rule
    before = _launch_state(cli["spec"])
    with pytest.raises(sc.Blocked, match="--authorized-by is required"):
        _cli_extension(cli["needed"], authorized_by=authorized_by)
    assert calls == [] and _launch_state(cli["spec"]) == before


def test_cli_extension_refused_when_summary_inputs_are_missing(cli, monkeypatch):
    spec = cli["spec"]
    store = ebp.make_store(spec)
    run_dir = ebp._run_path(store.by_key()[(cli["needed"], CAPS[0], 101)][-1]["run_dir"])
    history = run_dir / "history.jsonl"
    saved, before = history.read_bytes(), _launch_state(spec)
    history.unlink()
    try:
        with pytest.raises(sc.Blocked, match="extension decision refused"):
            _cli_extension(cli["needed"])
    finally:
        history.write_bytes(saved)
    assert _launch_state(spec) == before


def test_cli_extension_refused_when_the_summary_finds_an_integrity_failure(cli):
    spec = cli["spec"]
    store = ebp.make_store(spec)
    run_dir = ebp._run_path(store.by_key()[(cli["needed"], CAPS[0], 100)][-1]["run_dir"])
    steps = run_dir / "diagnostics" / "steps.jsonl"
    saved, before = steps.read_bytes(), _launch_state(spec)
    rows = [json.loads(line) for line in saved.decode().splitlines()]
    rows[0]["loss"] = rows[0]["loss"] + 1e-9  # inside the G5 epochs: summary-side G5 fails
    steps.write_text("".join(json.dumps(r) + "\n" for r in rows))
    try:
        with pytest.raises(sc.Blocked, match=r"extension decision refused.*integrity"):
            _cli_extension(cli["needed"])
    finally:
        steps.write_bytes(saved)
    assert _launch_state(spec) == before


def test_cli_extension_refused_while_cap_runs_are_incomplete(one_arm, monkeypatch):
    spec = one_arm()
    assert ebp.run_stage(spec, "validation", out=quiet([])) == 0  # seeds 101 and 102 missing
    monkeypatch.setattr(ebp, "production_spec", lambda: spec)
    calls, before = _spy_decision(monkeypatch), _launch_state(spec)
    with pytest.raises(sc.Blocked, match="every cap-3 run of every arm completed"):
        _cli_extension("A")
    assert calls == [] and _launch_state(spec) == before
    with pytest.raises(sc.Blocked, match="incomplete cap-3 runs"):  # the summary refuses on its own as well
        _summary().extension_decision(spec, ["A"])


@pytest.mark.parametrize("bad", [
    lambda arms: None, lambda arms: list(arms), lambda arms: {a: "yes" for a in arms},
    lambda arms: {a: 1 for a in arms}, lambda arms: {}, lambda arms: {"Z": True},
], ids=["none", "list", "string", "int", "empty", "other-arm"])
def test_cli_extension_refuses_an_invalid_decision(cli, monkeypatch, bad):
    _spy_decision(monkeypatch, result=bad)
    before = _launch_state(cli["spec"])
    with pytest.raises(sc.Blocked, match="invalid extension decision"):
        _cli_extension(cli["needed"])
    assert _launch_state(cli["spec"]) == before


def test_cli_extension_refuses_when_the_decision_itself_fails(cli, monkeypatch):
    def broken(arms):
        raise RuntimeError("decision crashed")

    _spy_decision(monkeypatch, result=broken)
    before = _launch_state(cli["spec"])
    with pytest.raises(sc.Blocked, match="extension decision failed.*decision crashed"):
        _cli_extension(cli["needed"])
    assert _launch_state(cli["spec"]) == before


def _invocations(spec):
    return spec.paths.invocations.read_bytes()


@pytest.mark.parametrize("failure", ["missing-module", "missing-function", "not-callable", "import-error"])
def test_cli_extension_refuses_on_an_import_or_interface_failure_before_writing(cli, monkeypatch, failure):
    if failure == "missing-module":
        monkeypatch.setattr(ebp, "SUMMARY_MODULE", "training.wavenet_study.no_such_summary_module")
    elif failure == "missing-function":
        monkeypatch.delattr(_summary(), "extension_decision")
    elif failure == "not-callable":
        monkeypatch.setattr(_summary(), "extension_decision", 42)
    else:
        import importlib

        real_import = importlib.import_module

        def failing(name, *a, **k):
            if name == ebp.SUMMARY_MODULE:
                raise ImportError("simulated import failure")
            return real_import(name, *a, **k)

        monkeypatch.setattr(importlib, "import_module", failing)
    spec = cli["spec"]
    before, invocations = _launch_state(spec), _invocations(spec)
    with pytest.raises(sc.Blocked, match="could not be loaded|not callable"):
        _cli_extension(cli["needed"])
    assert _launch_state(spec) == before and _invocations(spec) == invocations  # refused before any lock or record


def test_validation_and_completion_never_load_the_summary_script(cli, monkeypatch):
    def forbidden():
        raise AssertionError("validation and completion must not load the summary script")

    monkeypatch.setattr(ebp, "_load_extension_decision", forbidden)
    before = _launch_state(cli["spec"])
    for stage in ("validation", "completion"):  # both already done: nothing left to run
        assert ebp.main(["run", "--threads", "1", "--stage", stage]) == 0
    assert _launch_state(cli["spec"]) == before


def test_the_summary_script_is_imported_lazily_and_loads_when_the_runner_is_the_main_script(tmp_path):
    script = tmp_path / "load_check.py"
    script.write_text(textwrap.dedent(f"""
        import runpy, sys
        sys.path.insert(0, {str(REPO)!r})
        g = runpy.run_path({str(REPO / "training/wavenet_study/epoch_budget_pilot.py")!r}, run_name="__cli__")
        print("summary imported at load:", g["SUMMARY_MODULE"] in sys.modules)
        fn = g["_load_extension_decision"]()
        print("loaded:", fn.__module__, fn.__name__, callable(fn))
    """))
    out = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, cwd=tmp_path,
                         check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == ["summary imported at load: False",
                                       f"loaded: {ebp.SUMMARY_MODULE} extension_decision True"]
