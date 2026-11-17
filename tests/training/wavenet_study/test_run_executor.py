"""Stage 2: the single-run executor (protocol §9, §10.2, §10.4, §12.3).

Synthetic names, tiny models and the accepted trainers, unchanged. Every run directory lives
in pytest's temporary folder. The Git guard of conftest.py is active for all of them.

Reference runs ("plain") are produced by calling the accepted trainer directly, exactly as the
LR pilot did; executor runs must reproduce them bit for bit (the property G5 relies on).
"""

from __future__ import annotations

import json
import math

import pytest
import torch
from torch.optim.optimizer import _global_optimizer_post_hooks

from training.wavenet_study import study_common as sc
from utils.data import make_context_windows
from utils.diagnostics.schedule import DiagnosticsSchedule
from utils.reproducibility import derive_seed, make_generator, seed_everything
from utils.run_manager import RunManager
from utils.vocabulary import Vocabulary

TRAIN_NAMES = ["abc", "bca", "cab", "abd", "bad", "dab", "ace", "cea", "eca", "bed", "deb", "fab",
               "abf", "caf", "fec", "edf", "fed", "def", "bef", "cbd", "adc", "ebf", "fda", "dcf"]
DEV_NAMES = ["abe", "fac", "dbc", "ecd"]
SEED = 11
VOCAB = Vocabulary.from_names(TRAIN_NAMES)
TRAIN = make_context_windows(TRAIN_NAMES, VOCAB, context_length=8)
DEV = make_context_windows(DEV_NAMES, VOCAB, context_length=8)
BATCH = next(b for b in (5, 6, 7, 9) if TRAIN.num_examples % b != 1)
STEPS_PER_EPOCH = math.ceil(TRAIN.num_examples / BATCH)


def _arm(arm):
    from models.feedforward import batchnorm_mlp, mlp
    from models.hierarchical import wavenet

    if arm == "A":
        return mlp, mlp.MLPLanguageModel, mlp.MLPConfig(context_length=8, embedding_dim=3, hidden_dim=5), \
            mlp.MLPTrainingConfig, mlp.train_mlp, "none"
    if arm == "B":
        return batchnorm_mlp, batchnorm_mlp.BatchNormMLPLanguageModel, \
            batchnorm_mlp.BatchNormMLPConfig(context_length=8, embedding_dim=3, hidden_dim=5), \
            batchnorm_mlp.BatchNormMLPTrainingConfig, batchnorm_mlp.train_batchnorm_mlp, "batchnorm"
    norm = "none" if arm == "C" else "batchnorm"
    return wavenet, wavenet.WaveNetLanguageModel, \
        wavenet.WaveNetConfig(context_length=8, embedding_dim=3, hidden_dim=4, normalization=norm), \
        wavenet.WaveNetTrainingConfig, wavenet.train_wavenet, norm


def build(root, arm, *, cap, lr=0.1, train_fn=None, attempt="a1"):
    """Set-up exactly like the LR pilot's build path (global seeding, explicit init generator)."""
    module, model_cls, model_cfg, train_cls, fn, norm = _arm(arm)
    seed_everything(SEED)
    model = model_cls(VOCAB, model_cfg, generator=make_generator(derive_seed(SEED, "init")))
    training = train_cls(lr=lr, max_epochs=cap, batch_size=BATCH, eval_batch_size=16)
    name = getattr(model, "model_name", None) or module.MODEL_NAME
    run = RunManager.create(name, config=module.run_config(None, model.config, training), results_root=root)
    schedule = DiagnosticsSchedule(enabled=True, every_n_steps=2, gradients=True, parameters=True, activations=False)
    prepared = sc.PreparedRun(model, TRAIN, DEV, training, train_fn or fn, run, schedule, None)
    context = sc.RunContext(f"{arm}-c{cap}-s{SEED}-{attempt}", arm, SEED, cap, norm, {"stage": "synthetic_test"})
    return prepared, context, fn


def plain_run(root, arm, *, cap):
    prepared, _, fn = build(root, arm, cap=cap)
    with prepared.run.track():
        fn(prepared.model, TRAIN, DEV, training=prepared.training, seed=SEED, run=prepared.run,
           diagnostics=prepared.diagnostics, config=None)
    return prepared


def records(run_dir):
    return (sc.read_jsonl_strict(run_dir / "history.jsonl"),
            sc.read_jsonl_strict(run_dir / "diagnostics" / "steps.jsonl"))


def schemas_of(run_dir):
    history, steps = records(run_dir)
    return sc.RecordSchemas(sc.record_schema(history), sc.record_schema(steps))


def deep_equal(a, b, path="") -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            deep_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            deep_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert isinstance(b, float) and sc.float_bits(a) == sc.float_bits(b), path
    else:
        assert a == b, path


IGNORED_CHECKPOINT_KEYS = {"created_at"}


def assert_bit_identical_runs(ref, new, norm):
    (rh, rs), (nh, ns) = records(ref.run.run_dir), records(new.run.run_dir)
    assert sc.compare_keyed_records(rh, nh, key_fields=("epoch",), exclude=[("seconds",)],
                                    null_allowed=sc.history_null_allowed_for(norm)) is None
    assert sc.compare_keyed_records(rs, ns, null_allowed=sc.step_null_allowed) is None
    for fname in sc.CHECKPOINT_FILES.values():
        a = sc.load_payload(ref.run.run_dir / fname)
        b = sc.load_payload(new.run.run_dir / fname)
        assert a.keys() == b.keys()
        for key in a:  # model and optimiser state, RNG state, sampler, step, epoch, best metric, config
            if key not in IGNORED_CHECKPOINT_KEYS:
                deep_equal(a[key], b[key], f"{fname}:{key}")
    for (n1, p1), (n2, p2) in zip(ref.model.named_parameters(), new.model.named_parameters(), strict=True):
        assert n1 == n2 and torch.equal(p1, p2)
        assert (p1.grad is None) == (p2.grad is None) and (p1.grad is None or torch.equal(p1.grad, p2.grad))
    deep_equal(dict(ref.model.state_dict()), dict(new.model.state_dict()))
    deep_equal({m: mod.training for m, mod in ref.model.named_modules()},
               {m: mod.training for m, mod in new.model.named_modules()})


# ============================================================ clean runs: parity and records
@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    """Per arm: a plain cap-3 reference, an executor cap-3 run (G5-style check at epoch 3 against
    the reference) and an executor cap-5 run (G5 at 3 against the plain reference, G6 at 3
    against the executor cap-3 run)."""
    torch.set_num_threads(1)
    out = {}
    for arm in "ABCD":
        root = tmp_path_factory.mktemp(f"clean_{arm}")
        ledger = sc.CategoryBLedger(root)
        ref = plain_run(root / "plain", arm, cap=3)
        ref_h, ref_s = records(ref.run.run_dir)
        schemas = schemas_of(ref.run.run_dir)
        g5 = sc.PrefixCheck("G5", 3, tuple(ref_h), tuple(ref_s),
                            sc.checkpoint_state_hash(ref.run.run_dir / "model_final.pt"), "plain cap-3 run")
        global_before = len(_global_optimizer_post_hooks)
        prepared3, ctx3, _ = build(root / "exe3", arm, cap=3)
        epochs_seen = []
        res3 = sc.execute_run(prepared3, ctx3, schemas=schemas, ledger=ledger, prefix_checks=[g5],
                              on_epoch=epochs_seen.append)
        h3, s3 = records(prepared3.run.run_dir)
        hash3 = sc.read_jsonl_strict(prepared3.run.run_dir / sc.STATE_HASH_FILE)[2]["sha256"]
        g6 = sc.PrefixCheck("G6", 3, tuple(h3), tuple(s3), hash3, "executor cap-3 run")
        prepared5, ctx5, _ = build(root / "exe5", arm, cap=5)
        res5 = sc.execute_run(prepared5, ctx5, schemas=schemas, ledger=ledger, prefix_checks=[g5, g6])
        out[arm] = {"ref": ref, "p3": prepared3, "r3": res3, "p5": prepared5, "r5": res5, "ledger": ledger,
                    "schemas": schemas, "norm": ctx3.normalization, "global_before": global_before,
                    "epochs_seen": epochs_seen, "root": root}
    return out


@pytest.mark.parametrize("arm", list("ABCD"))
def test_executor_reproduces_the_plain_trainer_bit_for_bit(clean, arm):
    c = clean[arm]
    assert c["r3"]["status"] == "completed", c["r3"]["failure"]
    assert_bit_identical_runs(c["ref"], c["p3"], c["norm"])


@pytest.mark.parametrize("arm", list("ABCD"))
def test_runner_side_g5_and_g6_pass_and_longer_cap_keeps_the_prefix(clean, arm):
    c = clean[arm]
    assert [(i["gate"], i["status"]) for i in c["r3"]["integrity"]] == [("G5", "pass")]
    assert c["r5"]["status"] == "completed", c["r5"]["failure"]
    assert [(i["gate"], i["status"]) for i in c["r5"]["integrity"]] == [("G5", "pass"), ("G6", "pass")]
    on_disk = json.loads((c["p5"].run.run_dir / sc.INTEGRITY_FILE).read_text())
    assert [i["status"] for i in on_disk["checks"]] == ["pass", "pass"]
    h5, _ = records(c["p5"].run.run_dir)
    assert len(h5) == 5


@pytest.mark.parametrize("arm", list("ABCD"))
def test_completed_runs_pass_g3_g4_g7_and_record_hashes_at_every_epoch(clean, arm):
    c = clean[arm]
    for prepared, result, cap in ((c["p3"], c["r3"], 3), (c["p5"], c["r5"], 5)):
        run_dir = prepared.run.run_dir
        meta = json.loads((run_dir / "metadata.json").read_text())
        assert sc.gate_g3(attempt_status=result["status"], metadata_status=meta["status"]).status == "pass"
        assert sc.gate_g4(run_dir, cap=cap).status == "pass"
        assert sc.gate_g7(run_dir).status == "pass"
        assert [r["epoch"] for r in sc.read_jsonl_strict(run_dir / sc.STATE_HASH_FILE)] == list(range(cap))
        assert meta["git"] == {"available": False, "reason": "git disabled by project policy"}


@pytest.mark.parametrize("arm", list("ABCD"))
def test_clean_runs_register_no_category_b_and_report_the_structural_category_c(clean, arm):
    c = clean[arm]
    r = c["r3"]
    assert r["category_a"] is None and r["category_b_events"] == [] and c["ledger"].load()["events"] == []
    n_params = len(list(c["ref"].model.parameters()))
    kinds = [(x["kind"], x["record"], x["global_step"]) for x in r["category_c"]]
    assert kinds.count(("structural_null", "steps", 0)) == n_params
    assert kinds.count(("structural_null", "history", None)) == (3 if arm == "C" else 0)
    assert any(k == ("epsilon_regularised", "steps", 0) for k in kinds)
    assert all(x["global_step"] in (0, None) for x in r["category_c"])
    assert r["category_b_ledger_heads"] == c["ledger"].heads()


@pytest.mark.parametrize("arm", list("ABCD"))
def test_finiteness_cost_is_measured_per_epoch(clean, arm):
    c = clean[arm]
    timing = json.loads((c["p3"].run.run_dir / sc.TIMING_FILE).read_text())
    n_points = len(sc.leaf_modules(c["p3"].model))
    assert len(timing["epoch_records"]) == 3
    for rec in timing["epoch_records"]:
        f = rec["finiteness"]
        assert f["parameter_checks"] == STEPS_PER_EPOCH
        assert f["module_checks_training"] == STEPS_PER_EPOCH * n_points
        assert f["module_checks_evaluation"] > 0
        assert f["parameter_check_seconds"] >= 0 and f["module_check_seconds"] >= 0
        assert rec["state_hash_seconds"] >= 0 and rec["record_check_seconds"] >= 0
    assert timing["finiteness_open_bucket"] == {k: 0 if "checks" in k else 0.0
                                                for k in timing["finiteness_open_bucket"]}
    assert len(timing["epochs"]) == 3 and timing["training_call_seconds"] > 0


@pytest.mark.parametrize("arm", list("ABCD"))
def test_hooks_and_wrapper_are_removed_and_progress_has_no_learning_metrics(clean, arm):
    c = clean[arm]
    for prepared in (c["p3"], c["p5"]):
        assert all(not m._forward_hooks for m in prepared.model.modules())  # clean runs: no hooks at all
        assert "append_history" not in vars(prepared.run)
    assert len(_global_optimizer_post_hooks) == c["global_before"]
    assert [set(e) for e in c["epochs_seen"]] == [{"epoch", "cap", "seconds"}] * 3


# ============================================================ fault injection
def _execute(tmp_path, arm, *, schemas, cap=2, train_fn=None, prefix_checks=(), attempt="a1", **kw):
    prepared, ctx, _ = build(tmp_path / f"r_{attempt}", arm, cap=cap, train_fn=train_fn, attempt=attempt)
    ledger = sc.CategoryBLedger(tmp_path)
    result = sc.execute_run(prepared, ctx, schemas=schemas, ledger=ledger, prefix_checks=prefix_checks, **kw)
    meta = json.loads((prepared.run.run_dir / "metadata.json").read_text())
    on_disk = json.loads((prepared.run.run_dir / sc.RUN_RESULT_FILE).read_text())
    assert on_disk["status"] == result["status"]
    assert sc.gate_g3(attempt_status=result["status"], metadata_status=meta["status"]).status == "pass"
    assert not monitor_hooks_left(prepared.model)  # test-side injection hooks may remain
    assert "append_history" not in vars(prepared.run)
    return prepared, result, ledger


def monitor_hooks_left(model):
    return [h for m in model.modules() for h in m._forward_hooks.values()
            if "FinitenessMonitor" in getattr(h, "__qualname__", "")]


def _with_setup(fn, setup):
    """A train_fn that installs a test-side fault *after* the monitor is attached, then trains."""
    def train_fn(model, *args, **kwargs):
        setup(model)
        return fn(model, *args, **kwargs)
    return train_fn


@pytest.fixture(scope="module")
def ref_schemas(clean):
    return {arm: clean[arm]["schemas"] for arm in "ABCD"}


def _counter_hook(module, *, at, grad_mode, value, before_monitor=False):
    """Replace one output element at the ``at``-th forward of the given grad mode.

    Installed inside ``train_fn``, i.e. after the monitor attached: with ``before_monitor`` the
    hook is prepended (the monitor sees the corrupted output); otherwise it runs after the
    monitor's check (only downstream code, e.g. the trainer's own guards, sees it)."""
    calls = {"n": 0}

    def hook(_m, _i, output):
        if torch.is_grad_enabled() == grad_mode:
            calls["n"] += 1
            if calls["n"] == at:
                out = output.clone()
                out.view(-1)[0] = value
                return out
        return None
    return module.register_forward_hook(hook, prepend=before_monitor)


def test_inf_preactivation_in_training_is_category_a5_and_stops_the_run(tmp_path, ref_schemas):
    _, _, _, _, fn, _ = _arm("A")

    def setup(model):  # the 4th SGD forward of epoch 2 gets an infinite pre-activation
        _counter_hook(model.hidden, at=STEPS_PER_EPOCH + 4, grad_mode=True, value=float("inf"),
                      before_monitor=True)

    prepared, r, _ = _execute(tmp_path, "A", train_fn=_with_setup(fn, setup), schemas=ref_schemas["A"])
    assert r["status"] == "mechanically_unstable"
    a = r["category_a"]
    assert (a["code"], a["source"]) == ("A5", "finiteness_monitor")
    d = a["detail"]
    assert (d["kind"], d["name"], d["forward"], d["epoch_1based"], d["global_step"]) == \
        ("module_output", "hidden", "training", 2, STEPS_PER_EPOCH + 3)
    assert len(records(prepared.run.run_dir)[0]) == 1  # epoch 2 never written
    assert r["epochs_completed"] == 1 and r["steps_completed"] == STEPS_PER_EPOCH + 3


def test_nonfinite_output_in_an_evaluation_forward_is_category_a5(tmp_path, ref_schemas):
    _, _, _, _, fn, _ = _arm("C")

    def setup(model):  # the first evaluation forward of epoch 1
        _counter_hook(model.act3, at=1, grad_mode=False, value=float("nan"), before_monitor=True)

    _, r, _ = _execute(tmp_path, "C", train_fn=_with_setup(fn, setup), schemas=ref_schemas["C"])
    d = r["category_a"]["detail"]
    assert r["status"] == "mechanically_unstable" and r["category_a"]["code"] == "A5"
    assert (d["name"], d["forward"], d["epoch_1based"], d["steps_completed"]) == \
        ("act3", "evaluation", 1, STEPS_PER_EPOCH)


def test_nonfinite_parameter_after_a_step_is_category_a3(tmp_path, ref_schemas):
    from torch.optim.optimizer import register_optimizer_step_post_hook

    _, _, _, _, fn, _ = _arm("D")
    handles = []

    def setup(model):  # corrupt a weight right after the 5th step, before the monitor's check
        calls = {"n": 0}

        def corrupt(_opt, _a, _k):
            calls["n"] += 1
            if calls["n"] == 5:
                with torch.no_grad():
                    model.hidden3.weight[0, 0] = float("nan")
        handles.append(register_optimizer_step_post_hook(corrupt))
        _global_optimizer_post_hooks.move_to_end(handles[0].id, last=False)  # run first

    try:
        _, r, _ = _execute(tmp_path, "D", train_fn=_with_setup(fn, setup), schemas=ref_schemas["D"])
    finally:
        for h in handles:
            h.remove()
    d = r["category_a"]["detail"]
    assert r["status"] == "mechanically_unstable" and r["category_a"]["code"] == "A3"
    assert (d["kind"], d["name"], d["global_step"], d["epoch_1based"]) == \
        ("parameter_after_step", "hidden3.weight", 4, 1)


@pytest.mark.parametrize("fault,code", [("loss", "A1"), ("gradient", "A2")])
def test_accepted_trainer_guards_are_category_a_with_their_codes(tmp_path, ref_schemas, fault, code):
    _, _, _, _, fn, _ = _arm("A")

    def setup(model):
        if fault == "loss":  # appended after the monitor's hook: the monitor saw a finite output
            _counter_hook(model.out, at=3, grad_mode=True, value=float("nan"))
        else:
            state = {"n": 0}

            def grad_hook(g):
                state["n"] += 1
                return g.clone().fill_(float("inf")) if state["n"] == 2 else None
            model.hidden.weight.register_hook(grad_hook)

    _, r, _ = _execute(tmp_path, "A", train_fn=_with_setup(fn, setup), schemas=ref_schemas["A"])
    assert r["status"] == "mechanically_unstable"
    assert (r["category_a"]["code"], r["category_a"]["source"]) == (code, "trainer_guard")


@pytest.mark.parametrize("arm", ["B", "D"])
def test_nonfinite_recalibration_statistics_are_category_a4(tmp_path, ref_schemas, arm):
    _, _, _, _, fn, _ = _arm(arm)
    target = "hidden" if arm == "B" else "hidden1"

    def setup(model):  # after the monitor: only the recalibration statistics see the NaN
        _counter_hook(getattr(model, target), at=1, grad_mode=False, value=float("nan"))

    _, r, _ = _execute(tmp_path, arm, train_fn=_with_setup(fn, setup), schemas=ref_schemas[arm])
    assert r["status"] == "mechanically_unstable"
    assert (r["category_a"]["code"], r["category_a"]["source"]) == ("A4", "trainer_guard")


def test_monitor_detection_inside_recalibration_keeps_its_attribution(tmp_path, ref_schemas):
    _, _, _, _, fn, _ = _arm("B")

    def setup(model):  # before the monitor: it sees the infinity in the first recalibration forward
        _counter_hook(model.hidden, at=1, grad_mode=False, value=float("inf"), before_monitor=True)

    _, r, _ = _execute(tmp_path, "B", train_fn=_with_setup(fn, setup), schemas=ref_schemas["B"])
    assert "recalibration failed" in r["failure"]["message"]  # the trainer re-wrapped our error
    assert (r["category_a"]["code"], r["category_a"]["detail"]["name"]) == ("A5", "hidden")
    assert r["category_a"]["detail"]["forward"] == "evaluation"


def test_nonfinite_recorded_nll_is_category_a(tmp_path, ref_schemas, monkeypatch):
    from utils import run_manager

    original = run_manager.append_jsonl

    def inject(path, record):
        if str(path).endswith("history.jsonl") and record.get("epoch") == 1:
            record = {**record, "dev_nll": float("nan")}  # written as "nan" by to_jsonable
        return original(path, record)

    monkeypatch.setattr(run_manager, "append_jsonl", inject)
    prepared, r, _ = _execute(tmp_path, "A", schemas=ref_schemas["A"])
    assert r["status"] == "mechanically_unstable"
    a = r["category_a"]
    assert a["code"] == "recorded_value" and a["detail"][0]["path"] == ["dev_nll"]
    assert a["detail"][0]["value"] == "nan" and a["detail"][0]["epoch"] == 1
    assert records(prepared.run.run_dir)[0][1]["dev_nll"] == "nan"  # evidence preserved


def test_category_b_is_recorded_with_location_and_never_stops_training(tmp_path, ref_schemas, monkeypatch):
    from models.feedforward import mlp

    original = mlp.append_jsonl

    def inject(path, record):
        if record.get("step") == 4:
            record = json.loads(json.dumps(record))
            record["updates"]["per_parameter"]["hidden.weight"]["relative_update"] = None
        if record.get("step") == STEPS_PER_EPOCH + 2:
            record = {**record, "unexpected_metric": 1.5}
        return original(path, record)

    monkeypatch.setattr(mlp, "append_jsonl", inject)
    prepared, r, ledger = _execute(tmp_path, "A", schemas=ref_schemas["A"])
    assert r["status"] == "completed" and r["category_a"] is None
    events = ledger.load()["events"]
    assert r["category_b_events"] == [e["event_id"] for e in events] and len(events) == 3
    e0 = events[0]
    assert (e0["epoch_1based"], e0["global_step"], e0["record"], e0["field"], e0["observed"], e0["status"]) == \
        (1, 4, "steps", ["updates", "per_parameter", "hidden.weight", "relative_update"], None, "unresolved")
    assert e0["reason"].startswith("undefined_value") and e0["evidence"]["related_fields"]["update_norm"] >= 0
    assert e0["arm"] == "A" and e0["cap"] == 2 and e0["attempt_id"] == "A-c2-s11-a1"
    later = {e["reason"].split(":")[0] for e in events[1:]}
    assert later == {"unclassified_field", "extra_field"} and {e["epoch_1based"] for e in events[1:]} == {2}
    assert len(records(prepared.run.run_dir)[0]) == 2  # training continued to the cap
    assert sc.gate_g4(prepared.run.run_dir, cap=2).status == "pass"
    assert sc.gate_g7(prepared.run.run_dir).status == "pass"
    assert r["category_b_ledger_heads"] == ledger.heads() and ledger.blocking_events() == r["category_b_events"]


def test_schema_deviation_in_a_history_row_is_category_b(tmp_path, ref_schemas):
    schemas = ref_schemas["A"]
    narrowed = sc.RecordSchemas(frozenset(p for p in schemas.history if p != ("lr",)), schemas.steps)
    _, r, ledger = _execute(tmp_path, "A", schemas=narrowed)
    assert r["status"] == "completed"
    kinds = {(e["record"], e["field"][0], e["reason"].split(":")[0]) for e in ledger.load()["events"]}
    assert kinds == {("history", "lr", "extra_field")}


# ============================================================ runner-side integrity
def _reference(clean, arm, *, perturb=None, bad_hash=False, epochs=2):
    c = clean[arm]
    h, s = records(c["p3"].run.run_dir)
    hashes = sc.read_jsonl_strict(c["p3"].run.run_dir / sc.STATE_HASH_FILE)
    if perturb is not None:
        h = json.loads(json.dumps(h))
        h[1]["train_nll"] = math.nextafter(h[1]["train_nll"], math.inf)  # one ulp
    digest = "0" * 64 if bad_hash else hashes[epochs - 1]["sha256"]
    return sc.PrefixCheck("G5", epochs, tuple(h), tuple(s), digest, "synthetic reference")


@pytest.mark.parametrize("fault,path", [("value", ["train_nll"]), ("hash", ["state_sha256"])])
def test_prefix_mismatch_stops_the_run_as_integrity_blocked(tmp_path, clean, ref_schemas, fault, path):
    check = _reference(clean, "B", perturb=True if fault == "value" else None, bad_hash=fault == "hash")
    prepared, r, _ = _execute(tmp_path, "B", cap=3, prefix_checks=[check], schemas=ref_schemas["B"])
    assert r["status"] == "integrity_blocked" and r["category_a"] is None
    assert r["integrity"][0]["status"] == "fail" and r["integrity"][0]["first_mismatch"]["path"] == path
    assert r["failure"]["integrity"]["gate"] == "G5"
    run_dir = prepared.run.run_dir
    assert len(records(run_dir)[0]) == 2  # stopped right after epoch 2's history write
    assert sc.load_payload(run_dir / "checkpoint_last.pt")["epoch"] == 1  # epoch-2 commit never written
    assert not (run_dir / "model_final.pt").exists()
    on_disk = json.loads((run_dir / sc.INTEGRITY_FILE).read_text())
    assert on_disk["checks"][0]["status"] == "fail"


def test_prefix_check_longer_than_the_cap_is_not_applicable(tmp_path, clean, ref_schemas):
    check = _reference(clean, "A", epochs=2)
    _, r, _ = _execute(tmp_path, "A", cap=1, prefix_checks=[check], schemas=ref_schemas["A"])
    assert r["status"] == "completed" and r["integrity"][0]["status"] == "not_applicable"


# ============================================================ software errors and interruption
def test_exception_in_training_is_a_software_error(tmp_path, ref_schemas):
    def broken(*_a, **_k):
        raise ValueError("simulated software failure")

    _, r, _ = _execute(tmp_path, "A", train_fn=broken, schemas=ref_schemas["A"])
    assert r["status"] == "error" and r["category_a"] is None
    assert r["failure"]["type"] == "ValueError" and r["failure"]["phase"] == "training"


def test_step_count_disagreement_is_a_software_error(tmp_path, ref_schemas):
    def lying(model, train, dev, *, run, **_k):
        run.append_history({"epoch": 0, "steps": 99})

    _, r, _ = _execute(tmp_path, "A", train_fn=lying, schemas=ref_schemas["A"])
    assert r["status"] == "error" and "step count mismatch" in r["failure"]["message"]


def test_failure_while_attaching_is_an_error_and_leaves_no_hooks(tmp_path, ref_schemas):
    before = len(_global_optimizer_post_hooks)
    _, r, _ = _execute(tmp_path, "A", schemas=ref_schemas["A"], finiteness_points=[("bogus", object())])
    assert r["status"] == "error" and r["phase"] == "attach"
    assert len(_global_optimizer_post_hooks) == before


def test_floating_point_error_outside_training_is_not_category_a():
    monitor = sc.FinitenessMonitor(torch.nn.Sequential(torch.nn.Linear(1, 1)))
    assert sc._classify_outcome(FloatingPointError("x"), "attach", monitor) == ("error", None)


def test_keyboard_interrupt_is_recorded_and_returned(tmp_path, ref_schemas):
    def interrupted(*_a, **_k):
        raise KeyboardInterrupt

    _, r, _ = _execute(tmp_path, "A", train_fn=interrupted, schemas=ref_schemas["A"])
    assert r["status"] == "interrupted"


# ============================================================ argument validation (nothing attached)
def test_invalid_arguments_are_refused_before_anything_runs(tmp_path, ref_schemas):
    prepared, ctx, _ = build(tmp_path / "v", "A", cap=2)
    ledger = sc.CategoryBLedger(tmp_path)
    s = ref_schemas["A"]
    import dataclasses

    with pytest.raises(TypeError):
        sc.execute_run(prepared, ctx, schemas=s, ledger=None)
    with pytest.raises(ValueError, match="cap"):
        sc.execute_run(prepared, dataclasses.replace(ctx, cap=3), schemas=s, ledger=ledger)
    with pytest.raises(ValueError, match="normalization"):
        sc.execute_run(prepared, dataclasses.replace(ctx, normalization="batchnorm"), schemas=s, ledger=ledger)
    dup = _dummy_check("G5")
    with pytest.raises(ValueError, match="duplicate"):
        sc.execute_run(prepared, ctx, schemas=s, ledger=ledger, prefix_checks=[dup, dup])
    with pytest.raises(ValueError, match="schema"):
        sc.execute_run(prepared, ctx, schemas=sc.RecordSchemas(frozenset(), s.steps), ledger=ledger)
    assert json.loads((prepared.run.run_dir / "metadata.json").read_text())["status"] == "created"
    assert not (prepared.run.run_dir / sc.RUN_RESULT_FILE).exists()
    r = sc.execute_run(prepared, ctx, schemas=s, ledger=ledger)
    assert r["status"] == "completed"
    with pytest.raises(ValueError, match="fresh"):
        sc.execute_run(prepared, ctx, schemas=s, ledger=ledger)


def _dummy_check(gate):
    return sc.PrefixCheck(gate, 1, (), (), "0" * 64, "dummy")
