"""Stage 1: append-only, hash-chained category B ledger (protocol §9.7; approval A5)."""

from __future__ import annotations

import json

import pytest

from training.wavenet_study import study_common as sc


def _event(eid="B-x-001", **over):
    e = {"event_id": eid, "created_at": "2026-11-11T00:00:00+00:00", "attempt_id": "D-c30-s100-a1",
         "run_dir": "runs/wavenet_batchnorm/run_001", "arm": "D", "seed": 100, "cap": 30, "epoch_1based": 4,
         "global_step": 17127, "record": "steps",
         "field": ["updates", "per_parameter", "bn2.weight", "relative_update"],
         "observed": "inf", "reason": "non-finite derived ratio", "evidence": {"line": 228}, "status": "unresolved"}
    e.update(over)
    return e


def _res(eid="B-x-001", outcome="accepted_as_expected", **over):
    r = {"event_id": eid, "outcome": outcome, "reviewer": "Nur", "timestamp": "2026-11-12T10:00:00+00:00",
         "rationale": "explained", "evidence": "derivation"}
    r.update(over)
    return r


def test_register_resolve_and_status_with_full_history(tmp_path):
    led = sc.CategoryBLedger(tmp_path)
    led.register(_event("B-1"))
    led.register(_event("B-2"))
    assert led.blocking_events() == ["B-1", "B-2"]
    led.resolve(_res("B-1", "rejected"))
    led.resolve(_res("B-1", "accepted_as_expected"))
    led.resolve(_res("B-2", "corrected"))
    st = led.status()
    assert st["B-1"]["status"] == "accepted_as_expected" and len(st["B-1"]["resolutions"]) == 2
    assert led.blocking_events() == []


def test_unresolved_and_rejected_events_block(tmp_path):
    led = sc.CategoryBLedger(tmp_path)
    led.register(_event("B-1"))
    led.register(_event("B-2"))
    led.resolve(_res("B-2", "rejected"))
    assert led.blocking_events() == ["B-1", "B-2"]


@pytest.mark.parametrize("field", ["reviewer", "timestamp", "rationale", "evidence"])
def test_resolution_requires_all_four_fields(tmp_path, field):
    led = sc.CategoryBLedger(tmp_path)
    led.register(_event())
    with pytest.raises(sc.LedgerError):
        led.resolve(_res(**{field: "  "}))
    bad = _res()
    del bad[field]
    with pytest.raises(sc.LedgerError):
        led.resolve(bad)


def test_bad_outcome_unknown_event_duplicate_and_non_unresolved_initial_status_are_refused(tmp_path):
    led = sc.CategoryBLedger(tmp_path)
    led.register(_event())
    with pytest.raises(sc.LedgerError):
        led.resolve(_res(outcome="ignored"))
    with pytest.raises(sc.LedgerError):
        led.resolve(_res(eid="B-missing"))
    with pytest.raises(sc.LedgerError):
        led.register(_event())
    with pytest.raises(sc.LedgerError):
        led.register(_event("B-9", status="accepted_as_expected"))
    with pytest.raises(sc.LedgerError):
        led.register({k: v for k, v in _event("B-8").items() if k != "reason"})


def _lines(path):
    return path.read_text().splitlines()


def test_chain_detects_edits_reordering_deletion_and_formatting(tmp_path):
    led = sc.CategoryBLedger(tmp_path)
    for i in range(3):
        led.register(_event(f"B-{i}"))
    path = tmp_path / sc.CategoryBLedger.EVENTS_FILE
    original = path.read_text()
    lines = _lines(path)

    obj = json.loads(lines[1])
    obj["observed"] = "nan"  # edit a value, keep the line canonical
    path.write_text("\n".join([lines[0], sc.canonical_json(obj), lines[2]]) + "\n")
    with pytest.raises(sc.LedgerError, match="chain"):
        led.load()

    path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    with pytest.raises(sc.LedgerError, match="chain"):
        led.load()

    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(sc.LedgerError, match="chain"):
        led.load()

    path.write_text(original.replace('":"', '": "', 1))
    with pytest.raises(sc.LedgerError, match="canonical"):
        led.load()

    path.write_text(original[:-1])  # incomplete last line
    with pytest.raises(sc.LedgerError, match="incomplete"):
        led.load()


def test_truncating_a_suffix_is_only_visible_through_the_chain_head(tmp_path):
    led = sc.CategoryBLedger(tmp_path)
    led.register(_event("B-0"))
    head_after_one = led.heads()["events"]
    led.register(_event("B-1"))
    head = led.heads()["events"]
    path = tmp_path / sc.CategoryBLedger.EVENTS_FILE
    path.write_text(_lines(path)[0] + "\n")
    assert led.heads()["events"] == head_after_one != head  # recorded head exposes the truncation


def test_append_returns_the_new_head_and_never_rewrites(tmp_path):
    led = sc.CategoryBLedger(tmp_path)
    h1 = led.register(_event("B-0"))
    first = (tmp_path / sc.CategoryBLedger.EVENTS_FILE).read_text()
    h2 = led.register(_event("B-1"))
    after = (tmp_path / sc.CategoryBLedger.EVENTS_FILE).read_text()
    assert after.startswith(first) and h1 != h2 == led.heads()["events"]
    assert json.loads(_lines(tmp_path / sc.CategoryBLedger.EVENTS_FILE)[1])["prev_sha256"] == h1


def test_empty_ledger_has_genesis_heads(tmp_path):
    assert sc.CategoryBLedger(tmp_path).heads() == {"events": sc.GENESIS, "resolutions": sc.GENESIS}


def test_resolution_for_unknown_event_in_file_is_rejected_on_load(tmp_path):
    led = sc.CategoryBLedger(tmp_path)
    sc.HashChainedLog(tmp_path / sc.CategoryBLedger.RESOLUTIONS_FILE).append(_res("B-ghost"))
    with pytest.raises(sc.LedgerError, match="unknown event"):
        led.load()
