"""MS7: decision verification (D-49 P7-7, O-4; plan §8.4). Synthetic records, plus one read-only parse
of the real research record."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from training.rnn_study import decisions as dc
from training.rnn_study.common import REPO_ROOT

HEADER = "| ID | Date | Decision | Status | Reason | Consequences | Reference |\n|---|---|---|---|---|---|---|\n"
ROW = "| {id} | 2026-11-20 | Decision with `code \\| pipe` and `a | b` | {status} | why | then | ref |\n"
OK = "ACCEPTED (user approval, 2026-11-20)"


def record(rows: str, *, heading="## 11. Decisions and Rationale", tail="## 12. Limitations\n", extra="") -> str:
    return f"# Record\n\n{heading}\n\nText.\n\n{HEADER}{rows}\n{extra}{tail}body\n"


def write(tmp_path, text) -> Path:
    p = tmp_path / "record.md"
    p.write_text(text, encoding="utf-8")
    return p


def sha(line: str) -> str:
    return hashlib.sha256(line.rstrip("\n").encode()).hexdigest()


def test_valid_table_and_code_span_pipes(tmp_path) -> None:
    r1, r2 = ROW.format(id="D-01", status="ACCEPTED"), ROW.format(id="D-02", status=OK)
    rows = dc.parse_decision_table(record(r1 + r2))
    assert [r.id for r in rows] == ["D-01", "D-02"] and all(len(r.cells) == 7 for r in rows)
    assert rows[1].cells[2] == "Decision with `code | pipe` and `a | b`"
    assert rows[1].sha256 == sha(r2)
    path = write(tmp_path, record(r1 + r2))
    got = dc.verify_decisions(path, {"auth": {"id": "D-02", "row_sha256": sha(r2)}}, ("auth",), "D-02")
    assert got["auth"].id == "D-02"


@pytest.mark.parametrize("status", ["ACCEPTED", "PENDING", "PROPOSED", "ACCEPTED (direction); details PENDING",
                                    "ACCEPTED (user approval, 2026-11-20); final runs PENDING",
                                    "**ACCEPTED (user approval, 2026-11-20)**", "accepted (user approval, 2026-11-20)"])
def test_status_must_match_the_exact_form(tmp_path, status) -> None:
    row = ROW.format(id="D-05", status=status)
    path = write(tmp_path, record(row))
    with pytest.raises(dc.DecisionError, match="status"):
        dc.verify_decisions(path, {"k": {"id": "D-05", "row_sha256": sha(row)}}, ("k",), "D-05")


@pytest.mark.parametrize("form", ["approval", "authorisation", "decision"])
def test_three_accepted_forms(tmp_path, form) -> None:
    row = ROW.format(id="D-05", status=f"ACCEPTED (user {form}, 2026-11-20)")
    path = write(tmp_path, record(row))
    dc.verify_decisions(path, {"k": {"id": "D-05", "row_sha256": sha(row)}}, ("k",), "D-05")


def test_missing_edited_and_unreferenced_decisions(tmp_path) -> None:
    row = ROW.format(id="D-05", status=OK)
    path = write(tmp_path, record(row))
    with pytest.raises(dc.DecisionError, match="not in the research record"):
        dc.verify_decisions(path, {"k": {"id": "D-06", "row_sha256": sha(row)}}, ("k",), "D-05")
    with pytest.raises(dc.DecisionError, match="changed"):
        dc.verify_decisions(path, {"k": {"id": "D-05", "row_sha256": "0" * 64}}, ("k",), "D-05")
    with pytest.raises(dc.DecisionError, match="not referenced"):
        dc.verify_decisions(path, {}, ("k",), "D-05")


def test_staleness_blocks_until_reviewed(tmp_path) -> None:
    r1, r2 = ROW.format(id="D-05", status=OK), ROW.format(id="D-06", status=OK)
    path = write(tmp_path, record(r1 + r2))
    with pytest.raises(dc.DecisionError, match="after D-05"):
        dc.verify_decisions(path, {"k": {"id": "D-05", "row_sha256": sha(r1)}}, ("k",), "D-05")
    dc.verify_decisions(path, {"k": {"id": "D-05", "row_sha256": sha(r1)}}, ("k",), "D-06")
    with pytest.raises(dc.DecisionError, match="not in the record"):
        dc.verify_decisions(path, {"k": {"id": "D-05", "row_sha256": sha(r1)}}, ("k",), "D-09")


@pytest.mark.parametrize("text,match", [
    (record("| D-01 | d | x | ACCEPTED | r | c |\n"), "cells"),                       # 6 cells
    (record("| D-01 | d | x | ACCEPTED | r | c | ref | extra |\n"), "cells"),        # 8 cells
    (record(ROW.format(id="D-01", status=OK) * 2), "duplicate"),
    (record(ROW.format(id="X-01", status=OK)), "invalid decision id"),
    (record(ROW.format(id="D-01", status=OK), heading="## 11. Decisions"), "exactly one line"),
    (record(ROW.format(id="D-01", status=OK), tail="## 13. Other\n"), "closes"),
    (record(ROW.format(id="D-01", status=OK), extra="\n" + HEADER + ROW.format(id="D-02", status=OK)), "header"),
    (record(ROW.format(id="D-01", status=OK), extra="\nnote\n" + ROW.format(id="D-02", status=OK)), "outside the table"),
    (record("| D-01 | d | `unclosed | x | s | r | c | ref |\n"), "unclosed"),
])
def test_malformed_tables_fail_closed(text, match) -> None:
    with pytest.raises(dc.DecisionError, match=match):
        dc.parse_decision_table(text)


def test_escaped_pipe_is_not_a_separator() -> None:
    cells = dc.split_cells("| D-01 | a \\| b | x | s | r | c | ref |")
    assert len(cells) == 7 and cells[1] == "a | b"


@pytest.mark.skipif(not (REPO_ROOT / "PROJECT_RESEARCH_RECORD.md").exists(),
                    reason="the project's research record is kept locally and is not in the repository")
def test_real_record_parses_and_d47_d49_are_accepted() -> None:
    """Read-only: the project's research record parses with the structural rules, every row has 7
    cells, and D-47 – D-49 satisfy the exact status form (O-4)."""
    rows = dc.parse_decision_table((REPO_ROOT / "PROJECT_RESEARCH_RECORD.md").read_text(encoding="utf-8"))
    ids = [r.id for r in rows]
    assert {"D-01", "D-47", "D-48", "D-49"} <= set(ids) and all(len(r.cells) == 7 for r in rows)
    for did in ("D-47", "D-48", "D-49"):
        row = next(r for r in rows if r.id == did)
        assert dc.require_accepted(rows, did, row.sha256) is row
