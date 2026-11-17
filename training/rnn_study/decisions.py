"""Decision verification against the research record (D-49, P7-7; plan §8.4).

Drivers fail closed unless every decision they depend on is present in the decision table of
``PROJECT_RESEARCH_RECORD.md`` §11, carries exactly the status form ``ACCEPTED (user approval,
YYYY-MM-DD)`` (or ``authorisation`` / ``decision``, O-4), has the pinned row SHA-256, and no newer
decision exists than the one recorded as reviewed (``record_reviewed_through``).

Parsing is structural: the section heading, the exact header row, and a code-span-aware cell splitter
(a ``|`` inside backticks or escaped as ``\\|`` is not a separator); every row must have exactly 7
cells. No substring matching on decision text is used. No new approval file is introduced; the study
conditions only *reference* decisions.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SECTION_HEADING = "## 11. Decisions and Rationale"
NEXT_SECTION_PREFIX = "## 12."
HEADER = ("ID", "Date", "Decision", "Status", "Reason", "Consequences", "Reference")
ID_RE = re.compile(r"^D-\d{2,}( addendum)?$")
STATUS_RE = re.compile(r"^ACCEPTED \(user (approval|authorisation|decision), \d{4}-\d{2}-\d{2}\)$")


class DecisionError(RuntimeError):
    """The record's decision table is malformed, or a required decision is missing or not accepted."""


@dataclass(frozen=True)
class DecisionRow:
    id: str
    cells: tuple[str, ...]
    text: str
    sha256: str
    index: int

    @property
    def status(self) -> str:
        return self.cells[3]


def split_cells(line: str) -> list[str]:
    """Cells of one Markdown table row. Backtick code spans (matching run lengths) and ``\\|`` do not
    split. Leading and trailing pipes are required."""
    s = line.rstrip("\n")
    if not (s.startswith("|") and s.endswith("|")):
        raise DecisionError(f"not a table row: {s[:60]!r}")
    body, cells, cur, i, fence = s[1:-1], [], [], 0, 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if ch == "`":
            j = i
            while j < len(body) and body[j] == "`":
                j += 1
            run = j - i
            if fence == 0:
                fence = run
            elif fence == run:
                fence = 0
            cur.append(body[i:j])
            i = j
            continue
        if ch == "|" and fence == 0:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if fence:
        raise DecisionError(f"unclosed code span in row: {s[:60]!r}")
    cells.append("".join(cur).strip())
    return cells


def parse_decision_table(text: str) -> list[DecisionRow]:
    """Parse §11's decision table (plan §8.4 steps 1 – 4); any irregularity raises :class:`DecisionError`."""
    lines = text.split("\n")
    starts = [i for i, ln in enumerate(lines) if ln == SECTION_HEADING]
    if len(starts) != 1:
        raise DecisionError(f"expected exactly one line {SECTION_HEADING!r}, found {len(starts)}")
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(NEXT_SECTION_PREFIX)), None)
    if end is None:
        raise DecisionError(f"no {NEXT_SECTION_PREFIX!r} heading closes the decisions section")
    section = lines[start + 1: end]
    header_at = [i for i, ln in enumerate(section) if ln.startswith("|") and tuple(split_cells(ln)) == HEADER]
    if len(header_at) != 1:
        raise DecisionError(f"expected exactly one decision table header, found {len(header_at)}")
    h = header_at[0]
    if h + 1 >= len(section) or set(section[h + 1].replace("|", "").strip()) - set("-: "):
        raise DecisionError("the header row is not followed by a separator row")
    rows, seen = [], set()
    i = h + 2
    while i < len(section) and section[i].startswith("|"):
        cells = split_cells(section[i])
        if len(cells) != len(HEADER):
            raise DecisionError(f"row {i - h - 1} has {len(cells)} cells, expected {len(HEADER)}: {section[i][:60]!r}")
        if not ID_RE.match(cells[0]):
            raise DecisionError(f"invalid decision id {cells[0]!r}")
        if cells[0] in seen:
            raise DecisionError(f"duplicate decision id {cells[0]}")
        seen.add(cells[0])
        rows.append(DecisionRow(cells[0], tuple(cells), section[i],
                                hashlib.sha256(section[i].encode("utf-8")).hexdigest(), len(rows)))
        i += 1
    stray = [ln for ln in section[i:] + section[:h] if ln.startswith("| D-")]
    if stray:
        raise DecisionError(f"decision-like rows outside the table: {stray[0][:60]!r}")
    if not rows:
        raise DecisionError("the decision table is empty")
    return rows


def require_accepted(rows: list[DecisionRow], decision_id: str, pinned_sha256: str) -> DecisionRow:
    """The decision exists, its status matches the exact accepted form, and its row is unchanged."""
    row = next((r for r in rows if r.id == decision_id), None)
    if row is None:
        raise DecisionError(f"{decision_id} is not in the research record")
    if not STATUS_RE.match(row.status):
        raise DecisionError(f"{decision_id} status {row.status!r} is not of the form "
                            "'ACCEPTED (user approval|authorisation|decision, YYYY-MM-DD)'")
    if row.sha256 != pinned_sha256:
        raise DecisionError(f"{decision_id}: the record row changed since it was pinned "
                            f"({row.sha256[:12]}… != {pinned_sha256[:12]}…)")
    return row


def check_reviewed_through(rows: list[DecisionRow], reviewed_through: str) -> None:
    """Fail when the record holds any decision after ``reviewed_through`` (staleness, plan §8.4 step 4)."""
    idx = next((r.index for r in rows if r.id == reviewed_through), None)
    if idx is None:
        raise DecisionError(f"record_reviewed_through {reviewed_through!r} is not in the record")
    newer = [r.id for r in rows if r.index > idx]
    if newer:
        raise DecisionError(f"the record has decisions after {reviewed_through} not yet reviewed into the study "
                            f"conditions: {newer}")


def verify_decisions(record_path: str | Path, references: Mapping[str, Mapping[str, str]],
                     required: tuple[str, ...], reviewed_through: str) -> dict[str, DecisionRow]:
    """Verify every ``required`` reference key. ``references`` maps a key (for example
    ``"authorisation_L2"``) to ``{"id": "D-NN", "row_sha256": "…"}``."""
    rows = parse_decision_table(Path(record_path).read_text(encoding="utf-8"))
    check_reviewed_through(rows, reviewed_through)
    out = {}
    for key in required:
        ref = references.get(key)
        if not ref:
            raise DecisionError(f"required decision {key!r} is not referenced in the study conditions")
        out[key] = require_accepted(rows, ref["id"], ref["row_sha256"])
    return out


def row_sha256(record_path: str | Path, decision_id: str) -> str:
    """SHA-256 of one row's exact text, for pinning a decision in the study conditions."""
    rows = parse_decision_table(Path(record_path).read_text(encoding="utf-8"))
    row = next((r for r in rows if r.id == decision_id), None)
    if row is None:
        raise DecisionError(f"{decision_id} is not in the research record")
    return row.sha256
