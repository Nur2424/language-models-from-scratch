"""Every relative link in the main README (and the figures README) must resolve: the file or folder
exists, and a #section anchor matches a heading of the target, as GitHub generates it."""
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "figures/README.md")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def github_slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, markup removed, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("*", "").replace("_", "_")
    text = text.strip().lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in " -_" or cat[0] in "LN":
            out.append("-" if ch == " " else ch)
    return "".join(out)


def anchors(markdown: Path) -> set[str]:
    seen: dict[str, int] = {}
    result = set()
    in_code = False
    for line in markdown.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        m = HEADING.match(line)
        if m and not in_code:
            slug = github_slug(m.group(2))
            n = seen.get(slug, 0)
            result.add(slug if n == 0 else f"{slug}-{n}")
            seen[slug] = n + 1
    return result


def relative_links(markdown: Path) -> list[str]:
    text = re.sub(r"```.*?```", "", markdown.read_text(encoding="utf-8"), flags=re.S)
    return [t for t in LINK.findall(text) if not re.match(r"[a-z]+:", t)]


CASES = [(doc, link) for doc in DOCUMENTS for link in relative_links(REPO / doc)]


def test_github_slug_matches_known_examples():
    assert github_slug("5. Test results") == "5-test-results"
    assert github_slug("6. Where the advantage comes from (T-RQ1b)") == "6-where-the-advantage-comes-from-t-rq1b"
    assert github_slug("Context length and BatchNorm: a paired comparison") == \
        "context-length-and-batchnorm-a-paired-comparison"
    assert github_slug("Primary endpoint: test NLL (nats per character)") == \
        "primary-endpoint-test-nll-nats-per-character"


@pytest.mark.parametrize("doc,link", CASES, ids=[f"{d}:{l}" for d, l in CASES])
def test_relative_link_resolves(doc, link):
    path, _, anchor = link.partition("#")
    base = (REPO / doc).parent
    target = (base / unquote(path)).resolve() if path else (REPO / doc).resolve()
    assert target.exists(), f"{doc}: {link} -> missing {target}"
    if anchor:
        if target.is_dir():
            target = target / "README.md"
        assert target.suffix == ".md", f"{doc}: anchor on a non-Markdown target {link}"
        assert anchor in anchors(target), f"{doc}: no heading for #{anchor} in {target.relative_to(REPO)}"
