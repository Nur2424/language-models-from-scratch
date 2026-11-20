"""Build the README figures from the published result files.

Usage (from the repository root)::

    python3 figures/make_figures.py

Writes figures/out/readme_fig1.{png,pdf} and figures/out/readme_fig2.{png,pdf}.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from figures import readme_fig1, readme_fig2  # noqa: E402


def main() -> int:
    for build in (readme_fig1.make, readme_fig2.make):
        for path in build():
            print(path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
