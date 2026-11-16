"""Repository-relative path resolution.

Every path in configs and metadata is interpreted relative to the repository
root, never relative to the current working directory. This removes the class
of bugs where a script only works when launched from one particular folder.
"""

from __future__ import annotations

from pathlib import Path

#: Absolute path of the repository root (the directory containing ``utils/``).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = REPO_ROOT / "data"
CONFIGS_DIR: Path = REPO_ROOT / "configs"
RESULTS_DIR: Path = REPO_ROOT / "results"
DEFAULT_NAMES_PATH: Path = DATA_DIR / "names.txt"


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Return an absolute path.

    Absolute inputs are returned unchanged (resolved). Relative inputs are
    joined onto ``base``, which defaults to the repository root.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path(base) if base is not None else REPO_ROOT).joinpath(p).resolve()


def relative_to_repo(path: str | Path) -> str:
    """Return ``path`` relative to the repository root when it lies inside it.

    Used when recording paths in metadata so runs are portable between
    machines. Paths outside the repository are returned as absolute POSIX
    strings.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()
