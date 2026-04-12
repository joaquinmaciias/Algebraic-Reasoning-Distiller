from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def find_parent_with_markers(
    start: Path,
    markers: Iterable[str] = (".git", ".venv", "pyproject.toml"),
) -> Path:
    """Return the first parent directory containing any marker directory.

    The search starts at `start` itself and walks upwards until the filesystem
    root is reached. Raises a RuntimeError if none of the markers are found.

    Args:
        start: Initial path from which the upward search begins.
        markers: Directory names whose presence defines a match.

    Returns:
        The first Path containing at least one marker.

    Raises:
        RuntimeError: If no parent directory contains any marker.
    """
    current: Path = start.resolve()

    for parent in (current, *current.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent

    raise RuntimeError(
        "No parent directory containing any of the markers was found. "
        f"Searched from: {current}"
    )


def check_cwd(expected_dir: Path) -> None:
    """Ensure the current working directory matches the expected directory.

    Args:
        expected_dir: Directory from which the script must be executed.

    Raises:
        RuntimeError: If the current working directory is different from
            `expected_dir`.
    """
    current: Path = Path.cwd().resolve()
    expected: Path = expected_dir.resolve()

    if current != expected:
        raise RuntimeError(
            "Script must be executed from the expected directory.\n"
            f"Current working directory: {current}\n"
            f"Expected directory: {expected}"
        )
