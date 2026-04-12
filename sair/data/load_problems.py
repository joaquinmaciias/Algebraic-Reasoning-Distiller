"""Problem loading utilities.

The SAIR Stage 1 problems ship as JSONL files with one record per line,
each exposing at least ``id``, ``equation1``, ``equation2`` and
``answer``. We accept any JSONL in a directory so the user can drop
additional HuggingFace-downloaded splits next to the judge examples
without touching code.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from sair.config import SAIR_JUDGE_EXAMPLES, SAIR_PROBLEMS_DIR
from sair.schemas import Problem


def _parse_record(obj: dict) -> Problem | None:
    """Best-effort conversion of a raw JSON dict to a Problem."""
    try:
        return Problem(
            id=str(obj["id"]),
            equation1=str(obj["equation1"]),
            equation2=str(obj["equation2"]),
            answer=bool(obj["answer"]) if "answer" in obj else None,
            difficulty=obj.get("difficulty"),
            eq1_id=obj.get("eq1_id"),
            eq2_id=obj.get("eq2_id"),
            index=obj.get("index"),
        )
    except Exception:
        return None


def load_problems_from_jsonl(path: Path) -> list[Problem]:
    """Load and validate problems from a single JSONL file."""
    if not path.exists():
        return []
    out: list[Problem] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed: Problem | None = _parse_record(obj)
            if parsed is not None:
                out.append(parsed)
    return out


def _candidate_directories() -> Iterable[Path]:
    """Yield directories that may contain problem JSONL files."""
    if SAIR_PROBLEMS_DIR.exists():
        yield SAIR_PROBLEMS_DIR
    if SAIR_JUDGE_EXAMPLES.exists():
        yield SAIR_JUDGE_EXAMPLES


def load_all_problems() -> list[Problem]:
    """Load every JSONL problem file from the known data directories.

    Deduplicates by ``id`` — first occurrence wins. Order is: local
    ``sair/data/problems`` first (user-downloaded splits), then the
    judge repo's bundled examples.
    """
    seen: set[str] = set()
    out: list[Problem] = []
    for d in _candidate_directories():
        for fp in sorted(d.glob("*.jsonl")):
            for prob in load_problems_from_jsonl(fp):
                if prob.id in seen:
                    continue
                seen.add(prob.id)
                out.append(prob)
    return out


def split_problems(
    problems: list[Problem],
    *,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> tuple[list[Problem], list[Problem]]:
    """Deterministic train/eval split by id hash."""
    shuffled: list[Problem] = list(problems)
    random.Random(seed).shuffle(shuffled)
    cut: int = int(round(len(shuffled) * float(train_ratio)))
    return shuffled[:cut], shuffled[cut:]
