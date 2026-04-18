"""Parse SAIR competition problem text files into a JSONL dataset.

Each line in the source files has the format:
    #N: <equation1> → <equation2> True|False

The arrow may appear as the Unicode character → or as â (garbled UTF-8).

Usage
-----
    # Place the .txt files in sair/data/competition/ then run:
    python -m sair.scripts.parse_competition_problems

    # Or specify a custom input directory and output file:
    python -m sair.scripts.parse_competition_problems \\
        --input-dir sair/data/competition \\
        --output    sair/data/problems/competition_1669.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from utils.paths import check_cwd
from sair.config import REPO_DIR, SAIR_PROBLEMS_DIR


# Matches both the real Unicode arrow and common garbled variants.
# The â character (U+00E2) is the first byte of the UTF-8 sequence for →
# when mis-decoded as Latin-1, so we match it followed by optional junk.
_ARROW_RE = re.compile(r"\s*(?:→|â[^\w\s]*)\s*")


def _parse_line(line: str, *, source: str, lineno: int) -> dict | None:
    """Parse one problem line.  Returns None if the line should be skipped."""
    line = line.strip()
    if not line or not line.startswith("#"):
        return None

    try:
        colon_idx = line.index(":")
        num_str = line[1:colon_idx].strip()
        num = int(num_str)
        rest = line[colon_idx + 1 :].strip()
    except (ValueError, IndexError):
        return None

    # Split on the arrow (handles → and â variants).
    parts = _ARROW_RE.split(rest, maxsplit=1)
    if len(parts) != 2:
        return None

    eq1 = parts[0].strip()
    eq2_label = parts[1].strip()

    # The label (True/False) may be attached to eq2 without a space.
    if eq2_label.endswith("True"):
        eq2 = eq2_label[:-4].strip()
        answer = True
    elif eq2_label.endswith("False"):
        eq2 = eq2_label[:-5].strip()
        answer = False
    else:
        # No recognisable label — skip rather than guess.
        return None

    if not eq1 or not eq2:
        return None

    problem_id = f"{Path(source).stem}_{num}"
    return {
        "id": problem_id,
        "equation1": eq1,
        "equation2": eq2,
        "answer": answer,
    }


def parse_file(path: Path) -> list[dict]:
    """Parse all valid problems from one text file."""
    records: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            record = _parse_line(line, source=path.name, lineno=lineno)
            if record is not None:
                records.append(record)
    return records


def build(
    *,
    input_dir: Path,
    output_path: Path,
) -> dict[str, int]:
    """Parse all .txt files in input_dir and write combined JSONL."""
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    total = 0
    skipped_dup = 0
    per_file: dict[str, int] = {}
    counts = {"true": 0, "false": 0}

    with output_path.open("w", encoding="utf-8") as out_fh:
        for txt in txt_files:
            records = parse_file(txt)
            written = 0
            for rec in records:
                if rec["id"] in seen_ids:
                    skipped_dup += 1
                    continue
                seen_ids.add(rec["id"])
                out_fh.write(json.dumps(rec) + "\n")
                written += 1
                total += 1
                counts["true" if rec["answer"] else "false"] += 1
            per_file[txt.name] = written
            print(f"  {txt.name}: {written} problems parsed")

    summary = {
        "total": total,
        "true": counts["true"],
        "false": counts["false"],
        "skipped_duplicates": skipped_dup,
        "output": str(output_path),
    }
    print(f"\n[parse] {summary}")
    return summary


def _parse_args() -> argparse.Namespace:
    default_input = REPO_DIR / "sair" / "data" / "competition"
    default_output = SAIR_PROBLEMS_DIR / "competition_all.jsonl"
    parser = argparse.ArgumentParser(
        description="Parse competition .txt files into SAIR JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    return parser.parse_args()


if __name__ == "__main__":
    check_cwd(expected_dir=REPO_DIR)
    args = _parse_args()
    build(input_dir=Path(args.input_dir), output_path=Path(args.output))
