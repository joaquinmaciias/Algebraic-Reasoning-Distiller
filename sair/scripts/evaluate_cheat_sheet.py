"""Score a candidate cheat sheet against the held-out eval problems.

Uses ``sair.agents.evaluator.evaluate_with_local_model`` which in turn
calls the official SAIR ``judge_response`` extractor (via ``judge.py``
from the cloned judge repo). The per-problem results are dumped to
``sair/artifacts/<cheat_sheet_stem>_results.jsonl`` so error analysis
is trivial.

Usage
-----
    python -m sair.scripts.evaluate_cheat_sheet sair/cheat_sheets/v1.md
    python -m sair.scripts.evaluate_cheat_sheet sair/cheat_sheets/v1.md \
        --eval-file sair/data/training/eval_problems.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.paths import check_cwd

from sair.agents.distiller import load_distiller_model
from sair.agents.evaluator import evaluate_with_local_model
from sair.config import (
    REPO_DIR,
    SAIR_ARTIFACTS_DIR,
    SAIR_INFERENCE_CONFIG,
    SAIR_TRAINING_DIR,
)
from sair.data.load_problems import load_problems_from_jsonl
from sair.schemas import Problem, RunMetrics


def _resolve_eval_file(user_path: Path | None) -> Path:
    """Pick the eval JSONL from the user or the default training dir."""
    if user_path is not None:
        return user_path
    return SAIR_TRAINING_DIR / "eval_problems.jsonl"


def run(
    *,
    cheat_sheet_path: Path,
    eval_file: Path,
) -> RunMetrics:
    """Evaluate a cheat sheet file and write per-problem results."""
    if not cheat_sheet_path.exists():
        raise FileNotFoundError(f"cheat sheet not found: {cheat_sheet_path}")
    if not eval_file.exists():
        raise FileNotFoundError(
            f"eval problems not found: {eval_file}. "
            "Run `python -m sair.data.build_training_dataset` first."
        )

    cheat_sheet_text: str = cheat_sheet_path.read_text(encoding="utf-8")
    problems: list[Problem] = load_problems_from_jsonl(eval_file)
    print(f"[eval] {len(problems)} eval problems, cheat sheet {cheat_sheet_path.name}")

    print("[eval] loading distiller model...")
    cfg: SAIR_INFERENCE_CONFIG = SAIR_INFERENCE_CONFIG()
    model, tokenizer, adapter_label = load_distiller_model(cfg=cfg)
    print(f"[eval] adapter: {adapter_label}")

    metrics: RunMetrics
    per_problem: list[dict]
    metrics, per_problem = evaluate_with_local_model(
        problems=problems,
        cheat_sheet_text=cheat_sheet_text,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    SAIR_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path: Path = (
        SAIR_ARTIFACTS_DIR / f"{cheat_sheet_path.stem}_results.jsonl"
    )
    with results_path.open("w", encoding="utf-8") as fh:
        for row in per_problem:
            fh.write(json.dumps(row) + "\n")

    summary_path: Path = (
        SAIR_ARTIFACTS_DIR / f"{cheat_sheet_path.stem}_metrics.json"
    )
    summary_path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")

    print(
        f"[eval] accuracy = {metrics.accuracy:.3f} "
        f"({metrics.correct}/{metrics.total}, "
        f"unparseable={metrics.unparseable})"
    )
    for bucket, stats in metrics.per_bucket.items():
        total: float = stats.get("total", 0.0)
        correct: float = stats.get("correct", 0.0)
        acc: float = (correct / total) if total > 0 else 0.0
        print(f"[eval]  {bucket}: {acc:.3f} ({int(correct)}/{int(total)})")
    print(f"[eval] per-problem  -> {results_path}")
    print(f"[eval] summary      -> {summary_path}")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SAIR cheat sheet.")
    parser.add_argument(
        "cheat_sheet",
        type=Path,
        help="Path to the cheat sheet markdown file.",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="JSONL of eval problems (defaults to the training split).",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = _parse_args()
    run(
        cheat_sheet_path=Path(args.cheat_sheet),
        eval_file=_resolve_eval_file(args.eval_file),
    )


if __name__ == "__main__":
    check_cwd(expected_dir=REPO_DIR)
    main()
