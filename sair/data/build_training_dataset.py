"""Build SFT and GRPO training datasets for the SAIR distiller.

This script runs the offline pipeline (counterexample + symbolic prover +
retriever) over every labeled problem, then emits two JSONL files:

- ``sft_dataset.jsonl`` — supervised examples of the form
  ``{"prompt": "<problem>", "completion": "<think>...</think><answer>...</answer>"}``
  where the completion is templated from the ground-truth evidence.
- ``grpo_dataset.jsonl`` — per-problem records used by the GRPO loop.
  Each record contains the user prompt and the ground-truth verdict so
  the reward function can grade generations.

Usage
-----
    python -m sair.data.build_training_dataset
    python -m sair.data.build_training_dataset --max-problems 500 --no-retriever
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sair.config import SAIR_TRAINING_DIR
from sair.data.load_problems import load_all_problems, split_problems
from sair.graph import run_pipeline_sequential
from sair.schemas import EvidenceBundle, Problem, TrainingExample


def _render_completion(bundle: EvidenceBundle, *, verdict: bool) -> str:
    """Template a supervised completion from an EvidenceBundle.

    Format:
        <think>
        (evidence reasoning)
        </think>
        <answer>
        VERDICT: TRUE|FALSE
        REASONING: ...
        PROOF: ...   (if TRUE)
        COUNTEREXAMPLE: ...   (if FALSE)
        </answer>
    """
    reasoning: str = bundle.rendered_reasoning() or "No additional evidence."
    verdict_label: str = "TRUE" if verdict else "FALSE"

    if verdict:
        body: str = (
            f"VERDICT: {verdict_label}\n"
            f"REASONING: Forward reasoning from E1 yields E2.\n"
            f"PROOF:\n{reasoning}"
        )
    else:
        body = (
            f"VERDICT: {verdict_label}\n"
            f"REASONING: A small finite magma satisfies E1 but not E2.\n"
            f"COUNTEREXAMPLE:\n{reasoning}"
        )

    return f"<think>\n{reasoning}\n</think>\n<answer>\n{body}\n</answer>"


def _render_user_prompt(problem: Problem) -> str:
    """Build the user message for the distiller training examples."""
    return (
        f"<equation1>{problem.equation1}</equation1>\n"
        f"<equation2>{problem.equation2}</equation2>"
    )


def _is_consistent(bundle: EvidenceBundle, problem: Problem) -> bool:
    """True if the solver agrees with the labeled answer (when it decided)."""
    if bundle.consensus_verdict is None:
        return True
    if problem.answer is None:
        return True
    return bundle.consensus_verdict == problem.answer


def build(
    *,
    max_problems: int | None = None,
    train_ratio: float = 0.8,
    use_retriever: bool = False,
    seed: int = 0,
) -> dict[str, int]:
    """Build the two JSONL training files and return a counts summary."""
    SAIR_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    problems: list[Problem] = load_all_problems()
    labeled: list[Problem] = [p for p in problems if p.answer is not None]
    print(f"[build] loaded {len(problems)} problems ({len(labeled)} with labels)")

    if max_problems is not None:
        labeled = labeled[: int(max_problems)]

    train: list[Problem]
    eval_split: list[Problem]
    train, eval_split = split_problems(
        labeled, train_ratio=train_ratio, seed=seed
    )
    print(f"[build] split: train={len(train)} eval={len(eval_split)}")

    sft_path: Path = SAIR_TRAINING_DIR / "sft_dataset.jsonl"
    grpo_path: Path = SAIR_TRAINING_DIR / "grpo_dataset.jsonl"
    eval_path: Path = SAIR_TRAINING_DIR / "eval_problems.jsonl"

    n_sft: int = 0
    n_skipped: int = 0
    n_solver_agree: int = 0

    with sft_path.open("w", encoding="utf-8") as sft_fh, grpo_path.open(
        "w", encoding="utf-8"
    ) as grpo_fh:
        for prob in train:
            bundle: EvidenceBundle = run_pipeline_sequential(
                prob, use_retriever=use_retriever
            )
            if not _is_consistent(bundle, prob):
                # Solver contradicted the label — skip this example so we
                # don't poison the SFT set with wrong reasoning.
                n_skipped += 1
                continue

            if bundle.consensus_verdict == prob.answer:
                n_solver_agree += 1

            # Supervised example (always uses the labeled verdict, even if
            # the solver couldn't decide — that way the completion is
            # always correct, just with weaker evidence).
            assert prob.answer is not None
            completion: str = _render_completion(bundle, verdict=prob.answer)
            prompt: str = _render_user_prompt(prob)
            example: TrainingExample = TrainingExample(
                problem_id=prob.id,
                prompt=prompt,
                completion=completion,
                verdict=prob.answer,
                source="counterexample"
                if bundle.consensus_verdict is False
                else "symbolic_proof"
                if bundle.consensus_verdict is True
                else "external_label",
            )
            sft_fh.write(example.model_dump_json() + "\n")
            n_sft += 1

            # GRPO record (raw problem + label, no scripted completion)
            grpo_record: dict = {
                "id": prob.id,
                "prompt": prompt,
                "answer": prob.answer,
                "difficulty": prob.difficulty,
            }
            grpo_fh.write(json.dumps(grpo_record) + "\n")

    with eval_path.open("w", encoding="utf-8") as eval_fh:
        for prob in eval_split:
            eval_fh.write(prob.model_dump_json() + "\n")

    summary: dict[str, int] = {
        "sft_examples": n_sft,
        "skipped_inconsistent": n_skipped,
        "solver_agreed": n_solver_agree,
        "eval_problems": len(eval_split),
    }
    print(f"[build] {summary}")
    print(f"[build] SFT   -> {sft_path}")
    print(f"[build] GRPO  -> {grpo_path}")
    print(f"[build] EVAL  -> {eval_path}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SAIR training datasets.")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--no-retriever", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = _parse_args()
    build(
        max_problems=args.max_problems,
        train_ratio=float(args.train_ratio),
        use_retriever=not bool(args.no_retriever),
        seed=int(args.seed),
    )
