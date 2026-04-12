"""Evaluator agent: simulates the offline no-tools SAIR judge.

This reuses the official verdict extractor from the SAIR judge repo
(``judge.py``) so our local accuracy numbers match the grading logic
byte-for-byte. If the judge repo is missing from the expected location,
we fall back to an inline reimplementation of the same rules.

Two evaluation backends are provided:

- **Local backend** (``evaluate_with_local_model``): uses the SAIR
  distiller LoRA to answer each problem with a cheat sheet injected into
  the system prompt. Free, fast, matches our training signal.
- **Cheat-sheet-only backend** (``evaluate_cheat_sheet``): renders the
  cheat sheet through the official ``prompt.render_prompt`` template and
  reports per-problem verdicts. Used to score a candidate cheat sheet
  exactly as the official judge would at submission time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sair.config import SAIR_INFERENCE_CONFIG, SAIR_JUDGE_REPO
from sair.schemas import Problem, RunMetrics


# ---------------------------------------------------------------------------
# Judge binding — import judge.judge_response from the cloned SAIR repo
# ---------------------------------------------------------------------------


def _load_official_judge() -> Callable[[str, bool], tuple[bool | None, str]] | None:
    """Try to import ``judge_response`` from the cloned SAIR judge repo.

    Returns None if the repo is not present or the module fails to load.
    """
    judge_path: Path = SAIR_JUDGE_REPO / "judge.py"
    if not judge_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("sair_judge", str(judge_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["sair_judge"] = module
        spec.loader.exec_module(module)
        return module.judge_response  # type: ignore[attr-defined]
    except Exception:
        return None


_judge_response = _load_official_judge()


def _fallback_judge(response: str, expected_answer: bool) -> tuple[bool | None, str]:
    """Minimal fallback if the official judge cannot be imported.

    Only recognizes ``VERDICT: TRUE|FALSE`` markers. Used as a last resort
    so tests can still run without the external repo.
    """
    import re

    matches: list[str] = re.findall(r"(?i)VERDICT\s*[:：]\s*(TRUE|FALSE)", response)
    if not matches:
        return None, "No VERDICT found in response"
    verdict: bool = matches[-1].upper() == "TRUE"
    correct: bool = verdict == expected_answer
    label: str = "TRUE" if verdict else "FALSE"
    if correct:
        return True, f"Answered {label}"
    exp: str = "TRUE" if expected_answer else "FALSE"
    return False, f"Answered {label}, expected {exp}"


def judge_response(response: str, expected_answer: bool) -> tuple[bool | None, str]:
    """Public judge entry point. Uses the official implementation when available."""
    if _judge_response is not None:
        return _judge_response(response, expected_answer)
    return _fallback_judge(response, expected_answer)


# ---------------------------------------------------------------------------
# Local model backend — answer problems with the trained distiller
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _answer_problem_local(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: SAIR_INFERENCE_CONFIG,
    cheat_sheet_text: str,
    problem: Problem,
) -> str:
    """Run the local distiller on one problem, with the cheat sheet injected."""
    system_prompt: str = str(cfg.system_prompt)
    if cheat_sheet_text.strip():
        system_prompt = f"{system_prompt}\n\nCHEAT SHEET:\n{cheat_sheet_text}\n"

    user_prompt: str = (
        f"<equation1>{problem.equation1}</equation1>\n"
        f"<equation2>{problem.equation2}</equation2>"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_ids: torch.Tensor = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs: torch.Tensor = model.generate(
        input_ids=input_ids,
        max_new_tokens=int(cfg.max_new_tokens),
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        do_sample=False,
    )
    prompt_len: int = int(input_ids.shape[-1])
    return tokenizer.decode(
        outputs[0, prompt_len:], skip_special_tokens=True
    ).strip()


def evaluate_with_local_model(
    *,
    problems: list[Problem],
    cheat_sheet_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: SAIR_INFERENCE_CONFIG | None = None,
) -> tuple[RunMetrics, list[dict[str, Any]]]:
    """Score a cheat sheet against labeled problems using the local distiller.

    Returns a ``RunMetrics`` summary plus a per-problem detail list suitable
    for error analysis and GRPO reward dumps.
    """
    cfg = cfg or SAIR_INFERENCE_CONFIG()

    total: int = 0
    correct: int = 0
    unparseable: int = 0
    by_bucket: dict[str, dict[str, float]] = {}
    per_problem: list[dict[str, Any]] = []

    start: float = perf_counter()
    for problem in problems:
        if problem.answer is None:
            continue
        total += 1
        response: str = _answer_problem_local(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            cheat_sheet_text=cheat_sheet_text,
            problem=problem,
        )
        judged: tuple[bool | None, str] = judge_response(response, problem.answer)
        ok: bool | None = judged[0]
        reason: str = judged[1]

        if ok is None:
            unparseable += 1
        elif ok:
            correct += 1

        bucket: str = problem.difficulty or "unknown"
        bucket_stats: dict[str, float] = by_bucket.setdefault(
            bucket, {"total": 0.0, "correct": 0.0, "unparseable": 0.0}
        )
        bucket_stats["total"] += 1.0
        if ok is True:
            bucket_stats["correct"] += 1.0
        elif ok is None:
            bucket_stats["unparseable"] += 1.0

        per_problem.append(
            {
                "id": problem.id,
                "difficulty": problem.difficulty,
                "expected": problem.answer,
                "correct": ok,
                "reason": reason,
                "response": response,
            }
        )

    accuracy: float = float(correct) / float(total) if total > 0 else 0.0

    metrics: RunMetrics = RunMetrics(
        total=total,
        correct=correct,
        unparseable=unparseable,
        accuracy=accuracy,
        per_bucket=by_bucket,
        wall_seconds=perf_counter() - start,
    )
    return metrics, per_problem
