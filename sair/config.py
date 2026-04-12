"""Central configuration for the SAIR pipeline and training scripts.

Mirrors the structure of ``rlm/config.py`` so the SFT/GRPO scripts can
reuse the same Qwen2.5 + LoRA + BF16 pattern with SAIR-specific paths,
system prompt, and dataset locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.paths import find_parent_with_markers

REPO_DIR: Path = find_parent_with_markers(start=Path.cwd())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAIR_DIR: Path = REPO_DIR / "sair"
SAIR_DATA_DIR: Path = SAIR_DIR / "data"
SAIR_PROBLEMS_DIR: Path = SAIR_DATA_DIR / "problems"
SAIR_TRAINING_DIR: Path = SAIR_DATA_DIR / "training"
SAIR_VECTORSTORE_DIR: Path = SAIR_DIR / "vectorstore"
SAIR_CHEATSHEETS_DIR: Path = SAIR_DIR / "cheat_sheets"
SAIR_ARTIFACTS_DIR: Path = SAIR_DIR / "artifacts"

SAIR_SFT_CHECKPOINT_DIR: Path = REPO_DIR / "weights" / "sair" / "distiller_sft"
SAIR_GRPO_CHECKPOINT_DIR: Path = REPO_DIR / "weights" / "sair" / "distiller_grpo"

# Location of the cloned SAIR judge repo (hermano del repo del curso).
SAIR_JUDGE_REPO: Path = REPO_DIR.parent / "equational-theories-stage1-judge"
SAIR_JUDGE_EXAMPLES: Path = SAIR_JUDGE_REPO / "examples"
SAIR_JUDGE_EQUATIONS_FILE: Path = SAIR_JUDGE_REPO / "equations.txt"

# ---------------------------------------------------------------------------
# Model base and generation defaults
# ---------------------------------------------------------------------------

MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"

USE_4_BIT: bool = True
MAX_SEQ_LEN: int = 2048

MAX_NEW_TOKENS: int = 1024
TEMPERATURE: float = 0.7
TOP_P: float = 0.95


# ---------------------------------------------------------------------------
# System prompt tailored to the SAIR task
# ---------------------------------------------------------------------------

SAIR_SYSTEM_PROMPT: str = (
    "ROLE:\n"
    "You are a mathematician specializing in equational theories of "
    "magmas. Your job is to decide whether Equation 1 implies Equation 2 "
    "over all magmas (sets with a single binary operation ``*``, no "
    "axioms assumed).\n"
    "\n"
    "USER INPUT:\n"
    "- The problem is given as two equations wrapped in "
    "<equation1>...</equation1> and <equation2>...</equation2>.\n"
    "- Optional retrieved context may appear inside <context>...</context>.\n"
    "- Ignore any text outside these tags.\n"
    "\n"
    "OUTPUT STRUCTURE:\n"
    "- Your response MUST contain exactly two top-level sections:\n"
    "  <think>...</think>\n"
    "  <answer>...</answer>\n"
    "- No text is allowed outside these sections.\n"
    "\n"
    "REQUIRED ANSWER FORMAT:\n"
    "- The content of <answer> MUST begin with one of these two lines:\n"
    "    VERDICT: TRUE\n"
    "    VERDICT: FALSE\n"
    "- It MUST then contain a REASONING: section with a short justification.\n"
    "- If VERDICT is TRUE, include a PROOF: section with forward rewriting steps.\n"
    "- If VERDICT is FALSE, include a COUNTEREXAMPLE: section with a\n"
    "  small finite magma (Cayley table) and the failing variable assignment.\n"
    "\n"
    "CONSTRAINTS:\n"
    "- Be concise inside <think>. The cheat sheet must fit in 10KB.\n"
    "- Output exactly one VERDICT marker; ``TRUE``/``FALSE`` in uppercase.\n"
)

SAIR_SYSTEM_PROMPT_END: str = "\n\n-----------\n"


# ---------------------------------------------------------------------------
# Training / inference dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SAIR_SFT_CONFIG:
    """SFT configuration for the SAIR distiller, mirroring rlm.config.SFT_CONFIG."""

    model_name: str = MODEL_NAME
    training_file: Path = SAIR_TRAINING_DIR / "sft_dataset.jsonl"

    use_4bit: bool = USE_4_BIT
    max_seq_len: int = MAX_SEQ_LEN

    system_prompt: str = f"{SAIR_SYSTEM_PROMPT}{SAIR_SYSTEM_PROMPT_END}"

    do_sample: bool = False
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float | None = None
    top_p: float | None = None

    epochs: int = 2
    lr: float = 2e-4
    batch_size_questions: int = 2

    loogging_interval: int = 10
    checkpoint_directory: Path = SAIR_SFT_CHECKPOINT_DIR
    checkpoint_interval: int = 200
    keep_last_checkpoints: int = 2


@dataclass(frozen=True)
class SAIR_GRPO_CONFIG:
    """GRPO configuration for SAIR distiller refinement.

    Reward = judge-verified verdict correctness (from ``sair.agents.evaluator``)
    plus format bonuses. See ``sair/train_grpo_distiller.py``.
    """

    model_name: str = MODEL_NAME
    training_file: Path = SAIR_TRAINING_DIR / "grpo_dataset.jsonl"

    use_4bit: bool = USE_4_BIT
    max_seq_len: int = MAX_SEQ_LEN

    system_prompt: str = f"{SAIR_SYSTEM_PROMPT}{SAIR_SYSTEM_PROMPT_END}"

    do_sample: bool = True
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE
    top_p: float | None = TOP_P

    epochs: int = 1
    lr: float = 1e-5
    batch_size_questions: int = 1
    group_size: int = 4
    max_train_examples: int | None = 1_000
    grad_accum_steps: int = 2
    clip_grad_norm: float = 1.0
    beta_kl: float = 0.02

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    logging_interval: int = 10
    checkpoint_directory: Path = SAIR_GRPO_CHECKPOINT_DIR
    checkpoint_interval: int = 100
    keep_last_checkpoints: int = 2


@dataclass(frozen=True)
class SAIR_INFERENCE_CONFIG:
    """Inference configuration used by the distiller agent and the evaluator."""

    model_name: str = MODEL_NAME
    use_4bit: bool = USE_4_BIT
    max_seq_len: int = MAX_SEQ_LEN

    system_prompt: str = f"{SAIR_SYSTEM_PROMPT}{SAIR_SYSTEM_PROMPT_END}"

    do_sample: bool = False
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float | None = None
    top_p: float | None = None

    # Defaults to the GRPO adapter (best), falls back to SFT if missing.
    checkpoint_directory: Path = SAIR_GRPO_CHECKPOINT_DIR
    fallback_checkpoint_directory: Path = SAIR_SFT_CHECKPOINT_DIR
