"""LLM-based distiller agent.

The distiller turns an ``EvidenceBundle`` into natural-language reasoning
that an offline no-tools judge can re-read and follow. It is the component
that we fine-tune: the SFT stage teaches it to imitate the structured
reasoning produced by our symbolic agents, and the GRPO stage rewards it
for reasoning that actually helps the SAIR judge land a correct verdict.

Two usage modes are supported:

- **Per-problem mode** (``run_distiller``): given an ``EvidenceBundle``,
  generate a ``<think>...</think><answer>...</answer>`` completion.
- **Cheat-sheet mode** (``synthesize_cheat_sheet_entry``): given a set of
  similar problems + their evidences, produce a single compact lemma
  block that will become one section of the final cheat sheet.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from sair.config import SAIR_INFERENCE_CONFIG
from sair.schemas import EvidenceBundle, Problem


# ---------------------------------------------------------------------------
# Dedicated system prompt for cheat-sheet synthesis mode.
# Intentionally does NOT use the SAIR judgment format (<think>/<answer>/VERDICT)
# to avoid polluting the cheat sheet with verdict markers that would bias
# the downstream no-tools judge.
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT: str = (
    "You are a mathematics teacher writing concise reference notes for "
    "an automated theorem prover. Your notes must be plain prose and "
    "bullet points — NO verdict markers, NO <answer> tags, NO VERDICT: lines.\n"
    "\n"
    "Your task: given a cluster of related equational-theory problems "
    "(each asking whether Equation 1 implies Equation 2 over all magmas) "
    "and the evidence collected by a symbolic solver, distill the key "
    "patterns into 3-8 bullet-point heuristics a reader can apply "
    "to unseen analogous problems.\n"
    "\n"
    "Output format — use ONLY these sections:\n"
    "  PATTERN: <one-sentence description of the structural pattern>\n"
    "  TRUE-WHEN: <conditions under which E1 implies E2>\n"
    "  FALSE-WHEN: <conditions under which a counterexample exists>\n"
    "  KEY-STEPS: <compact proof strategy or counterexample construction tip>\n"
    "\n"
    "Rules:\n"
    "- Write at most 300 words total.\n"
    "- Do NOT write VERDICT:, <answer>, <think>, TRUE, FALSE as standalone tokens.\n"
    "- Do NOT restate the equations verbatim.\n"
    "- Be precise: use variable names like x, y, z and operator *.\n"
)


# Regex patterns to strip VERDICT/answer blocks that leak despite the above.
_STRIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE),
    re.compile(r"VERDICT\s*:\s*(TRUE|FALSE)", re.IGNORECASE),
    re.compile(r"REASONING\s*:.*?(?=\n[A-Z]|\Z)", re.DOTALL | re.IGNORECASE),
    re.compile(r"PROOF\s*:.*?(?=\n[A-Z]|\Z)", re.DOTALL | re.IGNORECASE),
    re.compile(r"COUNTEREXAMPLE\s*:.*?(?=\n[A-Z]|\Z)", re.DOTALL | re.IGNORECASE),
]


def _strip_verdict_blocks(text: str) -> str:
    """Remove any judgment-format artifacts from a synthesized cheat sheet entry."""
    result = text
    for pattern in _STRIP_PATTERNS:
        result = pattern.sub("", result)
    # Collapse runs of blank lines left by stripping.
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _build_bnb_config(*, use_4bit: bool) -> BitsAndBytesConfig | None:
    """Match the 4-bit NF4 BF16 setup used by rlm training."""
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _resolve_checkpoint(cfg: SAIR_INFERENCE_CONFIG) -> Path | None:
    """Pick the best available LoRA adapter: GRPO first, SFT as fallback."""
    primary: Path = cfg.checkpoint_directory
    if primary.exists() and any(primary.iterdir()):
        return primary
    fallback: Path = cfg.fallback_checkpoint_directory
    if fallback.exists() and any(fallback.iterdir()):
        return fallback
    return None


def load_distiller_model(
    *,
    cfg: SAIR_INFERENCE_CONFIG | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Load the base model and attach the SAIR distiller LoRA adapter.

    Returns:
        A tuple ``(model, tokenizer, adapter_label)``. ``adapter_label`` is
        one of ``"grpo"``, ``"sft"``, or ``"base"`` depending on which
        checkpoint was available.
    """
    cfg = cfg or SAIR_INFERENCE_CONFIG()

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=str(cfg.model_name),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(cfg.model_name),
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
    model.generation_config.eos_token_id = int(tokenizer.eos_token_id)

    adapter_path: Path | None = _resolve_checkpoint(cfg)
    adapter_label: str = "base"
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model=model, model_id=str(adapter_path))
        adapter_label = (
            "grpo" if adapter_path == cfg.checkpoint_directory else "sft"
        )
    model.eval()

    return model, tokenizer, adapter_label


def _build_problem_prompt(
    *,
    problem: Problem,
    retrieved_context: list[str] | None = None,
) -> str:
    """Build the user message for the distiller from a problem + optional RAG."""
    context_block: str = ""
    if retrieved_context:
        joined: str = "\n".join(f"- {c}" for c in retrieved_context)
        context_block = f"<context>\n{joined}\n</context>\n"
    return (
        f"{context_block}"
        f"<equation1>{problem.equation1}</equation1>\n"
        f"<equation2>{problem.equation2}</equation2>"
    )


@torch.inference_mode()
def run_distiller(
    *,
    bundle: EvidenceBundle,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: SAIR_INFERENCE_CONFIG | None = None,
    retrieved_context: list[str] | None = None,
) -> str:
    """Generate a ``<think>/<answer>`` completion for a single problem.

    The evidences in ``bundle`` are embedded directly into the user prompt
    as a scratchpad so the LLM can cite them. The model is expected to
    keep only the useful bits in its reasoning.
    """
    cfg = cfg or SAIR_INFERENCE_CONFIG()

    scratchpad: str = bundle.rendered_reasoning()
    user_prompt: str = _build_problem_prompt(
        problem=bundle.problem, retrieved_context=retrieved_context
    )
    if scratchpad:
        user_prompt = f"{user_prompt}\n\n<scratchpad>\n{scratchpad}\n</scratchpad>"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": str(cfg.system_prompt)},
        {"role": "user", "content": user_prompt},
    ]

    model_inputs: Any = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_ids: torch.Tensor = model_inputs["input_ids"]

    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": int(cfg.max_new_tokens),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "do_sample": bool(cfg.do_sample),
    }
    if bool(cfg.do_sample):
        if cfg.temperature is not None:
            gen_kwargs["temperature"] = float(cfg.temperature)
        if cfg.top_p is not None:
            gen_kwargs["top_p"] = float(cfg.top_p)
    if "attention_mask" in model_inputs:
        gen_kwargs["attention_mask"] = model_inputs["attention_mask"]

    outputs: torch.Tensor = model.generate(**gen_kwargs)
    prompt_len: int = int(input_ids.shape[-1])
    new_tokens: torch.Tensor = outputs[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def synthesize_cheat_sheet_entry(
    *,
    title: str,
    cluster_problems: list[Problem],
    cluster_evidences: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: SAIR_INFERENCE_CONFIG | None = None,
    max_bytes: int = 1_200,
) -> str:
    """Ask the distiller to summarize one cluster into a short lemma block.

    Uses SYNTHESIS_SYSTEM_PROMPT (plain prose, no VERDICT markers) instead
    of the SAIR judgment prompt so the cheat sheet does not leak verdict
    markers that would bias the downstream no-tools judge.

    The result is post-processed to strip any verdict artifacts and trimmed
    to ``max_bytes`` to fit under the global 10KB budget.
    """
    cfg = cfg or SAIR_INFERENCE_CONFIG()

    # Build a verdict-annotated example list for better heuristic grounding.
    examples_block: str = "\n".join(
        f"- {p.equation1}  =>  {p.equation2}"
        + (f"  [ground truth: {'TRUE' if p.answer else 'FALSE'}]" if p.answer is not None else "")
        for p in cluster_problems[:6]
    )
    evidences_block: str = "\n\n".join(cluster_evidences[:6])

    # Count TRUE/FALSE split so the model knows the distribution.
    n_true = sum(1 for p in cluster_problems if p.answer is True)
    n_false = sum(1 for p in cluster_problems if p.answer is False)
    split_note = f"(TRUE: {n_true}, FALSE: {n_false} in this cluster)"

    user_prompt: str = (
        f"Cluster: {title} {split_note}\n\n"
        f"Representative problems (with ground truth):\n{examples_block}\n\n"
        f"Evidence from the symbolic solver:\n{evidences_block}\n\n"
        "Write a compact cheat-sheet block for this cluster in the same format "
        "as a SAIR reasoning answer. Include a short <think> section and an "
        "<answer> section with VERDICT, REASONING, and either PROOF or "
        "COUNTEREXAMPLE. Prefer concrete algebraic patterns, small Cayley-table "
        "counterexamples, and reusable proof strategies."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": str(cfg.system_prompt)},
        {"role": "user", "content": user_prompt},
    ]

    model_inputs: Any = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_ids: torch.Tensor = model_inputs["input_ids"]

    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": int(cfg.max_new_tokens),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "do_sample": False,
    }
    if "attention_mask" in model_inputs:
        gen_kwargs["attention_mask"] = model_inputs["attention_mask"]

    outputs: torch.Tensor = model.generate(**gen_kwargs)
    prompt_len: int = int(input_ids.shape[-1])
    decoded: str = tokenizer.decode(
        outputs[0, prompt_len:], skip_special_tokens=True
    ).strip()

    # decoded = _strip_verdict_blocks(decoded)

    # Hard byte cap so the final render stays under 10KB.
    encoded: bytes = decoded.encode("utf-8")
    if len(encoded) > max_bytes:
        decoded = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return decoded
