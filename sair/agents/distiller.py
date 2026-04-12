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

    input_ids: torch.Tensor = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

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

    The result is trimmed to ``max_bytes`` so it composes well with other
    entries under the global 10KB cheat sheet budget.
    """
    cfg = cfg or SAIR_INFERENCE_CONFIG()

    examples_block: str = "\n".join(
        f"- {p.equation1}  =>  {p.equation2}" for p in cluster_problems[:6]
    )
    evidences_block: str = "\n\n".join(cluster_evidences[:6])

    user_prompt: str = (
        f"Cluster title: {title}\n\n"
        f"Representative problems:\n{examples_block}\n\n"
        f"Evidence collected by the offline solver:\n{evidences_block}\n\n"
        "Write a compact ``lemma / heuristic`` block that an offline "
        "no-tools judge could read in ~400 tokens to quickly decide "
        "analogous problems. Use bullet points. Do not restate the "
        "problems verbatim."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": str(cfg.system_prompt)},
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
    decoded: str = tokenizer.decode(
        outputs[0, prompt_len:], skip_special_tokens=True
    ).strip()

    # Hard byte cap so the final render stays under 10KB.
    encoded: bytes = decoded.encode("utf-8")
    if len(encoded) > max_bytes:
        decoded = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return decoded
