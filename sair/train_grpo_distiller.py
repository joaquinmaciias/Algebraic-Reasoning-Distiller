"""GRPO training for the SAIR distiller.

Adapts ``rlm/train_grpo.py`` with a SAIR-specific reward function: the
reward is driven by the official SAIR judge verdict extractor
(``judge_response``) applied to the model's generation, plus a small
format shaping bonus. This is the "RLVF" stage of the project — the
verification feedback is the judge, so there are no human labels beyond
the dataset ground-truth booleans.

Usage
-----
    python -m sair.train_grpo_distiller
    python -m sair.train_grpo_distiller weights/sair/distiller_sft
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.paths import check_cwd

from sair.agents.evaluator import judge_response
from sair.config import REPO_DIR, SAIR_SFT_CHECKPOINT_DIR
from sair.config import SAIR_GRPO_CONFIG as CONFIG


# ---------------------------------------------------------------------------
# Reward function — verdict correctness + format shaping
# ---------------------------------------------------------------------------


_THINK_BLOCK_RE = re.compile(r"<think>\s*.*?\s*</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*.*?\s*</answer>", re.DOTALL | re.IGNORECASE)


def _has_think(text: str) -> bool:
    return _THINK_BLOCK_RE.search(text) is not None


def _has_answer(text: str) -> bool:
    return _ANSWER_BLOCK_RE.search(text) is not None


def _has_structured_sections(text: str) -> bool:
    """True if the answer contains the REASONING and PROOF/COUNTEREXAMPLE headers."""
    has_reasoning: bool = bool(re.search(r"(?i)REASONING\s*:", text))
    has_proof_or_counter: bool = bool(
        re.search(r"(?i)(PROOF|COUNTEREXAMPLE)\s*:", text)
    )
    return has_reasoning and has_proof_or_counter


def sair_reward_function(*, generated_text: str, ground_truth_answer: bool) -> float:
    """Reward = 0.70 correctness + 0.15 format + 0.10 structure + 0.05 think.

    The dominant term is verdict correctness as graded by the SAIR judge's
    own verdict extractor, so optimizing this reward is aligned with the
    official evaluation setup.
    """
    reward: float = 0.0

    correct: bool | None = judge_response(generated_text, ground_truth_answer)[0]
    if correct is True:
        reward += 0.70
    elif correct is False:
        reward += 0.0  # wrong verdict, no correctness credit
    else:
        reward += 0.05  # unparseable — tiny credit so the loss isn't flat

    if _has_think(generated_text):
        reward += 0.05
    if _has_answer(generated_text):
        reward += 0.10
    if _has_structured_sections(generated_text):
        reward += 0.10

    return float(max(0.0, min(1.0, reward)))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_grpo_dataset(path: Path) -> Dataset:
    """Load the GRPO JSONL produced by build_training_dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"GRPO dataset not found at {path}. "
            "Run `python -m sair.data.build_training_dataset` first."
        )
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Small utilities copied in spirit from rlm/train_grpo.py
# ---------------------------------------------------------------------------


def _build_bnb_config(*, use_4bit: bool) -> BitsAndBytesConfig | None:
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _maybe_wrap_with_lora(
    *, base_model: torch.nn.Module, cfg: CONFIG
) -> torch.nn.Module:
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg: LoraConfig = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(base_model, lora_cfg)


def _ensure_only_trainable_params(*, model: torch.nn.Module) -> None:
    if not isinstance(model, PeftModel):
        return
    for name, param in model.named_parameters():
        param.requires_grad = bool("lora_" in name)


def _load_base_and_policy(
    *, cfg: CONFIG, adapter_path: Path | None
) -> tuple[torch.nn.Module, torch.nn.Module, Any]:
    tokenizer: Any = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    base_model: Any = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.config.pad_token_id = int(tokenizer.pad_token_id)
    base_model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
    base_model.generation_config.eos_token_id = int(tokenizer.eos_token_id)

    if bool(cfg.use_4bit):
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=False
        )
        base_model.config.use_cache = False

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    if adapter_path is None:
        policy_model: torch.nn.Module = _maybe_wrap_with_lora(
            base_model=base_model, cfg=cfg
        )
    else:
        policy_model = PeftModel.from_pretrained(
            model=base_model, model_id=str(adapter_path), is_trainable=True
        )

    _ensure_only_trainable_params(model=policy_model)
    return base_model, policy_model, tokenizer


def _build_prompt_text(*, tokenizer: Any, cfg: CONFIG, user_prompt: str) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return f"{prompt}<think>"


def _safe_std(*, x: torch.Tensor) -> torch.Tensor:
    if int(x.numel()) <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return x.std(unbiased=False)


def _gather_generated_logp_stats(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum log-probs and token counts for the generated portion of each sequence."""
    outputs: Any = model(input_ids=input_ids, attention_mask=attention_mask)
    logits: torch.Tensor = outputs.logits

    log_probs_all: torch.Tensor = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_tokens: torch.Tensor = input_ids[:, 1:]
    attn: torch.Tensor = attention_mask[:, 1:]

    start_tgt: int = max(int(prompt_len) - 1, 0)
    bsz: int = int(input_ids.shape[0])
    sums: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []

    for b in range(bsz):
        gen_mask: torch.Tensor = torch.zeros_like(attn[b], dtype=torch.bool)
        if start_tgt < int(gen_mask.numel()):
            gen_mask[start_tgt:] = True
        gen_mask = gen_mask & attn[b].to(dtype=torch.bool)

        picked: torch.Tensor = (
            log_probs_all[b]
            .gather(dim=-1, index=target_tokens[b].unsqueeze(-1))
            .squeeze(-1)
        )
        sums.append(picked[gen_mask].sum())
        counts.append(torch.tensor(int(gen_mask.sum().item()), device=input_ids.device))

    return torch.stack(sums, dim=0), torch.stack(counts, dim=0)


def _decode_generated_only(
    *, tokenizer: Any, sequences: torch.Tensor, prompt_len: int
) -> list[str]:
    bsz: int = int(sequences.shape[0])
    cont_ids: list[torch.Tensor] = []
    for b in range(bsz):
        seq: torch.Tensor = sequences[b]
        cont_ids.append(seq[int(prompt_len) :] if int(prompt_len) < int(seq.numel()) else seq[:0])
    return tokenizer.batch_decode(cont_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main GRPO loop
# ---------------------------------------------------------------------------


def main(*, adapter_path: Path | None = None) -> None:
    """Run GRPO with the SAIR judge reward on top of an SFT adapter."""
    cfg: CONFIG = CONFIG()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ref_model, policy_model, tokenizer = _load_base_and_policy(
        cfg=cfg, adapter_path=adapter_path
    )
    policy_model.train()

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        params=[p for p in policy_model.parameters() if bool(p.requires_grad)],
        lr=float(cfg.lr),
    )

    ds: Dataset = _load_grpo_dataset(cfg.training_file)
    if cfg.max_train_examples is not None and len(ds) > int(cfg.max_train_examples):
        ds = ds.select(range(int(cfg.max_train_examples)))
    print(f"[grpo] training on {len(ds)} problems")

    loader: DataLoader[dict[str, list[Any]]] = DataLoader(
        ds,
        batch_size=int(cfg.batch_size_questions),
        shuffle=True,
        drop_last=False,
        collate_fn=lambda xs: {
            "prompt": [str(x["prompt"]) for x in xs],
            "answer": [bool(x["answer"]) for x in xs],
        },
    )

    out_dir: Path = Path(cfg.checkpoint_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    device: torch.device = next(policy_model.parameters()).device
    step: int = 0
    optimizer.zero_grad(set_to_none=True)

    group_size: int = int(cfg.group_size)
    max_new_tokens: int = int(cfg.max_new_tokens)
    temperature: float = float(cfg.temperature)
    top_p: float | None = cfg.top_p
    beta_kl: float = float(cfg.beta_kl)

    for epoch in range(int(cfg.epochs)):
        pbar: tqdm[dict[str, list[Any]]] = tqdm(
            loader, desc=f"epoch {epoch + 1}/{int(cfg.epochs)}", dynamic_ncols=True
        )

        for batch in pbar:
            batch_prompts: list[str] = batch["prompt"]
            batch_answers: list[bool] = batch["answer"]

            total_loss: torch.Tensor = torch.tensor(0.0, device=device, dtype=torch.float32)
            batch_rewards_all: list[float] = []

            for user_prompt, gt in zip(batch_prompts, batch_answers, strict=True):
                prompt_text: str = _build_prompt_text(
                    tokenizer=tokenizer, cfg=cfg, user_prompt=user_prompt
                )
                prompt_inputs: Any = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=int(cfg.max_seq_len),
                )
                prompt_input_ids: torch.Tensor = prompt_inputs["input_ids"].to(device)
                prompt_attention_mask: torch.Tensor = prompt_inputs["attention_mask"].to(device)
                prompt_len: int = int(prompt_input_ids.shape[1])

                gen_kwargs: dict[str, Any] = {
                    "input_ids": prompt_input_ids,
                    "attention_mask": prompt_attention_mask,
                    "do_sample": True,
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "num_return_sequences": int(group_size),
                    "pad_token_id": int(tokenizer.pad_token_id),
                    "eos_token_id": int(tokenizer.eos_token_id),
                }
                if top_p is not None:
                    gen_kwargs["top_p"] = float(top_p)

                policy_model.eval()
                with torch.no_grad():
                    gen_ids: torch.Tensor = policy_model.generate(**gen_kwargs)
                policy_model.train()

                # Pad the group sequences to a common length
                max_len: int = int(max(int(s.shape[0]) for s in gen_ids))
                padded_ids: list[torch.Tensor] = []
                padded_masks: list[torch.Tensor] = []
                for s in gen_ids:
                    pad_len: int = int(max_len - int(s.shape[0]))
                    if pad_len > 0:
                        pad_ids: torch.Tensor = torch.full(
                            (pad_len,),
                            fill_value=int(tokenizer.pad_token_id),
                            device=s.device,
                            dtype=s.dtype,
                        )
                        s2: torch.Tensor = torch.cat([s, pad_ids], dim=0)
                        m2: torch.Tensor = torch.cat(
                            [
                                torch.ones_like(s, dtype=torch.long),
                                torch.zeros_like(pad_ids, dtype=torch.long),
                            ],
                            dim=0,
                        )
                    else:
                        s2 = s
                        m2 = torch.ones_like(s, dtype=torch.long)
                    padded_ids.append(s2.unsqueeze(0))
                    padded_masks.append(m2.unsqueeze(0))

                group_input_ids: torch.Tensor = torch.cat(padded_ids, dim=0)
                group_attention_mask: torch.Tensor = torch.cat(padded_masks, dim=0)

                decoded_cont: list[str] = _decode_generated_only(
                    tokenizer=tokenizer, sequences=group_input_ids, prompt_len=prompt_len
                )
                rewards_list: list[float] = [
                    sair_reward_function(generated_text=t, ground_truth_answer=gt)
                    for t in decoded_cont
                ]
                batch_rewards_all.extend(rewards_list)

                rewards_tensor: torch.Tensor = torch.tensor(
                    rewards_list, device=device, dtype=torch.float32
                )
                mean_reward: torch.Tensor = rewards_tensor.mean()
                std_reward: torch.Tensor = _safe_std(x=rewards_tensor)
                advantages: torch.Tensor = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
                adv_detached: torch.Tensor = advantages.detach()

                sum_logp_pi, n_gen = _gather_generated_logp_stats(
                    model=policy_model,
                    input_ids=group_input_ids,
                    attention_mask=group_attention_mask,
                    prompt_len=prompt_len,
                )
                with torch.no_grad():
                    sum_logp_ref, _ = _gather_generated_logp_stats(
                        model=ref_model,
                        input_ids=group_input_ids,
                        attention_mask=group_attention_mask,
                        prompt_len=prompt_len,
                    )

                n_gen_safe: torch.Tensor = torch.clamp(
                    n_gen.to(dtype=torch.float32), min=1.0
                )
                mean_logp_pi: torch.Tensor = sum_logp_pi.to(dtype=torch.float32) / n_gen_safe
                mean_logp_ref: torch.Tensor = sum_logp_ref.to(dtype=torch.float32) / n_gen_safe

                approx_kl: torch.Tensor = mean_logp_pi - mean_logp_ref
                loss_q: torch.Tensor = -(
                    adv_detached * mean_logp_pi
                ).mean() + beta_kl * (approx_kl.mean())
                total_loss = total_loss + loss_q

            total_loss = total_loss / float(max(len(batch_prompts), 1))
            total_loss = total_loss / float(cfg.grad_accum_steps)
            total_loss.backward()

            step += 1
            if step % int(cfg.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=[
                        p for p in policy_model.parameters() if bool(p.requires_grad)
                    ],
                    max_norm=float(cfg.clip_grad_norm),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            avg_r: float = float(sum(batch_rewards_all) / max(len(batch_rewards_all), 1))
            pbar.set_postfix({
                "loss": f"{float(total_loss.detach().cpu().item()):.4f}",
                "r": f"{avg_r:.3f}",
            })

            if cfg.checkpoint_interval > 0 and step % int(cfg.checkpoint_interval) == 0:
                ckpt: Path = out_dir / f"checkpoint-step-{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                policy_model.save_pretrained(save_directory=str(ckpt))
                tokenizer.save_pretrained(save_directory=str(ckpt))

    if not isinstance(policy_model, PeftModel):
        raise RuntimeError("Training finished but policy_model is not a PEFT model.")
    policy_model.save_pretrained(save_directory=str(out_dir))
    tokenizer.save_pretrained(save_directory=str(out_dir))
    print(f"[grpo] saved to {out_dir}")


def _parse_optional_checkpoint_arg(*, argv: list[str]) -> Path | None:
    if len(argv) < 2:
        return None
    raw: str = argv[1].strip()
    return Path(raw) if raw else None


if __name__ == "__main__":
    check_cwd(expected_dir=REPO_DIR)
    path: Path | None = _parse_optional_checkpoint_arg(argv=sys.argv)
    if path is None and SAIR_SFT_CHECKPOINT_DIR.exists():
        path = SAIR_SFT_CHECKPOINT_DIR
    main(adapter_path=path)
