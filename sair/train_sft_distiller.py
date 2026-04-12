"""SFT training for the SAIR distiller.

Mirrors the pattern in ``rlm/train_sft.py`` but reads the JSONL dataset
produced by ``sair/data/build_training_dataset.py`` and formats it with
the SAIR system prompt. LoRA adapters are saved to
``weights/sair/distiller_sft``.

Usage
-----
    python -m sair.train_sft_distiller
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from utils.paths import check_cwd

from sair.config import REPO_DIR
from sair.config import SAIR_SFT_CONFIG as CONFIG


def formatting_prompts_func(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    cfg: CONFIG,
) -> str:
    """Format one SFT record into a chat-template supervised string.

    Input record shape (produced by sair.data.build_training_dataset):
        {"prompt": "<equation1>...", "completion": "<think>...</answer>", ...}
    """
    user_prompt: str = str(example["prompt"])
    assistant_text: str = str(example["completion"]).strip()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]

    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _build_bnb_config(*, use_4bit: bool) -> BitsAndBytesConfig | None:
    """NF4 4-bit + BF16 compute, matching the rlm training defaults."""
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _load_jsonl_dataset(path: Path) -> Dataset:
    """Load a JSONL file produced by build_training_dataset into a HF Dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"SFT dataset not found at {path}. "
            "Run `python -m sair.data.build_training_dataset` first."
        )
    return load_dataset("json", data_files=str(path), split="train")


def train() -> None:
    """Run LoRA SFT on the SAIR distillation dataset."""
    cfg: CONFIG = CONFIG()

    # Force BF16 mixed precision to avoid GradScaler issues with 4-bit models.
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

    tokenizer: Any = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    model: Any = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    peft_config: LoraConfig = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

    dataset: Dataset = _load_jsonl_dataset(cfg.training_file)
    print(f"[sft] loaded {len(dataset)} training examples from {cfg.training_file}")

    sft_args: SFTConfig = SFTConfig(
        output_dir=str(cfg.checkpoint_directory),
        num_train_epochs=int(cfg.epochs),
        per_device_train_batch_size=int(cfg.batch_size_questions),
        gradient_accumulation_steps=2,
        learning_rate=float(cfg.lr),
        fp16=False,
        bf16=True,
        logging_steps=cfg.loogging_interval,
        save_steps=cfg.checkpoint_interval,
        save_total_limit=cfg.keep_last_checkpoints,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=1.0,
        max_length=int(cfg.max_seq_len),
        packing=False,
    )

    formatting_func: Any = partial(
        formatting_prompts_func, tokenizer=tokenizer, cfg=cfg
    )

    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_args,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(output_dir=str(cfg.checkpoint_directory))
    tokenizer.save_pretrained(save_directory=str(cfg.checkpoint_directory))
    print(f"[sft] saved to {cfg.checkpoint_directory}")


if __name__ == "__main__":
    check_cwd(expected_dir=REPO_DIR)
    train()
