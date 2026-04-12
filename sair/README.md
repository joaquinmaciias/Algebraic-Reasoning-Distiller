# SAIR Stage 1 — Distiller Pipeline

End-to-end pipeline for the **SAIR Foundation Mathematics Distillation
Challenge (Stage 1: Equational Theories)**. The deliverable is a single
≤10KB cheat sheet that the offline no-tools judge injects into its
prompt. We train a Qwen2.5-7B-Instruct LoRA distiller on evidence
produced by a multi-agent symbolic pipeline (counterexample search,
symbolic prover, RAG retriever), then use that distiller to synthesize
the final cheat sheet.

## Layout

```
sair/
  agents/
    counterexample.py    # finite magma refutation search
    prover.py            # symbolic rewriting prover
    retriever.py         # Chroma RAG over equations + implications
    distiller.py         # Qwen2.5 + LoRA distiller (load / run / synth)
    evaluator.py         # wraps the official judge_response
  data/
    ingest_equations.py          # builds sair/vectorstore
    load_problems.py             # JSONL problem loader
    build_training_dataset.py    # SFT + GRPO dataset builder
  scripts/
    generate_cheat_sheet.py      # final cheat sheet generation
    evaluate_cheat_sheet.py      # score a cheat sheet vs eval split
  config.py              # paths, model name, system prompt, configs
  equations.py           # equation parser + magma term evaluator
  graph.py               # LangGraph + sequential multi-agent runner
  magma.py               # finite magma enumeration
  schemas.py             # pydantic contracts
  train_sft_distiller.py     # SFT entry point
  train_grpo_distiller.py    # GRPO entry point (judge-verified reward)
```

## Prerequisites (DGX, one-time)

```bash
# From the repo root
uv sync

# Clone the SAIR judge repo as a sibling of this repo if not already
# (path must be ../equational-theories-stage1-judge)
```

## DGX command sequence

Run these from the repo root, in order. Each command is stand-alone
and can be resumed independently.

### 1. Build the RAG index
```bash
python -m sair.data.ingest_equations
```
Outputs: `sair/vectorstore/` (Chroma collection `sair_equations`).

### 2. Build SFT + GRPO datasets
```bash
python -m sair.data.build_training_dataset
# Optional: restrict to a subset for a quick smoke run
python -m sair.data.build_training_dataset --max-problems 200 --no-retriever
```
Outputs:
- `sair/data/training/sft_dataset.jsonl`
- `sair/data/training/grpo_dataset.jsonl`
- `sair/data/training/eval_problems.jsonl`

### 3. Supervised fine-tune the distiller
```bash
python -m sair.train_sft_distiller
```
Outputs: `weights/sair/distiller_sft/` (LoRA adapter).

### 4. GRPO refinement with judge-verified reward
```bash
python -m sair.train_grpo_distiller
```
Reward = `0.70 * judge_correct + 0.15 * format + 0.10 * structure + 0.05 * think`.
Outputs: `weights/sair/distiller_grpo/`.

### 5. Generate the final cheat sheet
```bash
python -m sair.scripts.generate_cheat_sheet --version v1
```
Loads the GRPO adapter (SFT fallback), clusters problems by structural
key, synthesizes one lemma block per cluster, and writes
`sair/cheat_sheets/v1.md` under the 10KB budget.

### 6. Evaluate the cheat sheet
```bash
python -m sair.scripts.evaluate_cheat_sheet sair/cheat_sheets/v1.md
```
Uses the official `judge_response` (from the cloned judge repo) on the
held-out eval split. Writes:
- `sair/artifacts/v1_metrics.json`
- `sair/artifacts/v1_results.jsonl`

### 7. (Optional) Iterate
Inspect `v1_results.jsonl`, tweak prompts or cluster keys in
`sair/scripts/generate_cheat_sheet.py`, then re-run steps 5 and 6 with
`--version v2`.

## FastAPI endpoint

The same pipeline is exposed at `/phase5/sair` via
`sair.graph.run_sair_pipeline`. It's independent from the offline
training scripts — useful only for interactive debugging.

## Key design notes

- **The judge runs offline with no tools**, so the multi-agent pipeline
  only runs at dev time. The final deliverable is the cheat sheet, not
  the agents.
- **Reward alignment**: GRPO uses the *official* `judge_response`
  extractor (loaded from `../equational-theories-stage1-judge/judge.py`),
  so the reward signal is byte-identical to the grader.
- **Graceful degradation**: if the retriever index is missing, agents
  return empty snippets; if the GRPO adapter is missing, the distiller
  falls back to the SFT adapter; if both are missing, it uses the base
  model.
