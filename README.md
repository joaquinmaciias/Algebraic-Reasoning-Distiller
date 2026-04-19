# Algebraic Reasoning Distiller

> **SAIR Foundation — Mathematics Distillation Challenge, Stage 1: Equational Theories**  
> Final project for *Modelos Generativos Profundos* (MGP), Master in Artificial Intelligence, UPM 2025/2026.

A multi-agent system that decides whether one equational law implies another over all **magmas** — sets with a single binary operation `*` and no axioms assumed.

The system combines symbolic reasoning (term rewriting + finite magma search) with a fine-tuned LLM distiller trained via SFT → GRPO. All heavy computation happens offline; at evaluation time only a ≤ 10 KB static cheat sheet and a single LLM call are used.

---

## The problem

Given two equations expressed with `*`:

```
E1: x * y = y * x
E2: (x * y) * z = z * (x * y)
```

Decide whether **E1 ⟹ E2 holds in every magma**.

- **TRUE** — every magma satisfying E1 also satisfies E2.
- **FALSE** — there exists a finite magma where E1 holds but E2 does not.

Competition constraints: no tools during evaluation, ≤ 10 min, ≤ $0.01 per problem. Only resource allowed: a ≤ 10 KB static cheat sheet injected as a system-prompt prefix.

---

## Architecture

```
                        ┌────────────────────────────────┐
        E1, E2 ──────►  │        Planner (parser)         │
                        └───────────────┬────────────────┘
                                        │ AST
             ┌──────────────────────────┼──────────────────────┐
             ▼                          ▼                       ▼
   ┌──────────────────┐      ┌──────────────────┐   ┌──────────────────┐
   │  Counterexample  │      │ Symbolic Prover  │   │  RAG Retriever   │
   │  (magma search)  │      │ (term rewriting) │   │ (Chroma + SBERT) │
   └────────┬─────────┘      └────────┬─────────┘   └────────┬─────────┘
            │                         │                       │
            └─────────────────────────┼───────────────────────┘
                                      ▼
                             ┌────────────────┐
                             │   Aggregator   │  counterex > proof > retrieved
                             └───────┬────────┘
                                     │ EvidenceBundle
                                     ▼
                             ┌────────────────┐
                             │  LLM Distiller │  Qwen2.5-7B + LoRA (SFT → GRPO)
                             └───────┬────────┘
                                     │
                          ┌──────────┴──────────┐
                          ▼                      ▼
                  Per-problem verdict      Cheat sheet (≤ 10 KB)
                  <think>…</think>         written offline, injected at test time
                  <answer>VERDICT:…</answer>
```

### Agents

| Agent | What it does |
|---|---|
| **Planner** | Parses equations into ASTs using a recursive-descent grammar. Produces nested tuples `("mul", left, right)` consumed by all downstream agents. |
| **Counterexample** | Exhaustive enumeration over order-2/3 magmas (16 and 19,683 respectively); random sampling of 20,000 tables for order 4. Hard-FALSE when a witness is found. |
| **Symbolic Prover** | Three strategies: alpha-equivalence, substitution instance, one-step term rewriting. Hard-TRUE (confidence 0.9). Conservative — negative means nothing. |
| **RAG Retriever** | ChromaDB vector store of 4,694 canonical equations embedded with `all-MiniLM-L6-v2`. Returns top-5 similar equations as context for the distiller. Degrades gracefully if index is absent. |
| **Aggregator** | Priority policy: counterexample (conf 1.0) > symbolic proof (conf 0.9) > unknown (delegates to LLM). |
| **LLM Distiller** | `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4 + BF16 compute. LoRA adapters (r=16, α=32) on all attention and MLP projections. Outputs structured `<think>…</think><answer>VERDICT: TRUE|FALSE…</answer>`. |

---

## Dataset

The training corpus was assembled from three sources (~1,761 labelled problems total):

| Source | Problems | Notes |
|---|---|---|
| SAIR judge repo | 20 | Canonical labelled examples (`problems_hard3_20.jsonl`) |
| Playground hard3 | 72 | Collected from SAIR online playground (40 TRUE / 32 FALSE) |
| Competition dumps | ~1,669 | Parsed from `hard.txt`, `hard2.txt`, `hard3.txt`, `normales.txt` |

The competition files use the format `#N: E1 → E2 True|False`. The parser (`sair/scripts/parse_competition_problems.py`) normalises garbled UTF-8 arrow characters and deduplicates by problem ID.

Split: **80% train / 20% eval** (deterministic, seed 0).

---

## Training pipeline

### Stage 1 — Supervised Fine-Tuning (SFT)

The offline pipeline runs on every training problem and produces an `EvidenceBundle`. Each bundle is templated into a supervised completion:

```
<think>{evidence_reasoning}</think>
<answer>
VERDICT: {TRUE|FALSE}
REASONING: {short_justification}
{PROOF:|COUNTEREXAMPLE:} {details}
</answer>
```

Problems where the pipeline verdict contradicts the ground-truth label are excluded (~1,330 examples used).

**Hyperparameters:** 2 epochs, lr=2e-4, batch 2, grad accum 2, cosine scheduler, BF16 mixed precision.

### Stage 2 — GRPO (Group Relative Policy Optimisation)

GRPO treats the SAIR judge as a verification oracle — no human preference data needed. This is **Reinforcement Learning from Verifier Feedback (RLVF)**.

**Reward function:**
```
R = 0.70 × [verdict correct]
  + 0.05 × [<think> present]
  + 0.10 × [<answer> present]
  + 0.10 × [REASONING + PROOF/COUNTEREXAMPLE present]
  + 0.05 × [fallback for unparseable response]
```

The policy generates G=2 completions per problem; advantages are normalised within the group; KL penalty (β=0.02) prevents divergence from the SFT reference.

**Hyperparameters:** 1 epoch, lr=1e-5, group size 2, max 512 new tokens, gradient clipping 1.0.

### Stage 3 — Cheat sheet distillation

Problems are clustered by structural signature `vars{n}_m{d1}_n{d2}`. Key design decisions to avoid verdict bias:

- **Balanced TRUE/FALSE selection** — equal number of TRUE-majority and FALSE-majority clusters selected (v1 collapsed accuracy to 20% due to FALSE-dominated clusters).
- **Dedicated synthesis prompt** — the distiller is instructed to act as a teacher, not a judge; forbidden to emit `VERDICT:`, `<think>`, `<answer>` tokens.
- **Post-processing** — regex pass strips any residual verdict artifacts before the byte cap.
- **Preamble entry** — a fixed `HOW TO USE THIS CHEAT SHEET` entry (priority 999) explains to the inference-time model how to interpret the lemma packs.

Each cluster generates up to 1,200 bytes; entries are packed greedily to fill the 10,000-byte budget.

---

## Repository layout

```
Algebraic-Reasoning-Distiller/
├── api/
│   └── app.py                        # FastAPI: POST /sair, GET /health
├── container/                        # DGX Docker container (Ubuntu + uv)
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   ├── variables.sh                  # ← set CONTAINER_USER before first run
│   ├── up.sh / down.sh / attach.sh
│   └── utils/
├── sair/
│   ├── agents/
│   │   ├── counterexample.py         # finite magma refutation
│   │   ├── prover.py                 # alpha-equiv, substitution, rewrite
│   │   ├── retriever.py              # ChromaDB RAG lookup
│   │   ├── distiller.py              # Qwen2.5 LoRA inference + cheat-sheet synthesis
│   │   └── evaluator.py              # wraps the official SAIR judge
│   ├── data/
│   │   ├── competition/              # raw .txt competition problem dumps
│   │   ├── problems/                 # parsed JSONL problem sets
│   │   ├── training/                 # SFT / GRPO / eval splits (generated)
│   │   ├── ingest_equations.py       # builds the ChromaDB vector store
│   │   ├── load_problems.py          # deduplicating JSONL loader
│   │   └── build_training_dataset.py # pipeline → SFT + GRPO JSONL
│   ├── scripts/
│   │   ├── parse_competition_problems.py   # arrow-normalising .txt parser
│   │   ├── generate_cheat_sheet.py         # balanced clustering + synthesis
│   │   ├── evaluate_cheat_sheet.py         # score a cheat sheet on eval split
│   │   └── demo_agents.py                  # console demo of each agent
│   ├── config.py                     # paths, hyperparameters, system prompts
│   ├── equations.py                  # equation parser + magma evaluator
│   ├── graph.py                      # LangGraph wiring + sequential runner
│   ├── magma.py                      # finite magma enumeration
│   ├── schemas.py                    # Pydantic data contracts
│   ├── train_sft_distiller.py        # SFT training script
│   └── train_grpo_distiller.py       # GRPO training script
├── utils/
│   └── paths.py
├── weights/                          # LoRA checkpoints (gitignored)
├── pyproject.toml
└── .python-version                   # 3.12
```

---

## Quick start

### Prerequisites

- DGX cluster access (or any machine with an NVIDIA GPU, ≥ 24 GB VRAM recommended)
- Python 3.12, [`uv`](https://github.com/astral-sh/uv)
- Git + SSH key added to GitHub

### 1. Boot the container (DGX)

```bash
cd ~/dgx-uv-container
# Edit variables.sh: set CONTAINER_USER to a non-numeric username (e.g. "joaquin")
nano variables.sh

bash up.sh       # build + start (first run ~5 min)
bash attach.sh   # open shell inside the container
```

### 2. Clone and set up (inside container)

```bash
cd ~/data
git clone https://github.com/<your-user>/Algebraic-Reasoning-Distiller.git
cd Algebraic-Reasoning-Distiller

# Install dependencies (PyTorch cu124 required for driver >= 12.4)
uv sync
```

> **Note:** the `pyproject.toml` already pins torch to the `cu124` index via `[tool.uv.sources]`. If you see a CUDA version mismatch, verify with `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`.

### 3. Clone the SAIR judge (sibling directory)

```bash
cd ~/data
git clone https://github.com/sair-foundation/equational-theories-stage1-judge.git
```

Required for the GRPO reward function (`sair/agents/evaluator.py` loads it via `importlib`).

### 4. Run the agents demo (no GPU needed)

```bash
cd ~/data/Algebraic-Reasoning-Distiller
python -m sair.scripts.demo_agents
```

Shows each agent's contribution on a set of hand-picked problems — parser ASTs, RAG snippets, prover strategies, counterexample tables, and aggregator verdicts. No training or LLM required.

---

## Full training pipeline

Run from the repo root. Use `tmux` to keep jobs alive after SSH disconnect.

```bash
# Step 0 — parse competition .txt files into JSONL
python -m sair.scripts.parse_competition_problems

# Step 1 — build the ChromaDB RAG index (~1 min)
python -m sair.data.ingest_equations

# Step 2 — generate SFT + GRPO datasets (~10–30 min)
python -m sair.data.build_training_dataset

# Step 3 — supervised fine-tuning (~25–35 min on H100 with full corpus)
python -m sair.train_sft_distiller

# Step 4 — GRPO refinement (~1–2 s/step on H100)
python -m sair.train_grpo_distiller

# Step 5 — generate the cheat sheet (balanced clusters, synthesis prompt)
python -m sair.scripts.generate_cheat_sheet --version v2

# Step 6 — evaluate against the held-out split
python -m sair.scripts.evaluate_cheat_sheet sair/cheat_sheets/v2.md
```

---

## REST API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8182
```

**`POST /sair`**

```json
{
  "equation1": "x * y = y * x",
  "equation2": "x * x = x"
}
```

```json
{
  "verdict": false,
  "response": "VERDICT: FALSE\nREASONING: ...\nCOUNTEREXAMPLE: ...",
  "trace": [{"step": 0, "agent": "counterexample", "content": "..."}],
  "details": {"consensus_confidence": 1.0, "n_evidences": 1}
}
```

**`GET /health`** — liveness probe.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| **Distill-then-deploy** | Competition forbids tools at eval time. All heavy compute (magma search, RAG, multi-agent reasoning) is done offline; only the cheat sheet + one LLM call are used at test time. |
| **Qwen2.5-7B over larger models** | Fits in 4-bit quantisation on a single GPU with room to train. The $0.01/problem budget rules out frontier API calls. |
| **LoRA over full fine-tuning** | ~42M trainable parameters vs 7.6B. Enables training with GRPO (which requires two model copies) on a single GPU. |
| **SFT then GRPO** | SFT teaches format compliance (prerequisite for the judge to parse responses). GRPO then optimises verdict correctness using the judge as a binary reward signal. |
| **Balanced cheat sheet clustering** | v1 used top-k-by-size clusters; FALSE-dominated clusters leaked verdict tokens into synthesis and caused 20% accuracy on the playground. v2 enforces equal TRUE/FALSE cluster counts. |
| **Dedicated synthesis prompt** | Prevents the distiller from acting as a judge during cheat sheet generation. Bans `VERDICT:`, `<think>`, `<answer>` tokens in synthesis mode. |
| **Exhaustive order-2/3 + sampled order-4** | All FALSE cases in the dataset are refutable at order ≤ 3. Order-4 exhaustive enumeration ($4^{16} \approx 4.3 \times 10^9$ magmas) is infeasible, so 20,000 random samples are used as a best-effort pass. |
| **Graceful degradation** | Missing RAG index → empty retrieval. Missing GRPO weights → SFT fallback. Missing SFT → base model. Pipeline never crashes on missing components. |

---

## Evaluation results (preliminary)

| Configuration | Accuracy | Parseability |
|---|---|---|
| Base model (no fine-tuning) | 0.50 | 0.75 |
| SFT only (20-problem corpus) | 0.75 | 1.00 |
| SFT + GRPO (20-problem corpus) | 0.75 | 1.00 |
| SFT + GRPO + cheat sheet v1 | 0.20 | 1.00 |
| SFT + GRPO + cheat sheet v2 | *in progress* | *in progress* |
| Symbolic agents only | 0.50 | — |

v1 cheat sheet regression was caused by FALSE-dominated cluster selection (see design decisions above). v2 retraining on the full ~1,761-problem corpus is ongoing.

---

## Infrastructure notes (DGX)

- **PyTorch version:** pin to `cu124` build. The default wheels resolved by `uv` target CUDA 13.0 which is incompatible with driver 12.8 on the cluster.
- **Single-GPU training:** export `CUDA_VISIBLE_DEVICES=0` before training. `device_map="auto"` with 4-bit quantisation distributes parameters across all visible GPUs, which conflicts with gradient synchronisation.
- **Container username:** `variables.sh` `CONTAINER_USER` must be a non-numeric string (Linux `useradd` rejects names starting with a digit).

---

## License

MIT
