"""SAIR multi-agent package.

Each submodule defines one role in the offline distillation pipeline:

- ``planner``: decides which strategies to apply for a given problem.
- ``retriever``: pulls relevant equations / implications from the RAG index.
- ``prover``: attempts forward symbolic proofs (lightweight rewriter).
- ``counterexample``: searches finite magmas for refutations.
- ``distiller``: LLM that produces natural-language reasoning and cheat
  sheet fragments from the collected evidence.
- ``evaluator``: simulates the offline no-tools SAIR judge against a
  candidate cheat sheet.
"""
