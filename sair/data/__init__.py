"""Data pipeline for the SAIR multi-agent system.

Submodules:
- ``load_problems``: read problems from JSONL files.
- ``ingest_equations``: build the Chroma RAG index from equations.txt /
  export_raw_implications.
- ``build_training_dataset``: run the offline solver over labeled
  problems and produce SFT / GRPO training files.
"""
