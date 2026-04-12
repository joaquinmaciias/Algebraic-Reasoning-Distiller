"""RAG retriever agent over the SAIR equation corpus.

This is a thin wrapper around the existing Chroma vector store from
``rag/rag_engine.py``, specialized for the SAIR collections. The index
is populated by ``sair/data/ingest_equations.py`` with two kinds of
documents:

- ``kind="equation"``: a single equation from ``equations.txt`` with its
  canonical id as metadata.
- ``kind="implication"``: a snippet from ``export_raw_implications``
  describing a known implication or non-implication.

The retriever is tolerant of an empty index: if nothing has been ingested
yet, it returns an empty list so the pipeline can still run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sair.config import SAIR_VECTORSTORE_DIR


def _try_load_vectordb() -> Any | None:
    """Best-effort load of the Chroma vectorstore. Returns None on failure."""
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        from langchain_community.vectorstores import Chroma
    except Exception:
        return None

    if not Path(SAIR_VECTORSTORE_DIR).exists():
        return None

    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma(
            persist_directory=str(SAIR_VECTORSTORE_DIR),
            embedding_function=embeddings,
            collection_name="sair_equations",
        )
    except Exception:
        return None


_vectordb_cache: Any | None = None
_vectordb_loaded: bool = False


def retrieve_relevant_snippets(
    *,
    query: str,
    k: int = 5,
    kind_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top-k relevant corpus snippets for a query.

    Args:
        query: Natural language or equation string to search with.
        k: Number of snippets to retrieve.
        kind_filter: Optional metadata filter (``"equation"`` or
            ``"implication"``).

    Returns:
        A list of dicts with ``text`` and ``metadata`` keys. Empty if the
        index is missing or not yet built.
    """
    global _vectordb_cache, _vectordb_loaded
    if not _vectordb_loaded:
        _vectordb_cache = _try_load_vectordb()
        _vectordb_loaded = True

    if _vectordb_cache is None:
        return []

    filt: dict[str, Any] | None = {"kind": kind_filter} if kind_filter else None
    try:
        docs = (
            _vectordb_cache.similarity_search(query, k=k, filter=filt)
            if filt
            else _vectordb_cache.similarity_search(query, k=k)
        )
    except Exception:
        return []

    return [
        {"text": d.page_content, "metadata": dict(d.metadata or {})} for d in docs
    ]
