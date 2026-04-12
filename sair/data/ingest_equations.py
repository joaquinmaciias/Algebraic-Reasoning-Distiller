"""Ingest equations / implications into a Chroma vector store.

Builds ``sair/vectorstore`` so the RAG retriever can find relevant
equations and known implications for any problem. The two input sources
are optional:

- ``equations.txt``: numbered list of canonical equations (one per line
  or ``equation N := ...`` Lean4 style).
- ``export_raw_implications``: raw dump of known implications (one per
  line, free-form).

If neither file exists, the script prints a warning and creates an empty
index so the retriever still degrades gracefully.

Usage
-----
    python -m sair.data.ingest_equations
"""

from __future__ import annotations

import re
from pathlib import Path

from sair.config import (
    SAIR_DATA_DIR,
    SAIR_JUDGE_EQUATIONS_FILE,
    SAIR_JUDGE_REPO,
    SAIR_VECTORSTORE_DIR,
)


_EQUATION_LINE_RE = re.compile(
    r"^\s*(?:equation\s+)?(\d+)\s*(?::=|:|\.|=>)?\s*(.*?)\s*$", re.IGNORECASE
)


def _read_equations_file() -> list[tuple[str, str]]:
    """Read the canonical equations list from any known location.

    Returns:
        List of ``(id, text)`` tuples. Empty if no file was found.
    """
    candidates: list[Path] = [
        SAIR_JUDGE_EQUATIONS_FILE,
        SAIR_DATA_DIR / "equations.txt",
        SAIR_JUDGE_REPO / "data" / "equations.txt",
    ]
    for path in candidates:
        if path.exists():
            print(f"[ingest] reading equations from {path}")
            return _parse_equations(path)
    print("[ingest] WARNING: no equations.txt found; retriever will be empty.")
    return []


def _parse_equations(path: Path) -> list[tuple[str, str]]:
    """Parse an equations.txt file into (id, text) tuples."""
    out: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line: str = raw.strip()
            if not line or line.startswith("#"):
                continue
            m: re.Match[str] | None = _EQUATION_LINE_RE.match(line)
            if m is not None:
                out.append((m.group(1), m.group(2)))
            else:
                out.append((str(len(out) + 1), line))
    print(f"[ingest] parsed {len(out)} equations")
    return out


def _read_implications_file() -> list[str]:
    """Read the known-implications dump, if available."""
    candidates: list[Path] = [
        SAIR_JUDGE_REPO / "export_raw_implications",
        SAIR_JUDGE_REPO / "data" / "export_raw_implications",
        SAIR_DATA_DIR / "export_raw_implications",
    ]
    for path in candidates:
        if path.exists():
            print(f"[ingest] reading implications from {path}")
            with path.open(encoding="utf-8") as fh:
                lines: list[str] = [line.strip() for line in fh if line.strip()]
            print(f"[ingest] parsed {len(lines)} implication records")
            return lines
    print("[ingest] no implications file found; skipping that collection.")
    return []


def main() -> None:
    """Build the Chroma vectorstore for the SAIR retriever."""
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        from langchain_core.documents import Document
        try:
            # Prefer the dedicated langchain-chroma package (langchain>=0.2.9)
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]
    except ImportError as exc:
        raise SystemExit(
            "LangChain dependencies are missing. Run `pip install -r requirements.txt` first."
        ) from exc

    SAIR_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    equations: list[tuple[str, str]] = _read_equations_file()
    implications: list[str] = _read_implications_file()

    documents: list[Document] = []
    for eq_id, eq_text in equations:
        documents.append(
            Document(
                page_content=f"Equation {eq_id}: {eq_text}",
                metadata={"kind": "equation", "equation_id": eq_id},
            )
        )
    for i, impl in enumerate(implications):
        documents.append(
            Document(
                page_content=impl,
                metadata={"kind": "implication", "implication_id": str(i)},
            )
        )

    print(f"[ingest] indexing {len(documents)} documents into {SAIR_VECTORSTORE_DIR}")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if documents:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(SAIR_VECTORSTORE_DIR),
            collection_name="sair_equations",
        )
        print("[ingest] done.")
    else:
        # Create an empty Chroma collection so the retriever can connect.
        Chroma(
            persist_directory=str(SAIR_VECTORSTORE_DIR),
            embedding_function=embeddings,
            collection_name="sair_equations",
        )
        print("[ingest] created empty index (no documents to add).")


if __name__ == "__main__":
    main()
