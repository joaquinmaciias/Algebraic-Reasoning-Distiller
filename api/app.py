"""FastAPI application for the Algebraic Reasoning Distiller.

Exposes the SAIR Stage 1 multi-agent pipeline as a REST endpoint.
The pipeline runs: planner → (retriever + prover + counterexample) → aggregator
and returns the consensus verdict with a full evidence trace.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8182
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Ensure repo root is on the path so 'sair' and 'utils' are importable.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sair.graph import run_sair_pipeline
from sair.schemas import SAIREquationPair, SAIRResponse

app: FastAPI = FastAPI(
    title="Algebraic Reasoning Distiller",
    description=(
        "Multi-agent pipeline for the SAIR Stage 1 competition: "
        "decides whether Equation 1 implies Equation 2 over all magmas."
    ),
    version="1.0.0",
)

PORT: int = 8182


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return structured JSON for unhandled exceptions (easier debugging on DGX)."""
    tb: str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return JSONResponse(
        status_code=500,
        content={
            "verdict": None,
            "response": f"ERROR: {exc}",
            "trace": [],
            "details": {
                "path": str(request.url.path),
                "error": str(exc),
                "traceback": tb,
            },
        },
    )


@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 when the server is up."""
    return {"status": "ok"}


@app.post("/sair", response_model=SAIRResponse, tags=["SAIR"])
async def sair_endpoint(request: SAIREquationPair) -> dict[str, Any]:
    """Decide whether Equation 1 implies Equation 2 over all magmas.

    The pipeline runs three agents in sequence:
    - **Counterexample agent**: exhaustive search over finite magmas of order ≤3,
      random sampling for order 4.
    - **Symbolic prover**: alpha-equivalence, substitution instance, and
      one-step term rewriting.
    - **RAG retriever**: nearest-neighbour lookup in the Chroma equation index.

    The aggregator fuses results with priority:
    counterexample (FALSE, confidence 1.0) > symbolic proof (TRUE, confidence 0.9)
    > unknown (retrieved context only).
    """
    result: SAIRResponse = run_sair_pipeline(
        equation1=request.equation1,
        equation2=request.equation2,
    )
    return result.model_dump()


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=PORT, reload=False)
