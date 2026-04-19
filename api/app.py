"""FastAPI application for the Algebraic Reasoning Distiller.

Exposes the SAIR Stage 1 multi-agent pipeline as a REST endpoint.
The pipeline runs: planner → (retriever + prover + counterexample) → aggregator
and returns the consensus verdict with a full evidence trace.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8182
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Ensure repo root is on the path so 'sair' and 'utils' are importable.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from sair.agents.distiller import load_distiller_model, run_distiller
from sair.config import SAIR_INFERENCE_CONFIG
from sair.graph import run_pipeline_sequential, run_sair_pipeline
from sair.schemas import Problem, SAIREquationPair, SAIRResponse

app: FastAPI = FastAPI(
    title="Algebraic Reasoning Distiller",
    description=(
        "Multi-agent pipeline for the SAIR Stage 1 competition: "
        "decides whether Equation 1 implies Equation 2 over all magmas."
    ),
    version="1.0.0",
)

PORT: int = 8182
PLAYGROUND_DIR: Path = REPO_ROOT / "playground"
DATA_DIR: Path = REPO_ROOT / "data"
CHEATSHEET_DIR: Path = REPO_ROOT / "sair" / "cheat_sheets"
DEFAULT_CHEATSHEET: Path = CHEATSHEET_DIR / "cheatsheet_1.txt"
ALLOWED_DATASETS: tuple[str, ...] = ("hard", "hard2", "hard3", "normal")
VALID_INFERENCE_MODES: tuple[str, ...] = ("auto", "symbolic", "distiller")
DEFAULT_INFERENCE_MODE: str = "auto"

DISTILLER_RUNTIME: dict[str, Any] = {
    "configured_mode": DEFAULT_INFERENCE_MODE,
    "active_mode": "symbolic",
    "enabled": False,
    "adapter": None,
    "error": None,
    "checkpoint_used": None,
    "model": None,
    "tokenizer": None,
    "cfg": None,
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if PLAYGROUND_DIR.exists():
    app.mount(
        "/playground",
        StaticFiles(directory=str(PLAYGROUND_DIR), html=True),
        name="playground",
    )


def _resolve_dataset_path(name: str) -> tuple[str, Path]:
    normalized: str = name.strip().lower()
    if normalized.endswith(".json"):
        normalized = normalized[:-5]

    if normalized not in ALLOWED_DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {name}")

    path: Path = DATA_DIR / f"{normalized}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {path.name}")
    return normalized, path


def _resolve_cheatsheet_path() -> Path:
    if DEFAULT_CHEATSHEET.exists():
        return DEFAULT_CHEATSHEET

    if CHEATSHEET_DIR.exists():
        candidates: list[Path] = sorted(
            [*CHEATSHEET_DIR.glob("*.txt"), *CHEATSHEET_DIR.glob("*.md")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    raise HTTPException(status_code=404, detail="No cheat sheet file found")


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {path.name}: {exc}") from exc


def _resolve_inference_mode() -> str:
    raw_mode: str = os.getenv("SAIR_INFERENCE_MODE", DEFAULT_INFERENCE_MODE).strip().lower()
    if raw_mode in VALID_INFERENCE_MODES:
        return raw_mode
    return DEFAULT_INFERENCE_MODE


def _dir_has_files(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _extract_verdict(response_text: str) -> bool | None:
    matches: list[str] = re.findall(r"(?i)VERDICT\s*[:：]\s*(TRUE|FALSE)", response_text)
    if not matches:
        return None
    return matches[-1].upper() == "TRUE"


def _init_distiller_runtime() -> None:
    mode: str = _resolve_inference_mode()
    DISTILLER_RUNTIME["configured_mode"] = mode

    if mode == "symbolic":
        DISTILLER_RUNTIME["active_mode"] = "symbolic"
        return

    cfg: SAIR_INFERENCE_CONFIG = SAIR_INFERENCE_CONFIG()
    has_grpo: bool = _dir_has_files(cfg.checkpoint_directory)
    has_sft: bool = _dir_has_files(cfg.fallback_checkpoint_directory)

    should_try_distiller: bool = mode == "distiller" or (mode == "auto" and (has_grpo or has_sft))
    if not should_try_distiller:
        DISTILLER_RUNTIME["active_mode"] = "symbolic"
        DISTILLER_RUNTIME["error"] = (
            "No trained checkpoints found. Expected one of: "
            f"{cfg.checkpoint_directory} or {cfg.fallback_checkpoint_directory}"
        )
        return

    if mode == "distiller" and not (has_grpo or has_sft):
        raise RuntimeError(
            "SAIR_INFERENCE_MODE=distiller but no trained checkpoints were found. "
            f"Expected one of: {cfg.checkpoint_directory} or {cfg.fallback_checkpoint_directory}"
        )

    model: Any
    tokenizer: Any
    adapter_label: str
    model, tokenizer, adapter_label = load_distiller_model(cfg=cfg)

    DISTILLER_RUNTIME["enabled"] = True
    DISTILLER_RUNTIME["active_mode"] = "distiller"
    DISTILLER_RUNTIME["adapter"] = adapter_label
    DISTILLER_RUNTIME["checkpoint_used"] = (
        str(cfg.checkpoint_directory)
        if adapter_label == "grpo"
        else (str(cfg.fallback_checkpoint_directory) if adapter_label == "sft" else "base")
    )
    DISTILLER_RUNTIME["model"] = model
    DISTILLER_RUNTIME["tokenizer"] = tokenizer
    DISTILLER_RUNTIME["cfg"] = cfg


def _run_sair_with_distiller(equation1: str, equation2: str) -> SAIRResponse:
    problem: Problem = Problem(id="adhoc", equation1=equation1, equation2=equation2)
    bundle = run_pipeline_sequential(problem=problem, use_retriever=True)

    llm_response: str = run_distiller(
        bundle=bundle,
        model=DISTILLER_RUNTIME["model"],
        tokenizer=DISTILLER_RUNTIME["tokenizer"],
        cfg=DISTILLER_RUNTIME["cfg"],
    )

    parsed_verdict: bool | None = _extract_verdict(llm_response)
    verdict: bool | None = parsed_verdict if parsed_verdict is not None else bundle.consensus_verdict

    trace: list[dict[str, Any]] = [
        {"step": i, "agent": ev.kind, "content": ev.content}
        for i, ev in enumerate(bundle.evidences)
    ]
    trace.append(
        {
            "step": len(trace),
            "agent": "distiller",
            "content": llm_response,
        }
    )

    return SAIRResponse(
        verdict=verdict,
        response=llm_response,
        trace=trace,
        details={
            "backend": "distiller",
            "adapter": DISTILLER_RUNTIME["adapter"],
            "checkpoint": DISTILLER_RUNTIME["checkpoint_used"],
            "consensus_fallback_verdict": bundle.consensus_verdict,
        },
    )


@app.on_event("startup")
async def startup_event() -> None:
    try:
        _init_distiller_runtime()
        print(
            "[api] inference backend="
            f"{DISTILLER_RUNTIME['active_mode']} "
            f"(configured={DISTILLER_RUNTIME['configured_mode']}, "
            f"adapter={DISTILLER_RUNTIME['adapter']})"
        )
    except Exception as exc:
        DISTILLER_RUNTIME["error"] = str(exc)
        raise


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


@app.get("/runtime", tags=["System"])
async def runtime() -> dict[str, Any]:
    """Show which backend (/sair) is currently using."""
    return {
        "ok": True,
        "configured_mode": DISTILLER_RUNTIME["configured_mode"],
        "active_mode": DISTILLER_RUNTIME["active_mode"],
        "enabled": DISTILLER_RUNTIME["enabled"],
        "adapter": DISTILLER_RUNTIME["adapter"],
        "checkpoint": DISTILLER_RUNTIME["checkpoint_used"],
        "error": DISTILLER_RUNTIME["error"],
    }


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root traffic to the playground UI."""
    return RedirectResponse(url="/playground/")


@app.get("/datasets", tags=["Playground"])
async def datasets() -> dict[str, Any]:
    """List datasets available for the playground batch evaluation."""
    items: list[dict[str, Any]] = []
    for name in ALLOWED_DATASETS:
        path: Path = DATA_DIR / f"{name}.json"
        if path.exists():
            items.append({"name": name, "filename": path.name})
    return {"ok": True, "datasets": items}


@app.get("/datasets/{name}", tags=["Playground"])
async def dataset_by_name(name: str) -> Any:
    """Return one dataset as JSON."""
    _, path = _resolve_dataset_path(name)
    return _load_json_file(path)


@app.get("/cheatsheet", tags=["Playground"])
async def cheatsheet() -> dict[str, Any]:
    """Return the cheat sheet text shown in the playground editor."""
    path: Path = _resolve_cheatsheet_path()
    return {
        "ok": True,
        "name": path.name,
        "content": path.read_text(encoding="utf-8"),
    }


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
    if DISTILLER_RUNTIME["enabled"]:
        result = _run_sair_with_distiller(
            equation1=request.equation1,
            equation2=request.equation2,
        )
    else:
        result = run_sair_pipeline(
            equation1=request.equation1,
            equation2=request.equation2,
        )
    return result.model_dump()


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=PORT, reload=False)
