"""Pydantic schemas shared by the SAIR pipeline, the API endpoint, and
the training data format.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# API contracts (used by `/phase5/sair`)
# ---------------------------------------------------------------------------


class SAIREquationPair(BaseModel):
    """Request payload for the SAIR Stage 1 endpoint."""

    equation1: str = Field(
        ...,
        description="First equation (E1), e.g. 'x = x * y'.",
        examples=["x = x * y"],
    )
    equation2: str = Field(
        ...,
        description="Second equation (E2), e.g. 'x = x * x'.",
        examples=["x = x * x"],
    )


class SAIRResponse(BaseModel):
    """Response payload for the SAIR Stage 1 endpoint."""

    verdict: Optional[bool] = None
    response: str
    trace: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal pipeline schemas
# ---------------------------------------------------------------------------


class Problem(BaseModel):
    """A single SAIR Stage 1 problem with optional ground truth."""

    id: str
    equation1: str
    equation2: str
    answer: Optional[bool] = None
    difficulty: Optional[str] = None
    eq1_id: Optional[int] = None
    eq2_id: Optional[int] = None
    index: Optional[int] = None


class Evidence(BaseModel):
    """One piece of evidence produced by an agent for a single problem."""

    kind: str  # "counterexample" | "symbolic_proof" | "retrieved" | "llm_reasoning"
    verdict: Optional[bool] = None
    confidence: float = 0.0
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    """Aggregated evidence collected by all agents for one problem."""

    problem: Problem
    evidences: list[Evidence] = Field(default_factory=list)
    consensus_verdict: Optional[bool] = None
    consensus_confidence: float = 0.0

    def rendered_reasoning(self) -> str:
        """Render the full evidence bundle as a reasoning text block."""
        parts: list[str] = []
        for ev in self.evidences:
            if ev.content:
                parts.append(ev.content)
        return "\n\n".join(parts)


class TrainingExample(BaseModel):
    """One SFT example derived from an EvidenceBundle."""

    problem_id: str
    prompt: str  # user message — the rendered problem statement
    completion: str  # expected assistant response (think + answer)
    verdict: bool  # ground truth label
    source: str  # "counterexample" | "symbolic_proof" | "external_label"


class CheatSheetEntry(BaseModel):
    """A single section of the cheat sheet with a priority and a body."""

    title: str
    body: str
    priority: int = 0  # higher wins when trimming to fit the 10KB budget


class CheatSheet(BaseModel):
    """The final deliverable: the ≤10KB prompt-prefix shipped to the judge."""

    version: str
    entries: list[CheatSheetEntry] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    def render(self, *, max_bytes: int = 10_000) -> str:
        """Assemble the cheat sheet text under a hard byte budget.

        Entries are emitted by descending priority; the first entry that
        would overflow the budget is dropped and subsequent entries are
        only included if they still fit.
        """
        ordered: list[CheatSheetEntry] = sorted(
            self.entries, key=lambda e: e.priority, reverse=True
        )
        pieces: list[str] = []
        used: int = 0
        for entry in ordered:
            block: str = f"## {entry.title}\n{entry.body}\n"
            size: int = len(block.encode("utf-8"))
            if used + size > max_bytes:
                continue
            pieces.append(block)
            used += size
        return "\n".join(pieces).strip()


class RunMetrics(BaseModel):
    """Aggregate metrics from a pipeline or evaluation run."""

    total: int = 0
    correct: int = 0
    unparseable: int = 0
    accuracy: float = 0.0
    per_bucket: dict[str, dict[str, float]] = Field(default_factory=dict)
    wall_seconds: float = 0.0
    cost_estimate_usd: float = 0.0
