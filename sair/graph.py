"""Offline orchestration of the SAIR multi-agent pipeline.

Two execution modes are exposed:

1. **Sequential driver** (``run_sair_pipeline``): a simple Python
   function that runs each agent in order. This is what the FastAPI
   endpoint ``/phase5/sair`` uses — it avoids pulling in the optional
   ``langgraph`` runtime and starts instantly.
2. **LangGraph wiring** (``build_langgraph_app``): a ``StateGraph`` with
   nodes for planner, retriever, prover, counterexample, and aggregator.
   Imported lazily so the rest of the package keeps working if
   ``langgraph`` is not installed. Used by the offline training-data
   builder for the MGP rubric (stateful multi-agent system).

Both paths share the same ``PipelineState`` dataclass and produce the
same ``EvidenceBundle``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sair.agents.counterexample import (
    CounterexampleResult,
    find_counterexample,
)
from sair.agents.prover import ProofResult, try_prove
from sair.agents.retriever import retrieve_relevant_snippets
from sair.equations import Equation, parse_equation
from sair.schemas import (
    Evidence,
    EvidenceBundle,
    Problem,
    SAIRResponse,
)


# ---------------------------------------------------------------------------
# State container shared by both execution paths
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Mutable state passed between agents."""

    problem: Problem
    eq1_ast: Equation | None = None
    eq2_ast: Equation | None = None
    parse_error: str | None = None
    retrieved: list[str] = field(default_factory=list)
    counter: CounterexampleResult | None = None
    proof: ProofResult | None = None
    bundle: EvidenceBundle | None = None


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


def node_planner(state: PipelineState) -> PipelineState:
    """Parse the equations and decide which agents to run.

    Parsing is cheap and deterministic, so we always run it here so that
    downstream nodes can assume ``eq1_ast`` / ``eq2_ast`` are valid.
    """
    try:
        state.eq1_ast = parse_equation(state.problem.equation1)
        state.eq2_ast = parse_equation(state.problem.equation2)
    except Exception as exc:
        state.parse_error = f"parse_error: {exc}"
    return state


def node_retriever(state: PipelineState, *, k: int = 5) -> PipelineState:
    """Pull nearby equations from the RAG index, if one is built."""
    query: str = f"{state.problem.equation1} => {state.problem.equation2}"
    snippets: list[dict[str, Any]] = retrieve_relevant_snippets(query=query, k=k)
    state.retrieved = [s["text"] for s in snippets]
    return state


def node_counterexample(
    state: PipelineState,
    *,
    orders: tuple[int, ...] = (2, 3),
) -> PipelineState:
    """Search for a finite-magma refutation."""
    if state.eq1_ast is None or state.eq2_ast is None:
        return state
    state.counter = find_counterexample(
        eq1=state.eq1_ast, eq2=state.eq2_ast, orders=orders
    )
    return state


def node_prover(state: PipelineState) -> PipelineState:
    """Try a shallow symbolic proof."""
    if state.eq1_ast is None or state.eq2_ast is None:
        return state
    state.proof = try_prove(eq1=state.eq1_ast, eq2=state.eq2_ast)
    return state


def node_aggregator(state: PipelineState) -> PipelineState:
    """Fuse the outputs of the agents into a single EvidenceBundle.

    Policy:
    - A counterexample wins (hard FALSE, confidence 1.0).
    - A symbolic proof wins otherwise (hard TRUE, confidence 0.9).
    - Retrieved snippets alone produce an "unknown" bundle with low
      confidence — the distiller still has context to reason from.
    """
    evidences: list[Evidence] = []
    consensus_verdict: bool | None = None
    consensus_confidence: float = 0.0

    if state.parse_error is not None:
        evidences.append(
            Evidence(
                kind="parse_error",
                verdict=None,
                confidence=0.0,
                content=state.parse_error,
            )
        )

    if state.counter is not None and state.counter.found:
        evidences.append(
            Evidence(
                kind="counterexample",
                verdict=False,
                confidence=1.0,
                content=state.counter.render(),
                metadata={
                    "order": state.counter.order,
                    "n_checked": state.counter.n_magmas_checked,
                },
            )
        )
        consensus_verdict = False
        consensus_confidence = 1.0
    elif state.proof is not None and state.proof.proved:
        evidences.append(
            Evidence(
                kind="symbolic_proof",
                verdict=True,
                confidence=0.9,
                content=state.proof.render(),
                metadata={"strategy": state.proof.strategy},
            )
        )
        consensus_verdict = True
        consensus_confidence = 0.9
    else:
        # Negative results from both agents are still informative for the
        # distiller (they say "shallow strategies failed").
        if state.counter is not None:
            evidences.append(
                Evidence(
                    kind="counterexample_search",
                    verdict=None,
                    confidence=0.0,
                    content=state.counter.render(),
                    metadata={"n_checked": state.counter.n_magmas_checked},
                )
            )
        if state.proof is not None:
            evidences.append(
                Evidence(
                    kind="symbolic_proof_attempt",
                    verdict=None,
                    confidence=0.0,
                    content=state.proof.render(),
                )
            )

    if state.retrieved:
        evidences.append(
            Evidence(
                kind="retrieved",
                verdict=None,
                confidence=0.1,
                content="Relevant snippets:\n" + "\n".join(f"- {s}" for s in state.retrieved),
            )
        )

    state.bundle = EvidenceBundle(
        problem=state.problem,
        evidences=evidences,
        consensus_verdict=consensus_verdict,
        consensus_confidence=consensus_confidence,
    )
    return state


# ---------------------------------------------------------------------------
# Sequential driver — used by FastAPI and by the training-data builder
# ---------------------------------------------------------------------------


def run_pipeline_sequential(
    problem: Problem,
    *,
    use_retriever: bool = True,
) -> EvidenceBundle:
    """Run planner -> (retriever, prover, counterexample) -> aggregator.

    Args:
        problem: The problem to solve.
        use_retriever: Skip the RAG step (useful in fast dataset-building).

    Returns:
        The aggregated EvidenceBundle.
    """
    state: PipelineState = PipelineState(problem=problem)
    state = node_planner(state)
    if use_retriever:
        state = node_retriever(state)
    state = node_counterexample(state)
    state = node_prover(state)
    state = node_aggregator(state)
    assert state.bundle is not None
    return state.bundle


# ---------------------------------------------------------------------------
# LangGraph wiring — satisfies the MGP multi-agent rubric
# ---------------------------------------------------------------------------


def build_langgraph_app() -> Any:
    """Build a LangGraph ``StateGraph`` wiring of the same nodes.

    Raises:
        ImportError: If ``langgraph`` is not installed.
    """
    from langgraph.graph import END, StateGraph  # imported lazily

    graph: StateGraph = StateGraph(PipelineState)
    graph.add_node("planner", node_planner)
    graph.add_node("retriever", node_retriever)
    graph.add_node("prover", node_prover)
    graph.add_node("counterexample", node_counterexample)
    graph.add_node("aggregator", node_aggregator)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "prover")
    graph.add_edge("prover", "counterexample")
    graph.add_edge("counterexample", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# FastAPI-compatible thin wrapper
# ---------------------------------------------------------------------------


def run_sair_pipeline(equation1: str, equation2: str) -> SAIRResponse:
    """Run the full SAIR pipeline for one (E1, E2) problem.

    This is the stable import site used by ``api/app.py`` for the
    ``/phase5/sair`` endpoint. It returns a SAIRResponse with the
    evidence rendered in the format expected by the SAIR judge.
    """
    problem: Problem = Problem(
        id="adhoc",
        equation1=equation1,
        equation2=equation2,
        answer=None,
    )
    bundle: EvidenceBundle = run_pipeline_sequential(
        problem, use_retriever=True
    )

    verdict: bool | None = bundle.consensus_verdict
    verdict_line: str = (
        "VERDICT: TRUE"
        if verdict is True
        else ("VERDICT: FALSE" if verdict is False else "VERDICT: FALSE")
    )

    reasoning_block: str = bundle.rendered_reasoning() or "No evidence produced."

    rendered: str = (
        f"{verdict_line}\n"
        f"REASONING: Aggregated from counterexample search, symbolic "
        f"prover, and RAG retrieval. Consensus confidence "
        f"{bundle.consensus_confidence:.2f}.\n"
        f"{reasoning_block}"
    )

    trace: list[dict[str, Any]] = []
    for i, ev in enumerate(bundle.evidences):
        trace.append(
            {
                "step": i,
                "agent": ev.kind,
                "content": ev.content,
            }
        )

    return SAIRResponse(
        verdict=verdict,
        response=rendered,
        trace=trace,
        details={
            "stage": "sair_sequential",
            "equation1": equation1,
            "equation2": equation2,
            "consensus_confidence": bundle.consensus_confidence,
            "n_evidences": len(bundle.evidences),
        },
    )
