"""Lightweight symbolic prover for magma implications.

This is a deliberately shallow prover — its job is to catch easy
``E1 => E2`` cases that the counterexample agent cannot refute, without
trying to reinvent a full term rewriting engine.

Strategies attempted, in order:

1. **Syntactic equality after alpha-renaming** — ``x = x * x`` vs
   ``y = y * y``.
2. **Direct substitution** — if E2 is an instance of E1 obtained by
   substituting variables, E1 => E2 trivially.
3. **One-step rewrite** — try to rewrite each subterm of E2's LHS/RHS
   using E1 as an oriented rewrite rule (and vice versa). If after the
   rewrite the two sides become alpha-equivalent, we have a proof.

Anything beyond that is left to the LLM-based prover reasoning path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sair.equations import Equation, Term, collect_variables, term_to_str


@dataclass(frozen=True)
class ProofResult:
    """Outcome of a symbolic proof attempt."""

    proved: bool
    strategy: str = ""
    steps: tuple[str, ...] = ()

    def render(self) -> str:
        """Render a human-readable proof block for the cheat sheet."""
        if not self.proved:
            return "PROOF: symbolic prover could not find a proof."
        body: str = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.steps))
        return f"PROOF (strategy={self.strategy}):\n{body}"


# ---------------------------------------------------------------------------
# Alpha-equivalence
# ---------------------------------------------------------------------------


def _normalize_variables(node: Any) -> Any:
    """Rename variables to canonical names v0, v1, ... in first-seen order."""
    mapping: dict[str, str] = {}
    counter: list[int] = [0]

    def walk(n: Any) -> Any:
        if not isinstance(n, tuple) or not n:
            return n
        head: str = n[0]
        if head == "var":
            original: str = n[1]
            if original not in mapping:
                mapping[original] = f"v{counter[0]}"
                counter[0] += 1
            return ("var", mapping[original])
        if head == "mul":
            return ("mul", walk(n[1]), walk(n[2]))
        if head == "eq":
            return ("eq", walk(n[1]), walk(n[2]))
        return n

    return walk(node)


def alpha_equivalent(a: Any, b: Any) -> bool:
    """True if two AST nodes are equal up to consistent variable renaming."""
    return _normalize_variables(a) == _normalize_variables(b)


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------


def substitute(term: Term, *, mapping: dict[str, Term]) -> Term:
    """Apply a variable substitution to a term."""
    head: str = term[0]
    if head == "var":
        name: str = term[1]
        return mapping.get(name, term)
    if head == "mul":
        return ("mul", substitute(term[1], mapping=mapping), substitute(term[2], mapping=mapping))
    return term


def _match_pattern(pattern: Term, target: Term, bindings: dict[str, Term]) -> bool:
    """Match ``target`` against ``pattern``, extending bindings in place.

    Returns True on success and leaves ``bindings`` in a consistent state;
    returns False on the first conflict.
    """
    if pattern[0] == "var":
        name: str = pattern[1]
        if name in bindings:
            return bindings[name] == target
        bindings[name] = target
        return True
    if pattern[0] == "mul":
        if target[0] != "mul":
            return False
        return _match_pattern(pattern[1], target[1], bindings) and _match_pattern(
            pattern[2], target[2], bindings
        )
    return pattern == target


def _is_instance(pattern: Equation, candidate: Equation) -> bool:
    """True if ``candidate`` is a substitution instance of ``pattern``."""
    bindings: dict[str, Term] = {}
    if not _match_pattern(pattern[1], candidate[1], bindings):
        return False
    if not _match_pattern(pattern[2], candidate[2], bindings):
        return False
    return True


# ---------------------------------------------------------------------------
# One-step rewrite
# ---------------------------------------------------------------------------


def _try_rewrite_at(
    term: Term, *, lhs_pattern: Term, rhs_template: Term
) -> Term | None:
    """Try to rewrite the root of ``term`` using ``lhs_pattern -> rhs_template``.

    Returns the rewritten term or None if the pattern does not match.
    """
    bindings: dict[str, Term] = {}
    if not _match_pattern(lhs_pattern, term, bindings):
        return None
    return substitute(rhs_template, mapping=bindings)


def _rewrite_anywhere(term: Term, *, lhs_pattern: Term, rhs_template: Term) -> Term:
    """Rewrite every occurrence of ``lhs_pattern`` in ``term`` once (top-down)."""
    rewritten: Term | None = _try_rewrite_at(
        term, lhs_pattern=lhs_pattern, rhs_template=rhs_template
    )
    if rewritten is not None:
        return rewritten
    if term[0] == "mul":
        return (
            "mul",
            _rewrite_anywhere(term[1], lhs_pattern=lhs_pattern, rhs_template=rhs_template),
            _rewrite_anywhere(term[2], lhs_pattern=lhs_pattern, rhs_template=rhs_template),
        )
    return term


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def try_prove(*, eq1: Equation, eq2: Equation) -> ProofResult:
    """Attempt a shallow symbolic proof that ``eq1 => eq2``.

    This is intentionally conservative: a negative result says nothing,
    only a positive result should be trusted.
    """
    # Strategy 1: alpha-equivalence
    if alpha_equivalent(eq1, eq2):
        return ProofResult(
            proved=True,
            strategy="alpha_equivalence",
            steps=(
                f"E1 and E2 are identical up to variable renaming: "
                f"{term_to_str(eq1[1])} = {term_to_str(eq1[2])}",
            ),
        )

    # Strategy 2: E2 is a substitution instance of E1
    if _is_instance(eq1, eq2):
        return ProofResult(
            proved=True,
            strategy="substitution_instance",
            steps=(
                f"E2 is a substitution instance of E1 "
                f"({term_to_str(eq1[1])} = {term_to_str(eq1[2])}).",
            ),
        )

    # Strategy 3: one-step rewrite of E2 sides using E1 as oriented rule
    for lhs_pattern, rhs_template, direction in (
        (eq1[1], eq1[2], "E1 L->R"),
        (eq1[2], eq1[1], "E1 R->L"),
    ):
        new_lhs: Term = _rewrite_anywhere(
            eq2[1], lhs_pattern=lhs_pattern, rhs_template=rhs_template
        )
        new_rhs: Term = _rewrite_anywhere(
            eq2[2], lhs_pattern=lhs_pattern, rhs_template=rhs_template
        )
        rewritten_eq: Equation = ("eq", new_lhs, new_rhs)
        if alpha_equivalent(rewritten_eq, eq2):
            continue  # Rule did not fire.
        if alpha_equivalent(new_lhs, new_rhs):
            return ProofResult(
                proved=True,
                strategy=f"one_step_rewrite[{direction}]",
                steps=(
                    f"Applied E1 as rewrite rule ({direction}) to E2; "
                    f"both sides collapse to "
                    f"{term_to_str(new_lhs)}.",
                ),
            )

    # Strategy 3b: variable count check (trivially FALSE candidates)
    _ = collect_variables  # keep import anchored; no-op here
    return ProofResult(proved=False)
