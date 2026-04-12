"""Counterexample-finding agent.

Given two equations E1 and E2, this agent tries to refute ``E1 => E2`` by
finding a finite magma where E1 holds but E2 fails. This is by far the
cheapest and strongest signal source in the SAIR pipeline — most FALSE
problems in the public subsets collapse to an order 2 or 3 counterexample
in milliseconds.

The agent produces a structured ``CounterexampleResult`` that the
aggregator can embed verbatim into the distiller's training prompt and,
later, into the final cheat sheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable

from sair.equations import (
    Equation,
    MagmaTable,
    collect_variables,
    equation_holds,
    find_failing_assignment,
)
from sair.magma import iterate_magmas, render_table


@dataclass(frozen=True)
class CounterexampleResult:
    """Witness that ``E1 => E2`` is false, or a "searched but not found" record."""

    found: bool
    order: int | None = None
    table: MagmaTable | None = None
    failing_assignment: dict[str, int] = field(default_factory=dict)
    n_magmas_checked: int = 0
    elapsed_seconds: float = 0.0

    def render(self) -> str:
        """Render a human-readable witness block for the cheat sheet."""
        if not self.found or self.table is None or self.order is None:
            return (
                f"COUNTEREXAMPLE: none found after checking "
                f"{self.n_magmas_checked} magmas in {self.elapsed_seconds:.2f}s."
            )
        assignment_pretty: str = ", ".join(
            f"{name}={val}" for name, val in sorted(self.failing_assignment.items())
        )
        return (
            f"COUNTEREXAMPLE: order-{self.order} magma refutes E1 => E2.\n"
            f"Cayley table:\n{render_table(self.table)}\n"
            f"Failing assignment: {assignment_pretty}"
        )


def find_counterexample(
    *,
    eq1: Equation,
    eq2: Equation,
    orders: Iterable[int] = (2, 3),
    exhaustive_limit: int = 3,
    random_samples_order4: int = 20_000,
    seed: int | None = 0,
) -> CounterexampleResult:
    """Search finite magmas for a refutation of ``eq1 => eq2``.

    Args:
        eq1: Hypothesis equation (must hold on the witness magma).
        eq2: Target equation (must fail on the witness magma).
        orders: Magma orders to try, in order.
        exhaustive_limit: Highest order enumerated exhaustively.
        random_samples_order4: Number of random samples for order-4 fallback.
        seed: RNG seed for reproducibility.

    Returns:
        A ``CounterexampleResult``. ``found=False`` means the search
        exhausted the budget without finding a witness — it does NOT prove
        E1 => E2.
    """
    vars1: list[str] = collect_variables(eq1)
    vars2: list[str] = collect_variables(eq2)

    start: float = perf_counter()
    n_checked: int = 0

    for order in orders:
        for table in iterate_magmas(
            order,
            exhaustive_limit=exhaustive_limit,
            random_samples=random_samples_order4,
            seed=seed,
        ):
            n_checked += 1
            # Short-circuit: if E1 fails on this magma we don't care about E2.
            if not equation_holds(eq1, table=table, variables=vars1):
                continue
            # E1 holds. Does E2 also hold?
            assignment: dict[str, int] | None = find_failing_assignment(
                eq2, table=table, variables=vars2
            )
            if assignment is not None:
                # Witness: E1 holds, E2 fails on this magma.
                return CounterexampleResult(
                    found=True,
                    order=order,
                    table=table,
                    failing_assignment=assignment,
                    n_magmas_checked=n_checked,
                    elapsed_seconds=perf_counter() - start,
                )

    return CounterexampleResult(
        found=False,
        n_magmas_checked=n_checked,
        elapsed_seconds=perf_counter() - start,
    )
