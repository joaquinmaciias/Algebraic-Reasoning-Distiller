"""Finite magma enumeration for counterexample search.

A magma of order N is fully described by an NxN Cayley table. Our search
space grows as ``N ** (N * N)`` — 16 for N=2, 19,683 for N=3, 4.3B for
N=4 — so we use exhaustive enumeration up to N=3 and random sampling for
N>=4.
"""

from __future__ import annotations

import random
from typing import Iterator

from sair.equations import MagmaTable


def enumerate_magmas_exhaustive(order: int) -> Iterator[MagmaTable]:
    """Yield every Cayley table of the given order, in lexicographic order.

    Intended for small orders (N<=3). Caller is responsible for not calling
    this with N>=4 unless they really mean it.
    """
    n_cells: int = order * order
    digits: list[int] = [0] * n_cells
    while True:
        # Build an immutable NxN table from the flat digit list.
        rows: list[tuple[int, ...]] = []
        for r in range(order):
            start: int = r * order
            rows.append(tuple(digits[start : start + order]))
        yield tuple(rows)

        # Mixed-radix increment in base `order`.
        k: int = n_cells - 1
        while k >= 0:
            digits[k] += 1
            if digits[k] < order:
                break
            digits[k] = 0
            k -= 1
        if k < 0:
            return


def sample_random_magmas(
    order: int,
    *,
    n_samples: int,
    seed: int | None = None,
) -> Iterator[MagmaTable]:
    """Yield ``n_samples`` uniformly random Cayley tables of the given order.

    Used for large orders where exhaustive enumeration is infeasible.
    """
    rng: random.Random = random.Random(seed)
    for _ in range(n_samples):
        rows: list[tuple[int, ...]] = [
            tuple(rng.randrange(order) for _ in range(order)) for _ in range(order)
        ]
        yield tuple(rows)


def iterate_magmas(
    order: int,
    *,
    exhaustive_limit: int = 3,
    random_samples: int = 20_000,
    seed: int | None = None,
) -> Iterator[MagmaTable]:
    """Iterate over magmas using the best strategy for the given order.

    Args:
        order: Magma order (number of elements).
        exhaustive_limit: Maximum order for which exhaustive enumeration is
            used. Higher orders fall back to random sampling.
        random_samples: Number of random samples when the order exceeds
            ``exhaustive_limit``.
        seed: RNG seed for reproducible random sampling.
    """
    if order <= exhaustive_limit:
        yield from enumerate_magmas_exhaustive(order)
    else:
        yield from sample_random_magmas(
            order, n_samples=random_samples, seed=seed
        )


def render_table(table: MagmaTable) -> str:
    """Render a Cayley table as an ASCII grid suitable for the cheat sheet."""
    order: int = len(table)
    header_cells: list[str] = [" "] + [str(i) for i in range(order)]
    header: str = " | ".join(header_cells)
    sep: str = "-" * len(header)
    body_lines: list[str] = []
    for r in range(order):
        row_cells: list[str] = [str(r)] + [str(table[r][c]) for c in range(order)]
        body_lines.append(" | ".join(row_cells))
    return "\n".join([header, sep, *body_lines])
