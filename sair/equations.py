"""Parser and evaluator for SAIR equational theory expressions.

The SAIR Stage 1 problems express magma equations in a tiny grammar::

    equation := term '=' term
    term     := atom | atom '*' term
    atom     := var | '(' term ')'
    var      := single lowercase letter (x, y, z, w, ...)

The parser produces a small immutable AST of nested tuples, which keeps
evaluation extremely cheap in hot loops (counterexample search) and makes
hashing / memoization trivial.

AST node forms
--------------
- ``('var', 'x')``                variable reference
- ``('mul', left, right)``        binary magma multiplication
- ``('eq', lhs, rhs)``            top-level equality (root only)

All nodes are plain tuples so they work transparently as dict keys.
"""

from __future__ import annotations

from typing import Any, Union

Term = tuple[Any, ...]
Equation = tuple[str, Term, Term]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_SINGLE_CHAR_TOKENS: set[str] = {"*", "=", "(", ")"}


def _tokenize(text: str) -> list[str]:
    """Split an equation string into atomic tokens.

    Variables are single letters, so we do not need a general identifier
    scanner. Whitespace is discarded.
    """
    tokens: list[str] = []
    i: int = 0
    n: int = len(text)
    while i < n:
        ch: str = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in _SINGLE_CHAR_TOKENS:
            tokens.append(ch)
            i += 1
            continue
        if ch.isalpha():
            # Single-letter variable names only — enforced by the grammar.
            tokens.append(ch)
            i += 1
            continue
        raise ValueError(f"Unexpected character {ch!r} at position {i} in {text!r}")
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------


class _Parser:
    """Minimal recursive-descent parser for the SAIR equation grammar."""

    def __init__(self, tokens: list[str]) -> None:
        self._toks: list[str] = tokens
        self._pos: int = 0

    def _peek(self) -> Union[str, None]:
        if self._pos >= len(self._toks):
            return None
        return self._toks[self._pos]

    def _eat(self, expected: Union[str, None] = None) -> str:
        if self._pos >= len(self._toks):
            raise ValueError("Unexpected end of input")
        tok: str = self._toks[self._pos]
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r}")
        self._pos += 1
        return tok

    def parse_equation(self) -> Equation:
        """Parse a full ``<term> = <term>`` equation."""
        lhs: Term = self.parse_term()
        self._eat("=")
        rhs: Term = self.parse_term()
        if self._peek() is not None:
            raise ValueError(f"Trailing tokens after equation: {self._toks[self._pos :]}")
        return ("eq", lhs, rhs)

    def parse_term(self) -> Term:
        """Parse a term. ``*`` is right-associative, matching the dataset convention."""
        atom: Term = self.parse_atom()
        if self._peek() == "*":
            self._eat("*")
            right: Term = self.parse_term()
            return ("mul", atom, right)
        return atom

    def parse_atom(self) -> Term:
        tok: Union[str, None] = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of input while parsing atom")
        if tok == "(":
            self._eat("(")
            inner: Term = self.parse_term()
            self._eat(")")
            return inner
        if len(tok) == 1 and tok.isalpha():
            self._eat()
            return ("var", tok)
        raise ValueError(f"Unexpected token {tok!r} while parsing atom")


def parse_equation(text: str) -> Equation:
    """Parse an equation string into an AST.

    Example
    -------
    >>> parse_equation("x = x * (x * y)")
    ('eq', ('var', 'x'), ('mul', ('var', 'x'), ('mul', ('var', 'x'), ('var', 'y'))))
    """
    return _Parser(_tokenize(text)).parse_equation()


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------


def collect_variables(node: Term | Equation) -> list[str]:
    """Return the ordered unique list of variable names used in the node."""
    seen: dict[str, None] = {}
    stack: list[Any] = [node]
    while stack:
        cur: Any = stack.pop()
        if not isinstance(cur, tuple) or not cur:
            continue
        head: str = cur[0]
        if head == "var":
            seen.setdefault(cur[1], None)
        elif head == "mul":
            stack.append(cur[1])
            stack.append(cur[2])
        elif head == "eq":
            stack.append(cur[1])
            stack.append(cur[2])
    return list(seen.keys())


def term_to_str(node: Term) -> str:
    """Pretty-print a term back to its surface syntax."""
    head: str = node[0]
    if head == "var":
        return str(node[1])
    if head == "mul":
        left: str = term_to_str(node[1])
        right: str = term_to_str(node[2])
        # Parenthesize non-atomic operands for readability.
        left_p: str = left if node[1][0] == "var" else f"({left})"
        right_p: str = right if node[2][0] == "var" else f"({right})"
        return f"{left_p} * {right_p}"
    raise ValueError(f"Unknown term head: {head}")


def equation_to_str(eq: Equation) -> str:
    """Pretty-print an equation AST back to its surface syntax."""
    return f"{term_to_str(eq[1])} = {term_to_str(eq[2])}"


# ---------------------------------------------------------------------------
# Evaluation on a finite magma table
# ---------------------------------------------------------------------------

# A magma "table" is a tuple of tuples of ints where table[i][j] = i * j.
MagmaTable = tuple[tuple[int, ...], ...]


def evaluate_term(
    node: Term,
    *,
    table: MagmaTable,
    assignment: dict[str, int],
) -> int:
    """Evaluate a term on a magma table under a variable assignment.

    Args:
        node: Term AST.
        table: Square magma table.
        assignment: Map from variable name to element index.

    Returns:
        The element index produced by the term.
    """
    head: str = node[0]
    if head == "var":
        return assignment[node[1]]
    if head == "mul":
        lv: int = evaluate_term(node[1], table=table, assignment=assignment)
        rv: int = evaluate_term(node[2], table=table, assignment=assignment)
        return table[lv][rv]
    raise ValueError(f"Cannot evaluate node with head {head!r}")


def equation_holds(
    eq: Equation,
    *,
    table: MagmaTable,
    variables: list[str] | None = None,
) -> bool:
    """Check whether an equation is universally satisfied by a magma table.

    Iterates over every possible assignment of ``variables`` to the elements
    of the table and checks ``lhs == rhs``. Returns False on the first
    assignment that breaks the equation.
    """
    vars_list: list[str] = variables if variables is not None else collect_variables(eq)
    order: int = len(table)
    lhs: Term = eq[1]
    rhs: Term = eq[2]

    # Iterative Cartesian product without itertools to keep the hot loop lean.
    n_vars: int = len(vars_list)
    if n_vars == 0:
        return evaluate_term(lhs, table=table, assignment={}) == evaluate_term(
            rhs, table=table, assignment={}
        )

    indices: list[int] = [0] * n_vars
    while True:
        assignment: dict[str, int] = {
            vars_list[k]: indices[k] for k in range(n_vars)
        }
        lv: int = evaluate_term(lhs, table=table, assignment=assignment)
        rv: int = evaluate_term(rhs, table=table, assignment=assignment)
        if lv != rv:
            return False

        # Increment the mixed-radix counter.
        k: int = n_vars - 1
        while k >= 0:
            indices[k] += 1
            if indices[k] < order:
                break
            indices[k] = 0
            k -= 1
        if k < 0:
            return True


def find_failing_assignment(
    eq: Equation,
    *,
    table: MagmaTable,
    variables: list[str] | None = None,
) -> dict[str, int] | None:
    """Return the first variable assignment that falsifies the equation, or None.

    Used by the counterexample agent to produce a witness alongside the
    magma table when reporting a refutation.
    """
    vars_list: list[str] = variables if variables is not None else collect_variables(eq)
    order: int = len(table)
    lhs: Term = eq[1]
    rhs: Term = eq[2]

    n_vars: int = len(vars_list)
    if n_vars == 0:
        ok: bool = evaluate_term(lhs, table=table, assignment={}) == evaluate_term(
            rhs, table=table, assignment={}
        )
        return None if ok else {}

    indices: list[int] = [0] * n_vars
    while True:
        assignment: dict[str, int] = {
            vars_list[k]: indices[k] for k in range(n_vars)
        }
        lv: int = evaluate_term(lhs, table=table, assignment=assignment)
        rv: int = evaluate_term(rhs, table=table, assignment=assignment)
        if lv != rv:
            return assignment
        k: int = n_vars - 1
        while k >= 0:
            indices[k] += 1
            if indices[k] < order:
                break
            indices[k] = 0
            k -= 1
        if k < 0:
            return None
