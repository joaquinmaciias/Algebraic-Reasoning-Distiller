"""Interactive console demo of each SAIR agent.

Runs a small, hand-picked set of problems through every agent in the
pipeline and prints what each one contributes. Does NOT train, fine-tune
or load the LLM distiller — it only exercises the symbolic / retrieval
components so the contribution of each piece is visible.

Run
---
    python -m sair.scripts.demo_agents
"""

from __future__ import annotations

from typing import Any

from sair.agents.counterexample import find_counterexample
from sair.agents.prover import try_prove
from sair.agents.retriever import retrieve_relevant_snippets
from sair.equations import (
    collect_variables,
    equation_to_str,
    parse_equation,
    term_to_str,
)
from sair.magma import render_table


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

BAR = "=" * 72


def banner(title: str) -> None:
    print("\n" + BAR)
    print(f"  {title}")
    print(BAR)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Demo cases
# ---------------------------------------------------------------------------

# Each case is (name, eq1, eq2, expected_verdict_or_none).
DEMO_CASES: list[tuple[str, str, str, bool | None]] = [
    (
        "Alpha-equivalencia trivial",
        "x * y = y * x",
        "a * b = b * a",
        True,
    ),
    (
        "Instancia por sustitucion",
        "x * y = y * x",
        "(a * b) * c = c * (a * b)",
        True,
    ),
    (
        "Contraejemplo sencillo (no conmutativa)",
        "x = x",
        "x * y = y * x",
        False,
    ),
    (
        "Caso dificil sin veredicto simbolico",
        "x * (y * z) = (x * y) * z",
        "x * y = y * x",
        False,
    ),
]


# ---------------------------------------------------------------------------
# Agent demos
# ---------------------------------------------------------------------------


def demo_parser() -> None:
    banner("1. PLANIFICADOR (parser)")
    print("Convierte texto -> AST (tuplas anidadas).")
    print("La estructura la consumen todos los demas agentes.\n")

    samples: list[str] = [
        "x * y = y * x",
        "x = x * (x * y)",
        "(a * b) * c = c * (a * b)",
    ]
    for text in samples:
        section(f"Equation: {text}")
        ast = parse_equation(text)
        print(f"  AST:        {ast}")
        print(f"  pretty:     {equation_to_str(ast)}")
        print(f"  variables:  {collect_variables(ast)}")
        print(f"  lhs str:    {term_to_str(ast[1])}")
        print(f"  rhs str:    {term_to_str(ast[2])}")


def demo_retriever() -> None:
    banner("2. RECUPERADOR RAG")
    print("Busca ecuaciones similares en ChromaDB (embeddings MiniLM).")
    print("Si el indice no esta construido, devuelve [] (degrada con elegancia).\n")

    queries: list[str] = [
        "x * y = y * x",
        "x * (y * z) = (x * y) * z",
    ]
    for q in queries:
        section(f"Query: {q}")
        snippets: list[dict[str, Any]] = retrieve_relevant_snippets(query=q, k=5)
        if not snippets:
            print("  (indice vacio o no construido; el pipeline seguiria sin contexto RAG)")
            print("  Construye el indice con: python -m sair.data.ingest_equations")
            continue
        for i, s in enumerate(snippets, start=1):
            text: str = s.get("text", "")
            meta: dict[str, Any] = s.get("metadata", {})
            print(f"  [{i}] {text}")
            if meta:
                print(f"       metadata: {meta}")


def demo_prover() -> None:
    banner("3. PROBADOR SIMBOLICO")
    print("Intenta demostrar E1 => E2 con tres estrategias:")
    print("  (a) alfa-equivalencia")
    print("  (b) instancia por sustitucion")
    print("  (c) reescritura de un paso")
    print("Es conservador: un negativo no refuta.\n")

    for name, e1, e2, expected in DEMO_CASES:
        section(name)
        print(f"  E1: {e1}")
        print(f"  E2: {e2}")
        print(f"  verdad esperada: {expected}")
        ast1 = parse_equation(e1)
        ast2 = parse_equation(e2)
        result = try_prove(eq1=ast1, eq2=ast2)
        if result.proved:
            print(f"  -> PROBADO (estrategia={result.strategy})")
            for step in result.steps:
                print(f"     * {step}")
        else:
            print("  -> sin prueba (no concluye nada; pasa al contraejemplo/LLM)")


def demo_counterexample() -> None:
    banner("4. BUSCADOR DE CONTRAEJEMPLOS")
    print("Enumera magmas finitos (orden 2 y 3) buscando uno que cumpla E1 y viole E2.")
    print("Un contraejemplo concreto refuta la implicacion de forma absoluta.\n")

    for name, e1, e2, expected in DEMO_CASES:
        section(name)
        print(f"  E1: {e1}")
        print(f"  E2: {e2}")
        print(f"  verdad esperada: {expected}")
        ast1 = parse_equation(e1)
        ast2 = parse_equation(e2)
        result = find_counterexample(eq1=ast1, eq2=ast2, orders=(2, 3))
        print(f"  magmas revisados: {result.n_magmas_checked}")
        print(f"  tiempo: {result.elapsed_seconds:.3f}s")
        if result.found and result.table is not None:
            print(f"  -> CONTRAEJEMPLO encontrado en orden {result.order}")
            print(f"     asignacion fallida: {result.failing_assignment}")
            print("     tabla de Cayley:")
            for line in render_table(result.table).splitlines():
                print(f"       {line}")
        else:
            print("  -> no encontrado (no prueba que sea TRUE, solo que no se refuta rapido)")


def demo_aggregator() -> None:
    banner("5. AGREGADOR")
    print("Fusiona las salidas con politica por prioridad:")
    print("  contraejemplo -> FALSE (conf 1.0)")
    print("  prueba        -> TRUE  (conf 0.9)")
    print("  ninguno       -> UNKNOWN (pasa al LLM destilador)\n")

    for name, e1, e2, expected in DEMO_CASES:
        section(name)
        print(f"  E1: {e1}")
        print(f"  E2: {e2}")
        ast1 = parse_equation(e1)
        ast2 = parse_equation(e2)
        proof = try_prove(eq1=ast1, eq2=ast2)
        counter = find_counterexample(eq1=ast1, eq2=ast2, orders=(2, 3))

        if counter.found:
            verdict: str = "FALSE"
            confidence: float = 1.0
            source: str = f"counterexample (orden {counter.order})"
        elif proof.proved:
            verdict = "TRUE"
            confidence = 0.9
            source = f"symbolic_proof ({proof.strategy})"
        else:
            verdict = "UNKNOWN"
            confidence = 0.0
            source = "shallow strategies failed -> delega en LLM destilador"

        match: str = "OK" if (
            (verdict == "TRUE" and expected is True)
            or (verdict == "FALSE" and expected is False)
            or (verdict == "UNKNOWN")
        ) else "MISMATCH"
        print(f"  esperado: {expected}  |  veredicto: {verdict} (conf {confidence})  [{match}]")
        print(f"  fuente:   {source}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + BAR)
    print("  SAIR multi-agent demo")
    print("  (sin entrenamiento, sin LLM — solo los agentes simbolicos y RAG)")
    print(BAR)
    demo_parser()
    demo_retriever()
    demo_prover()
    demo_counterexample()
    demo_aggregator()
    print("\n" + BAR)
    print("  Demo terminada.")
    print(BAR + "\n")


if __name__ == "__main__":
    main()
