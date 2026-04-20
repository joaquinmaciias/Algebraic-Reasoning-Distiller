"""Generate the final SAIR cheat sheet using the trained distiller.

Pipeline:
1. Load labeled training problems.
2. Run the offline pipeline on each one to collect evidence.
3. Cluster problems by a simple structural key (variable count + shape).
4. Ask the distiller to synthesize one compact lemma per cluster.
5. Assemble the sections into a ``CheatSheet`` and render to ≤10KB.
6. Write the result to ``sair/cheat_sheets/vN.md``.

Usage
-----
    python -m sair.scripts.generate_cheat_sheet
    python -m sair.scripts.generate_cheat_sheet --version v1 --max-clusters 8
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from sair.config import (
    REPO_DIR,
    SAIR_CHEATSHEETS_DIR,
    SAIR_INFERENCE_CONFIG,
)
from sair.data.load_problems import load_all_problems
from sair.graph import run_pipeline_sequential
from sair.schemas import (
    CheatSheet,
    CheatSheetEntry,
    EvidenceBundle,
    Problem,
)
from utils.paths import check_cwd

STATIC_CHEAT_SHEET_BODY: str = """You are deciding whether Equation 1 implies Equation 2 over all magmas.
A magma is a non-empty set with one binary operation * and no axioms.

OUTPUT:
- VERDICT: TRUE or FALSE
- REASONING: brief but explicit
- PROOF: required for TRUE
- COUNTEREXAMPLE: required for FALSE

FAST TRUE CHECKS:
- Eq2 is syntactically identical on both sides, or Eq2 is Eq1 up to side swap and variable renaming.
- Eq1 rewrites Eq2's left side to its right side by direct substitution instances.
- Eq1 forces a projection law such as a*b=b or a*b=a, and Eq2 simplifies under that law.

FAST FALSE CHECKS:
- A single finite magma satisfying Eq1 and violating Eq2 proves FALSE.
- If Eq1 is balanced in variable multiplicities but Eq2 is unbalanced, prioritize projection/XOR countermodels.
- If Eq2 changes leftmost or rightmost leaf behavior not controlled by Eq1, try projection countermodels first.

COUNTERMODEL FAMILY:
- Left projection: x*y=x. Any term evaluates to its leftmost leaf.
- Right projection: x*y=y. Any term evaluates to its rightmost leaf.
- XOR on {0,1}: x*y=x+y mod 2. A law holds iff each variable has equal parity on both sides.
- Constant-zero: x*y=0. Useful when Eq1 collapses all compound terms but Eq2 exposes a variable.
- Left shift mod 3: x*y=x+1 mod 3. Tracks left spine depth.
- Right shift mod 3: x*y=y+1 mod 3. Tracks right spine depth.
- Boolean AND/OR on {0,1}. Useful for absorption/idempotence-shaped laws.
- Near-projections on {0,1,2}: start from x*y=x or x*y=y and perturb one row/column entry.

PROOF STRATEGY:
1. Parse terms as binary trees.
2. Canonicalize variables by first appearance and allow equation side swap.
3. Instantiate Eq1 as a rewrite rule in both directions.
4. Try to rewrite both sides of Eq2 to a common term.
5. If Eq1 extracts an operation law, simplify Eq2 under that law.

FALSE VALIDATION:
- Give the domain and full table.
- State that Eq1 holds for all assignments.
- Give one assignment where Eq2 fails and show the two evaluated values.

CAUTION:
- Do not conclude TRUE from "no small counterexample".
- Do not conclude TRUE from one operation form unless Eq1 proves that form.
- Prefer FALSE only with a verified table; prefer TRUE only with a derivation."""


def _structural_key(problem: Problem) -> str:
    """Cheap clustering key based on variable count and shape hashes."""
    e1: str = problem.equation1
    e2: str = problem.equation2
    n_vars: int = len({c for c in (e1 + e2) if c.isalpha()})
    shape1: int = e1.count("*")
    shape2: int = e2.count("*")
    return f"vars{n_vars}_m{shape1}_n{shape2}"


def _cluster_problems(
    problems: list[Problem],
    *,
    max_clusters: int,
    max_per_cluster: int,
) -> dict[str, list[Problem]]:
    """Group problems by structural key with balanced TRUE/FALSE representation.

    Half the cluster budget is allocated to TRUE-majority clusters and the
    other half to FALSE-majority clusters so the cheat sheet does not bias
    the downstream judge toward either verdict.
    """
    if max_clusters <= 0:
        print("[cheatsheet] static-only mode: no learned clusters selected")
        return {}

    buckets: dict[str, list[Problem]] = defaultdict(list)
    for p in problems:
        buckets[_structural_key(p)].append(p)

    # Split clusters by majority verdict.
    true_clusters: list[tuple[str, list[Problem]]] = []
    false_clusters: list[tuple[str, list[Problem]]] = []
    for key, items in buckets.items():
        n_true = sum(1 for p in items if p.answer is True)
        n_false = sum(1 for p in items if p.answer is False)
        if n_true >= n_false:
            true_clusters.append((key, items))
        else:
            false_clusters.append((key, items))

    # Sort each group by bucket size (larger = more representative).
    true_clusters.sort(key=lambda kv: len(kv[1]), reverse=True)
    false_clusters.sort(key=lambda kv: len(kv[1]), reverse=True)

    half: int = max(1, max_clusters // 2)
    selected: dict[str, list[Problem]] = {}
    for key, items in true_clusters[:half]:
        selected[key] = items[:max_per_cluster]
    for key, items in false_clusters[:half]:
        selected[key] = items[:max_per_cluster]

    print(
        f"[cheatsheet] cluster split: {len(true_clusters[:half])} TRUE-majority, "
        f"{len(false_clusters[:half])} FALSE-majority"
    )
    return selected


def _render_cluster_title(key: str, problems: list[Problem]) -> str:
    """Human-readable section title."""
    return f"Lemma pack {key} ({len(problems)} problems)"


def _collect_evidences(
    problems: list[Problem],
    *,
    use_retriever: bool,
    cluster_key: str,
) -> list[tuple[Problem, EvidenceBundle]]:
    """Run the offline pipeline on every problem in a cluster."""
    out: list[tuple[Problem, EvidenceBundle]] = []
    for prob in tqdm(
        problems,
        desc=f"[cheatsheet] evidence {cluster_key}",
        unit="problem",
        dynamic_ncols=True,
        leave=False,
    ):
        bundle: EvidenceBundle = run_pipeline_sequential(
            prob, use_retriever=use_retriever
        )
        out.append((prob, bundle))
    return out


def _priority_for(cluster_problems: list[Problem]) -> int:
    """Higher priority for larger clusters so they survive the byte cap."""
    return len(cluster_problems)


def generate(
    *,
    version: str,
    max_clusters: int,
    max_per_cluster: int,
    use_retriever: bool,
    max_bytes: int,
    entry_max_bytes: int,
    entry_max_new_tokens: int,
    entry_max_examples: int,
    entry_max_evidence_chars: int,
) -> Path:
    """Build the cheat sheet and write it to disk. Returns the output path."""
    SAIR_CHEATSHEETS_DIR.mkdir(parents=True, exist_ok=True)

    problems: list[Problem] = [
        p for p in load_all_problems() if p.answer is not None
    ]
    print(f"[cheatsheet] loaded {len(problems)} labeled problems")

    clusters: dict[str, list[Problem]] = _cluster_problems(
        problems,
        max_clusters=max_clusters,
        max_per_cluster=max_per_cluster,
    )
    print(f"[cheatsheet] selected {len(clusters)} clusters")

    cfg: SAIR_INFERENCE_CONFIG = SAIR_INFERENCE_CONFIG()
    adapter_label = "none"
    model = None
    tokenizer = None
    synthesize_cheat_sheet_entry = None
    if clusters:
        from sair.agents.distiller import (
            load_distiller_model,
            synthesize_cheat_sheet_entry,
        )

        print("[cheatsheet] loading distiller model...")
        model, tokenizer, adapter_label = load_distiller_model(cfg=cfg)
        print(f"[cheatsheet] distiller adapter: {adapter_label}")

    entries: list[CheatSheetEntry] = [
        CheatSheetEntry(
            title="MAGMA IMPLICATION CHEAT SHEET",
            body=STATIC_CHEAT_SHEET_BODY,
            priority=999,
        )
    ]

    cluster_items = list(clusters.items())
    for key, cluster in tqdm(
        cluster_items,
        desc="[cheatsheet] clusters",
        unit="cluster",
        dynamic_ncols=True,
    ):
        tqdm.write(f"[cheatsheet] cluster {key}: {len(cluster)} problems")
        collected: list[tuple[Problem, EvidenceBundle]] = _collect_evidences(
            cluster,
            use_retriever=use_retriever,
            cluster_key=key,
        )
        evidence_texts: list[str] = [
            b.rendered_reasoning() or "No evidence." for _, b in collected
        ]
        if synthesize_cheat_sheet_entry is None:
            raise RuntimeError("cheat-sheet synthesizer was not loaded")
        body: str = synthesize_cheat_sheet_entry(
            title=key,
            cluster_problems=[p for p, _ in collected],
            cluster_evidences=evidence_texts,
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            max_bytes=int(entry_max_bytes),
            max_new_tokens=int(entry_max_new_tokens),
            max_examples=int(entry_max_examples),
            max_evidence_chars=int(entry_max_evidence_chars),
            show_progress=True,
        )
        entries.append(
            CheatSheetEntry(
                title=_render_cluster_title(key, cluster),
                body=body,
                priority=_priority_for(cluster),
            )
        )

    cheat_sheet: CheatSheet = CheatSheet(
        version=version,
        entries=entries,
        metrics={
            "n_clusters": len(entries),
            "adapter": adapter_label,
        },
    )
    rendered: str = cheat_sheet.render(max_bytes=max_bytes)
    out_path: Path = SAIR_CHEATSHEETS_DIR / f"{version}.md"
    out_path.write_text(rendered, encoding="utf-8")

    size_bytes: int = len(rendered.encode("utf-8"))
    print(f"[cheatsheet] wrote {out_path} ({size_bytes} bytes)")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAIR cheat sheet.")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--max-per-cluster", type=int, default=6)
    parser.add_argument("--no-retriever", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=10_000)
    parser.add_argument("--entry-max-bytes", type=int, default=900)
    parser.add_argument("--entry-max-new-tokens", type=int, default=192)
    parser.add_argument("--entry-max-examples", type=int, default=3)
    parser.add_argument("--entry-max-evidence-chars", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = _parse_args()
    generate(
        version=str(args.version),
        max_clusters=int(args.max_clusters),
        max_per_cluster=int(args.max_per_cluster),
        use_retriever=not bool(args.no_retriever),
        max_bytes=int(args.max_bytes),
        entry_max_bytes=int(args.entry_max_bytes),
        entry_max_new_tokens=int(args.entry_max_new_tokens),
        entry_max_examples=int(args.entry_max_examples),
        entry_max_evidence_chars=int(args.entry_max_evidence_chars),
    )


if __name__ == "__main__":
    check_cwd(expected_dir=REPO_DIR)
    main()
