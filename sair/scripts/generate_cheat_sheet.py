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
from typing import Any

from utils.paths import check_cwd

from sair.agents.distiller import (
    load_distiller_model,
    synthesize_cheat_sheet_entry,
)
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
    """Human-readable section title showing the TRUE/FALSE split."""
    n_true = sum(1 for p in problems if p.answer is True)
    n_false = sum(1 for p in problems if p.answer is False)
    return f"Lemma pack {key} (T:{n_true} F:{n_false})"


def _collect_evidences(
    problems: list[Problem],
    *,
    use_retriever: bool,
) -> list[tuple[Problem, EvidenceBundle]]:
    """Run the offline pipeline on every problem in a cluster."""
    out: list[tuple[Problem, EvidenceBundle]] = []
    for prob in problems:
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

    print("[cheatsheet] loading distiller model...")
    cfg: SAIR_INFERENCE_CONFIG = SAIR_INFERENCE_CONFIG()
    model, tokenizer, adapter_label = load_distiller_model(cfg=cfg)
    print(f"[cheatsheet] distiller adapter: {adapter_label}")

    # Preamble entry: tells the judge how to read the cheat sheet.
    preamble_body: str = (
        "Each section below contains heuristics for a structural cluster of "
        "equational-implication problems over magmas (binary op `*`, no axioms).\n"
        "- TRUE-WHEN: conditions under which E1 logically implies E2.\n"
        "- FALSE-WHEN: conditions under which a small counterexample exists.\n"
        "- KEY-STEPS: proof sketch or counterexample construction.\n"
        "Use these heuristics as a guide, not a lookup table. "
        "Always verify with a short reasoning chain before issuing a VERDICT."
    )
    entries: list[CheatSheetEntry] = [
        CheatSheetEntry(title="HOW TO USE THIS CHEAT SHEET", body=preamble_body, priority=999)
    ]

    for key, cluster in clusters.items():
        print(f"[cheatsheet] cluster {key}: {len(cluster)} problems")
        collected: list[tuple[Problem, EvidenceBundle]] = _collect_evidences(
            cluster, use_retriever=use_retriever
        )
        evidence_texts: list[str] = [
            b.rendered_reasoning() or "No evidence." for _, b in collected
        ]
        body: str = synthesize_cheat_sheet_entry(
            title=key,
            cluster_problems=[p for p, _ in collected],
            cluster_evidences=evidence_texts,
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            max_bytes=1_200,
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
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = _parse_args()
    generate(
        version=str(args.version),
        max_clusters=int(args.max_clusters),
        max_per_cluster=int(args.max_per_cluster),
        use_retriever=not bool(args.no_retriever),
        max_bytes=int(args.max_bytes),
    )


if __name__ == "__main__":
    check_cwd(expected_dir=REPO_DIR)
    main()
