"""
Modulo: tune_best_hybrid (Script)
Ibrido (miglior EVM dell'iterazione, greedy_minimax) con pesi cercati in un intervallo stretto.
Si esegue SOLO se il miglior albero EVM ha caso peggiore >= --min-worst (default 8): una
strategia aggressiva migliora con un piccolo malus sul bucket peggiore.
Scrive MODELS_DIR/hybrid_best.json: {"skipped": true, ...} oppure i pesi trovati.
Limite: i pesi sono scelti sugli stessi 5040 segreti usati per valutare (stima ottimistica).

Uso: python -m experiments.tune_best_hybrid [--lo 0.90] [--hi 1.0] [--min-worst 8]
"""
import argparse
import json

from experiments.tune_hybrid_weights import run_adaptive_search
from src.analysis.best_model import pick_best_model
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.paths import MODELS_DIR, TREES_DIR
from src.strategies.zoo import build_base_strategies


def main() -> None:
    parser = argparse.ArgumentParser(description="Pesi dell'ibrido (miglior EVM, minimax).")
    parser.add_argument("--lo", type=float, default=0.90)
    parser.add_argument("--hi", type=float, default=1.0)
    parser.add_argument("--coarse-step", type=float, default=0.02)
    parser.add_argument("--zoom-step", type=float, default=0.005)
    parser.add_argument("--micro-step", type=float, default=0.001)
    parser.add_argument("--min-worst", type=int, default=8)
    args = parser.parse_args()

    out = MODELS_DIR / "hybrid_best.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    kind, stats = pick_best_model(TREES_DIR)
    print(f"Miglior modello: {kind} (media {stats['avg']:.5f}, caso peggiore {stats['worst']})")
    if stats["worst"] < args.min_worst:
        out.write_text(json.dumps({"skipped": True, "base_model": kind,
                                   "reason": f"caso peggiore {stats['worst']} < {args.min_worst}"}, indent=2))
        print("Caso peggiore già basso: ibrido non necessario.")
        return

    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    base = build_base_strategies(all_perms, lookup, MODELS_DIR, strict=True)

    champ_avg, champ_def = run_adaptive_search(
        base[f"estimated_evm_{kind}"], base["greedy_minimax"], all_perms,
        args.lo, args.hi, args.coarse_step, args.zoom_step, args.micro_step,
    )
    result = {
        "skipped": False,
        "base_model": kind,
        "weight_a": champ_avg["weight_a"],
        "weight_b": champ_avg["weight_b"],
        "avg": champ_avg["avg_moves"],
        "worst": champ_avg["worst_case"],
        "champion_pareto_defensive": champ_def,
        "search": {"lo": args.lo, "hi": args.hi},
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Pesi: {kind} {result['weight_a']:.4f} / minimax {result['weight_b']:.4f} "
          f"-> media {result['avg']:.5f}, caso peggiore {result['worst']}")


if __name__ == "__main__":
    main()