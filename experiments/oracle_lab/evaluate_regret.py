"""
Modulo: evaluate_regret (Script)
Giudice finale di un oracolo (estimator): su stati con costo ESATTO noto, confronta la
mossa che l'oracolo sceglierebbe con la mossa ottima.

    regret(S) = costo_vero(mossa scelta dall'oracolo) - costo_esatto(S)  >= 0

Il costo vero di una mossa usa i costi esatti dei bucket. Per questo gli stati valutati
hanno N <= --max-n (default 10): i loro bucket sono piccoli e il costo esatto è calcolabile.

Limiti: (1) gli stati piccoli sono nel training dell'endgame: per un test fuori campione
passare con --states i raw_states di un'altra iterazione; (2) con max-n piccolo si
valuta solo l'endgame, non l'apertura.

Uso: python -m experiments.oracle_lab.evaluate_regret --models legacy_piecewise data/runs/iter_00/models/selected.json
"""
import argparse
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from experiments.oracle_lab.config import RAW_STATES_FILEPATH
from experiments.oracle_lab.storage import load_raw_states
from src.core.engine import calculate_buckets
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.estimators.base import BaseBucketEstimator
from src.estimators.segmented import SegmentedLinearEstimator
from src.strategies.costs import expected_cost
from src.strategies.deep_evm import PureDeepEVMStrategy
from src.strategies.entropy import EntropyStrategy
from src.strategies.selection import select_best

NEGATIVE_TOL = 1e-9
N_BINS = [(3, 5), (6, 10), (11, 20), (21, 10**9)]


@dataclass(frozen=True)
class StateRegret:
    n: int
    optimal: float
    chosen_true: float
    guess: str

    @property
    def regret(self) -> float:
        return max(self.chosen_true - self.optimal, 0.0)


def evaluate_state(
    solutions: list[str],
    estimator: BaseBucketEstimator,
    lookup: FeedbackLookup,
    all_perms: list[str],
    exact_cost: Callable[[list[str]], float],
    optimal: float,
) -> StateRegret:
    """
    PRE:  solutions ha N >= 3 e costo esatto `optimal` (= exact_cost(solutions)).
          exact_cost è esatto per ogni sottoinsieme proprio di solutions.
    POST: mossa scelta dall'oracolo (stesso tie-break delle strategie) e suo costo vero.
          Solleva RuntimeError se il costo vero è inferiore all'ottimo: è impossibile (bug).
    """
    n = len(solutions)
    scores = {
        g: expected_cost(calculate_buckets(g, solutions, lookup), n, estimator.estimate_cost)
        for g in all_perms
    }
    guess = select_best(scores, solutions, higher_is_better=False)
    true_cost = expected_cost(calculate_buckets(guess, solutions, lookup), n, exact_cost)
    if true_cost < optimal - NEGATIVE_TOL:
        raise RuntimeError(f"Costo vero {true_cost:.6f} < ottimo {optimal:.6f} (N={n}): bug")
    return StateRegret(n, optimal, true_cost, guess)


def summarize(results: list[StateRegret]) -> list[dict]:
    """POST: per ogni fascia di N non vuota (e per il totale) conteggio, regret medio/massimo, % ottime."""
    rows = []
    for label, subset in [
        *((f"N {lo}-{hi}" if hi < 10**9 else f"N >= {lo}", [r for r in results if lo <= r.n <= hi])
          for lo, hi in N_BINS),
        ("TOTALE", results),
    ]:
        if not subset:
            continue
        regrets = [r.regret for r in subset]
        rows.append(
            {
                "bin": label,
                "states": len(subset),
                "mean_regret": sum(regrets) / len(regrets),
                "max_regret": max(regrets),
                "pct_optimal": 100 * sum(x <= 1e-9 for x in regrets) / len(regrets),
            }
        )
    return rows


def load_model(spec: str) -> SegmentedLinearEstimator:
    """PRE: spec è 'legacy_geometric', 'legacy_piecewise' o il percorso di un modello JSON."""
    if spec == "legacy_geometric":
        return SegmentedLinearEstimator.legacy_geometric()
    if spec == "legacy_piecewise":
        return SegmentedLinearEstimator.legacy_piecewise()
    return SegmentedLinearEstimator.load(Path(spec))


def print_summary(name: str, rows: list[dict]) -> None:
    print(f"\n=== {name} ===")
    print(f"{'Fascia':<10} | {'Stati':>6} | {'Regret medio':>12} | {'Regret max':>10} | {'Ottime %':>8}")
    for r in rows:
        print(f"{r['bin']:<10} | {r['states']:>6} | {r['mean_regret']:>12.5f} | "
              f"{r['max_regret']:>10.5f} | {r['pct_optimal']:>7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regret di policy di uno o più oracoli.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--states", type=Path, default=RAW_STATES_FILEPATH)
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--max-states", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="salva il riepilogo in JSON")
    args = parser.parse_args()

    raw = load_raw_states(args.states)
    pool = sorted(
        (s for s in raw.values() if s.is_exact and 3 <= s.n <= args.max_n),
        key=lambda s: s.state_id,
    )
    if not pool:
        raise SystemExit("Nessuno stato esatto utilizzabile: esegui prima dataset_refiner.")
    states = random.Random(args.seed).sample(pool, min(args.max_states, len(pool)))
    print(f"Stati valutati: {len(states)} (su {len(pool)} esatti con N <= {args.max_n})")

    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    tablebase = {sid: s.best_cost for sid, s in raw.items() if s.is_exact}
    exact = PureDeepEVMStrategy(all_perms, lookup, EntropyStrategy(all_perms, lookup), tablebase)

    optimal = {s.state_id: exact.exact_cost(s.solutions) for s in states}

    summary: dict[str, list[dict]] = {}
    for spec in args.models:
        model = load_model(spec)
        results = [
            evaluate_state(s.solutions, model, lookup, all_perms, exact.exact_cost, optimal[s.state_id])
            for s in tqdm(states, desc=spec)
        ]
        summary[spec] = summarize(results)
        print_summary(spec, summary[spec])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nRiepilogo salvato in {args.out}")


if __name__ == "__main__":
    main()