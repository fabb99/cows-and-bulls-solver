"""
Modulo: dataset_refiner (Script)
Fase 1b: perfeziona gli stati di raw_states.json.
- N <= max_exact_n: costo ESATTO con PureDeepEVMStrategy (is_exact=True).
- max_exact_n < N <= max_tournament_n: mini-torneo tra strategie; vince chi costruisce
  il sotto-albero più economico (resta un upper bound, is_exact=False).

Uso: python -m experiments.oracle_lab.dataset_refiner [--max-exact-n 10] [--max-tournament-n 1500]
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from experiments.oracle_lab.config import COST_TOL, EXACT_SOURCE, RAW_STATES_FILEPATH
from experiments.oracle_lab.models import RawState
from experiments.oracle_lab.storage import load_raw_states, save_raw_states
from src.analysis.metrics import average_depth
from src.analysis.tree_builder import build_decision_tree
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.strategies.base import BaseStrategy
from src.strategies.deep_evm import PureDeepEVMStrategy
from src.strategies.entropy import EntropyStrategy
from src.strategies.greedy_minimax import GreedyMinimaxStrategy

SAVE_INTERVAL: int = 25


@dataclass
class RefineStats:
    newly_exact: int = 0  # stati diventati esatti in questa esecuzione
    improved: int = 0  # stati il cui costo è sceso


def evaluate_state_tournament(
    solutions: list[str],
    competitors: dict[str, BaseStrategy],
    current_best_cost: float,
) -> tuple[float, Optional[str]]:
    """
    PRE:  solutions ha N >= 3; competitors = {nome: strategia pronta}.
    POST: (costo, nome) della strategia che batte strettamente current_best_cost,
          oppure (current_best_cost, None) se nessuna lo batte.
    """
    best_cost, winner = current_best_cost, None
    for name, strategy in competitors.items():
        cost = average_depth(build_decision_tree(strategy, solutions))
        if cost < best_cost - COST_TOL:
            best_cost, winner = cost, name
    return best_cost, winner


def refine_states(
    raw_states: dict[str, RawState],
    exact_strategy: PureDeepEVMStrategy,
    competitors: dict[str, BaseStrategy],
    max_exact_n: int,
    max_tournament_n: int,
    save_path: Path = RAW_STATES_FILEPATH,
    save_interval: int = SAVE_INTERVAL,
) -> RefineStats:
    """
    PRE:  raw_states è il dizionario della Fase 1.
    POST: aggiorna raw_states in-place e lo salva periodicamente e alla fine.
          Solleva RuntimeError se un costo noto risulta INFERIORE al minimo esatto:
          è impossibile, e indica un bug (nel calcolo esatto o nell'albero sorgente).
    """
    stats = RefineStats()
    todo = sorted(
        (s for s in raw_states.values() if 3 <= s.n <= max_tournament_n), key=lambda s: s.n
    )  # prima gli stati piccoli: i loro costi esatti aiutano quelli grandi

    for i, state in enumerate(tqdm(todo, desc="Raffinamento stati")):
        if state.n <= max_exact_n:
            if not state.is_exact:
                exact = exact_strategy.exact_cost(state.solutions)
                if exact > state.best_cost + COST_TOL:
                    raise RuntimeError(
                        f"Costo noto {state.best_cost:.6f} < minimo esatto {exact:.6f} "
                        f"(N={state.n}, sorgente {state.source_tree}): bug"
                    )
                if exact < state.best_cost - COST_TOL:
                    stats.improved += 1
                state.best_cost = exact
                state.source_tree = EXACT_SOURCE
                state.is_exact = True
                stats.newly_exact += 1
                exact_strategy.precomputed_tablebase[state.state_id] = exact
        elif not state.is_exact:
            cost, winner = evaluate_state_tournament(state.solutions, competitors, state.best_cost)
            if winner is not None:
                state.best_cost, state.source_tree = cost, winner
                stats.improved += 1

        if (i + 1) % save_interval == 0:
            save_raw_states(raw_states, save_path)

    save_raw_states(raw_states, save_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Raffina i costi degli stati.")
    parser.add_argument("--max-exact-n", type=int, default=10)
    parser.add_argument("--max-tournament-n", type=int, default=1500)
    args = parser.parse_args()

    raw_states = load_raw_states(RAW_STATES_FILEPATH)
    if not raw_states:
        print(f"{RAW_STATES_FILEPATH} vuoto o assente: esegui prima data_miner.")
        return

    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])  # costruzione anticipata
    entropy = EntropyStrategy(all_perms, lookup)

    tablebase = {sid: s.best_cost for sid, s in raw_states.items() if s.is_exact}
    exact_strategy = PureDeepEVMStrategy(all_perms, lookup, entropy, tablebase)
    competitors: dict[str, BaseStrategy] = {
        "entropy": entropy,
        "minimax": GreedyMinimaxStrategy(all_perms, lookup),
    }

    stats = refine_states(
        raw_states, exact_strategy, competitors, args.max_exact_n, args.max_tournament_n
    )
    n_exact = sum(s.is_exact for s in raw_states.values())
    print(f"Nuovi stati esatti: {stats.newly_exact} | costi migliorati: {stats.improved}")
    print(f"Stati esatti totali: {n_exact} / {len(raw_states)}")


if __name__ == "__main__":
    main()