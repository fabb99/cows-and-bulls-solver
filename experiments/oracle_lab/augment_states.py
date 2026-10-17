"""
Modulo: augment_states (Script)
Aumenta raw_states con stati che gli alberi delle strategie buone non visitano.
  Modalità A: partite ε-greedy (entropia, con probabilità ε una mossa casuale).
  Modalità B: bucket di m mosse casuali NON giocate, per ogni stato visitato.
Ogni nuovo stato riceve come costo iniziale quello dell'albero entropia (upper bound
valido, is_exact=False); dataset_refiner lo migliorerà. Le quote per fascia di N
evitano che le fasce piccole (molto frequenti) dominino.

Uso: python -m experiments.oracle_lab.augment_states [--epsilon 0.15] [--m-buckets 5]
"""
import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from tqdm import tqdm

from experiments.oracle_lab.config import RAW_STATES_FILEPATH
from experiments.oracle_lab.models import RawState
from experiments.oracle_lab.storage import load_raw_states, save_raw_states
from src.core.constants import WIN_FEEDBACK
from src.core.engine import calculate_buckets, filter_solutions
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.state import state_hash, state_id
from src.strategies.base import BaseStrategy
from src.strategies.entropy import EntropyStrategy
from src.strategies.memoized import MemoizedStrategy

BINS: list[tuple[int, int]] = [(3, 10), (11, 40), (41, 150), (151, 500), (501, 1440)]
SOURCE_A = "augment_eps_game"
SOURCE_B = "augment_candidate_bucket"
_HISTORY = ["-"]  # history non vuota: evita la regola del Turno 1


def bin_index(n: int, bins: list[tuple[int, int]] = BINS) -> int | None:
    """POST: indice della fascia che contiene n, oppure None."""
    for i, (lo, hi) in enumerate(bins):
        if lo <= n <= hi:
            return i
    return None


class EntropyCostOracle:
    """Costo medio (mosse attese) dell'albero di una strategia su uno stato, memoizzato."""

    def __init__(self, strategy: BaseStrategy, lookup: FeedbackLookup) -> None:
        self.strategy = strategy
        self.lookup = lookup
        self._memo: dict[str, float] = {}

    def cost(self, solutions: list[str]) -> float:
        """
        PRE:  solutions non vuota.
        POST: stessa quantità di average_depth(build_decision_tree(strategy, solutions)):
              1 + sum_{bucket non vincente} P(bucket) * cost(bucket).
        """
        n = len(solutions)
        if n == 1:
            return 1.0
        key = state_hash(solutions)
        if key in self._memo:
            return self._memo[key]
        guess = solutions[0] if n == 2 else self.strategy.choose_guess(solutions, _HISTORY)
        total = 1.0
        for fb, bucket in calculate_buckets(guess, solutions, self.lookup).items():
            if fb != WIN_FEEDBACK:
                total += (len(bucket) / n) * self.cost(bucket)
        self._memo[key] = total
        return total


def play_epsilon_game(
    secret: str,
    strategy: BaseStrategy,
    all_perms: list[str],
    lookup: FeedbackLookup,
    epsilon: float,
    rng: random.Random,
    max_turns: int = 20,
) -> list[tuple[list[str], str]]:
    """
    PRE:  secret in all_perms; 0 <= epsilon <= 1.
    POST: traiettoria [(soluzioni prima della mossa, mossa giocata)], fino alla vittoria
          o a max_turns. Turno 1 = all_perms[0]; con N <= 2 la prima soluzione; altrimenti
          con probabilità epsilon una mossa casuale, altrimenti quella della strategia.
    """
    solutions, history = list(all_perms), []
    trajectory: list[tuple[list[str], str]] = []
    for _ in range(max_turns):
        if not history:
            guess = all_perms[0]
        elif len(solutions) <= 2:
            guess = solutions[0]
        elif rng.random() < epsilon:
            guess = rng.choice(all_perms)
        else:
            guess = strategy.choose_guess(solutions, history)
        trajectory.append((solutions, guess))
        history.append(guess)
        fb = lookup.get_feedback(guess, secret)
        if fb == WIN_FEEDBACK:
            break
        solutions = filter_solutions(guess, fb, solutions, lookup)
    return trajectory


@dataclass
class AugmentStats:
    games: int = 0
    added_a: int = 0
    added_b: int = 0
    added_by_bin: list[int] = field(default_factory=list)


def augment(
    raw_states: dict[str, RawState],
    strategy: BaseStrategy,
    oracle: EntropyCostOracle,
    all_perms: list[str],
    lookup: FeedbackLookup,
    *,
    quotas: list[int],
    bins: list[tuple[int, int]] = BINS,
    epsilon: float = 0.15,
    m_buckets: int = 5,
    max_n: int = 1440,
    max_games: int = 20000,
    seed: int = 0,
    progress: Callable[[], None] | None = None,
) -> AugmentStats:
    """
    PRE:  len(quotas) == len(bins); raw_states è modificato in-place.
    POST: aggiunge stati NUOVI (assenti da raw_states) con 3 <= N <= max_n, rispettando le
          quote per fascia; si ferma a quote piene o dopo max_games partite.
          Deterministico dato il seed.
    """
    rng = random.Random(seed)
    stats = AugmentStats(added_by_bin=[0] * len(bins))

    def try_add(solutions: list[str], source: str) -> bool:
        if len(solutions) > max_n:
            return False
        b = bin_index(len(solutions), bins)
        if b is None or stats.added_by_bin[b] >= quotas[b]:
            return False
        sid = state_id(solutions)
        if sid in raw_states:
            return False
        raw_states[sid] = RawState(list(solutions), oracle.cost(solutions), source)
        stats.added_by_bin[b] += 1
        return True

    for _ in range(max_games):
        if all(c >= q for c, q in zip(stats.added_by_bin, quotas, strict=True)):
            break
        secret = rng.choice(all_perms)
        stats.games += 1
        for solutions, guess in play_epsilon_game(
            secret, strategy, all_perms, lookup, epsilon, rng
        ):
            if len(solutions) < 3:
                continue
            stats.added_a += try_add(solutions, SOURCE_A)
            if m_buckets > 0:
                others = [g for g in rng.sample(all_perms, m_buckets + 1) if g != guess]
                for other in others[:m_buckets]:
                    for bucket in calculate_buckets(other, solutions, lookup).values():
                        if len(bucket) < len(solutions):
                            stats.added_b += try_add(bucket, SOURCE_B)
        if progress is not None:
            progress()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Aumenta raw_states con stati non visitati.")
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--m-buckets", type=int, default=5)
    parser.add_argument("--quota", type=int, default=300, help="nuovi stati per fascia di N")
    parser.add_argument("--max-n", type=int, default=1440)
    parser.add_argument("--max-games", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_states = load_raw_states(RAW_STATES_FILEPATH)
    before = len(raw_states)
    print(f"Stati già presenti: {before}")

    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    strategy = MemoizedStrategy(EntropyStrategy(all_perms, lookup))
    oracle = EntropyCostOracle(strategy, lookup)

    with tqdm(total=args.max_games, desc="Partite", unit=" partite") as pbar:
        stats = augment(
            raw_states, strategy, oracle, all_perms, lookup,
            quotas=[args.quota] * len(BINS), epsilon=args.epsilon, m_buckets=args.m_buckets,
            max_n=args.max_n, max_games=args.max_games, seed=args.seed,
            progress=pbar.update,
        )

    save_raw_states(raw_states, RAW_STATES_FILEPATH)
    print(f"\nPartite: {stats.games} | nuovi stati: {len(raw_states) - before} "
          f"(A: {stats.added_a}, B: {stats.added_b})")
    for (lo, hi), added in zip(BINS, stats.added_by_bin, strict=True):
        flag = "" if added >= args.quota else "  <- quota non raggiunta"
        print(f"  N {lo}-{hi}: {added}/{args.quota}{flag}")


if __name__ == "__main__":
    main()