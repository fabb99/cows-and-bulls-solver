"""
Modulo: frankenstein
RoutedStrategy: gioca le mosse prescritte da una policy {state_hash: mossa},
costruita da experiments/train_router.py. Per stati fuori policy delega a una
strategia di fallback.
"""
import json
from pathlib import Path
from typing import Optional

from src.core.lookup_table import FeedbackLookup
from src.core.state import state_hash
from src.strategies.base import BaseStrategy

POLICY_STYLES = ("best_average", "pareto_defensive")


class RoutedStrategy(BaseStrategy):
    higher_is_better = False

    def __init__(
        self,
        all_permutations: list[str],
        lookup: FeedbackLookup,
        policy: dict[str, str],
        fallback: BaseStrategy,
    ) -> None:
        """
        PRE:  policy = {state_hash(soluzioni): mossa}; fallback è una strategia valida.
        """
        super().__init__(all_permutations, lookup)
        self.policy = policy
        self.fallback = fallback
        self._legal = set(all_permutations)

    @classmethod
    def from_file(
        cls,
        path: Path,
        style: str,
        all_permutations: list[str],
        lookup: FeedbackLookup,
        fallback: BaseStrategy,
    ) -> "RoutedStrategy":
        """PRE: path è un routing_policy.json valido; style in POLICY_STYLES."""
        if style not in POLICY_STYLES:
            raise ValueError(f"style deve essere uno di {POLICY_STYLES}")
        with open(path, "r", encoding="utf-8") as file:
            policy = json.load(file)[style]
        return cls(all_permutations, lookup, policy, fallback)

    def choose_guess(
        self, possible_solutions: list[str], history: Optional[list[str]] = None
    ) -> str:
        """
        PRE:  possible_solutions non vuota.
        POST: con N <= 2 la prima soluzione; altrimenti la mossa in policy per questo
              esatto insieme di soluzioni (se legale), altrimenti quella del fallback.
        """
        if not possible_solutions:
            raise ValueError("possible_solutions è vuota")
        if len(possible_solutions) <= 2:
            return possible_solutions[0]
        guess = self.policy.get(state_hash(possible_solutions))
        if guess is not None and guess in self._legal:
            return guess
        return self.fallback.choose_guess(possible_solutions, history)

    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        raise NotImplementedError("RoutedStrategy non valuta candidati: segue la policy")