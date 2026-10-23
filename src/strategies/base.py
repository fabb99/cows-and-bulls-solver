"""
Modulo: base
Interfaccia astratta delle strategie, con il flusso di scelta comune (template method).
"""
from abc import ABC, abstractmethod
from typing import ClassVar

from src.core.lookup_table import FeedbackLookup
from src.core.signatures import get_representatives
from src.strategies.selection import select_best


class BaseStrategy(ABC):
    """
    Una sottoclasse implementa `evaluate_candidates` e dichiara `higher_is_better`.
    Può ridefinire `candidate_pool` o `choose_guess` se il flusso comune non basta.
    """

    higher_is_better: ClassVar[bool]

    def __init__(self, all_permutations: list[str], lookup: FeedbackLookup) -> None:
        """PRE: all_permutations è l'universo delle mosse; lookup copre quel dominio."""
        self.all_permutations = all_permutations
        self.lookup = lookup

    @abstractmethod
    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        """
        PRE:  candidates_to_test e possible_solutions non vuoti.
        POST: {candidato: punteggio}. Il verso del punteggio è `higher_is_better`.
        """

    def candidate_pool(self, possible_solutions: list[str], history: list[str]) -> list[str]:
        """
        POST: mosse da valutare. Al Turno 1 (history vuota e universo intero) tutte le
              mosse sono equivalenti per simmetria, quindi basta un rappresentante;
              altrimenti tutto l'universo.
        """
        if not history and len(possible_solutions) == len(self.all_permutations):
            return get_representatives(self.all_permutations, history, possible_solutions)
        return self.all_permutations

    def choose_guess(
        self, possible_solutions: list[str], history: list[str] | None = None
    ) -> str:
        """
        PRE:  possible_solutions non vuota.
        POST: con 1 o 2 soluzioni, la prima (con 2 è già ottimo: 1.5 mosse attese);
              altrimenti il candidato migliore secondo `select_best`.
        """
        if not possible_solutions:
            raise ValueError("possible_solutions è vuota")
        if len(possible_solutions) <= 2:
            return possible_solutions[0]
        pool = self.candidate_pool(possible_solutions, history or [])
        scores = self.evaluate_candidates(pool, possible_solutions)
        return select_best(scores, possible_solutions, self.higher_is_better)