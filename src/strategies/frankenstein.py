"""
Modulo: frankenstein
FrankensteinStrategy: segue una RICETTA {sequenza di feedback: nome di una strategia}.
A ogni turno la chiave è la sequenza dei feedback ricevuti finora; la strategia nominata
sceglie la mossa sullo stato reale. Non memorizza mosse: funziona con qualsiasi prima mossa
(es. casuale) e con qualsiasi segreto. Chiavi assenti -> strategia di fallback.
"""
import json
from collections.abc import Sequence
from pathlib import Path

from src.core.feedback import Feedback
from src.core.lookup_table import FeedbackLookup
from src.strategies.base import BaseStrategy

RECIPE_STYLES = ("best_average", "pareto_defensive")
START_KEY = "START"


def path_key(feedbacks: Sequence[Feedback]) -> str:
    """POST: 'START' se vuota, altrimenti i feedback 'b,c' uniti da '|' (es. '0,1|1,0')."""
    if not feedbacks:
        return START_KEY
    return "|".join(f"{b},{c}" for b, c in feedbacks)


class FrankensteinStrategy(BaseStrategy):
    higher_is_better = False

    def __init__(
        self,
        all_permutations: list[str],
        lookup: FeedbackLookup,
        recipe: dict[str, str],
        strategies: dict[str, BaseStrategy],
        fallback_name: str = "max_entropy",
    ) -> None:
        """
        PRE:  recipe = {path_key: nome}; ogni nome (e fallback_name) è in strategies.
        POST: ValueError se la ricetta nomina strategie sconosciute.
        """
        super().__init__(all_permutations, lookup)
        unknown = (set(recipe.values()) | {fallback_name}) - set(strategies)
        if unknown:
            raise ValueError(f"Strategie non disponibili: {sorted(unknown)}")
        self.recipe = recipe
        self.strategies = strategies
        self.fallback_name = fallback_name

    @classmethod
    def from_file(
        cls,
        path: Path,
        style: str,
        all_permutations: list[str],
        lookup: FeedbackLookup,
        strategies: dict[str, BaseStrategy],
        fallback_name: str = "max_entropy",
    ) -> "FrankensteinStrategy":
        """PRE: path è un recipe.json valido; style in RECIPE_STYLES."""
        if style not in RECIPE_STYLES:
            raise ValueError(f"style deve essere uno di {RECIPE_STYLES}")
        with open(path, encoding="utf-8") as file:
            recipe = json.load(file)[style]
        return cls(all_permutations, lookup, recipe, strategies, fallback_name)

    def feedback_history(self, possible_solutions: list[str], history: list[str]) -> list[Feedback]:
        """
        PRE:  possible_solutions è coerente con history (tutte compatibili con ogni mossa).
        POST: i feedback ricevuti, uno per mossa giocata.
        """
        if not history:
            return []
        sample = possible_solutions[0]
        return [self.lookup.get_feedback(move, sample) for move in history]

    def choose_guess(
        self, possible_solutions: list[str], history: list[str] | None = None
    ) -> str:
        """
        PRE:  possible_solutions non vuota.
        POST: con N <= 2 la prima soluzione; altrimenti la mossa della strategia che la
              ricetta assegna alla sequenza di feedback (o del fallback).
        """
        if not possible_solutions:
            raise ValueError("possible_solutions è vuota")
        if len(possible_solutions) <= 2:
            return possible_solutions[0]
        history = history or []
        key = path_key(self.feedback_history(possible_solutions, history))
        name = self.recipe.get(key, self.fallback_name)
        return self.strategies[name].choose_guess(possible_solutions, history)

    def evaluate_candidates(self, candidates_to_test, possible_solutions):
        raise NotImplementedError("FrankensteinStrategy non valuta candidati: segue la ricetta")