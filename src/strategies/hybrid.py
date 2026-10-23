"""
Modulo: hybrid
Strategia ibrida: somma pesata dei punteggi normalizzati di due strategie.
"""
import math

from src.strategies.base import BaseStrategy


class HybridWeightedStrategy(BaseStrategy):
    """
    Ogni strategia produce punteggi che vengono normalizzati in (0, 1] (1 = il migliore),
    poi combinati: weight_a * norm_a + weight_b * norm_b. Più alto è meglio.
    """

    higher_is_better = True

    def __init__(
        self,
        strategy_a: BaseStrategy,
        strategy_b: BaseStrategy,
        weight_a: float,
        weight_b: float,
    ) -> None:
        """
        PRE:  le due strategie condividono lookup e universo; pesi >= 0 con somma > 0
              (non serve che sommino a 1: contano solo i rapporti).
        """
        super().__init__(strategy_a.all_permutations, strategy_a.lookup)
        if strategy_b.lookup is not strategy_a.lookup:
            raise ValueError("Le due strategie devono condividere la stessa FeedbackLookup")
        if weight_a < 0 or weight_b < 0 or weight_a + weight_b <= 0:
            raise ValueError("Pesi non validi")
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.weight_a = weight_a
        self.weight_b = weight_b

    @staticmethod
    def _normalize_scores(scores: dict[str, float], is_higher_better: bool) -> dict[str, float]:
        """
        POST: il migliore vale 1.0, gli altri sono in [0, 1].
              higher_better: s / max.  lower_better: min / s  (un costo inf vale 0).
        PRE (lower_better): tutti i punteggi > 0, altrimenti ValueError.
        """
        if is_higher_better:
            best = max(scores.values())
            if best <= 0:  # tutti i punteggi nulli: nessuna preferenza
                best = 1.0
            return {c: s / best for c, s in scores.items()}

        best = min(scores.values())
        if best <= 0 or math.isinf(best):
            raise ValueError(f"Punteggio non normalizzabile (migliore = {best})")
        return {c: best / s for c, s in scores.items()}

    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        """POST: {candidato: punteggio ibrido in [0, weight_a + weight_b]}."""
        norm_a = self._normalize_scores(
            self.strategy_a.evaluate_candidates(candidates_to_test, possible_solutions),
            self.strategy_a.higher_is_better,
        )
        norm_b = self._normalize_scores(
            self.strategy_b.evaluate_candidates(candidates_to_test, possible_solutions),
            self.strategy_b.higher_is_better,
        )
        return {
            c: norm_a[c] * self.weight_a + norm_b[c] * self.weight_b
            for c in candidates_to_test
        }