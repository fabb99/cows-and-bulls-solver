"""
Modulo: routed (src/strategies)
Contiene RoutedStrategy: il "Frankenstein" definitivo.
Legge la Routing Policy da file e delega ogni mossa alla strategia specialista 
più forte per l'esatto storico dei feedback ricevuti finora.
"""
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
from math import inf

from src.strategies.base import BaseStrategy
from src.core.constants import DIM

DEFAULT_POLICY_FILEPATH = Path("data/models/routing_policy.json")


class RoutedStrategy(BaseStrategy):
    """
    Seleziona la strategia ottima in base allo storico dei feedback della partita corrente.
    """

    def __init__(
        self, 
        all_permutations: List[str], 
        lookup, 
        available_strategies: Dict[str, BaseStrategy],
        policy_filepath: Path = DEFAULT_POLICY_FILEPATH,
        style: str = "pareto_defensive", 
        fallback_strategy_name: str = "max_entropy"
    ):
        super().__init__(all_permutations, lookup)
        self.available_strategies = available_strategies
        self.fallback_strategy_name = fallback_strategy_name
        self.style = style
        
        # Carica il JSON e prende SOLO il ramo dello stile selezionato:
        full_policy = self._load_policy(policy_filepath)
        self.routing_policy = full_policy.get(style, full_policy)

    def _load_policy(self, filepath: Path) -> Dict[str, str]:
        """Carica il JSON della policy."""
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _deduce_feedback_history(self, possible_solutions: List[str], history_moves: List[str]) -> List[Tuple[int, int]]:
        """
        Deduce tutti i feedback ricevuti finora confrontando la prima soluzione valida
        con la lista delle mosse giocate.
        """
        if not history_moves or not possible_solutions:
            return []
            
        secret_sample = possible_solutions[0]
        feedbacks = []
        for move in history_moves:
            fb = self.lookup.get_feedback(move, secret_sample)
            feedbacks.append(fb)
            
        return feedbacks

    def choose_guess(self, possible_solutions: List[str], history: List[str] = None) -> str:
        """
        Seleziona il "Cervello" giusto per questo specifico turno e delega a lui la mossa.
        """
        if history is None:
            history = []
            
        # 1. Determina lo storico dei feedback (la nostra "posizione" nell'albero)
        if len(history) == 0:
            history_key = "START"
        else:
            feedback_list = self._deduce_feedback_history(possible_solutions, history)
            history_key = " -> ".join([str(fb) for fb in feedback_list])

        # 2. Cerca la strategia vincente nel file JSON
        #    Se non esiste (un ramo mai esplorato o fuori policy), usa il fallback (es. la strategia dell'ultimo turno noto)
        active_strategy_name = self.routing_policy.get(history_key, self.fallback_strategy_name)
        
        # Assicuriamoci che esista nel nostro dizionario
        if active_strategy_name not in self.available_strategies:
            active_strategy_name = self.fallback_strategy_name
            
        active_strategy = self.available_strategies[active_strategy_name]
        
        # 3. DELEGA COMPLETA AL CAMPIONE DI QUESTO RAMO
        return active_strategy.choose_guess(possible_solutions, history)

    def evaluate_candidates(self, candidates_to_test: List[str], possible_solutions: List[str]) -> Dict[str, float]:
        """
        Metodo pass-through per mantenere compatibilità con eventuali test esterni.
        Restituisce semplicemente 0 per tutte, poiché la scelta la deleghiamo totalmente.
        """
        return {c: 0.0 for c in candidates_to_test}