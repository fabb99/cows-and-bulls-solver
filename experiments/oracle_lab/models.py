"""
Modulo: models (experiments/oracle_lab)
Definisce le strutture dati tipizzate (Dataclass) per la pipeline dell'Oracolo:
- RawState: Lo stato grezzo estratto dagli alberi e raffinato dal Deep EVM (Fasi 1 e 2).
- FeatureRecord: Il record arricchito per l'analisi statistica e il training (Fasi 3 e 4).
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass(slots=True)
class RawState:
    """
    Rappresenta uno stato di gioco estratto o raffinato.
    
    Attributi:
        solutions: La lista delle combinazioni compatibili rimaste.
        best_cost: Il miglior costo medio reale registrato (M).
        source_tree: Il nome della strategia che ha generato questo costo (es. 'max_entropy').
    """
    solutions: List[str]
    best_cost: float
    source_tree: str

    @property
    def n(self) -> int:
        """Proprietà calcolata: Ritorna len(self.solutions)."""
        return len(self.solutions)

    @property
    def state_id(self) -> str:
        """Proprietà calcolata: Ritorna la chiave stringa canonica ordinata (soluzioni unite da virgola)."""
        return ",".join(sorted(self.solutions))

    def to_dict(self) -> Dict[str, Any]:
        """Serializzazione: Converte l'istanza in un dizionario pronto per json.dump."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawState":
        """Deserializzazione: Ricostruisce un'istanza di RawState da un dizionario JSON."""
        return cls(**data)

@dataclass(slots=True)
class FeatureRecord:
    """
    Rappresenta un campione arricchito pronto per il Machine Learning (EDA e Regressione).
    
    Attributi:
        state_id: L'identificatore univoco dello stato.
        n: Numero di soluzioni (dimensione del gruppo).
        residual_entropy: Incertezza residua (log2(N) - H).
        n_max: Dimensione del bucket peggiore della mossa migliore.
        target_cost: Il costo reale da prevedere (M).
        source_tree: Strategia di provenienza del costo.
    """
    state_id: str
    n: int
    residual_entropy: float
    n_max: int
    target_cost: float
    source_tree: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializzazione: Converte in dizionario per json.dump."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureRecord":
        """Deserializzazione: Ricostruisce un'istanza da un dizionario JSON."""
        return cls(**data)