"""
Modulo: models (experiments/oracle_lab)
Strutture dati tipizzate della pipeline dell'Oracolo:
- RawState: stato estratto dagli alberi e raffinato dal Deep EVM.
- FeatureRecord: stato arricchito con le feature del registry, pronto per il training.
"""
from dataclasses import asdict, dataclass
from typing import Any

from experiments.oracle_lab.config import EXACT_SOURCE
from src.core.state import state_id as compute_state_id


@dataclass(slots=True)
class RawState:
    """
    Attributi:
        solutions:   combinazioni ancora compatibili.
        best_cost:   miglior costo medio noto (mosse attese per chiudere lo stato).
        source_tree: chi ha prodotto best_cost (nome di albero/strategia o EXACT_SOURCE).
        is_exact:    True se best_cost è il minimo esatto di Bellman; False se è solo
                     un upper bound raggiunto da una strategia.
    """

    solutions: list[str]
    best_cost: float
    source_tree: str
    is_exact: bool = False

    @property
    def n(self) -> int:
        return len(self.solutions)

    @property
    def state_id(self) -> str:
        return compute_state_id(self.solutions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawState":
        """
        PRE:  data ha almeno solutions, best_cost, source_tree.
        POST: i dati legacy (senza is_exact) risultano esatti solo se source_tree == EXACT_SOURCE.
        """
        data = dict(data)
        data.setdefault("is_exact", data.get("source_tree") == EXACT_SOURCE)
        return cls(**data)


@dataclass(slots=True)
class FeatureRecord:
    """
    Attributi:
        state_id:    chiave canonica dello stato.
        n:           numero di soluzioni.
        features:    {nome: valore}, con nomi del registry src.features.bucket_features.
        target_cost: costo da predire (best_cost dello stato).
        is_exact:    True se target_cost è esatto, False se upper bound.
        source_tree: provenienza del target.
    """

    state_id: str
    n: int
    features: dict[str, float]
    target_cost: float
    is_exact: bool
    source_tree: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureRecord":
        return cls(**data)