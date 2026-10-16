"""
Modulo: storage (experiments/oracle_lab)
Lettura/scrittura dei dataset JSON. La scrittura è atomica (file temporaneo + rename):
un'interruzione a metà salvataggio non corrompe il file esistente.
"""
import json
import os
from pathlib import Path
from typing import Any

from experiments.oracle_lab.models import FeatureRecord, RawState


def _write_json_atomic(obj: Any, path: Path) -> None:
    """POST: `path` contiene obj in JSON compatto (nessuna indentazione: i file sono grandi)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(obj, file, separators=(",", ":"))
    os.replace(tmp, path)


def load_raw_states(path: Path) -> dict[str, RawState]:
    """POST: {state_id: RawState} dal file, oppure {} se il file non esiste."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {sid: RawState.from_dict(rec) for sid, rec in data.items()}


def save_raw_states(states: dict[str, RawState], path: Path) -> None:
    """POST: file sovrascritto in modo atomico."""
    _write_json_atomic({sid: s.to_dict() for sid, s in states.items()}, path)


def load_features(path: Path) -> list[FeatureRecord]:
    """POST: lista di FeatureRecord dal file, oppure [] se il file non esiste."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as file:
        return [FeatureRecord.from_dict(rec) for rec in json.load(file)]


def save_features(records: list[FeatureRecord], path: Path) -> None:
    """POST: file sovrascritto in modo atomico."""
    _write_json_atomic([r.to_dict() for r in records], path)