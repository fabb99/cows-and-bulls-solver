"""
Modulo: feature_builder (Script)
Fase 2: per ogni stato di raw_states.json calcola TUTTE le feature del registry
(src.features.bucket_features) e salva il dataset in FEATURES_DATASET_FILEPATH.
raw_states è la fonte di verità per target, is_exact e sorgente; le feature di un
record già calcolate non vengono ricalcolate (si calcolano solo quelle mancanti).

Uso: python -m experiments.oracle_lab.feature_builder
"""
from tqdm import tqdm

from experiments.oracle_lab.config import FEATURES_DATASET_FILEPATH, RAW_STATES_FILEPATH
from experiments.oracle_lab.models import FeatureRecord, RawState
from experiments.oracle_lab.storage import (
    load_features,
    load_raw_states,
    save_features,
)
from src.features.bucket_features import FEATURES

SAVE_INTERVAL: int = 100


def build_feature_record(state: RawState) -> FeatureRecord:
    """PRE: state valido. POST: record con tutte le feature del registry."""
    return FeatureRecord(
        state_id=state.state_id,
        n=state.n,
        features={name: fn(state.solutions) for name, fn in FEATURES.items()},
        target_cost=state.best_cost,
        is_exact=state.is_exact,
        source_tree=state.source_tree,
    )


def sync_record(record: FeatureRecord, state: RawState) -> None:
    """POST: target, is_exact e sorgente allineati a raw_states; feature mancanti calcolate."""
    record.target_cost = state.best_cost
    record.is_exact = state.is_exact
    record.source_tree = state.source_tree
    for name, fn in FEATURES.items():
        if name not in record.features:
            record.features[name] = fn(state.solutions)


def main() -> None:
    raw_states = load_raw_states(RAW_STATES_FILEPATH)
    cache = {r.state_id: r for r in load_features(FEATURES_DATASET_FILEPATH)}

    for i, (sid, state) in enumerate(tqdm(raw_states.items(), desc="Calcolo feature")):
        if sid in cache:
            sync_record(cache[sid], state)
        else:
            cache[sid] = build_feature_record(state)
        if (i + 1) % SAVE_INTERVAL == 0:
            save_features(list(cache.values()), FEATURES_DATASET_FILEPATH)

    save_features(list(cache.values()), FEATURES_DATASET_FILEPATH)
    n_exact = sum(r.is_exact for r in cache.values())
    print(f"Dataset: {len(cache)} stati ({n_exact} esatti) -> {FEATURES_DATASET_FILEPATH}")


if __name__ == "__main__":
    main()