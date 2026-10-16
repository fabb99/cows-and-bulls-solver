"""
Modulo: config (experiments/oracle_lab)
Costanti condivise dagli esperimenti. I percorsi vivono in src/paths.py.
"""
from src.paths import DATA_DIR, LEGACY_DIR, MODELS_DIR, RUN_DIR, TREES_DIR  # noqa: F401

RAW_STATES_FILEPATH = RUN_DIR / "raw_states.json"
FEATURES_DATASET_FILEPATH = RUN_DIR / "features_dataset.json"

# Soglie degli scaglioni (endgame / midgame / opening).
# PROVVISORIE: verranno aggiornate dopo lo studio di find_optimal_splits.
DEFAULT_SPLITS: list[int] = [13, 43]

# Due costi che differiscono meno di così sono considerati uguali.
COST_TOL: float = 1e-12

# Etichetta della sorgente per i costi esatti (minimo di Bellman).
EXACT_SOURCE = "exact_strategy"