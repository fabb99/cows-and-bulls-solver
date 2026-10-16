"""
Modulo: config (experiments/oracle_lab)
Percorsi e costanti condivise dagli esperimenti. Unica fonte di verità.
La cartella di run si sovrascrive con la variabile d'ambiente COWS_RUN_DIR
(es. COWS_RUN_DIR=data/runs/iter_01).
"""
import os
from pathlib import Path

DATA_DIR = Path("data")
LEGACY_DIR = DATA_DIR / "legacy_v1"

RUN_DIR = Path(os.environ.get("COWS_RUN_DIR", DATA_DIR / "runs" / "iter_00"))
TREES_DIR = RUN_DIR / "trees"
RAW_STATES_FILEPATH = RUN_DIR / "raw_states.json"
FEATURES_DATASET_FILEPATH = RUN_DIR / "features_dataset.json"
MODELS_DIR = RUN_DIR / "models"

# Soglie degli scaglioni (endgame / midgame / opening).
# PROVVISORIE: verranno aggiornate dopo lo studio di find_optimal_splits.
DEFAULT_SPLITS: list[int] = [13, 43]

# Due costi che differiscono meno di così sono considerati uguali.
COST_TOL: float = 1e-12

# Etichetta della sorgente per i costi esatti (minimo di Bellman).
EXACT_SOURCE = "exact_strategy"