"""
Modulo: paths
Percorsi dei dati. La cartella di run si seleziona con la variabile d'ambiente
COWS_RUN_DIR (es. COWS_RUN_DIR=data/runs/iter_01); di default data/runs/iter_00.
"""
import os
from pathlib import Path

DATA_DIR = Path("data")
LEGACY_DIR = DATA_DIR / "legacy_v1"

RUN_DIR = Path(os.environ.get("COWS_RUN_DIR", DATA_DIR / "runs" / "iter_00"))
TREES_DIR = RUN_DIR / "trees"
MODELS_DIR = RUN_DIR / "models"
ROUTING_POLICY_FILEPATH = MODELS_DIR / "routing_policy.json"
RECIPE_FILEPATH = MODELS_DIR / "recipe.json"