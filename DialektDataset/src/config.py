"""Zentrale Konfiguration: Pfade, Konstanten, Hyperparameter."""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "eval"
TEST_DIR = BASE_DIR / "test"

SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Hyperparameter
# ---------------------------------------------------------------------------
WHISPER_HPARAMS = {
    "model": "openai/whisper-large-v3",
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "effective_batch_size": 16,
    "epochs": 3,
    "fp16": True,
    "warmup_steps": 50,
    "save_total_limit": 2,
    "language": "de",
}

PARAKEET_HPARAMS = {
    "model": "nvidia/parakeet-tdt-0.6b-v3",
    "learning_rate": 1e-4,       # ~20× höher als vorher (Expertenempfehlung für Adapter/LN-Tuning)
    "weight_decay": 1e-3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "epochs": 50,                # früher 10; Early Stopping beendet Training bei Plateau
    "precision": "16-mixed",
    "early_stopping_patience": 8,
    "num_workers": 1,
    "min_lr": 1e-7,
    "freeze_encoder": True,      # LayerNorm wird im Code selektiv aufgetaut
    "speed_rates": [0.9, 1.0, 1.1],  # Datenverdreifachung via Speed-Perturbation
}
