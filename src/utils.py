"""Shared Helpers: Audio, Daten, WER, I/O."""

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import re

from jiwer import wer as jiwer_wer

from src.config import (
    BASE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    SEED, WHISPER_HPARAMS, PARAKEET_HPARAMS, log,
)

# ---------------------------------------------------------------------------
# WER text normalization
# ---------------------------------------------------------------------------
def _normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """Compute WER with text normalization (lowercase, no punctuation).

    Pairs where the normalized reference is empty (e.g. punctuation-only
    transcripts) are skipped to avoid a jiwer ValueError.
    """
    refs_norm = [_normalize_text(r) for r in references]
    hyps_norm = [_normalize_text(h) for h in hypotheses]

    filtered = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r.strip()]
    if not filtered:
        return 0.0
    skipped = len(references) - len(filtered)
    if skipped:
        log.warning(f"compute_wer: {skipped} Sample(s) mit leerer Referenz nach Normalisierung übersprungen.")
    refs, hyps = zip(*filtered)
    return jiwer_wer(list(refs), list(hyps))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_metadata(split_dir: Path) -> list[dict]:
    """Read metadata.csv -> list of {file_name, text, audio_path}."""
    csv_path = split_dir / "metadata.csv"
    audio_dir = split_dir / "audio"
    samples = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "file_name": row["file_name"],
                "text": row["text"],
                "audio_path": str(audio_dir / row["file_name"]),
            })
    return samples


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file, convert to mono float32 at target_sr."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Audio-Datei nicht gefunden: {path}")
    audio, sr = sf.read(path, dtype="float32")
    # Stereo -> Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample if needed
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"Gespeichert: {path}")


def save_predictions(
    path: Path,
    samples: list[dict],
    references: list[str],
    hypotheses: list[str],
) -> None:
    """Save per-sample predictions as JSONL for error analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample, ref, hyp in zip(samples, references, hypotheses):
            entry = {
                "file_name": sample["file_name"],
                "reference": ref,
                "hypothesis": hyp,
                "wer": round(
                    jiwer_wer(_normalize_text(ref), _normalize_text(hyp))
                    if _normalize_text(ref).strip() else 0.0,
                    4,
                ),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"Transkriptionen gespeichert: {path} ({len(samples)} Eintraege)")


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------
def create_run_dir(explicit_dir: str | None = None) -> Path:
    """Create a timestamped run directory, or reuse an existing one."""
    if explicit_dir:
        run_dir = Path(explicit_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = BASE_DIR / "results" / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_environment(run_dir: Path, args: argparse.Namespace) -> None:
    """Save environment and hyperparameter info for reproducibility."""
    env = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            if torch.cuda.is_available() else None
        ),
        "seed": SEED,
        "steps_requested": args.step or "all",
        "hyperparameters": {
            "whisper": WHISPER_HPARAMS,
            "parakeet": PARAKEET_HPARAMS,
        },
        "data": {
            "train_samples": len(load_metadata(TRAIN_DIR)),
            "val_samples": len(load_metadata(VAL_DIR)),
            "test_samples": len(load_metadata(TEST_DIR)),
        },
    }
    save_json(run_dir / "config.json", env)
