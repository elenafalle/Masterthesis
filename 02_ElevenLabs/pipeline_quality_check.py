#!/usr/bin/env python3
"""
Pipeline: Qualitätscheck für synthetische Datensätze (Baseline WER only)

Berechnet nur die Baseline WER für Voxtral, ElevenLabs und Reference Dataset.
Kein Fine-tuning, kein Error Clustering – nur Datenqualitätsprüfung.

Nutzung:
    python pipeline_quality_check.py --dataset voxtral
    python pipeline_quality_check.py --dataset elevenlabs
    python pipeline_quality_check.py --dataset reference
    python pipeline_quality_check.py --dataset voxtral --model whisper
    python pipeline_quality_check.py --dataset voxtral --model parakeet
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from src.config import SEED, WHISPER_HPARAMS, PARAKEET_HPARAMS, log
from src.utils import create_run_dir, log_environment, save_json
from src import whisper, parakeet

# ---------------------------------------------------------------------------
# Dataset paths — passe diese an deine Ordnerstruktur an
# ---------------------------------------------------------------------------
DATASET_PATHS = {
    "ElevenLabs":     Path("./all"),
}

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(all_results: dict, run_dir: Path) -> None:
    log.info(f"{'='*60}")
    log.info("  QUALITÄTSCHECK — BASELINE WER")
    log.info(f"{'='*60}")

    lines = []
    lines.append("")
    lines.append(f"  {'Modell':<35} {'WER':>10} {'WER %':>10}")
    lines.append("  " + "-" * 57)

    for key, r in all_results.items():
        lines.append(f"  {r['label']:<35} {r['wer']:>10.4f} {r['wer_percent']:>9.2f}%")

    lines.append("")
    table_str = "\n".join(lines)
    print(table_str)
    log.info(f"Vergleichstabelle:\n{table_str}")
    save_json(run_dir / "quality_check_results.json", all_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualitätscheck: Baseline WER für synthetische Datensätze",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ElevenLabs"],
        required=True,
        help="Datensatz für den Qualitätscheck.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["whisper", "parakeet", "both"],
        default="both",
        help="Welches Modell verwenden (default: both).",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existierenden Run-Ordner fortsetzen.",
    )
    parser.add_argument(
    "--step", type=int, nargs="+", default=None,
    help="Nicht verwendet, nur fuer Kompatibilitaet mit log_environment.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Temporär TEST_DIR überschreiben
    import src.config as cfg
    dataset_path = DATASET_PATHS[args.dataset]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Datensatz-Pfad nicht gefunden: {dataset_path}")

    original_test_dir = cfg.TEST_DIR
    cfg.TEST_DIR = dataset_path
    log.info(f"TEST_DIR überschrieben: {dataset_path}")
    cfg.TRAIN_DIR = dataset_path
    cfg.VAL_DIR = dataset_path

    run_dir = create_run_dir(args.run_dir)

    log.info(f"Pipeline: Qualitätscheck — {args.dataset.upper()}")
    log.info(f"Run-Verzeichnis: {run_dir}")
    log.info(f"Datensatz: {dataset_path}")
    log.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    save_json(run_dir / "config.json", {
    "dataset": args.dataset,
    "model": args.model,
    "dataset_path": str(dataset_path),
})

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    fh = logging.FileHandler(run_dir / "quality_check.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)

    all_results = {}

    # --- Baseline Whisper ---
    if args.model in ("whisper", "both"):
        all_results["baseline_whisper"] = whisper.evaluate(
            WHISPER_HPARAMS["model"],
            f"Baseline Whisper — {args.dataset}",
            run_dir / "whisper" / "baseline",
        )

    # --- Baseline Parakeet ---
    if args.model in ("parakeet", "both"):
        all_results["baseline_parakeet"] = parakeet.evaluate(
            PARAKEET_HPARAMS["model"],
            f"Baseline Parakeet — {args.dataset}",
            run_dir / "parakeet" / "baseline",
        )

    if all_results:
        print_comparison(all_results, run_dir)

    # TEST_DIR zurücksetzen
    cfg.TEST_DIR = original_test_dir

    log.info(f"Qualitätscheck abgeschlossen. Ergebnisse: {run_dir}")


if __name__ == "__main__":
    main()