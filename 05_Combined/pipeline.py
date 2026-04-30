"""
Pipeline: Whisper & Parakeet Fine-Tuning für RQ3
Führe dieses Skript aus: python3 pipeline.py --config voxtral

Erwartet folgende Ordnerstruktur:
  05_Combined/
    pipeline.py               ← dieses Skript
    src/                      ← Hilfsfunktionen
    train/                    ← Self-created Trainingsdaten
        metadata.csv
        audio/
    eval/                     ← Self-created Validierungsdaten
        metadata.csv
        audio/
    test/                     ← Self-created Testdaten
        metadata.csv
        audio/
    all_voxtral/              ← Voxtral synthetische Daten
        metadata.csv
        audio/
    all_elevenlabs/           ← ElevenLabs synthetische Daten
        metadata.csv
        audio/

Venv erstellen und Voraussetzungen installieren:
    python3 -m venv venv
    source venv/bin/activate          # Mac/Linux
    venv\\Scripts\\activate           # Windows
    pip install torch transformers peft jiwer bert-score soundfile librosa numpy pandas scipy matplotlib nemo_toolkit[asr] hydra-core fiddle cloudpickle lightning bitsandbytes

Nutzung:
    python3 -m src.merge_datasets             # Zuerst Datasets mergen
    python3 pipeline.py --config voxtral      # Self-created + Voxtral
    python3 pipeline.py --config elevenlabs   # Self-created + ElevenLabs
    python3 pipeline.py --config combined     # Self-created + Voxtral + ElevenLabs
    python3 pipeline.py --config voxtral --step 3 4      # Nur Fine-Tuning
    python3 pipeline.py --config voxtral --step 5 6      # Nur Evaluation
    python3 pipeline.py --config voxtral --step 7        # Nur Error Clustering
    python3 pipeline.py --config voxtral --step 5 6 7    # Evaluation + Error Clustering
    python3 pipeline.py --config voxtral --run-dir results/run_xyz  # Fortsetzen
"""

import argparse
import datetime
import json
import logging
import platform
import sys
from pathlib import Path

import torch

import src.config as cfg
from src.config import SEED, WHISPER_HPARAMS, PARAKEET_HPARAMS, log, BASE_DIR
from src.utils import create_run_dir, save_json
from src import whisper, parakeet
from src import error_clustering

# ---------------------------------------------------------------------------
# Konfigurationen — TRAIN_DIR pro Kombination
# ---------------------------------------------------------------------------
CONFIGS = {
    "voxtral":    BASE_DIR / "train_merge_voxtral",
    "elevenlabs": BASE_DIR / "train_merge_elevenlabs",
    "combined":   BASE_DIR / "train_merge_combined",
}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(all_results: dict, run_dir: Path) -> None:
    log.info(f"{'='*60}")
    log.info("  VERGLEICHSTABELLE")
    log.info(f"{'='*60}")

    lines = []
    lines.append("")
    lines.append(f"  {'Modell':<30} {'WER':>10} {'WER %':>10}")
    lines.append("  " + "-" * 52)

    order = ["finetuned_whisper", "finetuned_parakeet"]
    for key in order:
        if key in all_results:
            r = all_results[key]
            lines.append(f"  {r['label']:<30} {r['wer']:>10.4f} {r['wer_percent']:>9.2f}%")

    lines.append("")
    table_str = "\n".join(lines)
    print(table_str)
    log.info(f"Vergleichstabelle:\n{table_str}")
    save_json(run_dir / "comparison.json", all_results)


# ---------------------------------------------------------------------------
# Step 7: Error Clustering
# ---------------------------------------------------------------------------
def run_error_clustering(run_dir: Path, config: str) -> None:
    log.info("=" * 60)
    log.info("  STEP 7: ERROR CLUSTERING")
    log.info("=" * 60)

    targets = {
        "finetuned_whisper": (
            run_dir / "whisper" / "finetuned" / "predictions.jsonl",
            run_dir / "whisper" / "finetuned" / "error_clustering",
            f"Fine-tuned Whisper ({config})",
        ),
        "finetuned_parakeet": (
            run_dir / "parakeet" / "finetuned" / "predictions.jsonl",
            run_dir / "parakeet" / "finetuned" / "error_clustering",
            f"Fine-tuned Parakeet ({config})",
        ),
    }

    summaries = {}
    for key, (pred_path, out_dir, label) in targets.items():
        if not pred_path.exists():
            log.info(f"  [Skip] {label}: {pred_path} not found.")
            continue
        summary = error_clustering.run(pred_path, out_dir, label)
        if summary:
            summaries[key] = summary

    if summaries:
        error_clustering.compare(summaries, run_dir)
    else:
        log.warning("  Keine predictions.jsonl gefunden — Error Clustering übersprungen.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RQ3 Fine-Tuning Pipeline — Combined Datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["voxtral", "elevenlabs", "combined"],
        required=True,
        help="Trainingsdatensatz-Kombination.",
    )
    parser.add_argument(
        "--step", type=int, nargs="+", choices=[3, 4, 5, 6, 7],
        help="Nur bestimmte Schritte (3=FT Whisper, 4=FT Parakeet, 5=Eval Whisper, 6=Eval Parakeet, 7=Error Clustering). Standard: alle.",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existierenden Run-Ordner fortsetzen.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    steps = set(args.step) if args.step else {3, 4, 5, 6, 7}

    # TRAIN_DIR auf gewählte Kombination setzen
    train_dir = CONFIGS[args.config]
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train-Ordner nicht gefunden: {train_dir}\n"
            f"Bitte zuerst ausführen: python3 -m src.merge_datasets --config {args.config}"
        )
    cfg.TRAIN_DIR = train_dir
    log.info(f"TRAIN_DIR gesetzt: {train_dir}")

    run_dir = create_run_dir(args.run_dir)

    log.info(f"Pipeline: RQ3 Fine-Tuning — {args.config.upper()}")
    log.info(f"Run-Verzeichnis: {run_dir}")
    log.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Schritte: {sorted(steps)}")

    save_json(run_dir / "config.json", {
        "timestamp": datetime.datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
        "seed": SEED,
        "config": args.config,
        "train_dir": str(train_dir),
        "steps_requested": sorted(steps),
        "hyperparameters": {
            "whisper": WHISPER_HPARAMS,
            "parakeet": PARAKEET_HPARAMS,
        },
        "data": {
            "train_samples": len(list((cfg.TRAIN_DIR / "audio").glob("*.wav"))),
            "val_samples": len(list((cfg.VAL_DIR / "audio").glob("*.wav"))),
            "test_samples": len(list((cfg.TEST_DIR / "audio").glob("*.wav"))),
        },
    })

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    fh = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)

    all_results: dict = {}
    comparison_path = run_dir / "comparison.json"
    if comparison_path.exists():
        with open(comparison_path, encoding="utf-8") as f:
            all_results = json.load(f)
        log.info(f"Vorherige Ergebnisse geladen: {list(all_results.keys())}")

    # --- Step 3: Fine-tune Whisper ---
    if 3 in steps:
        whisper.finetune(run_dir)

    # --- Step 4: Fine-tune Parakeet ---
    if 4 in steps:
        parakeet.finetune(run_dir)

    # --- Step 5: Evaluate fine-tuned Whisper ---
    if 5 in steps:
        whisper_model_dir = run_dir / "whisper" / "model"
        if not whisper_model_dir.exists():
            log.warning(f"Whisper-Modell nicht gefunden: {whisper_model_dir} — Schritt 5 übersprungen.")
        else:
            all_results["finetuned_whisper"] = whisper.evaluate(
                str(whisper_model_dir),
                f"Fine-tuned Whisper ({args.config})",
                run_dir / "whisper" / "finetuned",
            )

    # --- Step 6: Evaluate fine-tuned Parakeet ---
    if 6 in steps:
        parakeet_model = run_dir / "parakeet" / "model" / "parakeet_finetuned.nemo"
        if not parakeet_model.exists():
            log.warning(f"Parakeet-Modell nicht gefunden: {parakeet_model} — Schritt 6 übersprungen.")
        else:
            all_results["finetuned_parakeet"] = parakeet.evaluate(
                str(parakeet_model),
                f"Fine-tuned Parakeet ({args.config})",
                run_dir / "parakeet" / "finetuned",
            )

    # --- WER Comparison ---
    if all_results:
        print_comparison(all_results, run_dir)

    # --- Step 7: Error Clustering ---
    if 7 in steps:
        run_error_clustering(run_dir, args.config)

    log.info(f"Pipeline abgeschlossen. Ergebnisse: {run_dir}")


if __name__ == "__main__":
    main()
