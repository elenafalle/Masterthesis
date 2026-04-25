"""
Pipeline: Whisper & Parakeet Evaluation + Fine-Tuning + Error Clustering + WER Breakdown
Führe dieses Skript aus: python pipeline.py

Erwartet folgende Ordnerstruktur:
  DialektDataset/
    pipeline.py               ← dieses Skript
    src/                      ← Hilfsfunktionen
    train/
        metadata.csv
        audio/
    eval/
        metadata.csv
        audio/
    test/
        metadata.csv
        audio/

Venv erstellen und Voraussetzungen installieren:
    python3 -m venv venv
    source venv/bin/activate          # Mac/Linux
    venv\\Scripts\\activate           # Windows
    pip install torch transformers peft jiwer bert-score soundfile librosa numpy pandas scipy matplotlib nemo_toolkit[asr] hydra-core fiddle cloudpickle lightning bitsandbytes

Nutzung:
    python pipeline.py                   # Alle 8 Schritte
    python pipeline.py --step 1          # Nur Baseline Whisper
    python pipeline.py --step 3 4        # Nur Fine-Tuning beider Modelle
    python pipeline.py --step 7          # Nur Error Clustering
    python pipeline.py --step 8          # Nur WER Breakdown
    python pipeline.py --step 7 8        # Error Clustering + WER Breakdown
    python pipeline.py --run-dir results/run_2026-02-20_143000  # Fortsetzen
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from src.config import SEED, WHISPER_HPARAMS, PARAKEET_HPARAMS, log
from src.utils import create_run_dir, log_environment, save_json
from src import whisper, parakeet
from src import error_clustering
from src import wer_breakdown


# ---------------------------------------------------------------------------
# Comparison table (WER)
# ---------------------------------------------------------------------------
def print_comparison(all_results: dict, run_dir: Path) -> None:
    log.info(f"{'='*60}")
    log.info("  VERGLEICHSTABELLE  (WER)")
    log.info(f"{'='*60}")

    lines = []
    lines.append("")
    lines.append(f"  {'Modell':<30} {'WER':>10} {'WER %':>10}")
    lines.append("  " + "-" * 52)

    order = ["baseline_whisper", "finetuned_whisper", "baseline_parakeet", "finetuned_parakeet"]
    for key in order:
        if key in all_results:
            r = all_results[key]
            lines.append(f"  {r['label']:<30} {r['wer']:>10.4f} {r['wer_percent']:>9.2f}%")

    lines.append("")
    if "baseline_whisper" in all_results and "finetuned_whisper" in all_results:
        bw = all_results["baseline_whisper"]["wer_percent"]
        fw = all_results["finetuned_whisper"]["wer_percent"]
        lines.append(f"  Whisper Verbesserung:  {bw:.2f}% -> {fw:.2f}% (Delta {bw - fw:+.2f}%)")

    if "baseline_parakeet" in all_results and "finetuned_parakeet" in all_results:
        bp = all_results["baseline_parakeet"]["wer_percent"]
        fp = all_results["finetuned_parakeet"]["wer_percent"]
        lines.append(f"  Parakeet Verbesserung: {bp:.2f}% -> {fp:.2f}% (Delta {bp - fp:+.2f}%)")
    lines.append("")

    table_str = "\n".join(lines)
    print(table_str)
    log.info(f"Vergleichstabelle:\n{table_str}")
    save_json(run_dir / "comparison.json", all_results)


# ---------------------------------------------------------------------------
# Step 7: Error Clustering
# ---------------------------------------------------------------------------
def run_error_clustering(run_dir: Path) -> None:
    log.info("=" * 60)
    log.info("  STEP 7: ERROR CLUSTERING")
    log.info("=" * 60)

    targets = {
        "baseline_whisper": (
            run_dir / "whisper"  / "baseline"  / "predictions.jsonl",
            run_dir / "whisper"  / "baseline"  / "error_clustering",
            "Baseline Whisper",
        ),
        "finetuned_whisper": (
            run_dir / "whisper"  / "finetuned" / "predictions.jsonl",
            run_dir / "whisper"  / "finetuned" / "error_clustering",
            "Fine-tuned Whisper",
        ),
        "baseline_parakeet": (
            run_dir / "parakeet" / "baseline"  / "predictions.jsonl",
            run_dir / "parakeet" / "baseline"  / "error_clustering",
            "Baseline Parakeet",
        ),
        "finetuned_parakeet": (
            run_dir / "parakeet" / "finetuned" / "predictions.jsonl",
            run_dir / "parakeet" / "finetuned" / "error_clustering",
            "Fine-tuned Parakeet",
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
        log.warning("  Keine predictions.jsonl gefunden — Error Clustering uebersprungen.")


# ---------------------------------------------------------------------------
# Step 8: WER Breakdown (Substitution / Deletion / Insertion)
# ---------------------------------------------------------------------------
def run_wer_breakdown(run_dir: Path) -> None:
    log.info("=" * 60)
    log.info("  STEP 8: WER BREAKDOWN (Sub / Del / Ins)")
    log.info("=" * 60)

    targets = {
        "baseline_whisper":   (run_dir / "whisper"  / "baseline"  / "predictions.jsonl", "Baseline Whisper"),
        "finetuned_whisper":  (run_dir / "whisper"  / "finetuned" / "predictions.jsonl", "Fine-tuned Whisper"),
        "baseline_parakeet":  (run_dir / "parakeet" / "baseline"  / "predictions.jsonl", "Baseline Parakeet"),
        "finetuned_parakeet": (run_dir / "parakeet" / "finetuned" / "predictions.jsonl", "Fine-tuned Parakeet"),
    }

    all_breakdowns = {}
    for key, (pred_path, label) in targets.items():
        if not pred_path.exists():
            log.info(f"  [Skip] {label}: {pred_path} not found.")
            continue
        bd = wer_breakdown.run(pred_path, label)
        if bd:
            all_breakdowns[key] = bd
            out_path = pred_path.parent / "wer_breakdown.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(bd, f, ensure_ascii=False, indent=2)
            log.info(f"  Saved: {out_path}")

    if all_breakdowns:
        wer_breakdown.compare(all_breakdowns, run_dir)
    else:
        log.warning("  Keine predictions.jsonl gefunden — WER Breakdown uebersprungen.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper & Parakeet Evaluation + Fine-Tuning + Error Clustering + WER Breakdown",
    )
    parser.add_argument(
        "--step", type=int, nargs="+", choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Nur bestimmte Schritte ausfuehren (z.B. --step 1 2). Standard: alle.",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existierenden Run-Ordner fortsetzen statt einen neuen zu erstellen.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    run_dir = create_run_dir(args.run_dir)
    steps = set(args.step) if args.step else {1, 2, 3, 4, 5, 6, 7, 8}

    log.info("Pipeline: Whisper & Parakeet Evaluation + Fine-Tuning + Error Clustering + WER Breakdown")
    log.info(f"Run-Verzeichnis: {run_dir}")
    log.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Schritte: {sorted(steps)}")

    log_environment(run_dir, args)

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

    # --- Step 1: Baseline Whisper ---
    if 1 in steps:
        all_results["baseline_whisper"] = whisper.evaluate(
            WHISPER_HPARAMS["model"],
            "Baseline Whisper (large-v3)",
            run_dir / "whisper" / "baseline",
        )

    # --- Step 2: Baseline Parakeet ---
    if 2 in steps:
        all_results["baseline_parakeet"] = parakeet.evaluate(
            PARAKEET_HPARAMS["model"],
            "Baseline Parakeet (TDT 0.6B v3)",
            run_dir / "parakeet" / "baseline",
        )

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
            log.warning(f"Whisper-Modell nicht gefunden: {whisper_model_dir} — Schritt 5 uebersprungen.")
        else:
            all_results["finetuned_whisper"] = whisper.evaluate(
                str(whisper_model_dir),
                "Fine-tuned Whisper",
                run_dir / "whisper" / "finetuned",
            )

    # --- Step 6: Evaluate fine-tuned Parakeet ---
    if 6 in steps:
        parakeet_model = run_dir / "parakeet" / "model" / "parakeet_finetuned.nemo"
        if not parakeet_model.exists():
            log.warning(f"Parakeet-Modell nicht gefunden: {parakeet_model} — Schritt 6 uebersprungen.")
        else:
            all_results["finetuned_parakeet"] = parakeet.evaluate(
                str(parakeet_model),
                "Fine-tuned Parakeet",
                run_dir / "parakeet" / "finetuned",
            )

    # --- WER Comparison ---
    if all_results:
        print_comparison(all_results, run_dir)

    # --- Step 7: Error Clustering ---
    if 7 in steps:
        run_error_clustering(run_dir)

    # --- Step 8: WER Breakdown ---
    if 8 in steps:
        run_wer_breakdown(run_dir)

    log.info(f"Pipeline abgeschlossen. Ergebnisse: {run_dir}")


if __name__ == "__main__":
    main()
