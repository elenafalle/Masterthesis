"""
WER breakdown into substitutions, deletions and insertions.

Integrates into the existing pipeline as Step 8.
"""

import json
from pathlib import Path
from collections import Counter

from jiwer import process_words

from src.config import log


def _load_jsonl(path: Path) -> list:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def run(predictions_path: Path, label: str) -> dict:
    """
    Compute WER breakdown from a predictions.jsonl file.

    Args:
        predictions_path: Path to predictions.jsonl
        label:            Human-readable label (e.g. 'Baseline Whisper')

    Returns:
        Breakdown dict with substitutions, deletions, insertions counts and percentages.
    """
    if not predictions_path.exists():
        log.warning(f"[WERBreakdown] {predictions_path} not found — skipping {label}.")
        return {}

    data = _load_jsonl(predictions_path)
    log.info(f"[WERBreakdown] {label}: {len(data)} segments loaded.")

    references = [d["reference"]  for d in data]
    hypotheses = [d["hypothesis"] for d in data]

    result = process_words(references, hypotheses)

    total_words  = result.hits + result.substitutions + result.deletions
    total_errors = result.substitutions + result.deletions + result.insertions

    def pct(n):
        return round(100 * n / total_words, 2) if total_words > 0 else 0.0

    bd = {
        "label":            label,
        "total_words":      total_words,
        "total_errors":     total_errors,
        "wer":              round(total_errors / total_words, 4) if total_words > 0 else 0.0,
        "wer_percent":      round(100 * total_errors / total_words, 2) if total_words > 0 else 0.0,
        "substitutions":    result.substitutions,
        "deletions":        result.deletions,
        "insertions":       result.insertions,
        "hits":             result.hits,
        "substitution_pct": pct(result.substitutions),
        "deletion_pct":     pct(result.deletions),
        "insertion_pct":    pct(result.insertions),
    }

    log.info(f"[WERBreakdown] {label}")
    log.info(f"  WER:           {bd['wer_percent']:>6.2f}%  ({total_errors} / {total_words} words)")
    log.info(f"  Substitutions: {bd['substitution_pct']:>6.2f}%  ({result.substitutions})")
    log.info(f"  Deletions:     {bd['deletion_pct']:>6.2f}%  ({result.deletions})")
    log.info(f"  Insertions:    {bd['insertion_pct']:>6.2f}%  ({result.insertions})")

    return bd


def compare(all_breakdowns: dict, run_dir: Path) -> None:
    """
    Print and save comparison table across all models.

    Args:
        all_breakdowns: {key: breakdown_dict}
        run_dir:        Root run directory for saving comparison
    """
    if not all_breakdowns:
        return

    log.info("=" * 65)
    log.info("  WER BREAKDOWN VERGLEICH")
    log.info("=" * 65)
    log.info(f"  {'Modell':<28} {'WER%':>7} {'Sub%':>7} {'Del%':>7} {'Ins%':>7}")
    log.info("  " + "-" * 60)

    order = ["baseline_whisper", "finetuned_whisper", "baseline_parakeet", "finetuned_parakeet"]
    for key in order:
        if key not in all_breakdowns:
            continue
        bd = all_breakdowns[key]
        log.info(
            f"  {bd['label']:<28}"
            f"  {bd['wer_percent']:>6.2f}%"
            f"  {bd['substitution_pct']:>6.2f}%"
            f"  {bd['deletion_pct']:>6.2f}%"
            f"  {bd['insertion_pct']:>6.2f}%"
        )

    out_path = run_dir / "wer_breakdown_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_breakdowns, f, ensure_ascii=False, indent=2)
    log.info(f"  Vergleich gespeichert: {out_path}")
