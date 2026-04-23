"""
src/error_clustering.py
=======================
Error Clustering for medical ASR evaluation.

Integrates into the existing pipeline as Step 7.
Called after fine-tuned model evaluation (Steps 5 / 6).

Categories:
    massive_error   BERTScore < 0.75  OR  medical value/unit changed
    medical_error   BERTScore 0.75-0.92
    acceptable      BERTScore > 0.92

Score computed:
    bert_f1   BERTScore using dbmdz/bert-base-german-cased

Motivated by Tobin et al. (2022) and Shor et al. (2023), who demonstrate
that BERTScore correlates more strongly with human judgements of ASR error
severity than WER, and that WER fails to distinguish clinically critical
errors from minor linguistic deviations.

The three-category scheme and BERTScore thresholds (0.75 and 0.92) were
validated through an inter-rater annotation study in which two raters with
medical backgrounds independently assessed 38 transcription pairs
(Cohen's Kappa κ = 0.51). Agreement was highest for massive_error
(73-91%) and acceptable (78-89%), supporting the validity of both thresholds.

Inputs expected per model:
    <run_dir>/<model>/baseline/predictions.jsonl
    <run_dir>/<model>/finetuned/predictions.jsonl

Each predictions.jsonl is a JSONL file (one JSON per line) with at least:
    {"file_name": "...", "reference": "...", "hypothesis": "...", "wer": 0.0}
    category defaults to "medical" if not present.

Outputs (written to <out_dir>/):
    error_clustering.csv
    error_clustering_summary.json
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
from bert_score import score as bert_score

from src.config import log

THRESHOLD_MASSIVE    = 0.75
THRESHOLD_ACCEPTABLE = 0.92

BERT_MODEL = "dbmdz/bert-base-german-cased"

_MEDICAL_VALUE_RE = re.compile(
    r"\b\d+[\.,]?\d*\s?"
    r"(mg|mcg|µg|g|kg|ml|l|mmhg|cmh2o|mmol|meq|mval|ie|iu|%|bpm|"
    r"liter|litern|einheit|einheiten|ampulle|ampullen|tablette|tabletten)\b",
    re.IGNORECASE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _extract_values(text: str) -> set:
    return {m.group().lower().replace(" ", "") for m in _MEDICAL_VALUE_RE.finditer(text)}


def _values_changed(ref: str, hyp: str) -> bool:
    ref_vals = _extract_values(ref)
    return bool(ref_vals) and not ref_vals.issubset(_extract_values(hyp))


def _classify(ref: str, hyp: str, f1: float, category: str) -> dict:
    val_changed = category == "medical" and _values_changed(ref, hyp)

    if f1 < THRESHOLD_MASSIVE or val_changed:
        error_type = "massive_error"
        reason = "meaning lost" if f1 < THRESHOLD_MASSIVE else "medical value/unit changed"
    elif f1 <= THRESHOLD_ACCEPTABLE:
        error_type = "medical_error"
        reason = "partial meaning preserved, medical term affected"
    else:
        error_type = "acceptable"
        reason = "meaning preserved"

    return {
        "error_type":    error_type,
        "reason":        reason,
        "value_changed": val_changed,
    }


def _compute_bertscore(hypotheses: list, references: list) -> list:
    import traceback
    try:
        _, _, F1 = bert_score(
            hypotheses,
            references,
            lang="de",
            model_type=BERT_MODEL,
            num_layers=9,
            device=DEVICE,
            verbose=False,
        )
        return [round(f.item(), 4) for f in F1]
    except Exception as e:
        log.warning(f"[ErrorClustering] BERTScore failed: {e}")
        log.warning(traceback.format_exc())
        return [None] * len(hypotheses)


def _load_jsonl(path: Path) -> list:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def run(predictions_path: Path, out_dir: Path, label: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        log.warning(f"[ErrorClustering] {predictions_path} not found - skipping {label}.")
        return {}

    data = _load_jsonl(predictions_path)
    log.info(f"[ErrorClustering] {label}: {len(data)} segments loaded.")

    references = [d["reference"]               for d in data]
    hypotheses = [d["hypothesis"]              for d in data]
    categories = [d.get("category", "medical") for d in data]
    audio_ids  = [d.get("file_name", f"seg_{i}") for i, d in enumerate(data)]

    log.info(f"[ErrorClustering] Computing BERTScore ({BERT_MODEL}) on {DEVICE}...")
    bert_f1_list = _compute_bertscore(hypotheses, references)

    rows = []
    for i, d in enumerate(data):
        f1  = bert_f1_list[i]
        cls = _classify(
            references[i], hypotheses[i],
            f1 if f1 is not None else 0.0,
            categories[i],
        )
        rows.append({
            "audio":         audio_ids[i],
            "category":      categories[i],
            "reference":     references[i],
            "hypothesis":    hypotheses[i],
            "wer":           d.get("wer", None),
            "bert_f1":       f1,
            "error_type":    cls["error_type"],
            "reason":        cls["reason"],
            "value_changed": cls["value_changed"],
        })

    csv_path = out_dir / "error_clustering.csv"
    fieldnames = [
        "audio", "category", "reference", "hypothesis",
        "wer", "bert_f1", "error_type", "reason", "value_changed",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total        = len(rows)
    error_counts = Counter(r["error_type"] for r in rows)
    cat_counts   = Counter(r["category"]   for r in rows)

    bert_vals = [r["bert_f1"] for r in rows if r["bert_f1"] is not None]
    wer_vals  = [r["wer"]     for r in rows if r["wer"]     is not None]

    avg_bert = sum(bert_vals) / len(bert_vals) if bert_vals else None
    avg_wer  = sum(wer_vals)  / len(wer_vals)  if wer_vals  else None

    summary = {
        "label":           label,
        "total":           total,
        "avg_wer":         round(avg_wer,  4) if avg_wer  is not None else None,
        "avg_bert_f1":     round(avg_bert, 4) if avg_bert is not None else None,
        "error_counts":    dict(error_counts),
        "error_pct": {
            k: round(100 * v / total, 1) for k, v in error_counts.items()
        },
        "category_counts": dict(cat_counts),
    }

    with open(out_dir / "error_clustering_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"[ErrorClustering] -- {label} --------------------------")
    for et in ["massive_error", "medical_error", "acceptable"]:
        n = error_counts.get(et, 0)
        log.info(f"  {et:<25}  {n:>4} ({100*n/total:>4.1f}%)")
    log.info(f"  Avg WER:       {avg_wer:.4f}"  if avg_wer  else "  Avg WER: n/a")
    log.info(f"  Avg BERTScore: {avg_bert:.4f}" if avg_bert else "  Avg BERTScore: n/a")
    log.info(f"  Results: {out_dir}")

    return summary


def plot(summaries: dict, run_dir: Path) -> None:
    """Generate and save error clustering stacked bar chart as PNG."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        log.warning("[ErrorClustering] matplotlib not installed — chart skipped.")
        return

    order  = ["baseline_whisper", "finetuned_whisper", "baseline_parakeet", "finetuned_parakeet"]
    labels = {
        "baseline_whisper":   "Baseline\nWhisper",
        "finetuned_whisper":  "Fine-tuned\nWhisper",
        "baseline_parakeet":  "Baseline\nParakeet",
        "finetuned_parakeet": "Fine-tuned\nParakeet",
    }
    categories = ["Massive error", "Medical error", "Acceptable"]
    pct_keys   = ["massive_error", "medical_error", "acceptable"]
    colors = ["#C0392B", "#F4A460", "#4A7C59"]

    models = [k for k in order if k in summaries]
    if not models:
        return

    x       = np.arange(len(models))
    bottoms = np.zeros(len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for cat, key, color in zip(categories, pct_keys, colors):
        vals = np.array([summaries[m].get("error_pct", {}).get(key, 0) for m in models])
        bars = ax.bar(x, vals, bottom=bottoms, width=0.5, color=color, label=cat, zorder=2, edgecolor="black", linewidth=0.8)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            h = bar.get_height()
            if h >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + h / 2,
                        f"{val:.1f}%",
                        ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")
            elif h > 0:
                mid_y   = bar.get_y() + h / 2
                x_start = bar.get_x() + bar.get_width() / 2 + 0.25
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(x_start, mid_y),
                    xytext=(x_start + 0.12, mid_y),
                    fontsize=10, fontweight="bold",
                    color="black", va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in models], fontsize=13)
    ax.set_ylabel("Percentage (%)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.yaxis.grid(True, color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)
    patches = [mpatches.Patch(color=c, label=cat) for cat, c in zip(categories, colors)]
    ax.legend(handles=patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=12)
    plt.tight_layout()
    out = run_dir / "error_clustering_chart.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"[ErrorClustering] Chart saved: {out}")


def compare(summaries: dict, run_dir: Path) -> None:
    if not summaries:
        return

    log.info("=" * 65)
    log.info("  ERROR CLUSTERING VERGLEICH")
    log.info("=" * 65)
    log.info(f"  {'Modell':<28} {'massive':>9} {'medical':>9} {'ok':>9} {'BERT F1':>9}")
    plot(summaries, run_dir)
    log.info("  " + "-" * 65)

    order = ["baseline_whisper", "finetuned_whisper", "baseline_parakeet", "finetuned_parakeet"]
    for key in order:
        if key not in summaries:
            continue
        s   = summaries[key]
        pct = s.get("error_pct", {})
        log.info(
            f"  {s['label']:<28}"
            f"  {pct.get('massive_error', 0):>8.1f}%"
            f"  {pct.get('medical_error', 0):>8.1f}%"
            f"  {pct.get('acceptable', 0):>8.1f}%"
            f"  {s.get('avg_bert_f1', 0) or 0:>9.4f}"
        )

    out_path = run_dir / "error_clustering_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    log.info(f"  Vergleich gespeichert: {out_path}")

