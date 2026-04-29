"""
Error Clustering

Integrates into the existing pipeline as Step 7.
Called after fine-tuned model evaluation (Steps 5 / 6).

Categories:
    massive_error   BERTScore < 0.75  OR  medical value/unit changed
    medical_error   BERTScore 0.75-0.92
    acceptable      BERTScore > 0.92

Score computed:
    cbert_f1   CBERTScore using GerMedBERT/medbert-512  (Klassifikation)
    bert_f1    BERTScore  using dbmdz/bert-base-german-cased  (Vergleich)

Agreement:
    Cohen's Kappa zwischen cbert- und bert-basiertem error_type

Two charts are generated:
    error_clustering_chart_cbert.png  — based on cBERT classification
    error_clustering_chart_bert.png   — based on BERT classification
"""

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from bert_score import score as bert_score

from src.config import log

THRESHOLD_MASSIVE    = 0.75
THRESHOLD_ACCEPTABLE = 0.92

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
BERT_MODEL_CBERT = "GerMedBERT/medbert-512"
BERT_MODEL_BERT  = "dbmdz/bert-base-german-cased"

BERT_MODEL_LAYERS: dict[str, int] = {
    "GerMedBERT/medbert-512":       8,
    "dbmdz/bert-base-german-cased": 9,
}

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


def _compute_bertscore(
    hypotheses: list,
    references: list,
    model: str,
    num_layers: int | None,
) -> list:
    import traceback
    try:
        kwargs = dict(
            lang="de",
            model_type=model,
            device=DEVICE,
            verbose=False,
        )
        if num_layers is not None:
            kwargs["num_layers"] = num_layers

        _, _, F1 = bert_score(hypotheses, references, **kwargs)
        return [round(f.item(), 4) for f in F1]
    except Exception as e:
        log.warning(f"[ErrorClustering] BERTScore failed ({model}): {e}")
        log.warning(traceback.format_exc())
        return [None] * len(hypotheses)


def _agreement(rows: list) -> dict:
    """Cohen's Kappa und prozentuale Übereinstimmung zwischen
    cbert- und bert-basiertem error_type."""
    from sklearn.metrics import cohen_kappa_score

    cbert_labels = [r["error_type"]      for r in rows]
    bert_labels  = [r["error_type_bert"] for r in rows]

    matches = sum(c == b for c, b in zip(cbert_labels, bert_labels))
    total   = len(rows)
    pct     = round(100 * matches / total, 2)
    kappa   = round(cohen_kappa_score(cbert_labels, bert_labels), 4)

    log.info(f"  Übereinstimmung cbert/bert: {pct:.1f}%  (Cohen κ = {kappa:.4f})")

    return {
        "agreement_pct": pct,
        "cohen_kappa":   kappa,
        "matching":      matches,
        "total":         total,
    }


def _load_jsonl(path: Path) -> list:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def run(
    predictions_path: Path,
    out_dir: Path,
    label: str,
    bert_model_cbert: str | None = None,
    bert_model_bert:  str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        log.warning(f"[ErrorClustering] {predictions_path} not found - skipping {label}.")
        return {}

    data = _load_jsonl(predictions_path)
    log.info(f"[ErrorClustering] {label}: {len(data)} segments loaded.")

    references = [d["reference"]                 for d in data]
    hypotheses = [d["hypothesis"]                for d in data]
    categories = [d.get("category", "medical")   for d in data]
    audio_ids  = [d.get("file_name", f"seg_{i}") for i, d in enumerate(data)]

    model_cbert = (bert_model_cbert
                   or os.environ.get("BERT_MODEL_CBERT")
                   or BERT_MODEL_CBERT)
    model_bert  = (bert_model_bert
                   or os.environ.get("BERT_MODEL_BERT")
                   or BERT_MODEL_BERT)

    log.info(f"[ErrorClustering] Computing BERTScores on {DEVICE} ...")
    log.info(f"  cbert: {model_cbert}")
    log.info(f"  bert:  {model_bert}")

    cbert_f1_list = _compute_bertscore(
        hypotheses, references,
        model_cbert,
        BERT_MODEL_LAYERS.get(model_cbert),
    )
    bert_f1_list = _compute_bertscore(
        hypotheses, references,
        model_bert,
        BERT_MODEL_LAYERS.get(model_bert),
    )

    rows = []
    for i, d in enumerate(data):
        cbert_f1 = cbert_f1_list[i]
        bert_f1  = bert_f1_list[i]

        # Klassifikation basiert auf cbert (medizinisches Modell)
        cls = _classify(
            references[i], hypotheses[i],
            cbert_f1 if cbert_f1 is not None else 0.0,
            categories[i],
        )

        # Klassifikation basiert auf bert (für Vergleich)
        bert_cls = _classify(
            references[i], hypotheses[i],
            bert_f1 if bert_f1 is not None else 0.0,
            categories[i],
        )

        rows.append({
            "audio":            audio_ids[i],
            "category":         categories[i],
            "reference":        references[i],
            "hypothesis":       hypotheses[i],
            "wer":              d.get("wer", None),
            "cbert_f1":         cbert_f1,
            "bert_f1":          bert_f1,
            "error_type":       cls["error_type"],
            "error_type_bert":  bert_cls["error_type"],
            "reason":           cls["reason"],
            "value_changed":    cls["value_changed"],
        })

    # Cohen's Kappa: Übereinstimmung cbert vs. bert
    agreement = _agreement(rows)

    csv_path = out_dir / "error_clustering.csv"
    fieldnames = [
        "audio", "category", "reference", "hypothesis",
        "wer", "cbert_f1", "bert_f1",
        "error_type", "error_type_bert", "reason", "value_changed",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total              = len(rows)
    error_counts_cbert = Counter(r["error_type"]      for r in rows)
    error_counts_bert  = Counter(r["error_type_bert"] for r in rows)
    cat_counts         = Counter(r["category"]        for r in rows)

    def _avg(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    avg_wer   = _avg(r["wer"]      for r in rows)
    avg_cbert = _avg(r["cbert_f1"] for r in rows)
    avg_bert  = _avg(r["bert_f1"]  for r in rows)

    summary = {
        "label":                label,
        "total":                total,
        "cbert_model":          model_cbert,
        "bert_model":           model_bert,
        "avg_wer":              avg_wer,
        "avg_cbert_f1":         avg_cbert,
        "avg_bert_f1":          avg_bert,
        "error_counts":         dict(error_counts_cbert),
        "error_counts_bert":    dict(error_counts_bert),
        "error_pct": {
            k: round(100 * v / total, 1) for k, v in error_counts_cbert.items()
        },
        "error_pct_bert": {
            k: round(100 * v / total, 1) for k, v in error_counts_bert.items()
        },
        "category_counts":      dict(cat_counts),
        "bert_cbert_agreement": agreement,
    }

    with open(out_dir / "error_clustering_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"[ErrorClustering] -- {label} --------------------------")
    for et in ["massive_error", "medical_error", "acceptable"]:
        n_c = error_counts_cbert.get(et, 0)
        n_b = error_counts_bert.get(et, 0)
        log.info(f"  {et:<25}  cbert: {n_c:>4} ({100*n_c/total:>4.1f}%)  bert: {n_b:>4} ({100*n_b/total:>4.1f}%)")
    log.info(f"  Avg WER:        {avg_wer:.4f}"   if avg_wer   else "  Avg WER: n/a")
    log.info(f"  Avg CBERTScore: {avg_cbert:.4f}" if avg_cbert else "  Avg CBERTScore: n/a")
    log.info(f"  Avg BERTScore:  {avg_bert:.4f}"  if avg_bert  else "  Avg BERTScore:  n/a")
    log.info(f"  Results: {out_dir}")

    return summary


def plot(summaries: dict, run_dir: Path, use_bert: bool = False) -> None:
    """Generate and save error clustering stacked bar chart as PNG.
    
    use_bert=False  → cBERT-based classification (error_pct)
    use_bert=True   → BERT-based classification (error_pct_bert)
    """
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
    colors     = ["#C0392B", "#F4A460", "#4A7C59"]

    pct_field = "error_pct_bert" if use_bert else "error_pct"
    suffix    = "bert"  if use_bert else "cbert"
    title     = "BERT (dbmdz/bert-base-german-cased)" if use_bert else "cBERT (GerMedBERT/medbert-512)"

    models = [k for k in order if k in summaries]
    if not models:
        return

    x       = np.arange(len(models))
    bottoms = np.zeros(len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11, color="#666666", pad=8)

    for cat, key, color in zip(categories, pct_keys, colors):
        vals = np.array([summaries[m].get(pct_field, {}).get(key, 0) for m in models])
        bars = ax.bar(x, vals, bottom=bottoms, width=0.5, color=color, label=cat,
                      zorder=2, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            if h >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + h / 2,
                        f"{val:.1f}%",
                        ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
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
    out = run_dir / f"error_clustering_chart_{suffix}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"[ErrorClustering] Chart saved: {out}")


def compare(summaries: dict, run_dir: Path) -> None:
    if not summaries:
        return

    log.info("=" * 75)
    log.info("  ERROR CLUSTERING VERGLEICH")
    log.info("=" * 75)
    log.info(f"  {'Modell':<28} {'massive':>9} {'medical':>9} {'ok':>9} {'CBERT F1':>9} {'BERT F1':>9} {'κ':>7}")

    # cBERT chart
    plot(summaries, run_dir, use_bert=False)
    # BERT chart
    plot(summaries, run_dir, use_bert=True)

    log.info("  " + "-" * 75)

    order = ["baseline_whisper", "finetuned_whisper", "baseline_parakeet", "finetuned_parakeet"]
    for key in order:
        if key not in summaries:
            continue
        s     = summaries[key]
        pct   = s.get("error_pct", {})
        agree = s.get("bert_cbert_agreement", {})
        log.info(
            f"  {s['label']:<28}"
            f"  {pct.get('massive_error', 0):>8.1f}%"
            f"  {pct.get('medical_error', 0):>8.1f}%"
            f"  {pct.get('acceptable',    0):>8.1f}%"
            f"  {s.get('avg_cbert_f1', 0) or 0:>9.4f}"
            f"  {s.get('avg_bert_f1',  0) or 0:>9.4f}"
            f"  {agree.get('cohen_kappa', 0) or 0:>7.4f}"
        )

    out_path = run_dir / "error_clustering_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    log.info(f"  Vergleich gespeichert: {out_path}")
