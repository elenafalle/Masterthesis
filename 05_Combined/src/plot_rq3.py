"""
src/plot_rq3.py
===============
Erstellt die RQ3 WER-Vergleichsgrafiken für die Masterarbeit.

Nutzung:
    python src/plot_rq3.py
    python src/plot_rq3.py --out results/rq3_wer_chart.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# WER ERGEBNISSE — hier eintragen
# ---------------------------------------------------------------------------

WHISPER_RESULTS = {
    "Self-created\nonly":         22.41,
    "Self-created\n+ Voxtral":    21.96,
    "Self-created\n+ ElevenLabs": 22.51,
    "Combined\n(all three)":      21.71,
}

PARAKEET_RESULTS = {
    "Self-created\nonly":         6.47,
    "Self-created\n+ Voxtral":    5.28,
    "Self-created\n+ ElevenLabs": 5.08,
    "Combined\n(all three)":      4.68,
}

# ---------------------------------------------------------------------------
# Farben
# ---------------------------------------------------------------------------

COLORS = ["#F9E4B7", "#BDD7EE", "#5B9BD5", "#B4A7D6"]
EDGE_COLORS = ["#F5C842", "#4472C4", "#4472C4", "#8E7CC3"]
LABELS = list(WHISPER_RESULTS.keys())

# Gemeinsame y-Achse für beide Einzelcharts
Y_MAX = max(max(WHISPER_RESULTS.values()), max(PARAKEET_RESULTS.values())) * 1.15


# ---------------------------------------------------------------------------
# Plot — beide Modelle zusammen
# ---------------------------------------------------------------------------

def plot(out_path: Path):
    whisper_vals = [WHISPER_RESULTS[l] for l in LABELS]
    parakeet_vals = [PARAKEET_RESULTS[l] for l in LABELS]

    n_bars = len(LABELS)
    bar_width = 0.15
    group_gap = 1.1

    x = np.array([0, group_gap])
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (label, color, edge) in enumerate(zip(LABELS, COLORS, EDGE_COLORS)):
        vals = [whisper_vals[i], parakeet_vals[i]]
        positions = x + offsets[i]

        bars = ax.bar(
            positions, vals,
            width=bar_width,
            color=color,
            edgecolor=edge,
            linewidth=1.5,
            label=label.replace("\n", " "),
        )

        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.2f}%",
                    ha="center", va="bottom",
                    fontsize=9, color="#444441", fontweight="bold"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(["Whisper", "Parakeet"], fontsize=13)
    ax.set_ylabel("WER %", fontsize=12, color="#5F5E5A")
    ax.set_ylim(0, max(filter(None, whisper_vals + parakeet_vals)) * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(colors="#888780")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D3D1C7")
    ax.yaxis.grid(True, color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(color=c, edgecolor=e, linewidth=1.5, label=l.replace("\n", " "))
        for c, e, l in zip(COLORS, EDGE_COLORS, LABELS)
    ]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Grafik gespeichert: {out_path}")


# ---------------------------------------------------------------------------
# Plot — einzelnes Modell (gemeinsame y-Achse)
# ---------------------------------------------------------------------------

def plot_single(results: dict, model_name: str, out_path: Path):
    labels = list(results.keys())
    vals = list(results.values())

    max_val = max(filter(None, vals))

    bar_width = 0.35
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Gemeinsame y-Achse für beide Charts
    ax.set_ylim(0, Y_MAX)

    bars = ax.bar(
        x, vals,
        width=bar_width,
        color=COLORS,
        edgecolor=EDGE_COLORS,
        linewidth=1.5,
    )

    for i, (bar, val) in enumerate(zip(bars, vals)):
        if val is not None:
            # Absolutwert in der Mitte des Balkens
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val / 2,
                f"{val:.2f}%",
                ha="center", va="center",
                fontsize=11, color="#444441", fontweight="bold"
            )
            # Delta über dem Balken
            if i > 0:
                delta = val - vals[0]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + Y_MAX * 0.02,
                    f"{delta:+.2f}%",
                    ha="center", va="bottom",
                    fontsize=9, color="#444441", fontstyle="italic"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("WER %", fontsize=12, color="#5F5E5A")
    ax.set_title(model_name, fontsize=14, color="#444441", pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(colors="#888780")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D3D1C7")
    ax.yaxis.grid(True, color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Grafik gespeichert: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RQ3 WER Vergleichsgrafik")
    parser.add_argument(
        "--out", type=str,
        default="results/rq3_wer_chart.png",
        help="Ausgabepfad für die kombinierte PNG-Datei"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Beide Modelle zusammen
    plot(Path(args.out))

    # Nur Whisper
    plot_single(
        results=WHISPER_RESULTS,
        model_name="Whisper large-v3",
        out_path=Path("results/rq3_whisper_chart.png")
    )

    # Nur Parakeet
    plot_single(
        results=PARAKEET_RESULTS,
        model_name="Parakeet TDT 0.6B v3",
        out_path=Path("results/rq3_parakeet_chart.png")
    )