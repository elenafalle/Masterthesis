import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

WHISPER_RESULTS = {
    "Self-created\nBaseline":     22.41,
    "Self-created\nonly":         23.36,
    "Self-created\n+ Voxtral":    21.96,
    "Self-created\n+ ElevenLabs": 22.51,
    "Combined\n(all three)":      21.71,
}

PARAKEET_RESULTS = {
    "Self-created\nBaseline":     30.08,
    "Self-created\nonly":         6.47,
    "Self-created\n+ Voxtral":    5.28,
    "Self-created\n+ ElevenLabs": 5.08,
    "Combined\n(all three)":      4.68,
}

COLORS      = ["#9CA3AF", "#2563EB", "#D97706", "#16A34A", "#DC2626"]
EDGE_COLORS = ["#6B7280", "#1D4ED8", "#B45309", "#15803D", "#B91C1C"]
LABELS = list(WHISPER_RESULTS.keys())

def plot(out_path: Path):
    whisper_vals  = [WHISPER_RESULTS[l]  for l in LABELS]
    parakeet_vals = [PARAKEET_RESULTS[l] for l in LABELS]

    n_bars    = len(LABELS)
    bar_width = 0.15
    group_gap = 1.1

    x       = np.array([0, group_gap])
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (label, color, edge) in enumerate(zip(LABELS, COLORS, EDGE_COLORS)):
        vals      = [whisper_vals[i], parakeet_vals[i]]
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
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,   # above the bar
                f"{val:.2f}%",
                ha="center", va="bottom",
                fontsize=8.5, color="#444441", fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Whisper", "Parakeet"], fontsize=13)
    ax.set_ylabel("WER %", fontsize=12, color="#5F5E5A")
    ax.set_ylim(0, max(whisper_vals + parakeet_vals) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(colors="#888780")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D3D1C7")
    ax.yaxis.grid(True, color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(facecolor=c, edgecolor=e, linewidth=1.5, label=l.replace("\n", " "))
        for c, e, l in zip(COLORS, EDGE_COLORS, LABELS)
    ]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=5,
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Gespeichert: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/rq3_wer_chart.png")
    args = parser.parse_args()
    plot(Path(args.out))
