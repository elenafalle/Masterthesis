"""
src/plot_rq3_clustering.py
==========================
Error Clustering Chart für RQ3 Parakeet Kombinationen.

Nutzung:
    python src/plot_rq3_clustering.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# ERROR CLUSTERING ERGEBNISSE
# ---------------------------------------------------------------------------

RESULTS = {
    "Self-created\nonly":         {"massive": 1.9, "medical": 12.8, "acceptable": 85.3},
    "Self-created\n+ Voxtral":    {"massive": 1.9, "medical":  7.1, "acceptable": 91.0},
    "Self-created\n+ ElevenLabs": {"massive": 1.5, "medical":  7.1, "acceptable": 91.4},
    "Combined\n(all three)":      {"massive": 1.9, "medical":  6.8, "acceptable": 91.4},
}

CATEGORIES = ["Massive error", "Medical error", "Acceptable"]
KEYS       = ["massive", "medical", "acceptable"]
COLORS     = ["#C0392B", "#F4A460", "#4A7C59"]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(out_path: Path):
    labels = list(RESULTS.keys())
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bottoms = np.zeros(len(labels))

    for cat, key, color in zip(CATEGORIES, KEYS, COLORS):
        vals = np.array([RESULTS[l][key] for l in labels])
        bars = ax.bar(
            x, vals,
            bottom=bottoms,
            width=0.5,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            label=cat,
            zorder=2,
        )
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            if h >= 4:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + h / 2,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold"
                )
            elif h > 0:
                mid_y = bar.get_y() + h / 2
                x_pos = bar.get_x() + bar.get_width() / 2 + 0.27
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2 + 0.25, mid_y),
                    xytext=(x_pos + 0.1, mid_y),
                    fontsize=10, fontweight="bold",
                    color="black", va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=13)
    ax.tick_params(colors="black")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D3D1C7")
    ax.yaxis.grid(True, color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=c, label=cat, edgecolor="black", linewidth=0.8)
               for cat, c in zip(CATEGORIES, COLORS)]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Grafik gespeichert: {out_path}")


if __name__ == "__main__":
    plot(Path("results/rq3_parakeet_clustering.png"))