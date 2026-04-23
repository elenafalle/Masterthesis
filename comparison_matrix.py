import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

COL_HEADER = "#E8E8E4"
COL_CELL   = "#FFFFFF"
COL_BORDER = "#CCCCC8"
COL_TEXT   = "#1A1A18"
COL_MUTED  = "#6B6B66"
COL_ARROW  = "#888880"
COL_BG_LET = "#1A1A18"

col_x  = [2.6, 6.3]
row_y  = [3.8, 0.4]
cw, ch = 3.5, 3.2
hh     = 1.1

def rounded_rect(ax, x, y, w, h, color, lw=0.6, radius=0.18):
    fancy = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=COL_BORDER, facecolor=color, zorder=2
    )
    ax.add_patch(fancy)

# column headers
for i, label in enumerate(["Self-created (dialect)", "Reference (MultiMed)"]):
    rounded_rect(ax, col_x[i], row_y[0] + ch + 0.15, cw, hh, COL_HEADER)
    ax.text(col_x[i] + cw/2, row_y[0] + ch + 0.15 + hh/2, label,
            ha="center", va="center", fontsize=14, fontweight="bold", color=COL_TEXT)

# row headers
for j, label in enumerate(["Baseline", "Fine-tuned"]):
    rounded_rect(ax, 0.1, row_y[j], 2.3, ch, COL_HEADER)
    ax.text(0.1 + 2.3/2, row_y[j] + ch/2, label,
            ha="center", va="center", fontsize=14, fontweight="bold", color=COL_TEXT)

data = [
    [("A", "22.41%", "30.08%"), ("B", "10.85%", "11.56%")],
    [("C", "23.36%",  "6.47%"), ("D",  "9.65%",  "7.71%")],
]

for j, row in enumerate(data):
    for i, (letter, wer_w, wer_p) in enumerate(row):
        x = col_x[i]
        y = row_y[j]
        rounded_rect(ax, x, y, cw, ch, COL_CELL)
        ax.text(x + cw/2, y + ch/2, letter,
                ha="center", va="center",
                fontsize=80, fontweight="bold", color=COL_BG_LET, alpha=0.05, zorder=3)
        mid_y = y + ch/2
        ax.plot([x + 0.2, x + cw - 0.2], [mid_y, mid_y],
                color=COL_BORDER, linewidth=0.5, zorder=4)
        ax.text(x + 0.25, mid_y + 0.58, "Whisper",
                va="center", fontsize=13, color=COL_MUTED, zorder=4)
        ax.text(x + cw - 0.2, mid_y + 0.58, wer_w,
                ha="right", va="center", fontsize=14, fontweight="bold", color=COL_TEXT, zorder=4)
        ax.text(x + 0.25, mid_y - 0.58, "Parakeet",
                va="center", fontsize=13, color=COL_MUTED, zorder=4)
        ax.text(x + cw - 0.2, mid_y - 0.58, wer_p,
                ha="right", va="center", fontsize=14, fontweight="bold", color=COL_TEXT, zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=COL_ARROW,
                                mutation_scale=10, lw=1.1), zorder=5)

arrow(ax, col_x[0] + cw + 0.05, row_y[0] + ch/2 + 0.3, col_x[1] - 0.05, row_y[0] + ch/2 + 0.3)
arrow(ax, col_x[0] + cw + 0.05, row_y[1] + ch/2 + 0.3, col_x[1] - 0.05, row_y[1] + ch/2 + 0.3)
arrow(ax, col_x[0] + cw/2 - 0.3, row_y[0] - 0.05, col_x[0] + cw/2 - 0.3, row_y[1] + ch + 0.05)
arrow(ax, col_x[1] + cw/2 - 0.3, row_y[0] - 0.05, col_x[1] + cw/2 - 0.3, row_y[1] + ch + 0.05)

plt.tight_layout()
plt.savefig("comparison_matrix.png", dpi=180, bbox_inches="tight")
print("Saved.")