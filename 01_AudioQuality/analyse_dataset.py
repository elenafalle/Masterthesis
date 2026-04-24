"""
Explorative Datenanalyse – dataset_export.json
Führe dieses Skript aus: python analyse_dataset.py
Voraussetzungen: pip install pandas scipy matplotlib librosa
"""

import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats

AUDIO_DIR = "./audio"  # ← hier deinen Pfad zum Audio-Ordner anpassen

# ─── 1. Daten laden & flach machen ───────────────────────────────────────────

with open("dataset_export.json") as f:
    data = json.load(f)

rows = []
for r in data["recordings"]:
    rows.append({
        "id":               r["id"],
        "audio_filename":   r["audio_filename"],
        "duration_ms":      r["duration_ms"],
        "file_size":        r["file_size"],
        "created_at":       r["created_at"],
        "sentence_text":    r["sentence"]["text"],
        "sentence_category":r["sentence"]["category"],
        "user_id":          r["user"]["id"],
        "username":         r["user"]["username"],
        "gender":           r["user"]["demographics"]["gender"],
        "age_group":        r["user"]["demographics"]["age_group"],
        "profession":       r["user"]["demographics"]["healthcare_profession"],
        "language_region":  r["user"]["demographics"]["language_region"],
    })

df = pd.DataFrame(rows)
df["created_at"] = pd.to_datetime(df["created_at"])

# ─── 2. Überblick ─────────────────────────────────────────────────────────────

print("=" * 60)
print("ÜBERBLICK")
print("=" * 60)
print(f"Gesamtaufnahmen  : {len(df)}")
print(f"Einzigartige User: {df['user_id'].nunique()}")
print(f"\nFehlende Werte pro Spalte:")
print(df.isnull().sum().to_string())

print("\nUSER-METADATEN:")
user_meta = (
    df.groupby("user_id", dropna=False)
    .agg(
        username=("username", "first"),
        aufnahmen=("id", "count"),
        gender=("gender", "first"),
        age_group=("age_group", "first"),
        profession=("profession", "first"),
        language_region=("language_region", "first"),
    )
    .sort_values("aufnahmen", ascending=False)
)
print(user_meta.to_string())

# ─── 5. Deskriptive Statistik nach Gruppe ─────────────────────────────────────

def group_stats(groupby_col: str, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"DURATION (ms) NACH {label.upper()}")
    print("=" * 60)
    tbl = (
        df.groupby(groupby_col, dropna=False)["duration_ms"]
        .agg(["count", "mean", "median", "std"])
        .rename(columns={"count": "n", "mean": "Ø ms", "median": "Median ms", "std": "Std ms"})
        .round(0)
        .sort_values("Median ms", ascending=False)
    )
    print(tbl.to_string())

group_stats("gender",          "Geschlecht")
group_stats("age_group",       "Altersgruppe")
group_stats("profession",      "Berufsgruppe")
group_stats("language_region", "Region")

# ─── 7. Statistischer Test: Geschlecht ────────────────────────────────────────

print("\n" + "=" * 60)
print("STATISTISCHER TEST: GESCHLECHT (ohne 0-ms-Aufnahmen)")
print("=" * 60)
df_clean = df[df["duration_ms"] > 0]
male   = df_clean[df_clean["gender"] == "male"]["duration_ms"]
female = df_clean[df_clean["gender"] == "female"]["duration_ms"]
u_stat, p_val = stats.mannwhitneyu(male, female, alternative="two-sided")
print(f"Mann-Whitney U = {u_stat:.0f}, p = {p_val:.4f}")
print("→ " + ("Signifikanter Unterschied (p < 0.05)" if p_val < 0.05 else "Kein signifikanter Unterschied"))

# ─── 8. Visualisierungen ──────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Aufnahme-Qualität nach Gruppen", fontsize=14, fontweight="bold")

GROUPS = [
    ("gender",          "Geschlecht",  axes[0, 0]),
    ("profession",      "Berufsgruppe",axes[0, 1]),
    ("age_group",       "Altersgruppe",axes[1, 0]),
    ("language_region", "Region",      axes[1, 1]),
]

for col, title, ax in GROUPS:
    order = (
        df_clean.groupby(col)["duration_ms"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data_by_group = [df_clean[df_clean[col] == g]["duration_ms"].values for g in order]
    bp = ax.boxplot(data_by_group, patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#93C2DB")
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Dauer (ms)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig("qualitaet_nach_gruppen.png", dpi=150, bbox_inches="tight")
print("\n→ Grafik gespeichert: qualitaet_nach_gruppen.png")
plt.show()

# ─── 8b. Aufnahmen pro Gruppe (einzelne Grafiken) ────────────────────────────

SINGLE_GROUPS = [
    ("gender",          "Geschlecht",   "aufnahmen_geschlecht.png",  "#93C2DB"),
    ("age_group",       "Altersgruppe", "aufnahmen_altersgruppe.png","#F4A97F"),
    ("profession",      "Berufsgruppe", "aufnahmen_berufsgruppe.png","#A8D5A2"),
    ("language_region", "Region",       "aufnahmen_region.png",      "#C9B8E8"),
]

for col, title, filename, color in SINGLE_GROUPS:
    counts = df[col].fillna("Keine Angabe").value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values,
                  color=color, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Anzahl Aufnahmen", fontsize=11)
    ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=10)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts.values) * 0.01,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\n→ Grafik gespeichert: {filename}")
    plt.close()

# ─── 10. Zusammenfassung ASR-Training ─────────────────────────────────────────

print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG – ASR-TRAINING")
print("=" * 60)

total_hours = df["duration_ms"].sum() / 1000 / 3600
print(f"Gesamt Audio (laut JSON): {total_hours:.2f} Stunden")

print("\nAudio pro Sprecher (Stunden):")
per_user = (
    df.groupby(["user_id", "username"])["duration_ms"]
    .sum()
    .div(1000 * 3600)
    .round(3)
    .sort_values(ascending=False)
)
print(per_user.to_string())

print("\nDemographische Balance:")
for col in ["gender", "age_group", "profession", "language_region"]:
    print(f"\n{col}:")
    tbl = df.groupby(col, dropna=False)["duration_ms"].agg(
        aufnahmen="count",
        stunden=lambda x: round(x.sum() / 1000 / 3600, 2)
    )
    print(tbl.to_string())

# ─── 12. Audio-Validierung (alle Dateien) ─────────────────────────────────────

print("\n" + "=" * 60)
print("AUDIO-VALIDIERUNG – ALLE DATEIEN")
print("=" * 60)

corrupt = []
ok = 0
total = len(df)

for i, (_, row) in enumerate(df.iterrows(), 1):
    filepath = Path(AUDIO_DIR) / row["audio_filename"]

    if i % 100 == 0:
        print(f"  {i}/{total} geprüft...")

    if not filepath.exists():
        corrupt.append({
            "audio_filename": row["audio_filename"],
            "username":       row["username"],
            "grund":          "datei_nicht_gefunden"
        })
        continue

    result = subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-v", "error", "-i", str(filepath), "-f", "null", "-"],
        capture_output=True, text=True
    )
    errors = [l for l in result.stderr.splitlines()
              if any(w in l for w in ["Error", "Invalid", "corrupt"])]
    if errors:
        corrupt.append({
            "audio_filename": row["audio_filename"],
            "username":       row["username"],
            "grund":          " | ".join(errors)
        })
    else:
        ok += 1

print(f"\n✓ OK:     {ok}")
print(f"✗ Defekt: {len(corrupt)}")

if corrupt:
    import shutil
    corrupt_df = pd.DataFrame(corrupt)
    corrupt_df.to_csv("corrupt_audio.csv", index=False)
    print("→ Defekte Dateien gespeichert: corrupt_audio.csv")
    print(corrupt_df.to_string())

    corrupt_dir = Path("corrupt_audio")
    corrupt_dir.mkdir(exist_ok=True)
    copied, missing = 0, 0
    for _, row in corrupt_df.iterrows():
        src = Path(AUDIO_DIR) / row["audio_filename"]
        if src.exists():
            shutil.copy2(src, corrupt_dir / row["audio_filename"])
            copied += 1
        else:
            missing += 1
    print(f"\n→ Ordner erstellt: {corrupt_dir.resolve()}")
    print(f"  {copied} Dateien kopiert, {missing} nicht gefunden")
else:
    print("→ Alle Dateien sind verwendbar!")

# ─── 13. ASR Audio-Qualität (RMS, SNR) ───────────────────────────────────────
#
# Ausschlusskriterien (Literaturbelege):
#
#   Kriterium               Schwellwert   Quelle
#   ──────────────────────────────────────────────────────────────────
#   Loudness (RMS) zu laut  > -10 dBFS   validiert anhand einer representativen Stichprobe
#   Loudness (RMS) zu leise < -40 dBFS   validiert anhand einer representativen Stichprobe
#   Background Noise (SNR)  < 10 dB      Liu et al. (2020), Applied Acoustics

print("\n" + "=" * 60)
print("ASR AUDIO-QUALITÄT – RMS & SNR")
print("=" * 60)

try:
    import librosa
    import tempfile
    import warnings
    warnings.filterwarnings("ignore")

    results = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        filepath = Path(AUDIO_DIR) / row["audio_filename"]

        if i % 100 == 0:
            print(f"  {i}/{total} analysiert...")

        if not filepath.exists():
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                ["/opt/homebrew/bin/ffmpeg", "-y", "-i", str(filepath),
                 "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
                capture_output=True
            )

            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            Path(tmp_path).unlink(missing_ok=True)

            # Lautstärke (RMS in dBFS)
            # RMS misst die rohe Signalenergie – für ASR relevanter als
            # wahrnehmungsgewichtetes LUFS, da ASR-Modelle auf dem
            # ungewichteten Rohsignal operieren
            # (vgl. Tomanek et al., 2024, JSLHR)
            rms = librosa.feature.rms(y=y)[0]
            rms_db = librosa.amplitude_to_db(rms.mean())

            # SNR-Schätzung: Signal- vs. Rauschenergie
            # Schwellwert < 10 dB nach Liu et al. (2020), Applied Acoustics
            signal_power = np.mean(y ** 2)
            frame_power = rms ** 2
            noise_power = np.percentile(frame_power, 10)
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

            results.append({
                "audio_filename": row["audio_filename"],
                "username":       row["username"],
                "rms_db":         round(float(rms_db), 2),
                "snr_db":         round(float(snr_db), 2),
            })

        except Exception as e:
            print(f"  Fehler bei {row['audio_filename']}: {e}")

    asr_df = pd.DataFrame(results)
    print(f"\nAnalysiert: {len(asr_df)} Dateien")

except ImportError:
    print("librosa nicht installiert – bitte ausführen: pip install librosa")
    asr_df = pd.DataFrame()

# ─── 14. ASR-Qualitätstabelle & Filterung ─────────────────────────────────────

print("\n" + "=" * 60)
print("ASR-QUALITÄT – FILTERUNG (RMS + SNR)")
print("=" * 60)

if not asr_df.empty:
    asr_df.to_csv("asr_quality_combined.csv", index=False)

    print(f"\nGesamtübersicht:")
    print(asr_df[["rms_db", "snr_db"]].describe().round(2).to_string())

    # Durchschnitt pro Sprecher
    print("\nDurchschnitt pro Sprecher:")
    print(
        asr_df.groupby("username")[["rms_db", "snr_db"]]
        .mean().round(2).sort_values("snr_db").to_string()
    )

    # Problematische Aufnahmen filtern
    problematic = asr_df[
        (asr_df["rms_db"] < -40) |
        (asr_df["rms_db"] > -10) |
        (asr_df["snr_db"] < 10)
    ].copy()

    print(f"\nProblematische Aufnahmen: {len(problematic)}")

    if len(problematic) > 0:
        problematic["grund"] = ""
        problematic.loc[problematic["rms_db"] < -40, "grund"] += "zu_leise "
        problematic.loc[problematic["rms_db"] > -10, "grund"] += "zu_laut "
        problematic.loc[problematic["snr_db"] < 10,  "grund"] += "zu_lautes_rauschen "
        problematic["grund"] = problematic["grund"].str.strip()
        problematic.to_csv("asr_problematic.csv", index=False)
        print("→ Gespeichert: asr_problematic.csv")
        print(problematic[[
            "username", "audio_filename",
            "rms_db", "snr_db", "grund"
        ]].to_string())

    print("\n→ Vollständige Ergebnisse gespeichert: asr_quality_combined.csv")

else:
    print("Schritt 13 nicht erfolgreich – kein Report möglich.")
