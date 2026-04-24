"""
Explorative Datenanalyse – ElevenLabs TTS Datensatz
Führe dieses Skript aus: python analyse_elevenlabs.py

Erwartet folgende Ordnerstruktur:
  elevenlabs/
    analyse_elevenlabs.py     ← dieses Skript
    metadata.csv              ← file_name, text
    audio/                    ← Audiodateien (.mp3)

Venv erstellen und Voraussetzungen installieren:
    python3 -m venv venv
    source venv/bin/activate          # Mac/Linux
    venv\\Scripts\\activate           # Windows
    pip install pandas scipy matplotlib librosa

Zusätzlich wird ffmpeg benötigt:
    Mac:   brew install ffmpeg
    Linux: sudo apt install ffmpeg
"""

import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats

# ─── KONFIGURATION ────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent.resolve()
AUDIO_DIR = BASE_DIR / "audio"
OUT_DIR   = BASE_DIR

print(f"\n→ Basis-Dir : {BASE_DIR}")
print(f"→ Audio-Dir : {AUDIO_DIR}")
print(f"→ Ausgabe   : {OUT_DIR}\n")

# ─── 1. Daten laden ───────────────────────────────────────────────────────────

df = pd.read_csv(BASE_DIR / "metadata.csv")

# Gender und Stimme aus Dateiname extrahieren
# Format: sample_00001_de_female_Fiona.mp3
def extract_gender(filename):
    if "female" in filename:
        return "female"
    elif "male" in filename:
        return "male"
    return "unknown"

def extract_voice(filename):
    stem = Path(filename).stem  # ohne .mp3
    parts = stem.split("_")
    return parts[-1] if parts else "unknown"

df["gender"] = df["file_name"].apply(extract_gender)
df["voice"]  = df["file_name"].apply(extract_voice)

print(f"Geladen: {len(df)} Aufnahmen ({df['gender'].value_counts().to_dict()})")

# ─── 2. Überblick ─────────────────────────────────────────────────────────────

print("=" * 60)
print(f"ÜBERBLICK – {BASE_DIR.name.upper()}")
print("=" * 60)
print(f"Gesamtaufnahmen  : {len(df)}")
print(f"Stimmen (voices) : {df['voice'].nunique()} – {df['voice'].unique().tolist()}")
print(f"\nFehlende Werte pro Spalte:")
print(df.isnull().sum().to_string())

print("\nAUFNAHMEN PRO STIMME:")
print(df.groupby(["gender", "voice"])["file_name"].count().to_string())

# ─── 8. Visualisierung: Balkendiagramm Geschlecht ────────────────────────────

COLORS = ["#93C2DB", "#F4A97F"]

counts = df["gender"].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(counts.index, counts.values,
              color=COLORS[:len(counts)], edgecolor="white", linewidth=0.8)
ax.set_title(f"Aufnahmen nach Geschlecht – {BASE_DIR.name}", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Anzahl Aufnahmen", fontsize=11)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out_path = OUT_DIR / "aufnahmen_geschlecht.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n→ Grafik gespeichert: {out_path}")
plt.close()

# ─── 12. Audio-Validierung ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("AUDIO-VALIDIERUNG – ALLE DATEIEN")
print("=" * 60)

corrupt = []
ok = 0
total = len(df)

for i, (_, row) in enumerate(df.iterrows(), 1):
    filepath = AUDIO_DIR / row["file_name"]

    if i % 100 == 0:
        print(f"  {i}/{total} geprüft...")

    if not filepath.exists():
        corrupt.append({
            "audio_filename": row["file_name"],
            "gender":         row["gender"],
            "grund":          "datei_nicht_gefunden"
        })
        continue

    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(filepath), "-f", "null", "-"],
        capture_output=True, text=True
    )
    errors = [l for l in result.stderr.splitlines()
              if any(w in l for w in ["Error", "Invalid", "corrupt"])]
    if errors:
        corrupt.append({
            "audio_filename": row["file_name"],
            "gender":         row["gender"],
            "grund":          " | ".join(errors)
        })
    else:
        ok += 1

print(f"\n✓ OK:     {ok}")
print(f"✗ Defekt: {len(corrupt)}")

if corrupt:
    corrupt_df = pd.DataFrame(corrupt)
    corrupt_df.to_csv(OUT_DIR / "corrupt_audio.csv", index=False)
    print(corrupt_df.to_string())
else:
    print("→ Alle Dateien sind verwendbar!")

# ─── 13. ASR Audio-Qualität (RMS, SNR) ───────────────────────────────────────

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
        filepath = AUDIO_DIR / row["file_name"]

        if i % 100 == 0:
            print(f"  {i}/{total} analysiert...")

        if not filepath.exists():
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(filepath),
                 "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
                capture_output=True
            )

            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            Path(tmp_path).unlink(missing_ok=True)

            rms = librosa.feature.rms(y=y)[0]
            rms_db = librosa.amplitude_to_db(rms.mean())

            signal_power = np.mean(y ** 2)
            frame_power = rms ** 2
            noise_power = np.percentile(frame_power, 10)
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

            results.append({
                "audio_filename": row["file_name"],
                "gender":         row["gender"],
                "voice":          row["voice"],
                "rms_db":         round(float(rms_db), 2),
                "snr_db":         round(float(snr_db), 2),
            })

        except Exception as e:
            print(f"  Fehler bei {row['file_name']}: {e}")

    asr_df = pd.DataFrame(results)
    print(f"\nAnalysiert: {len(asr_df)} Dateien")

except ImportError:
    print("librosa nicht installiert – pip install librosa")
    asr_df = pd.DataFrame()

# ─── 14. Filterung ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ASR-QUALITÄT – FILTERUNG (RMS + SNR)")
print("=" * 60)

if not asr_df.empty:
    asr_df.to_csv(OUT_DIR / "asr_quality_combined.csv", index=False)

    print(f"\nGesamtübersicht:")
    print(asr_df[["rms_db", "snr_db"]].describe().round(2).to_string())

    print("\nDurchschnitt pro Stimme:")
    print(asr_df.groupby(["gender", "voice"])[["rms_db", "snr_db"]].mean().round(2).to_string())

    problematic = asr_df[
        (asr_df["rms_db"] < -40) |
        (asr_df["rms_db"] > -10) |
        (asr_df["snr_db"] < 0)
    ].copy()

    print(f"\nProblematische Aufnahmen: {len(problematic)}")

    if len(problematic) > 0:
        problematic["grund"] = ""
        problematic.loc[problematic["rms_db"] < -40, "grund"] += "zu_leise "
        problematic.loc[problematic["rms_db"] > -10, "grund"] += "zu_laut "
        problematic.loc[problematic["snr_db"] < 0,  "grund"] += "defekte_datei "
        problematic["grund"] = problematic["grund"].str.strip()
        problematic.to_csv(OUT_DIR / "asr_problematic.csv", index=False)
        print(problematic[["gender", "voice", "audio_filename", "rms_db", "snr_db", "grund"]].to_string())

    print(f"\n→ Vollständige Ergebnisse gespeichert: {OUT_DIR / 'asr_quality_combined.csv'}")

else:
    print("Schritt 13 nicht erfolgreich – kein Report möglich.")
