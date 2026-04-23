"""
src/merge_datasets.py
=====================
Erstellt drei Combined Training Datasets für RQ3.

Konfigurationen:
    1. train_merge_voxtral/     — Self-created + Voxtral
    2. train_merge_elevenlabs/  — Self-created + ElevenLabs
    3. train_merge_combined/    — Self-created + Voxtral + ElevenLabs

Ausgabe je Konfiguration:
    ~/Combined/train_merge_<n>/
        ├── audio/          # Alle WAV/MP3-Dateien mit Prefix
        └── metadata.csv    # Kombinierte metadata

Validation und Test:
    ~/Combined/eval/
    ~/Combined/test/

Namenskonvention:
    sc_  = Self-created
    vx_  = Voxtral
    el_  = ElevenLabs

Nutzung:
    python -m src.merge_datasets              # alle drei Konfigurationen
    python -m src.merge_datasets --config voxtral
    python -m src.merge_datasets --config elevenlabs
    python -m src.merge_datasets --config combined
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, log

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
SC_TRAIN_DIR   = BASE_DIR / "train"
VOXTRAL_DIR    = BASE_DIR / "all_voxtral"
ELEVENLABS_DIR = BASE_DIR / "all_elevenlabs"


def _copy_source(source_dir: Path, audio_subdir: str, prefix: str,
                 meta_filename: str, output_audio: Path, rows: list) -> tuple:
    meta_path = source_dir / meta_filename
    if not meta_path.exists():
        log.warning(f"[MergeDatasets] metadata.csv nicht gefunden: {meta_path}")
        return 0, 0

    meta = pd.read_csv(meta_path)
    audio_dir = source_dir / audio_subdir if audio_subdir else source_dir
    copied = 0
    skipped = 0

    for _, row in meta.iterrows():
        src = audio_dir / row["file_name"]
        new_name = prefix + row["file_name"]
        dst = output_audio / new_name

        if not src.exists():
            log.warning(f"[MergeDatasets] Audio nicht gefunden: {src}")
            continue

        if dst.exists():
            skipped += 1
            rows.append({"file_name": new_name, "text": row["text"]})
            continue

        shutil.copy2(src, dst)
        rows.append({"file_name": new_name, "text": row["text"]})
        copied += 1

    return copied, skipped


def merge_single(config_name: str, include_voxtral: bool, include_elevenlabs: bool):
    output_dir   = BASE_DIR / f"train_merge_{config_name}"
    output_audio = output_dir / "audio"
    output_audio.mkdir(parents=True, exist_ok=True)

    log.info(f"[MergeDatasets] ==============================")
    log.info(f"[MergeDatasets] Konfiguration: {config_name}")
    log.info(f"[MergeDatasets] Ausgabe: {output_dir}")

    rows = []

    # 1. Self-created
    copied_sc, skipped_sc = _copy_source(
        SC_TRAIN_DIR, "audio", "sc_", "metadata.csv", output_audio, rows
    )
    log.info(f"[MergeDatasets] Self-created:  {copied_sc} neu  |  {skipped_sc} übersprungen")

    # 2. Voxtral
    copied_vx, skipped_vx = 0, 0
    if include_voxtral:
        copied_vx, skipped_vx = _copy_source(
            VOXTRAL_DIR, "audio", "vx_", "metadata.csv", output_audio, rows
        )
        log.info(f"[MergeDatasets] Voxtral:       {copied_vx} neu  |  {skipped_vx} übersprungen")

    # 3. ElevenLabs
    copied_el, skipped_el = 0, 0
    if include_elevenlabs:
        copied_el, skipped_el = _copy_source(
            ELEVENLABS_DIR, "audio", "el_", "metadata.csv", output_audio, rows
        )
        log.info(f"[MergeDatasets] ElevenLabs:    {copied_el} neu  |  {skipped_el} übersprungen")

    # metadata.csv speichern
    combined_df = pd.DataFrame(rows)
    combined_df.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8")

    total = len(combined_df)
    log.info(f"[MergeDatasets] Total:         {total} Einträge")
    log.info(f"[MergeDatasets] ==============================")

    return total


def merge_all(config: str = "all"):
    configs = {
        "voxtral":    (True,  False),
        "elevenlabs": (False, True),
        "combined":   (True,  True),
    }

    if config == "all":
        to_run = configs
    elif config in configs:
        to_run = {config: configs[config]}
    else:
        raise ValueError(f"Unbekannte Konfiguration: {config}")

    for name, (vx, el) in to_run.items():
        total = merge_single(name, vx, el)
        log.info(f"[MergeDatasets] {name}: {total} Samples gesamt")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge datasets for RQ3")
    parser.add_argument(
        "--config", type=str,
        default="all",
        choices=["voxtral", "elevenlabs", "combined", "all"],
        help="Welche Konfiguration erstellen (default: all)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_all(config=args.config)