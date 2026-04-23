# 01_AudioQuality — Stage 1: Audio Quality Assessment

This folder contains the audio quality analysis scripts for **Stage 1** of the evaluation procedure.

---

## What happens here

Before model evaluation, the acoustic suitability of all datasets was verified. Different quality criteria were applied depending on the dataset type:

| Dataset | Method | Criteria |
|---------|--------|----------|
| Self-created dialect dataset | RMS + SNR | Loudness and background noise filtering |
| ElevenLabs synthetic dataset | RMS only | Loudness filtering (no real background noise) |
| Voxtral synthetic dataset | RMS only | Loudness filtering (no real background noise) |
| Reference dataset (MultiMed) | Manual listening | Already validated in prior research |

---

## Quality Thresholds

| Criterion | Threshold | Classification |
|-----------|-----------|----------------|
| Loudness (RMS) | > -10 dBFS | Too loud |
| Loudness (RMS) | < -40 dBFS | Too quiet |
| Background Noise (SNR) | < 10 dB | Excessive background noise (self-recorded only) |
| Background Noise (SNR) | < 0 dB | Defective file (synthetic datasets only) |

Since synthetically generated data contains no real environmental noise, the SNR threshold was adjusted to < 0 dB for ElevenLabs and Voxtral to ensure that only technically defective files are removed.

---

## Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `analyse_dataset.py` | Self-created dialect dataset | RMS + SNR filtering, demographic analysis, audio validation |
| `analyze_elevenlabs.py` | ElevenLabs synthetic dataset | RMS filtering, audio validation, gender/voice analysis |
| `analyse_voxtral.py` | Voxtral synthetic dataset | RMS filtering, audio validation, gender/voice analysis |

---

## Usage

```bash
# Self-created dataset
python analyse_dataset.py

# ElevenLabs dataset
cd elevenlabs/
python analyze_elevenlabs.py

# Voxtral dataset
cd voxtral/
python analyse_voxtral.py
```

---

## Requirements

```bash
pip install pandas scipy matplotlib librosa
```

---

## Output

Each script produces the following output files:

- `asr_quality_combined.csv` — RMS and SNR values for all audio files
- `asr_problematic.csv` — flagged recordings with reason (if any)
- `aufnahmen_geschlecht.png` — bar chart of recordings by gender
- `corrupt_audio.csv` — technically defective files (if any)
