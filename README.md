# Masterthesis — Dialect-Aware ASR for Medical Emergency Communication

**"Creating datasets for domain-specific dialect solutions in automatic speech recognition"**
Elena Falle, St. Pölten University of Applied Sciences, 2026

---

## Overview

This repository contains all code developed for the master's thesis. The study evaluates the impact of Austrian dialect variation on ASR performance in simulated emergency medical communication using two models — Whisper large-v3 and Parakeet TDT 0.6B v3.

---

## Evaluation Pipeline — Five Stages

The experimental pipeline follows five structured stages:

| Stage | Description | Folder |
|-------|-------------|--------|
| 1 | Audio quality assessment (SNR, RMS filtering) | `01_AudioQuality/` |
| 2 | Baseline WER quality check for synthetic datasets | `02_ElevenLabs/`, `02_Voxtral/` |
| 3 | Self-created dialect dataset evaluation (baseline + fine-tuned) | `03_DialektDataset/` |
| 4 | Reference dataset evaluation (baseline + fine-tuned) | `04_ReferenceDataset/` |
| 5 | Combined dataset evaluation (augmentation with synthetic data) | `05_Combined/` |

---

## Repository Structure

```
├── 01_AudioQuality/       # Stage 1 — Audio quality assessment scripts
├── 02_ElevenLabs/         # Stage 2 — Quality check for ElevenLabs synthetic data
├── 02_Voxtral/            # Stage 2 — Quality check for Voxtral synthetic data
├── 03_DialektDataset/     # Stage 3 — Self-created dialect dataset (RQ1 & RQ2)
├── 04_ReferenceDataset/   # Stage 4 — Reference dataset evaluation (MultiMed)
├── 05_Combined/           # Stage 5 — Combined dataset experiments (RQ3)
└── README.md
```

---

## Models

- **Whisper large-v3**: https://huggingface.co/openai/whisper-large-v3
- **Parakeet TDT 0.6B v3**: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3

---

## Recommended Execution Order

```bash
# Stage 1: Audio quality check
cd 01_AudioQuality
python3 analyse_dataset.py          # self-created dataset
python3 analyze_elevenlabs.py       # ElevenLabs dataset
python3 analyse_voxtral.py          # Voxtral dataset

# Stage 2: Baseline WER quality check
cd 02_ElevenLabs
python3 pipeline_quality_check.py --dataset elevenlabs

cd 02_Voxtral
python3 pipeline_quality_check.py --dataset voxtral

# Stage 3: Dialect dataset (RQ1 & RQ2)
cd 03_DialektDataset
python3 pipeline.py

# Stage 4: Reference dataset
cd 04_ReferenceDataset
python3 pipeline.py

# Stage 5: Combined dataset (RQ3)
cd 05_Combined
python3 -m src.merge_datasets --config voxtral
python3 -m src.merge_datasets --config elevenlabs
python3 -m src.merge_datasets --config combined
python3 pipeline.py --config voxtral
python3 pipeline.py --config elevenlabs
python3 pipeline.py --config combined
```

---
