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
| 1 | Audio quality assessment (SNR, RMS filtering) | `DialektDataset/`, `ElevenLabs/`, `Voxtral/` |
| 2 | Baseline WER quality check for synthetic datasets | `ElevenLabs/`, `Voxtral/` |
| 3 | Self-created dialect dataset evaluation (baseline + fine-tuned) | `DialektDataset/` |
| 4 | Reference dataset evaluation (baseline + fine-tuned) | `ReferenceDataset/` |
| 5 | Combined dataset evaluation (augmentation with synthetic data) | `combined/` |

---

## Repository Structure

```
├── ElevenLabs/            # Stage 2 — Quality check for ElevenLabs synthetic data
├── Voxtral/               # Stage 2 — Quality check for Voxtral synthetic data
├── DialektDataset/        # Stage 3 — Self-created dialect dataset (RQ1 & RQ2)
├── ReferenceDataset/      # Stage 4 — Reference dataset evaluation (MultiMed)
├── combined/              # Stage 5 — Combined dataset experiments (RQ3)
└── README.md
```

---

## Models

- **Whisper large-v3**: https://huggingface.co/openai/whisper-large-v3
- **Parakeet TDT 0.6B v3**: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3

---

## Requirements

```bash
pip install torch transformers peft jiwer bert-score soundfile librosa numpy pandas scipy matplotlib
```

---

## Recommended Execution Order

```bash
# Stage 2: Quality check synthetic datasets
cd ElevenLabs && python pipeline_quality_check.py --dataset elevenlabs
cd Voxtral    && python pipeline_quality_check.py --dataset voxtral

# Stage 3: Dialect dataset
cd DialektDataset && python pipeline.py

# Stage 4: Reference dataset
cd ReferenceDataset && python pipeline.py

# Stage 5: Combined
cd combined && python pipeline_rq3.py --config voxtral
cd combined && python pipeline_rq3.py --config elevenlabs
cd combined && python pipeline_rq3.py --config combined
```

---

## Citation

Falle, E. (2026). *Creating datasets for domain-specific dialect solutions in automatic speech recognition*. Master's Thesis, St. Pölten University of Applied Sciences.
