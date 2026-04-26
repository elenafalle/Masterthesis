# ElevenLabs - Stage 2: Baseline Quality Check

This folder contains the pipeline for **Stage 2** of the evaluation procedure - the baseline WER quality check for the ElevenLabs synthetic dataset.

---

## What happens here

The ElevenLabs dataset was generated using the Eleven Multilingual v2 model and consists of 2,632 audio files (one female and one male voice) based on the same input texts as the self-created dialect dataset. Both ASR models are evaluated in their pretrained baseline configuration on the full dataset to assess its suitability as additional training data. A lower WER indicates higher transcription accuracy and greater intelligibility.

---

## Data Structure

```
ElevenLabs/
├── all/
│   ├── metadata.csv
│   └── audio/
├── src/
└── pipeline_quality_check.py
```

---

## src/ — Helper Functions

| File | Description |
|------|-------------|
| `config.py` | Paths and hyperparameters |
| `utils.py` | Audio loading, WER computation, Input/Output helpers |
| `whisper.py` | Whisper baseline evaluation |
| `parakeet.py` | Parakeet baseline evaluation |

---

## Usage

```bash
python pipeline_quality_check.py --dataset elevenlabs 

```

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Baseline Whisper evaluation - computes WER on the full ElevenLabs dataset |
| 2 | Baseline Parakeet evaluation - computes WER on the full ElevenLabs dataset |

No fine-tuning is performed: this pipeline is used exclusively for data quality assessment.
