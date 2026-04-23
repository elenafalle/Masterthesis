# ElevenLabs — Stage 2: Baseline Quality Check

This folder contains the pipeline for **Stage 2** of the evaluation procedure — the baseline WER quality check for the ElevenLabs synthetic dataset.

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

## Usage

```bash
# Both models
python pipeline_quality_check.py --dataset elevenlabs --model both

# Whisper only
python pipeline_quality_check.py --dataset elevenlabs --model whisper

# Parakeet only
python pipeline_quality_check.py --dataset elevenlabs --model parakeet
```
