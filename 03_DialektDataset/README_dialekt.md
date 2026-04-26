# DialektDataset — Stage 3: Self-created Dialect Dataset

This folder contains the pipeline for **Stage 3** of the evaluation procedure, corresponding to **RQ1 and RQ2** of the thesis.

---

## What happens here

Both models (Whisper large-v3 and Parakeet TDT 0.6B v3) are evaluated on the self-created Austrian dialect dataset in two configurations:
- **Baseline** — pretrained models without fine-tuning
- **Fine-tuned** — models adapted using domain-specific dialect data (LoRA for Whisper, LayerNorm for Parakeet)

In addition to WER, an error clustering analysis based on BERTScore is conducted to categorize transcription errors by clinical severity (massive error / medical error / acceptable).

---

## Data Structure

```
DialektDataset/
├── train/
│   ├── metadata.csv
│   └── audio/
├── eval/
│   ├── metadata.csv
│   └── audio/
├── test/
│   ├── metadata.csv
│   └── audio/
├── src/
└── pipeline.py
```
---

## src/ — Helper Functions

| File | Description |
|------|-------------|
| `config.py` | Paths and hyperparameters |
| `utils.py` | Audio loading, WER computation, Input/Output helpers |
| `whisper.py` | Whisper baseline evaluation and LoRA fine-tuning |
| `parakeet.py` | Parakeet baseline evaluation and LayerNorm fine-tuning|
| `error_clustering.py` | BERTScore-based error clustering: classifies transcription errors into massive, medical and acceptable (Step 7) |
| `wer_breakdown.py` | WER breakdown by substitutions, deletions and insertions (Step 8) |

---

## Usage

```bash
# Run full pipeline (Steps 1-8)
python3 pipeline.py

# Baseline only
python3 pipeline.py --step 1 2

# Fine-tuning only
python3 pipeline.py --step 3 4

# Evaluation only
python3 pipeline.py --step 5 6

# Error clustering only
python3 pipeline.py --step 7

# WER Breakdown only
python3 pipeline.py --step 8

# Continue existing run
python3 pipeline.py --step 8 --run-dir results/run_xyz
```

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Baseline Whisper evaluation |
| 2 | Baseline Parakeet evaluation |
| 3 | Fine-tune Whisper (LoRA) |
| 4 | Fine-tune Parakeet (LayerNorm) |
| 5 | Evaluate fine-tuned Whisper |
| 6 | Evaluate fine-tuned Parakeet |
| 7 | Error clustering (BERTScore) |
| 8 | WER breakdown (Sub / Del / Ins) |
