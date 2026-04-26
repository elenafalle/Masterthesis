# ReferenceDataset — Stage 4: Reference Dataset Evaluation

This folder contains the pipeline for **Stage 4** of the evaluation procedure — evaluation on the MultiMed reference dataset.

---

## What happens here

Both models are evaluated on the MultiMed reference dataset (German subset) in baseline and fine-tuned configurations. This enables a direct comparison between performance on dialectal speech (Stage 3) and standard German speech, allowing assessment of generalization and overfitting. The MultiMed dataset is the first multilingual medical ASR dataset and contains high-quality, human-annotated recordings sourced from professional medical YouTube channels.

---

## Data Structure

```
ReferenceDataset/
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
| `parakeet.py` | Parakeet baseline evaluation and LayerNorm fine-tuning |

---

## Usage

```bash
# Run full pipeline (baseline + fine-tuning + evaluation)
python3 pipeline.py

# Baseline only
python3 pipeline.py --step 1 2

# Fine-tuning only
python3 pipeline.py --step 3 4

# Evaluation only
python3 pipeline.py --step 5 6

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

---

## Reference

Le-Duc, K. et al. (2024). MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder. arXiv. https://doi.org/10.48550/ARXIV.2409.14074
