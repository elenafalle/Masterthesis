# combined — Stage 5: Combined Dataset Experiments (RQ3)

This folder contains the pipeline for **Stage 5** of the evaluation procedure, corresponding to **RQ3** of the thesis.

---

## What happens here

To investigate whether augmenting the self-created dialect dataset with synthetically generated speech data can further improve ASR performance, three combined training configurations are evaluated:

| Configuration | Training Data | Samples |
|---------------|--------------|---------|
| `voxtral` | Self-created + Voxtral | 3,867 |
| `elevenlabs` | Self-created + ElevenLabs | 3,867 |
| `combined` | Self-created + Voxtral + ElevenLabs | 6,499 |

Validation and test splits are used exclusively from the self-created dialect dataset to ensure evaluation under identical conditions as Stage 3.

---

## Data Structure

```
combined/
├── src/
├── pipeline.py
└── comparison_matrix.py
```

---

## src/ — Helper Functions

| File | Description |
|------|-------------|
| `config.py` | Paths and hyperparameters |
| `utils.py` | Audio loading, WER computation, Input/Output helpers |
| `whisper.py` | Whisper baseline evaluation and LoRA fine-tuning |
| `parakeet.py` | Parakeet baseline evaluation and LayerNorm fine-tuning |
| `error_clustering.py` | BERTScore-based error clustering - classifies transcription errors into massive, medical and acceptable |
| `merge_datasets.py` | Merges self-created dialect dataset with synthetic data (Voxtral, ElevenLabs) |
| `plot_rq3.py` | WER comparison chart across all RQ3 configurations |
| `plot_rq3_clustering.py` | Error clustering stacked bar chart for RQ3 Parakeet configurations |

## Other files

| File | Description |
|------|-------------|
| `comparison_matrix.py` | Generates the A/B/C/D comparison matrix showing WER results across all four evaluation configurations (baseline/fine-tuned × dialect/reference) |

---

## Pipeline Steps

The pipeline is run three times - once per configuration - to enable comparison across augmentation strategies:

| Step | Description |
|------|-------------|
| 0 | Merge datasets - run once per configuration before starting the pipeline |
| 3 | Fine-tune Whisper (LoRA) on merged dataset |
| 4 | Fine-tune Parakeet (LayerNorm) on merged dataset |
| 5 | Evaluate fine-tuned Whisper on dialect test set |
| 6 | Evaluate fine-tuned Parakeet on dialect test set |

Steps 1 and 2 (baseline evaluation) are not included - baselines were established in Stage 3 and Stage 4.

---

## Usage

```bash
# Configuration 1: Self-created + Voxtral
python3 -m src.merge_datasets --config voxtral
python3 pipeline.py --config voxtral

# Configuration 2: Self-created + ElevenLabs
python3 -m src.merge_datasets --config elevenlabs
python3 pipeline.py --config elevenlabs

# Configuration 3: Self-created + Voxtral + ElevenLabs
python3 -m src.merge_datasets --config combined
python3 pipeline.py --config combined

# Visualizations
python3 src/plot_rq3.py
python3 src/plot_rq3_clustering.py
python3 comparison_matrix.py
```
