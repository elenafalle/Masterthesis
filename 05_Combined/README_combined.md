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

## Usage

```bash
# Step 1: Merge datasets first
python3 -m src.merge_datasets --config voxtral
python3 -m src.merge_datasets --config elevenlabs
python3 -m src.merge_datasets --config combined

# Step 2: Run fine-tuning and evaluation
python3 pipeline.py --config voxtral
python3 pipeline.py --config elevenlabs
python3 pipeline.py --config combined

# Continue existing run
python3 pipeline.py --config voxtral --step 7 --run-dir results/run_xyz

# Visualizations
python3 src/plot_rq3.py
python3 src/plot_rq3_clustering.py
python3 comparison_matrix.py
```
