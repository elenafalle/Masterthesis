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

## Usage

```bash
# Run full pipeline (baseline + fine-tuning + evaluation)
python pipeline.py

# Baseline only
python pipeline.py --step 1 2

# Fine-tuning only
python pipeline.py --step 3 4

# Evaluation only
python pipeline.py --step 5 6

# Continue existing run
python pipeline.py --run-dir results/run_xyz
```

---

## Reference

Le-Duc, K. et al. (2024). MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder. arXiv. https://doi.org/10.48550/ARXIV.2409.14074
