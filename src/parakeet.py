"""Parakeet: Evaluation und Fine-Tuning."""

import json
import math
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

import src.config as cfg
from src.config import VAL_DIR, TEST_DIR, PARAKEET_HPARAMS, log
from src.utils import compute_wer, load_metadata, save_json, save_predictions


# ---------------------------------------------------------------------------
# LoRA für NeMo FastConformer
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Lightweight LoRA-Wrapper für nn.Linear.

    Berechnet: y = W·x + (B·A·x) * (alpha/r)
    Nur lora_A und lora_B werden trainiert; das Basisgewicht bleibt eingefroren.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_out = self.drop(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + lora_out * self.scale


def _apply_lora(encoder: nn.Module, target_names: list[str],
                r: int, alpha: float, dropout: float) -> None:
    """Ersetzt passende nn.Linear-Layer im Encoder durch LoRALinear-Wrapper."""
    replaced = 0
    for mod_name, module in list(encoder.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = mod_name.split(".")[-1]
        if leaf not in target_names:
            continue
        parts = mod_name.split(".")
        parent = encoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced += 1

    for name, param in encoder.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name

    n_lora = sum(p.numel() for n, p in encoder.named_parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in encoder.parameters())
    log.info(f"LoRA angewendet: {replaced} Layer | {n_lora/1e6:.1f}M / {n_total/1e6:.0f}M Encoder-Params trainierbar")


def _speed_perturb_manifest(
    original_manifest: Path,
    aug_manifest: Path,
    speed_rates: list[float],
    aug_audio_dir: Path,
) -> int:
    from scipy.signal import resample_poly as _resample_poly

    aug_audio_dir.mkdir(parents=True, exist_ok=True)
    with open(original_manifest, "r", encoding="utf-8") as f:
        originals = [json.loads(line) for line in f if line.strip()]

    entries = []
    for entry in originals:
        for rate in speed_rates:
            if abs(rate - 1.0) < 1e-6:
                entries.append(entry)
                continue

            stem = Path(entry["audio_filepath"]).stem
            out_path = aug_audio_dir / f"{stem}_sp{rate:.1f}.wav"
            if not out_path.exists():
                data, sr = sf.read(entry["audio_filepath"])
                up, down = 10, round(10 * rate)
                g = gcd(up, down)
                perturbed = _resample_poly(data, up // g, down // g).astype(np.float32)
                sf.write(str(out_path), perturbed, sr)

            info = sf.info(str(out_path))
            entries.append({
                "audio_filepath": str(out_path),
                "text": entry["text"],
                "duration": round(info.duration, 3),
            })

    aug_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(aug_manifest, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    rates_str = " + ".join(f"{r:.1f}×" for r in speed_rates)
    log.info(f"Speed-Perturbation [{rates_str}]: {len(originals)} → {len(entries)} Eintraege")
    return len(entries)


def _merge_lora(encoder: nn.Module) -> None:
    merged = 0
    for mod_name, module in list(encoder.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        with torch.no_grad():
            module.base.weight.data += (module.lora_B @ module.lora_A) * module.scale
        module.base.weight.requires_grad = True
        if module.base.bias is not None:
            module.base.bias.requires_grad = True
        parts = mod_name.split(".")
        parent = encoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.base)
        merged += 1
    log.info(f"LoRA zusammengeführt: {merged} Layer")


def evaluate(model_path_or_name: str, label: str, out_dir: Path) -> dict:
    """Evaluate a Parakeet model on the test set and save results."""
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    log.info(f"{'='*60}")
    log.info(f"  {label}")
    log.info(f"{'='*60}")

    if model_path_or_name.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(model_path_or_name)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path_or_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    log.info(f"Device: {device} | Modell: {model_path_or_name}")

    with open_dict(model.cfg.decoding):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(model.cfg.decoding, verbose=False)

    test_samples = load_metadata(TEST_DIR)
    audio_paths = [s["audio_path"] for s in test_samples]
    references = [s["text"] for s in test_samples]

    t0 = time.time()
    raw_output = model.transcribe(audio_paths, batch_size=4)
    elapsed = time.time() - t0

    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]
    hypotheses = []
    for item in raw_output:
        if isinstance(item, str):
            hypotheses.append(item)
        elif hasattr(item, "text"):
            hypotheses.append(item.text)
        else:
            hypotheses.append(str(item))

    word_error_rate = compute_wer(references, hypotheses)
    log.info(f"WER: {word_error_rate:.4f} ({word_error_rate*100:.2f}%) | Zeit: {elapsed:.1f}s")

    result = {
        "model": model_path_or_name,
        "label": label,
        "wer": round(word_error_rate, 6),
        "wer_percent": round(word_error_rate * 100, 2),
        "num_samples": len(test_samples),
        "elapsed_seconds": round(elapsed, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "eval.json", result)
    save_predictions(out_dir / "predictions.jsonl", test_samples, references, hypotheses)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _csv_to_nemo_manifest(split_dir: Path, manifest_path: Path) -> None:
    """Convert metadata.csv + audio/ to a NeMo-compatible JSONL manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    samples = load_metadata(split_dir)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for s in samples:
            info = sf.info(s["audio_path"])
            entry = {
                "audio_filepath": s["audio_path"],
                "text": s["text"],
                "duration": info.duration,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"Manifest: {manifest_path} ({len(samples)} Eintraege)")


def finetune(run_dir: Path) -> None:
    """Fine-tune Parakeet TDT 0.6B v3 on the training set."""
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl

    log.info(f"{'='*60}")
    log.info("  Fine-Tuning Parakeet")
    log.info(f"{'='*60}")

    hp = PARAKEET_HPARAMS
    parakeet_dir = run_dir / "parakeet"
    OUT_PATH = str(parakeet_dir / "model" / "parakeet_finetuned.nemo")

    manifests_dir = parakeet_dir / "nemo_manifests"
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"

    _csv_to_nemo_manifest(cfg.TRAIN_DIR, train_manifest)
    _csv_to_nemo_manifest(VAL_DIR, val_manifest)

    aug_train_manifest = manifests_dir / "train_manifest_augmented.jsonl"
    aug_audio_dir = parakeet_dir / "aug_audio"
    n_train = _speed_perturb_manifest(
        train_manifest, aug_train_manifest,
        speed_rates=hp.get("speed_rates", [0.9, 1.0, 1.1]),
        aug_audio_dir=aug_audio_dir,
    )

    log.info("Lade vortrainiertes Parakeet-Modell...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=hp["model"])

    log.info("Aktualisiere Konfiguration...")
    with open_dict(asr_model.cfg):
        asr_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
        asr_model.cfg.train_ds.manifest_filepath = str(aug_train_manifest.resolve())
        asr_model.cfg.validation_ds.manifest_filepath = str(val_manifest.resolve())
        asr_model.cfg.train_ds.batch_size = hp["batch_size"]
        asr_model.cfg.validation_ds.batch_size = hp["batch_size"]
        asr_model.cfg.train_ds.num_workers = hp["num_workers"]
        asr_model.cfg.validation_ds.num_workers = hp["num_workers"]
        asr_model.cfg.train_ds.text_field = "text"
        asr_model.cfg.validation_ds.text_field = "text"
        asr_model.cfg.train_ds.pretokenize = False
        asr_model.cfg.validation_ds.pretokenize = False
        asr_model.cfg.train_ds.max_duration = 40.0

        steps_per_epoch = max(1, (n_train // hp["batch_size"]) // hp["gradient_accumulation_steps"])
        total_steps = steps_per_epoch * hp["epochs"]
        warmup_steps = max(50, total_steps // 10)
        if "optim" in asr_model.cfg:
            asr_model.cfg.optim.lr = hp["learning_rate"]
            asr_model.cfg.optim.weight_decay = hp.get("weight_decay", 1e-3)
            asr_model.cfg.optim.sched = {
                "name": "CosineAnnealing",
                "max_steps": total_steps,
                "warmup_steps": warmup_steps,
                "min_lr": hp.get("min_lr", 1e-7),
                "last_epoch": -1,
            }
        log.info(f"LR-Scheduler: CosineAnnealing | total_steps={total_steps}"
                 f" | warmup={warmup_steps} (10%)")

    asr_model.change_decoding_strategy(asr_model.cfg.decoding, verbose=False)

    if hp.get("freeze_encoder", True):
        asr_model.encoder.freeze()
        n_ln = 0
        for _, module in asr_model.encoder.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True
                    n_ln += param.numel()
        log.info(f"Encoder eingefroren | LayerNorm aufgetaut: {n_ln / 1e6:.3f}M Params")
    trainable = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
    log.info(f"Gesamt trainierbare Parameter: {trainable / 1e6:.1f}M")

    asr_model.setup_training_data(asr_model.cfg.train_ds)
    asr_model.setup_validation_data(asr_model.cfg.validation_ds)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_wer",
        patience=hp["early_stopping_patience"],
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator=accelerator,
        max_epochs=hp["epochs"],
        precision=hp["precision"] if accelerator == "gpu" else 32,
        default_root_dir=str(parakeet_dir / "checkpoints"),
        enable_checkpointing=True,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=str(parakeet_dir / "tensorboard"), name="ParakeetFinetune",
        ),
        accumulate_grad_batches=hp["gradient_accumulation_steps"],
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback],
    )

    log.info("Starte Parakeet Training...")
    trainer.fit(asr_model)
    log.info("Parakeet Training beendet.")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    asr_model.save_to(OUT_PATH)
    log.info(f"Modell gespeichert: {OUT_PATH}")

    del asr_model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()