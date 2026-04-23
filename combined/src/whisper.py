"""Whisper: Evaluation und Fine-Tuning."""

import time
from pathlib import Path

import torch

import src.config as cfg
from src.config import SEED, VAL_DIR, TEST_DIR, WHISPER_HPARAMS, log
from src.utils import compute_wer, load_audio, load_metadata, save_json, save_predictions


def evaluate(model_path_or_name: str, label: str, out_dir: Path) -> dict:
    """Evaluate a Whisper model on the test set and save results."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    log.info(f"{'='*60}")
    log.info(f"  {label}")
    log.info(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device} | Modell: {model_path_or_name}")

    processor = WhisperProcessor.from_pretrained(model_path_or_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_path_or_name).to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=WHISPER_HPARAMS["language"], task="transcribe",
    )

    test_samples = load_metadata(TEST_DIR)
    references, hypotheses = [], []

    t0 = time.time()
    for i, sample in enumerate(test_samples):
        audio = load_audio(sample["audio_path"])
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        references.append(sample["text"])
        hypotheses.append(transcription)

        if (i + 1) % 20 == 0 or (i + 1) == len(test_samples):
            log.info(f"  Fortschritt: {i+1}/{len(test_samples)}")

    elapsed = time.time() - t0
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

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def finetune(run_dir: Path) -> None:
    """Fine-tune Whisper large-v3 on the training set."""
    from dataclasses import dataclass
    from typing import Any

    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    log.info(f"{'='*60}")
    log.info("  Fine-Tuning Whisper (LoRA)")
    log.info(f"{'='*60}")

    hp = WHISPER_HPARAMS
    MODEL_NAME = hp["model"]
    MODEL_DIR = str(run_dir / "whisper" / "model")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.generation_config.language = hp["language"]
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset -------------------------------------------------------
    class ASRDataset(torch.utils.data.Dataset):
        def __init__(self, split_dir: Path, processor_):
            self.samples = load_metadata(split_dir)
            self.processor = processor_

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            audio = load_audio(sample["audio_path"])
            input_features = self.processor(
                audio, sampling_rate=16000, return_tensors="np",
            ).input_features[0]
            labels = self.processor.tokenizer(sample["text"]).input_ids
            return {"input_features": input_features, "labels": labels}

    train_dataset = ASRDataset(cfg.TRAIN_DIR, processor)
    val_dataset = ASRDataset(VAL_DIR, processor)
    log.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # --- Data Collator -------------------------------------------------
    @dataclass
    class DataCollatorSpeechSeq2Seq:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100,
            )
            if (labels[:, 0] == self.decoder_start_token_id).all():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # --- Training ------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "whisper" / "checkpoints"),
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        warmup_steps=hp["warmup_steps"],
        num_train_epochs=hp["epochs"],
        fp16=torch.cuda.is_available() and hp["fp16"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=hp["save_total_limit"],
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        logging_dir=str(run_dir / "whisper" / "tensorboard"),
        remove_unused_columns=False,
        label_names=["labels"],
        seed=SEED,
        dataloader_num_workers=2,
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": compute_wer(label_str, pred_str)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    log.info("Starte Whisper Training...")
    trainer.train()
    log.info("Whisper Training abgeschlossen.")

    merged = model.merge_and_unload()
    merged.save_pretrained(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)
    log.info(f"Modell gespeichert: {MODEL_DIR}")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()