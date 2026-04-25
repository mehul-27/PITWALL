"""
Phase 6 — Fine-tune Llama 3.2 3B Instruct with QLoRA on Kaggle T4 GPU.
Uses Unsloth for efficient 4-bit loading and LoRA patching.
Saves adapter only (not full model) to local path and HuggingFace Hub.
HF_TOKEN must be set as environment variable — never hardcoded.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match master.md / config.py spec)
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Read JSONL file, return list of message dicts."""
    examples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    log.info("Loaded %d examples from %s", len(examples), path)
    return examples


def preprocess(examples: dict, tokenizer) -> dict:
    """
    Batched map function: convert each messages list to a formatted string
    using the model's native chat template.
    Handles both single-turn (3 messages) and multi-turn (>3 messages).
    """
    texts: list[str] = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.2 3B Instruct with QLoRA (Kaggle T4)"
    )
    parser.add_argument("--train",   required=True, help="Path to train.jsonl")
    parser.add_argument("--val",     required=True, help="Path to val.jsonl")
    parser.add_argument("--output",  required=True, help="Local path to save adapter")
    parser.add_argument("--hf_repo", required=True, help="HuggingFace repo to push adapter")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log.error("HF_TOKEN env var not set — cannot push to Hub")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load model + tokenizer (4-bit quantised via Unsloth)
    # ------------------------------------------------------------------
    log.info("Loading %s in 4-bit ...", BASE_MODEL_ID)

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    # ------------------------------------------------------------------
    # 2. Apply LoRA adapters
    # ------------------------------------------------------------------
    log.info("Applying QLoRA adapters ...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 3. Load + pre-tokenize dataset
    # ------------------------------------------------------------------
    log.info("Loading datasets ...")
    train_raw = load_jsonl(args.train)
    val_raw   = load_jsonl(args.val)

    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    train_dataset = Dataset.from_list(train_raw).map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=["messages"],
    )
    val_dataset = Dataset.from_list(val_raw).map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=["messages"],
    )

    log.info("Train: %d examples, Val: %d examples",
             len(train_dataset), len(val_dataset))

    # ------------------------------------------------------------------
    # 4. Configure trainer
    # ------------------------------------------------------------------
    log.info("Configuring SFTTrainer ...")

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        args=training_args,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    log.info("=" * 50)
    log.info("Starting training — 3 epochs, logging every 10 steps")
    log.info("=" * 50)

    trainer.train()

    log.info("Training complete.")

    # ------------------------------------------------------------------
    # 6. Save adapter locally
    # ------------------------------------------------------------------
    log.info("Saving adapter -> %s", args.output)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # ------------------------------------------------------------------
    # 7. Push adapter to HuggingFace Hub
    # ------------------------------------------------------------------
    log.info("Pushing adapter -> %s", args.hf_repo)
    model.push_to_hub(args.hf_repo, token=hf_token)
    tokenizer.push_to_hub(args.hf_repo, token=hf_token)
    log.info("Adapter pushed to HuggingFace Hub successfully.")

    log.info("Done. Adapter saved locally and pushed to Hub.")


if __name__ == "__main__":
    main()
