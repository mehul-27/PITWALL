import os
os.environ["UNSLOTH_USE_CUT_CROSS_ENTROPY"] = "0"

# ── Paste this entire file as a single Kaggle notebook cell ──────────────────
# Requires Kaggle secrets: HF_TOKEN
# Requires Kaggle dataset input: bajlesh/pitwall-training  (train.jsonl, val.jsonl)
# pip install before running (run this in a SEPARATE cell, then restart the kernel):
#   !pip install --upgrade unsloth "transformers>=4.51.0" trl peft accelerate

import json
import logging

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "/kaggle/input/datasets/bajlesh/pitwall-training/train.jsonl"
VAL_PATH    = "/kaggle/input/datasets/bajlesh/pitwall-training/val.jsonl"
OUTPUT_DIR  = "/kaggle/working/pitwall-adapter"
HF_REPO     = "YOUR_HF_USERNAME/pitwall-adapter"   # <-- edit before running

BASE_MODEL_ID  = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN not found. Add it via Kaggle Notebook Secrets (Add-ons → Secrets)."
    )

# ── 1. Load model + tokenizer ─────────────────────────────────────────────────
log.info("Loading %s (4-bit) ...", BASE_MODEL_ID)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

# ── 2. Apply QLoRA adapters ───────────────────────────────────────────────────
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

# ── 3. Load + pre-process dataset ─────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_chat_template_batch(examples: dict) -> dict:
    texts = []
    for msgs in examples["messages"]:
        texts.append(
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return {"text": texts}


log.info("Loading and preprocessing datasets ...")

train_raw = load_jsonl(TRAIN_PATH)
val_raw   = load_jsonl(VAL_PATH)

train_dataset = Dataset.from_list(train_raw).map(
    apply_chat_template_batch,
    batched=True,
    remove_columns=["messages"],
)
val_dataset = Dataset.from_list(val_raw).map(
    apply_chat_template_batch,
    batched=True,
    remove_columns=["messages"],
)

log.info("Train: %d  Val: %d", len(train_dataset), len(val_dataset))

# ── 4. Training arguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    evaluation_strategy="epoch",
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

# ── 5. Trainer ────────────────────────────────────────────────────────────────
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

# ── 6. Train ──────────────────────────────────────────────────────────────────
log.info("=" * 55)
log.info("Training — 3 epochs, batch 4×4 grad accum, lr 2e-4")
log.info("=" * 55)

trainer.train()

log.info("Training complete.")

# ── 7. Save adapter locally ───────────────────────────────────────────────────
log.info("Saving adapter -> %s", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── 8. Push to HuggingFace Hub ────────────────────────────────────────────────
log.info("Pushing adapter -> %s", HF_REPO)
model.push_to_hub(HF_REPO, token=HF_TOKEN)
tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)

log.info("Done. Adapter at %s and pushed to Hub.", OUTPUT_DIR)
