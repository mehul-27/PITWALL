"""
Baseline comparison — three-way eval on the same 50 test questions:
  1. Base Llama 3.2 3B (no adapter, no system prompt)
  2. Prompt-engineered Llama 3.2 3B (race engineer system prompt, no adapter)
  3. Fine-tuned Llama 3.2 3B (QLoRA adapter, race engineer system prompt)

Saves results to training/baseline_results.jsonl — one record per question.
Run from project root: python training/baseline.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ModuleNotFoundError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_MODEL_ID, INFERENCE_TEMPERATURE, MAX_NEW_TOKENS, SYSTEM_PROMPT, TEST_PATH

log = logging.getLogger(__name__)

N_QUESTIONS   = 50
RANDOM_SEED   = 42
OUTPUT_PATH   = Path(__file__).parent / "baseline_results.jsonl"

HF_TOKEN   = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")
if not HF_REPO_ID:
    raise EnvironmentError("HF_REPO_ID not set in .env")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _bnb_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_base_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading base model %s ...", BASE_MODEL_ID)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=_bnb_config(),
        device_map="auto",
        token=HF_TOKEN,
    )
    mdl.eval()
    return mdl, tok


def load_finetuned_model(base_model, tokenizer):
    from peft import PeftModel
    log.info("Attaching QLoRA adapter from %s ...", HF_REPO_ID)
    mdl = PeftModel.from_pretrained(base_model, HF_REPO_ID, token=HF_TOKEN)
    mdl.eval()
    return mdl, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, messages: list[dict]) -> str:
    import torch
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=INFERENCE_TEMPERATURE,
            do_sample=INFERENCE_TEMPERATURE > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Sample 50 questions from test set
    raw = [json.loads(l) for l in TEST_PATH.open(encoding="utf-8") if l.strip()]
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(raw, min(N_QUESTIONS, len(raw)))
    log.info("Sampled %d questions from test set", len(sample))

    # Load models once — share base weights, swap adapter
    base_model, tokenizer = load_base_model()
    ft_model, _ = load_finetuned_model(base_model, tokenizer)

    results: list[dict] = []

    for i, ex in enumerate(sample, 1):
        msgs = ex["messages"]
        # Extract user question (last user turn for multi-turn examples)
        user_q = next(
            m["content"] for m in reversed(msgs) if m["role"] == "user"
        )
        reference = next(
            (m["content"] for m in reversed(msgs) if m["role"] == "assistant"),
            "",
        )

        log.info("[%d/%d] %s", i, len(sample), user_q[:80])

        # 1 — base: no system prompt, just raw user question
        resp_base = generate(
            base_model, tokenizer,
            [{"role": "user", "content": user_q}],
        )

        # 2 — prompted: system prompt, no adapter
        resp_prompted = generate(
            base_model, tokenizer,
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": user_q}],
        )

        # 3 — fine-tuned: system prompt + adapter
        resp_finetuned = generate(
            ft_model, tokenizer,
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": user_q}],
        )

        results.append({
            "question":      user_q,
            "reference":     reference,
            "base":          resp_base,
            "prompted":      resp_prompted,
            "finetuned":     resp_finetuned,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Saved %d results -> %s", len(results), OUTPUT_PATH)


if __name__ == "__main__":
    main()
