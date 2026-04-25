"""
Model inference for PitWall.

Supports two local backends:
1. Ollama REST inference
2. Local Qwen base model + PEFT adapter loading via Transformers

The backend is selected through config.INFERENCE_BACKEND.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    INFERENCE_BACKEND,
    INFERENCE_TEMPERATURE,
    MAX_NEW_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    QWEN_ADAPTER_PATH,
    QWEN_BASE_MODEL_PATH,
    QWEN_LOAD_IN_4BIT,
    QWEN_LOCAL_DTYPE,
)

log = logging.getLogger(__name__)

_CHAT_ENDPOINT = f"{OLLAMA_URL}/api/chat"
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None


def _check_ollama() -> bool:
    """Return True if Ollama is reachable and the configured model exists."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            if OLLAMA_MODEL in models or any(m.startswith(OLLAMA_MODEL) for m in models):
                log.info("Ollama model '%s' is available.", OLLAMA_MODEL)
                return True
            log.warning(
                "Ollama is running but model '%s' not found. Available: %s",
                OLLAMA_MODEL,
                models,
            )
            return False
    except Exception as exc:
        log.warning("Cannot reach Ollama at %s: %s", OLLAMA_URL, exc)
        return False


def _resolve_dtype(torch_mod):
    if QWEN_LOCAL_DTYPE == "fp16":
        return torch_mod.float16
    if QWEN_LOCAL_DTYPE == "bf16":
        return torch_mod.bfloat16
    return None


def _load_local_qwen() -> bool:
    """Load local Qwen base model + adapter from disk."""
    global _LOCAL_MODEL, _LOCAL_TOKENIZER

    if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None:
        return True

    if not QWEN_BASE_MODEL_PATH or not QWEN_ADAPTER_PATH:
        log.warning(
            "Local Qwen backend selected but model paths are missing. "
            "Set PITWALL_QWEN_BASE_MODEL_PATH and PITWALL_QWEN_ADAPTER_PATH."
        )
        return False

    base_path = Path(QWEN_BASE_MODEL_PATH)
    adapter_path = Path(QWEN_ADAPTER_PATH)
    if not base_path.exists():
        log.warning("Qwen base model path not found: %s", base_path)
        return False
    if not adapter_path.exists():
        log.warning("Qwen adapter path not found: %s", adapter_path)
        return False

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        log.warning("Local Qwen dependencies unavailable: %s", exc)
        return False

    log.info("Loading local Qwen base model from %s", base_path)
    quant_cfg = None
    if QWEN_LOAD_IN_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_resolve_dtype(torch) or torch.float16,
        )

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
    else:
        model_kwargs["torch_dtype"] = _resolve_dtype(torch)

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        local_files_only=True,
    )
    model.eval()

    _LOCAL_MODEL = model
    _LOCAL_TOKENIZER = tokenizer
    _dev = next(model.parameters()).device
    log.info("Local Qwen inference ready. Base=%s Adapter=%s (primary device: %s)", base_path, adapter_path, _dev)
    return True


def _generate_ollama(
    messages: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = INFERENCE_TEMPERATURE,
) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": INFERENCE_TEMPERATURE,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        _CHAT_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "").strip()
            if not content:
                raise RuntimeError("Empty response from Ollama")
            return content
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def _generate_local_qwen(
    messages: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = INFERENCE_TEMPERATURE,
) -> str:
    if not _load_local_qwen():
        raise RuntimeError("Local Qwen model is not available")

    import torch

    tokenizer = _LOCAL_TOKENIZER
    model = _LOCAL_MODEL

    # String prompt + tokenizer() yields input_ids and attention_mask tensors.
    # apply_chat_template(..., return_tensors="pt") returns BatchEncoding — not .ne()-able.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
    in_len = input_ids.shape[-1]

    temp = INFERENCE_TEMPERATURE
    do_sample = temp > 0
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temp if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][in_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("Empty response from local Qwen model")
    return text


def _resolve_backend() -> tuple[str, bool]:
    backend = INFERENCE_BACKEND
    if backend == "local_qwen":
        available = _load_local_qwen()
        if not available:
            log.warning("Falling back to Ollama because local Qwen is unavailable.")
            return "ollama", _check_ollama()
        return "local_qwen", True
    return "ollama", _check_ollama()


ACTIVE_BACKEND, MODEL_AVAILABLE = _resolve_backend()

if MODEL_AVAILABLE:
    log.info("Inference ready via backend=%s", ACTIVE_BACKEND)
else:
    log.warning("No local inference backend available. Stub responses will be used.")


def generate(
    messages: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = INFERENCE_TEMPERATURE,
) -> str:
    """Generate a response with the configured local backend (temperature fixed to INFERENCE_TEMPERATURE)."""
    t = INFERENCE_TEMPERATURE
    if ACTIVE_BACKEND == "local_qwen":
        return _generate_local_qwen(messages, max_new_tokens=max_new_tokens, temperature=t)
    return _generate_ollama(messages, max_new_tokens=max_new_tokens, temperature=t)
