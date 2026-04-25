"""
Phase 7 — Evaluation: ROUGE-L, BLEU, factual accuracy, hallucination rate.
Runs fine-tuned model against all test.jsonl examples (known answers).
Produces training/eval_report.txt with metrics table + worst examples.

Run from the pitwall directory:
  python training/evaluate.py

Model load order (from config / .env):
  1. Local Qwen: PITWALL_QWEN_BASE_MODEL_PATH + PITWALL_QWEN_ADAPTER_PATH (if both set);
  2. Else: HF_REPO_ID + BASE_MODEL_ID (HuggingFace LoRA).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ModuleNotFoundError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BASE_MODEL_ID,
    INFERENCE_TEMPERATURE,
    MAX_NEW_TOKENS,
    QWEN_ADAPTER_PATH,
    QWEN_BASE_MODEL_PATH,
    QWEN_LOAD_IN_4BIT,
    QWEN_LOCAL_DTYPE,
    SYSTEM_PROMPT,
    TEST_PATH,
)

log = logging.getLogger(__name__)

REPORT_PATH   = Path(__file__).parent / "eval_report.txt"
RESULTS_PATH  = Path(__file__).parent / "eval_results.jsonl"
HALLUC_TOL    = 0.15   # ±15% relative tolerance for hallucination check
HALLUC_ABS    = 1.0    # absolute tolerance for small numbers (|val| < 2)
N_WORST       = 10     # worst examples shown in report

HF_TOKEN   = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _resolve_dtype(torch_mod):
    if QWEN_LOCAL_DTYPE == "fp16":
        return torch_mod.float16
    if QWEN_LOCAL_DTYPE == "bf16":
        return torch_mod.bfloat16
    return None


def _load_model_local_qwen():
    """Load Qwen base + PEFT from disk (same contract as app.inference)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not QWEN_BASE_MODEL_PATH or not QWEN_ADAPTER_PATH:
        return None, None
    base_path = Path(QWEN_BASE_MODEL_PATH)
    adapter_path = Path(QWEN_ADAPTER_PATH)
    if not base_path.exists() or not adapter_path.exists():
        log.error("Qwen path missing: base=%s adapter=%s", base_path, adapter_path)
        return None, None

    quant_cfg = None
    if QWEN_LOAD_IN_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_resolve_dtype(torch) or torch.float16,
        )

    log.info("Loading Qwen tokenizer from %s", base_path)
    tok = AutoTokenizer.from_pretrained(
        str(base_path), local_files_only=True, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs: dict = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
    else:
        model_kwargs["torch_dtype"] = _resolve_dtype(torch)

    log.info("Loading Qwen base model from %s", base_path)
    base = AutoModelForCausalLM.from_pretrained(str(base_path), **model_kwargs)
    log.info("Attaching adapter from %s", adapter_path)
    mdl = PeftModel.from_pretrained(
        base, str(adapter_path), local_files_only=True
    )
    mdl.eval()
    log.info("Qwen + adapter ready.")
    return mdl, tok


def _load_model_hf():
    if not HF_REPO_ID:
        return None, None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    log.info("Loading tokenizer (HF) ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)

    log.info("Loading base model (HF) ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        token=HF_TOKEN,
    )
    log.info("Attaching adapter %s ...", HF_REPO_ID)
    mdl = PeftModel.from_pretrained(base, HF_REPO_ID, token=HF_TOKEN)
    mdl.eval()
    log.info("Model ready (HF).")
    return mdl, tok


def load_model():
    """
    Prefer local Qwen (PITWALL_QWEN_BASE_MODEL_PATH + PITWALL_QWEN_ADAPTER_PATH);
    else HuggingFace adapter (HF_REPO_ID + BASE_MODEL_ID).

    If Qwen paths are set in .env, they are required (no silent fallback to HF).
    """
    if QWEN_BASE_MODEL_PATH or QWEN_ADAPTER_PATH:
        if not (QWEN_BASE_MODEL_PATH and QWEN_ADAPTER_PATH):
            raise EnvironmentError(
                "Set both PITWALL_QWEN_BASE_MODEL_PATH and PITWALL_QWEN_ADAPTER_PATH, "
                "or clear both to use HuggingFace (HF_REPO_ID)."
            )
        m, t = _load_model_local_qwen()
        if m is None or t is None:
            raise EnvironmentError(
                "Qwen paths in .env do not exist on disk. Check "
                f"PITWALL_QWEN_BASE_MODEL_PATH={QWEN_BASE_MODEL_PATH!r} and "
                f"PITWALL_QWEN_ADAPTER_PATH={QWEN_ADAPTER_PATH!r}."
            )
        return m, t
    m, t = _load_model_hf()
    if m is not None and t is not None:
        return m, t
    raise EnvironmentError(
        "Set PITWALL_QWEN_BASE_MODEL_PATH and PITWALL_QWEN_ADAPTER_PATH to local Qwen + "
        "adapter, or set HF_REPO_ID (and HF_TOKEN if needed) for Hub LoRA."
    )


def generate(model, tokenizer, messages: list[dict]) -> str:
    import torch
    do_sample = INFERENCE_TEMPERATURE > 0
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
    plen = input_ids.shape[-1]
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=INFERENCE_TEMPERATURE if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Number extraction  (tolerance-aware, noise-filtered)
# ---------------------------------------------------------------------------

# Matches lap-time format  M:SS.sss  (e.g. 1:19.327)
_LAPTIME_RE = re.compile(r"\b(\d{1,2}):(\d{2}\.\d+)\b")

# Matches plain numbers, including negatives and decimals
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Year numbers that are context, not factual claims
_YEAR_RANGE = set(range(2018, 2028))

# Ordinal suffixes: 1st, 2nd, 3rd, 4th, ..., 21st, etc.
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)


def extract_numbers(text: str) -> list[float]:
    """Extract meaningful numeric values from text.

    - Converts lap times (M:SS.sss) to total seconds before extraction.
    - Filters out season years (2018-2027) and ordinal rankings (1st, 2nd …).
    - Deduplicates while preserving order.
    """
    # ── Step 1: convert lap times to total seconds in-place ──────────────
    def _laptime_to_secs(m: re.Match) -> str:
        mins = int(m.group(1))
        secs = float(m.group(2))
        return f"{mins * 60 + secs:.3f}"

    normalised = _LAPTIME_RE.sub(_laptime_to_secs, text)

    # ── Step 2: collect ordinals so we can skip them ─────────────────────
    ordinals: set[str] = set()
    for m in _ORDINAL_RE.finditer(text):
        ordinals.add(m.group(1))

    # ── Step 3: extract all plain numbers ────────────────────────────────
    raw = _NUM_RE.findall(normalised)

    seen: set[float] = set()
    result: list[float] = []
    for tok in raw:
        val = float(tok)
        # skip years
        if val == int(val) and int(val) in _YEAR_RANGE:
            continue
        # skip ordinals (the bare digit that appeared with st/nd/rd/th)
        if tok in ordinals:
            continue
        if val not in seen:
            seen.add(val)
            result.append(val)
    return result


def _numbers_close(pred: float, ref: float) -> bool:
    """True if *pred* is acceptably close to *ref*.

    Uses whichever is more generous:
      • relative tolerance  (±HALLUC_TOL, default 15%)
      • absolute tolerance  (±HALLUC_ABS, default 1.0)  — important for small
        numbers where even a 0.5 difference exceeds 15%.
    """
    if ref == 0:
        return abs(pred) < max(0.001, HALLUC_ABS)
    if abs(pred - ref) / abs(ref) <= HALLUC_TOL:
        return True
    if abs(pred - ref) <= HALLUC_ABS:
        return True
    return False


def factual_accuracy(reference: str, prediction: str) -> bool:
    """True if prediction contains at least one reference number within tolerance."""
    ref_nums = extract_numbers(reference)
    if not ref_nums:
        return True  # no numbers to check — not counted
    pred_nums = extract_numbers(prediction)
    for r in ref_nums:
        for p in pred_nums:
            if _numbers_close(p, r):
                return True
    return False


def hallucination_score(reference: str, prediction: str) -> float:
    """Fraction of *meaningful* predicted numbers not grounded in reference.

    0.0 = no hallucinations, 1.0 = every predicted number is hallucinated.
    Uses ±15% relative tolerance OR ±1.0 absolute tolerance (whichever is
    more generous) so that small rounding differences are not penalised.
    """
    ref_nums = extract_numbers(reference)
    pred_nums = extract_numbers(prediction)
    if not pred_nums:
        return 0.0
    if not ref_nums:
        return 0.0  # can't judge — no ground truth numbers

    hallucinated = 0
    for p in pred_nums:
        grounded = any(_numbers_close(p, r) for r in ref_nums)
        if not grounded:
            hallucinated += 1
    return hallucinated / len(pred_nums)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rouge_l(reference: str, prediction: str) -> float:
    """LCS-based ROUGE-L F1."""
    ref_tokens  = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not ref_tokens or not pred_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    precision = lcs / n
    recall    = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_score(reference: str, prediction: str, max_n: int = 4) -> float:
    """Corpus-level BLEU-4 (single sentence, with brevity penalty)."""
    import math
    from collections import Counter

    ref_tokens  = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    log_score = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        clipped = sum(
            min(cnt, ref_ngrams[ng]) for ng, cnt in pred_ngrams.items()
        )
        total = max(len(pred_tokens) - n + 1, 0)
        if total == 0 or clipped == 0:
            return 0.0
        log_score += math.log(clipped / total)

    # Brevity penalty
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else \
        math.exp(1 - len(ref_tokens) / len(pred_tokens))

    return bp * math.exp(log_score / max_n)


def _fmt_nums(nums: list, limit: int = 8) -> str:
    """Format a number list for report display, truncating if needed."""
    shown = nums[:limit]
    suffix = f" ... ({len(nums)} total)" if len(nums) > limit else ""
    return "[" + ", ".join(f"{n}" for n in shown) + "]" + suffix


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    records: list[dict],
    rouge_scores: list[float],
    bleu_scores:  list[float],
    factual_hits: list[bool],
    halluc_scores: list[float],
) -> None:
    n = len(records)
    avg_rouge    = sum(rouge_scores)  / n
    avg_bleu     = sum(bleu_scores)   / n
    factual_acc  = sum(factual_hits)  / n
    avg_halluc   = sum(halluc_scores) / n

    lines: list[str] = []

    lines += [
        "=" * 70,
        "PitWall - Fine-tuned Model Evaluation Report",
        f"Test examples: {n}",
        "=" * 70,
        "",
        "METRICS SUMMARY",
        "-" * 40,
        f"  ROUGE-L (avg):          {avg_rouge:.4f}",
        f"  BLEU-4  (avg):          {avg_bleu:.4f}",
        f"  Factual accuracy:       {factual_acc:.2%}  "
        f"(+/-{int(HALLUC_TOL*100)}% or +/-{HALLUC_ABS} abs tolerance)",
        f"  Hallucination rate:     {avg_halluc:.2%}  "
        f"(noise-filtered; years/ordinals excluded, lap-times normalised)",
        "",
    ]

    # Per-score distribution (ASCII only: Windows console cp1252 cannot print U+2501 bars)
    def dist(scores: list[float], label: str) -> list[str]:
        buckets = [0, 0, 0, 0, 0]  # 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
        for s in scores:
            buckets[min(int(s * 5), 4)] += 1
        out = [f"  {label} distribution:"]
        edges = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        for edge, cnt in zip(edges, buckets):
            bar = "#" * cnt
            out.append(f"    {edge}  {bar} ({cnt})")
        return out

    lines += dist(rouge_scores,  "ROUGE-L")
    lines += [""]
    lines += dist(bleu_scores,   "BLEU-4")
    lines += [""]

    # Worst ROUGE-L examples
    worst_idx = sorted(range(n), key=lambda i: rouge_scores[i])[:N_WORST]
    lines += [
        f"WORST {N_WORST} EXAMPLES BY ROUGE-L",
        "-" * 70,
    ]
    for rank, idx in enumerate(worst_idx, 1):
        r = records[idx]
        lines += [
            f"#{rank}  ROUGE-L={rouge_scores[idx]:.3f}  "
            f"BLEU={bleu_scores[idx]:.3f}  "
            f"Factual={'Y' if factual_hits[idx] else 'N'}  "
            f"Halluc={halluc_scores[idx]:.2%}",
            f"  Q:   {r['question'][:100]}",
            f"  REF: {r['reference'][:120]}",
            f"  GEN: {r['generated'][:120]}",
            f"  ref_nums:  {_fmt_nums(r.get('ref_nums', []))}",
            f"  pred_nums: {_fmt_nums(r.get('pred_nums', []))}",
            "",
        ]

    # Factual failures
    failures = [i for i, h in enumerate(factual_hits) if not h]
    lines += [
        f"FACTUAL ACCURACY FAILURES ({len(failures)}/{n})",
        "-" * 70,
    ]
    for idx in failures[:N_WORST]:
        r = records[idx]
        lines += [
            f"  Q:   {r['question'][:100]}",
            f"  REF: {r['reference'][:120]}",
            f"  GEN: {r['generated'][:120]}",
            f"  ref_nums:  {_fmt_nums(r.get('ref_nums', []))}",
            f"  pred_nums: {_fmt_nums(r.get('pred_nums', []))}",
            "",
        ]

    lines += ["=" * 70]

    report_text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    log.info("Report saved -> %s", REPORT_PATH)
    _enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        # Windows cp1252 console: model text may include chars outside system encoding
        print("\n" + report_text.encode(_enc, errors="replace").decode(_enc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load test set
    raw = [json.loads(l) for l in TEST_PATH.open(encoding="utf-8") if l.strip()]
    log.info("Test set: %d examples", len(raw))

    # Load model
    model, tokenizer = load_model()

    records:       list[dict]  = []
    rouge_scores:  list[float] = []
    bleu_scores:   list[float] = []
    factual_hits:  list[bool]  = []
    halluc_scores: list[float] = []

    for i, ex in enumerate(raw, 1):
        msgs = ex["messages"]
        user_q    = next(m["content"] for m in reversed(msgs) if m["role"] == "user")
        reference = next(
            (m["content"] for m in reversed(msgs) if m["role"] == "assistant"), ""
        )

        log.info("[%d/%d] %s", i, len(raw), user_q[:80])

        generated = generate(
            model, tokenizer,
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": user_q}],
        )

        rl = rouge_l(reference, generated)
        bl = bleu_score(reference, generated)
        fa = factual_accuracy(reference, generated)
        hs = hallucination_score(reference, generated)

        ref_nums  = extract_numbers(reference)
        pred_nums = extract_numbers(generated)

        records.append({
            "question":  user_q,
            "reference": reference,
            "generated": generated,
            "rouge_l":   round(rl, 4),
            "bleu":      round(bl, 4),
            "factual":   fa,
            "halluc":    round(hs, 4),
            "ref_nums":  ref_nums,
            "pred_nums": pred_nums,
        })
        rouge_scores.append(rl)
        bleu_scores.append(bl)
        factual_hits.append(fa)
        halluc_scores.append(hs)

    # Save per-example results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Per-example results -> %s", RESULTS_PATH)

    # Write report
    write_report(records, rouge_scores, bleu_scores, factual_hits, halluc_scores)


if __name__ == "__main__":
    main()
