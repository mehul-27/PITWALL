"""
Re-score hallucination rate from existing eval_results.jsonl using the
updated tolerance-based metric.  No GPU / model needed -- just re-reads the
saved (reference, generated) pairs and recomputes scores.

Run from project root:
    python training/rescore_hallucination.py
"""

from __future__ import annotations

import json
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Re-use the improved metric from evaluate.py
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import (
    extract_numbers,
    hallucination_score,
    factual_accuracy,
    _numbers_close,
    HALLUC_TOL,
    HALLUC_ABS,
)

RESULTS_PATH = Path(__file__).parent / "eval_results.jsonl"
RESCORE_REPORT = Path(__file__).parent / "rescore_report.txt"


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run evaluate.py first.")
        sys.exit(1)

    records = [json.loads(l) for l in RESULTS_PATH.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} examples from {RESULTS_PATH.name}")

    old_halluc: list[float] = []
    new_halluc: list[float] = []
    old_factual: list[bool] = []
    new_factual: list[bool] = []

    detailed: list[dict] = []

    for r in records:
        ref = r["reference"]
        gen = r["generated"]

        old_h = r.get("halluc", 0.0)
        old_f = r.get("factual", True)

        new_h = hallucination_score(ref, gen)
        new_f = factual_accuracy(ref, gen)

        old_halluc.append(old_h)
        new_halluc.append(new_h)
        old_factual.append(old_f)
        new_factual.append(new_f)

        detailed.append({
            "question": r["question"][:80],
            "old_halluc": old_h,
            "new_halluc": round(new_h, 4),
            "ref_nums": extract_numbers(ref),
            "pred_nums": extract_numbers(gen),
        })

    n = len(records)
    avg_old = sum(old_halluc) / n
    avg_new = sum(new_halluc) / n
    old_fa = sum(old_factual) / n
    new_fa = sum(new_factual) / n

    lines = [
        "=" * 70,
        "PitWall -- Hallucination Metric Re-score Report",
        f"Examples: {n}",
        "=" * 70,
        "",
        "METRIC COMPARISON",
        "-" * 50,
        f"  Old hallucination rate:   {avg_old:.2%}  (old metric, 10% tol, no noise filter)",
        f"  New hallucination rate:   {avg_new:.2%}  "
        f"(+/-{int(HALLUC_TOL*100)}% rel / +/-{HALLUC_ABS} abs, noise-filtered)",
        f"  Delta improvement:        {avg_old - avg_new:+.2%}",
        "",
        f"  Old factual accuracy:     {old_fa:.2%}",
        f"  New factual accuracy:     {new_fa:.2%}",
        "",
    ]

    # Distribution comparison
    def bucket_dist(scores, label):
        buckets = [0, 0, 0, 0, 0]
        for s in scores:
            buckets[min(int(s * 5), 4)] += 1
        out = [f"  {label}:"]
        edges = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        for edge, cnt in zip(edges, buckets):
            bar = "#" * cnt
            out.append(f"    {edge}  {bar} ({cnt})")
        return out

    lines += bucket_dist(old_halluc, "OLD halluc distribution")
    lines += [""]
    lines += bucket_dist(new_halluc, "NEW halluc distribution")
    lines += [""]

    # Show examples with biggest improvement
    improvements = sorted(
        range(n),
        key=lambda i: old_halluc[i] - new_halluc[i],
        reverse=True,
    )

    lines += [
        "TOP 15 BIGGEST IMPROVEMENTS",
        "-" * 70,
    ]
    for rank, idx in enumerate(improvements[:15], 1):
        d = detailed[idx]
        lines += [
            f"#{rank}  old={d['old_halluc']:.2%} -> new={d['new_halluc']:.2%}  "
            f"(delta={d['old_halluc'] - d['new_halluc']:+.2%})",
            f"  Q: {d['question']}",
            f"  ref_nums:  {d['ref_nums'][:8]}{'...' if len(d['ref_nums']) > 8 else ''}",
            f"  pred_nums: {d['pred_nums'][:8]}{'...' if len(d['pred_nums']) > 8 else ''}",
            "",
        ]

    # Still-high hallucination examples
    still_high = sorted(range(n), key=lambda i: new_halluc[i], reverse=True)
    lines += [
        "TOP 10 STILL-HIGHEST HALLUCINATION EXAMPLES",
        "-" * 70,
    ]
    for rank, idx in enumerate(still_high[:10], 1):
        d = detailed[idx]
        lines += [
            f"#{rank}  new={d['new_halluc']:.2%}  (was {d['old_halluc']:.2%})",
            f"  Q: {d['question']}",
            f"  ref_nums:  {d['ref_nums'][:8]}{'...' if len(d['ref_nums']) > 8 else ''}",
            f"  pred_nums: {d['pred_nums'][:8]}{'...' if len(d['pred_nums']) > 8 else ''}",
            "",
        ]

    lines += ["=" * 70]

    report = "\n".join(lines)
    RESCORE_REPORT.write_text(report, encoding="utf-8")
    print(f"\nReport saved -> {RESCORE_REPORT}")
    print("\n" + report)


if __name__ == "__main__":
    main()
