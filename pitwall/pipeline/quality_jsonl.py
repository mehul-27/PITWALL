"""
PitWall JSONL data-quality utilities.
Filters bad tyre-degradation examples and normalises sector comparison wording.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_DIR

log = logging.getLogger(__name__)

DEG_RATE_RE = re.compile(
    r"(?:deg(?:radation)?(?: rate)?|degraded at|deg_rate)"
    r"[^-\d]{0,40}(-?\d+(?:\.\d+)?)\s*(?:s|seconds)?\s*/?\s*lap",
    re.IGNORECASE,
)
MAX_VIABLE_RE = re.compile(
    r"(?:max(?:imum)? viable(?: tyre)? stint(?: length)?|max viable tyre stint|"
    r"max_viable_laps|max stint|viable stint length)"
    r"[^.\d]{0,50}(?:around|of|~|up to)?[^.\d]{0,20}(\d+)\s*laps?",
    re.IGNORECASE,
)

NAME = r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?"
TIME = r"(\d+(?:\.\d+)?)"
SECTOR_WRONG_PATTERNS = [
    (
        re.compile(
            rf"\b({NAME})\s+is\s+{TIME}\s*s(?:econds)?\s+ahead of\s+({NAME})\s+in\s+Sector\s+([123])\b",
            re.IGNORECASE,
        ),
        lambda m: f"{m.group(3)} is {m.group(2)}s behind {m.group(1)} in Sector {m.group(4)}",
    ),
    (
        re.compile(
            rf"\b({NAME})\s+is\s+{TIME}\s*s(?:econds)?\s+faster than\s+({NAME})\s+in\s+Sector\s+([123])\b",
            re.IGNORECASE,
        ),
        lambda m: f"{m.group(1)} has {m.group(2)}s advantage over {m.group(3)} in Sector {m.group(4)}",
    ),
    (
        re.compile(
            rf"\b({NAME})\s+has\s+{TIME}\s*s(?:econds)?\s+deficit to\s+({NAME})\s+in\s+Sector\s+([123])\b",
            re.IGNORECASE,
        ),
        lambda m: f"{m.group(1)} is {m.group(2)}s behind {m.group(3)} in Sector {m.group(4)}",
    ),
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("%s:%d malformed JSON skipped: %s", path, line_no, exc)
    return examples


def _write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _example_text(example: dict[str, Any]) -> str:
    return "\n".join(
        str(msg.get("content", ""))
        for msg in example.get("messages", [])
        if msg.get("role") in {"user", "assistant"}
    )


def has_bad_degradation_stats(example: dict[str, Any]) -> tuple[bool, str | None]:
    text = _example_text(example)
    for match in DEG_RATE_RE.finditer(text):
        value = float(match.group(1))
        if value < 0:
            return True, f"deg_rate={value}"
    for match in MAX_VIABLE_RE.finditer(text):
        value = int(match.group(1))
        if value < 8 or value > 50:
            return True, f"max_viable_laps={value}"
    return False, None


def filter_bad_degradation(
    input_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    examples = _read_jsonl(input_path)
    kept: list[dict[str, Any]] = []
    removed = 0
    reasons: dict[str, int] = {}
    for ex in examples:
        bad, reason = has_bad_degradation_stats(ex)
        if bad:
            removed += 1
            reasons[reason or "unknown"] = reasons.get(reason or "unknown", 0) + 1
            continue
        kept.append(ex)
    _write_jsonl(output_path, kept)
    log.info(
        "Filtered %s -> %s kept=%d removed=%d reasons=%s",
        input_path, output_path, len(kept), removed, reasons,
    )
    return len(kept), removed


def _normalise_sector_text(text: str) -> tuple[str, int]:
    changes = 0
    for pattern, repl in SECTOR_WRONG_PATTERNS:
        text, count = pattern.subn(repl, text)
        changes += count
    return text, changes


def standardise_sector_direction(path: Path, output_path: Path | None = None) -> int:
    examples = _read_jsonl(path)
    changed_examples = 0
    total_changes = 0
    for ex in examples:
        changed = False
        for msg in ex.get("messages", []):
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            new_content, count = _normalise_sector_text(content)
            if count:
                msg["content"] = new_content
                changed = True
                total_changes += count
        if changed:
            changed_examples += 1

    target = output_path or path
    if output_path is None and total_changes:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
            log.info("Backup written -> %s", backup)
    _write_jsonl(target, examples)
    log.info(
        "Sector direction standardised %s -> %s changed_examples=%d replacements=%d",
        path, target, changed_examples, total_changes,
    )
    return total_changes


def _dataset_jsonl_paths() -> list[Path]:
    return sorted(
        p for p in DATASET_DIR.glob("*.jsonl")
        if not p.name.endswith(".bak") and "clean" not in p.stem
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PitWall JSONL quality fixes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_filter = sub.add_parser("filter-degradation")
    p_filter.add_argument("--input", type=Path, required=True)
    p_filter.add_argument("--output", type=Path)

    p_sector = sub.add_parser("standardise-sector")
    p_sector.add_argument("--input", type=Path)
    p_sector.add_argument("--output", type=Path)
    p_sector.add_argument("--all-dataset-jsonl", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd == "filter-degradation":
        output = args.output or args.input.with_name(args.input.stem + "_clean.jsonl")
        filter_bad_degradation(args.input, output)
    elif args.cmd == "standardise-sector":
        if args.all_dataset_jsonl:
            for path in _dataset_jsonl_paths():
                standardise_sector_direction(path)
        else:
            if args.input is None:
                raise SystemExit("--input is required unless --all-dataset-jsonl is set")
            standardise_sector_direction(args.input, args.output)


if __name__ == "__main__":
    main()
