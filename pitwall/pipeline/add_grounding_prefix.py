"""
Add selective grounding prefixes to failure-category answers.
Targets strategy and corner-type/corner-profile style examples only.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

STRATEGY_RE = re.compile(
    r"(strategy|pit window|undercut|overcut|stretch|stint|stay out|box|compound works best|offset)",
    re.IGNORECASE,
)
CORNER_RE = re.compile(
    r"(corner|corners|turn|slow-speed|medium-speed|high-speed|corner type|profile)",
    re.IGNORECASE,
)

DEG_RE = re.compile(r"(\d+(?:\.\d+)?)s/lap", re.IGNORECASE)
CLIFF_RE = re.compile(r"cliff(?: lap)?(?: at| is| around)?(?: lap)?\s*(\d+)", re.IGNORECASE)
MAX_VIABLE_RE = re.compile(r"(?:max(?:imum)? viable(?: stint| life)?|viable stint length(?: of)?|up to)\s*(\d+)\s+laps?", re.IGNORECASE)
TOP_SPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*km/h", re.IGNORECASE)
CORNER_COUNT_RE = re.compile(r"(\d+)\s+(slow|medium|high)(?:-speed)?\s+corners?", re.IGNORECASE)
AVG_MIN_SPEED_RE = re.compile(r"averages?\s+(\d+(?:\.\d+)?)\s*km/h\s+minimum speed", re.IGNORECASE)
DELTA_KPH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*km/h\s+(above|below)\s+the field", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _build_strategy_anchor(answer: str) -> str | None:
    facts: list[str] = []
    deg = DEG_RE.search(answer)
    cliff = CLIFF_RE.search(answer)
    viable = MAX_VIABLE_RE.search(answer)
    speeds = TOP_SPEED_RE.findall(answer)
    if deg:
        facts.append(f"deg {deg.group(1)}s/lap")
    if cliff:
        facts.append(f"cliff lap {cliff.group(1)}")
    if viable:
        facts.append(f"max viable {viable.group(1)} laps")
    if speeds:
        facts.append(f"top speed benchmark {speeds[0]} km/h")
    if not facts:
        return None
    return "Key facts: " + ", ".join(facts) + ". "


def _build_corner_anchor(answer: str) -> str | None:
    facts: list[str] = []
    counts = CORNER_COUNT_RE.findall(answer)
    avg_speed = AVG_MIN_SPEED_RE.search(answer)
    delta = DELTA_KPH_RE.search(answer)
    speeds = TOP_SPEED_RE.findall(answer)
    if counts:
        facts.extend(f"{count} {kind}-speed corners" for count, kind in counts[:3])
    if avg_speed:
        facts.append(f"avg min speed {avg_speed.group(1)} km/h")
    if delta:
        facts.append(f"{delta.group(1)} km/h {delta.group(2)} field")
    elif speeds:
        facts.append(f"reference speed {speeds[0]} km/h")
    if not facts:
        return None
    return "Key facts: " + ", ".join(facts) + ". "


def _should_anchor(question: str) -> tuple[bool, str | None]:
    if STRATEGY_RE.search(question):
        return True, "strategy"
    if CORNER_RE.search(question):
        return True, "corner"
    return False, None


def add_grounding_prefix(path: Path) -> tuple[int, int]:
    examples = _read_jsonl(path)
    changed = 0
    checked = 0
    for ex in examples:
        messages = ex.get("messages", [])
        if len(messages) < 3:
            continue
        question = next((m["content"] for m in messages if m.get("role") == "user"), "")
        assistant = next((m for m in messages if m.get("role") == "assistant"), None)
        if assistant is None:
            continue
        should_anchor, category = _should_anchor(question)
        if not should_anchor:
            continue
        checked += 1
        content = assistant.get("content", "")
        if content.startswith("Key facts: "):
            continue
        anchor = _build_strategy_anchor(content) if category == "strategy" else _build_corner_anchor(content)
        if not anchor:
            continue
        assistant["content"] = anchor + content
        changed += 1
    _write_jsonl(path, examples)
    log.info("Grounding prefix added for %d/%d targeted examples in %s", changed, checked, path)
    return changed, checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Add grounding prefixes to selected JSONL answers")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    for path in args.input:
        add_grounding_prefix(path)


if __name__ == "__main__":
    main()
