"""
When the model returns a vacuous telemetry reply (no lap times), append a
short factual line built from the SQL block so the user still sees the numbers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ALL_DRIVERS

_CANDIDATES = re.compile(
    r"(?:\b\d+:\d{2}\.\d{3}\b)|(?:\b\d{2,3}\.\d{3}\b)"
)


def _time_tokens_in_text(text: str) -> int:
    if not text:
        return 0
    return len(_CANDIDATES.findall(text))


def _response_looks_lap_vacuous(response: str) -> bool:
    """True if the reply is short and doesn't quote lap times from the table style."""
    if not response or not response.strip():
        return True
    if len(response) < 500 and _time_tokens_in_text(response) < 2:
        return True
    if len(response) < 200:
        return True
    return False


def _sec_to_lap_str(sec: float) -> str:
    if sec is None or sec < 0:
        return "N/A"
    if sec >= 60.0:
        m = int(sec // 60)
        s = sec - m * 60
        return f"{m}:{s:06.3f}"
    return f"{sec:.3f}"


def _cell_to_sec(tok: str) -> float | None:
    tok = tok.strip()
    m = re.match(r"^(\d+):(\d{2}\.\d+)$", tok)
    if m:
        return int(m.group(1)) * 60.0 + float(m.group(2))
    try:
        return float(tok)
    except ValueError:
        return None


def _parse_telemetry_lap_table(telemetry: str) -> list[tuple[str, int, float]]:
    """Parse Dr / Lap / LapTime rows after the 'Dr' header."""
    in_table = False
    out: list[tuple[str, int, float]] = []
    for line in telemetry.splitlines():
        if re.match(r"^\s*Dr\s+", line) and "LapTime" in line:
            in_table = True
            continue
        if in_table:
            s = line.strip()
            if not s or s.startswith("---") or s.startswith("==="):
                if out:
                    break
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            code, lap_s, t_s = parts[0], parts[1], parts[2]
            if code not in set(ALL_DRIVERS) or not lap_s.isdigit():
                continue
            tsec = _cell_to_sec(t_s)
            if tsec is not None:
                out.append((code, int(lap_s), tsec))
    return out


def _synthesize_lap_bullets(rows: list[tuple[str, int, float]]) -> str:
    if not rows:
        return ""
    ordered = sorted(rows, key=lambda x: x[2])
    lines: list[str] = [
        "**Data (retrieved, verbatim from query)**",
        "The model should have used these; numbers below match the log/SQL, not a guess:",
    ]
    for code, lap, tsec in rows:
        lines.append(f"- {code} - lap {lap}, {_sec_to_lap_str(tsec)}")
    if len(ordered) == 2:
        a, b = ordered[0], ordered[1]
        gap = abs(a[2] - b[2])
        faster = a[0]
        lines.append(
            f"- Overall gap: **{gap:.3f}s** ({faster} faster)."
        )
    return "\n".join(lines)


def maybe_append_telemetry_synthesis(
    response: str, telemetry: str, user_message: str
) -> str:
    """
    If the model forgot to include lap times, append a structured summary from the block.
    user_message is unused for now; kept for future heuristics.
    """
    _ = user_message
    if not _response_looks_lap_vacuous(response):
        return response
    rows = _parse_telemetry_lap_table(telemetry)
    if len(rows) < 1:
        return response
    extra = _synthesize_lap_bullets(rows)
    if not extra:
        return response
    return response.rstrip() + "\n\n" + extra
