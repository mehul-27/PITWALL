"""
Telemetry string validation — circuit-agnostic F1 plausibility bounds.
Runs on SQL-formatted result text before the model sees it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from delta_interpretation import (
    check_response_sector_sign_misinterpretation,
    telemetry_has_negative_sector_time_delta,
)

# Two-driver quali comparison pre-checks (identical / too-close times) run in retrieval
# before any model call. Re-export messages for tests and downstream tools.
from retrieval import (  # noqa: E402
    IDENTICAL_LAP_ERROR,
    MISSING_DRIVER_ERROR,
    SUSPECT_CLOSE_QUALI,
)

log = logging.getLogger(__name__)

BOUNDS_REPLACEMENT_MESSAGE = (
    "Telemetry query returned a value outside physically valid bounds for an F1 circuit. "
    "The data may be from a non-representative lap such as an outlap, inlap, safety car lap, "
    "or a lap with traffic. Cannot confirm this figure reliably. Please ask the user if they "
    "want to try a different session or lap filter."
)

_LAP_S = (60.0, 145.0)
_SECTOR_S = (10.0, 60.0)
_SECTOR_DELTA = (-2.0, 8.0)
_TOP_KMH = (270.0, 380.0)
_CORNER_KMH = (40.0, 330.0)
_DEG = (0.001, 0.300)
_LAPS = (5, 55)
_CLIFF = (3, 45)
_QR_DELTA = (0.5, 20.0)
_RANK = (1, 25)


def _to_sec(tok: str) -> float | None:
    t = tok.strip()
    m = re.match(r"^(\d+):(\d{2}\.\d+)$", t)
    if m:
        return int(m.group(1)) * 60.0 + float(m.group(2))
    m2 = re.match(r"^(\d+\.\d+)$", t)
    if m2:
        return float(m2.group(1))
    return None


def _log_bad(field: str, value: float | int, meta: dict[str, Any]) -> dict[str, Any]:
    r = {
        "field": field,
        "value": value,
        "circuit": meta.get("circuit"),
        "season": meta.get("year") or meta.get("season"),
        "session": meta.get("session_type"),
    }
    log.warning(
        "Telemetry validation: field=%r value=%r circuit=%r season=%r session=%r",
        field,
        value,
        r.get("circuit"),
        r.get("season"),
        r.get("session"),
    )
    return r


@dataclass
class TelemetryValidation:
    ok: bool
    failures: list[dict[str, Any]] = field(default_factory=list)


# Lap / sector time tokens: M:SS.ddd or SSS.ddd
_TIME_PAT = re.compile(r"(?:\d+:\d{2}\.\d+|\b\d{2,3}\.\d{2,3}\b)")


def validate_telemetry_block(text: str, meta: dict[str, Any] | None = None) -> TelemetryValidation:
    if not (text and text.strip()):
        return TelemetryValidation(ok=True)

    if "no valid lap" in text.lower() and "after filters" in text.lower():
        return TelemetryValidation(ok=True)

    meta = meta or {}
    failures: list[dict[str, Any]] = []

    for line in text.splitlines():
        if line.strip().startswith("(") or "===" in line or not line.strip():
            continue
        toks = _TIME_PAT.findall(line)
        if len(toks) < 4:
            continue
        conv = [_to_sec(t) for t in toks[:4]]
        for i, v in enumerate(conv):
            if v is None:
                continue
            if i == 0:
                if v < _LAP_S[0] or v > _LAP_S[1]:
                    failures.append(_log_bad("lap_time" if i == 0 else "time", v, meta))
            else:
                if v < _SECTOR_S[0] or v > _SECTOR_S[1]:
                    failures.append(_log_bad(f"sector{i}", v, meta))

    # Labelled free-text fields (if query blocks include these later)
    ft = text
    for m in re.finditer(
        r"(?i)(?:sector\s*delta|delta).{0,6}(-?\d+\.?\d*)\s*s",
        ft,
    ):
        v = float(m.group(1))
        if v < _SECTOR_DELTA[0] or v > _SECTOR_DELTA[1]:
            failures.append(_log_bad("sector_delta", v, meta))
    for m in re.finditer(
        r"(?i)(?:top\s*speed|speed\s*trap).{0,30}(\d{3})\s*km",
        ft,
    ):
        v = float(m.group(1))
        if v < _TOP_KMH[0] or v > _TOP_KMH[1]:
            failures.append(_log_bad("top_speed_kmh", v, meta))
    for m in re.finditer(
        r"(?i)(?:min(imum)?\s*speed|corner).{0,25}(\d{2,3})\s*km",
        ft,
    ):
        v = float(m.group(1))
        if v < _CORNER_KMH[0] or v > _CORNER_KMH[1]:
            failures.append(_log_bad("corner_min_kmh", v, meta))
    for m in re.finditer(
        r"(?i)(?:deg|degradation).{0,12}(\d+\.\d+).{0,8}(?:/|\s)lap|s/lap|per lap",
        ft,
    ):
        v = float(m.group(1))
        if v <= 0 or v < _DEG[0] or v > _DEG[1]:
            failures.append(_log_bad("tyre_deg_s_per_lap", v, meta))
    for m in re.finditer(
        r"(?i)max.{0,10}viable.{0,8}(\d{1,2})\s*laps",
        ft,
    ):
        v = int(m.group(1))
        if v < _LAPS[0] or v > _LAPS[1]:
            failures.append(_log_bad("max_viable_laps", v, meta))
    for m in re.finditer(
        r"(?i)cliff.{0,6}lap.{0,5}(\d{1,2})",
        ft,
    ):
        v = int(m.group(1))
        if v < _CLIFF[0] or v > _CLIFF[1]:
            failures.append(_log_bad("cliff_lap", v, meta))
    for m in re.finditer(
        r"(?i)quali.{0,12}race.{0,12}(\d+\.?\d*)\s*s",
        ft,
    ):
        v = float(m.group(1))
        if v < _QR_DELTA[0] or v > _QR_DELTA[1]:
            failures.append(_log_bad("quali_race_delta", v, meta))
    for m in re.finditer(
        r"(?i)(?:rank|position).{0,4}(?:#|no\.?)\s*(\d{1,2})\b",
        ft,
    ):
        v = int(m.group(1))
        if v < _RANK[0] or v > _RANK[1]:
            failures.append(_log_bad("driver_top_speed_rank", v, meta))

    return TelemetryValidation(ok=len(failures) == 0, failures=failures)
