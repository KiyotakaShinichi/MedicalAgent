"""Pure aggregate and lexical helpers for RAG baseline evaluation."""

from __future__ import annotations

import math
import re
from statistics import median
from typing import Any, Iterable, Mapping


def _document_text(item: Mapping[str, Any]) -> str:
    return " ".join([
        str(item.get("title") or ""),
        str(item.get("text") or ""),
        " ".join(str(tag) for tag in item.get("tags") or []),
        str(item.get("topic") or ""),
        str(item.get("section") or ""),
    ])


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9/-]+", (text or "").lower())


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return round(sum(vals) / max(len(vals), 1), 4)


def _rate(values: Iterable[bool]) -> float:
    vals = [bool(value) for value in values]
    return round(sum(1 for value in vals if value) / max(len(vals), 1), 4)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if percentile == 50:
        return round(float(median(ordered)), 3)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return round(ordered[max(0, min(index, len(ordered) - 1))], 3)
