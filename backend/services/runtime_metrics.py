"""Small vendor-neutral runtime metrics seam with low-cardinality labels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import Lock
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class HttpMetric:
    method: str
    route: str
    status_family: str
    duration_ms: float


@runtime_checkable
class RuntimeMetricsSink(Protocol):
    def record_http(self, metric: HttpMetric) -> None: ...
    def record_readiness(self, *, ready: bool) -> None: ...


class InMemoryRuntimeMetrics:
    """Process-local aggregate metrics; no identifiers or patient content."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._http: Counter[tuple[str, str, str]] = Counter()
        self._http_errors = 0
        self._duration_count = 0
        self._duration_sum_ms = 0.0
        self._duration_max_ms = 0.0
        self._readiness: Counter[str] = Counter()

    def record_http(self, metric: HttpMetric) -> None:
        with self._lock:
            self._http[(metric.method, metric.route, metric.status_family)] += 1
            if metric.status_family in {"4xx", "5xx"}:
                self._http_errors += 1
            self._duration_count += 1
            self._duration_sum_ms += metric.duration_ms
            self._duration_max_ms = max(self._duration_max_ms, metric.duration_ms)

    def record_readiness(self, *, ready: bool) -> None:
        with self._lock:
            self._readiness["ready" if ready else "not_ready"] += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            requests = [
                {
                    "method": method,
                    "route": route,
                    "status_family": family,
                    "count": count,
                }
                for (method, route, family), count in sorted(self._http.items())
            ]
            mean_ms = (
                self._duration_sum_ms / self._duration_count
                if self._duration_count
                else 0.0
            )
            return {
                "http_requests": requests,
                "http_error_count": self._http_errors,
                "latency": {
                    "count": self._duration_count,
                    "mean_ms": round(mean_ms, 3),
                    "max_ms": round(self._duration_max_ms, 3),
                },
                "readiness": dict(self._readiness),
            }


_sink: RuntimeMetricsSink = InMemoryRuntimeMetrics()


def get_runtime_metrics_sink() -> RuntimeMetricsSink:
    return _sink


def set_runtime_metrics_sink(sink: RuntimeMetricsSink) -> RuntimeMetricsSink:
    global _sink
    previous = _sink
    _sink = sink
    return previous


def record_http_request(*, method: str, route: str, status_code: int, duration_ms: float) -> None:
    try:
        _sink.record_http(
            HttpMetric(
                method=method.upper()[:12],
                route=route[:200],
                status_family=f"{max(0, min(9, status_code // 100))}xx",
                duration_ms=max(0.0, duration_ms),
            )
        )
    except Exception:
        return None


def record_readiness(*, ready: bool) -> None:
    try:
        _sink.record_readiness(ready=ready)
    except Exception:
        return None


__all__ = [
    "HttpMetric",
    "InMemoryRuntimeMetrics",
    "RuntimeMetricsSink",
    "get_runtime_metrics_sink",
    "record_http_request",
    "record_readiness",
    "set_runtime_metrics_sink",
]
