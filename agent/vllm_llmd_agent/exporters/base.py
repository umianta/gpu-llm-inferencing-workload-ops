"""Common exporter contract.

The agent normalizes everything to a flat list of :class:`MetricPoint` each
cycle. Exporters translate that list into their wire format. This keeps
collectors and attribution independent of any backend.

Metric names use dots (OTel convention, e.g. ``gpu.utilization``). The
Prometheus exporter converts dots to underscores automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class MetricPoint:
    name: str                       # dotted, e.g. "gpu.power.draw.watts"
    value: float
    unit: str = ""                  # UCUM-ish: "1", "By", "Cel", "W", "s", "%"
    kind: str = "gauge"             # "gauge" | "counter"
    labels: dict[str, str] = field(default_factory=dict)


class Exporter(Protocol):
    def start(self) -> None: ...

    def emit(self, points: list[MetricPoint]) -> None: ...

    def stop(self) -> None: ...
