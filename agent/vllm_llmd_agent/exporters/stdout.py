"""Stdout exporter — pretty-prints one snapshot. Used for --once sanity checks."""

from __future__ import annotations

import json

from .base import MetricPoint


class StdoutExporter:
    def start(self) -> None:  # noqa: D102
        pass

    def emit(self, points: list[MetricPoint]) -> None:  # noqa: D102
        for p in points:
            line = {
                "metric": p.name,
                "value": round(p.value, 4),
                "unit": p.unit,
                **p.labels,
            }
            print(json.dumps(line, sort_keys=True))

    def stop(self) -> None:  # noqa: D102
        pass
