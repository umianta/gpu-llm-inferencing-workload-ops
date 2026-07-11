"""Prometheus exporter — exposes an HTTP ``/metrics`` endpoint for scraping.

Uses ``prometheus_client``. Because label *sets* differ across cycles (pods come
and go), we register gauges/counters lazily keyed by (metric, label-key tuple)
and clear stale series each cycle to avoid unbounded cardinality growth.
"""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from .base import MetricPoint

logger = logging.getLogger("exporter.prometheus")


def _prom_name(dotted: str) -> str:
    return dotted.replace(".", "_").replace("-", "_")


class PrometheusExporter:
    def __init__(self, port: int) -> None:
        self.port = port
        self.registry = CollectorRegistry()
        self._metrics: dict[str, Gauge | Counter] = {}
        self._seen: dict[str, set[tuple[str, ...]]] = {}
        self._keys: dict[str, list[str]] = {}
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        start_http_server(self.port, registry=self.registry)
        self._started = True
        logger.info("prometheus endpoint on :%d/metrics", self.port)

    def _metric(self, point: MetricPoint, label_keys: list[str]) -> Gauge | Counter:
        name = _prom_name(point.name)
        existing = self._metrics.get(name)
        if existing is not None:
            return existing
        cls = Counter if point.kind == "counter" else Gauge
        # Counters expose <name>_total; strip a trailing _total to avoid doubling.
        metric_name = name[:-6] if (cls is Counter and name.endswith("_total")) else name
        metric = cls(
            metric_name,
            f"{point.name} ({point.unit})" if point.unit else point.name,
            labelnames=label_keys,
            registry=self.registry,
        )
        self._metrics[name] = metric
        self._keys[name] = label_keys
        return metric

    def emit(self, points: list[MetricPoint]) -> None:
        # Track which label-sets are present this cycle so we can drop only the
        # series that disappeared (departed pods/GPUs) — without clear()ing the
        # whole metric. clear() resets each series' _created timestamp, which
        # Prometheus reads as a counter reset and breaks rate()/increase().
        current: dict[str, set[tuple[str, ...]]] = {}
        for p in points:
            name = _prom_name(p.name)
            metric = self._metric(p, sorted(p.labels.keys()))
            # Align values to the metric's REGISTERED label keys (fixed at first
            # creation). Missing keys default to "" and extra keys are ignored —
            # so an occasional inconsistent label set can never crash the cycle.
            reg_keys = self._keys[name]
            values = tuple(p.labels.get(k, "") for k in reg_keys)
            child = metric.labels(*values) if values else metric
            if isinstance(child, Counter):
                # Counter has no set(); set the absolute (monotonic) value.
                child._value.set(p.value)  # type: ignore[attr-defined]
            else:
                child.set(p.value)
            current.setdefault(name, set()).add(values)

        # Remove series that were present last cycle but not this one.
        for name, metric in self._metrics.items():
            stale = self._seen.get(name, set()) - current.get(name, set())
            for labels in stale:
                if labels:
                    try:
                        metric.remove(*labels)
                    except KeyError:
                        pass
        self._seen = current

    def stop(self) -> None:
        pass
