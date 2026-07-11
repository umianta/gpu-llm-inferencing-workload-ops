"""OTLP exporter — pushes gauges over OTLP/HTTP to any OpenTelemetry backend.

Endpoint + auth come from the standard env vars, so the same container works
with any OTLP backend (Grafana Cloud, Datadog, a self-hosted Collector, etc.):

  OTEL_EXPORTER_OTLP_ENDPOINT   e.g. https://otlp.example.io
  OTEL_EXPORTER_OTLP_HEADERS    e.g. Authorization=Bearer <token>

All points are emitted as observable gauges (a per-cycle snapshot), matching
All points are emitted as observable gauges (a per-cycle snapshot) — one gauge
per metric per GPU.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from .base import MetricPoint

logger = logging.getLogger("exporter.otlp")


def _parse_headers() -> dict[str, str]:
    raw = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
    headers: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if "=" in part:
            key, _, value = part.partition("=")
            headers[key.strip()] = value.strip()
    return headers


class OtlpExporter:
    def __init__(self, endpoint: str, cluster: str, timeout: int = 10) -> None:
        self.endpoint = endpoint
        self.cluster = cluster
        self.timeout = timeout
        self._provider: MeterProvider | None = None
        self._latest: list[MetricPoint] = []
        self._registered: set[str] = set()

    def start(self) -> None:
        if not self.endpoint:
            logger.warning("OTLP endpoint not set; OTLP export disabled")
            return
        exporter = OTLPMetricExporter(
            endpoint=f"{self.endpoint.rstrip('/')}/v1/metrics",
            headers=_parse_headers(),
            timeout=self.timeout,
        )
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60000)
        resource = Resource.create(
            {"service.name": "vllm-llmd-agent", "cluster": self.cluster}
        )
        self._provider = MeterProvider(metric_readers=[reader], resource=resource)
        logger.info("OTLP exporter -> %s", self.endpoint)

    def _ensure_instrument(self, meter, point: MetricPoint) -> None:  # noqa: ANN001
        if point.name in self._registered:
            return

        def callback(options: CallbackOptions, _name=point.name) -> list[Observation]:  # noqa: ANN001
            return [
                Observation(p.value, attributes=p.labels)
                for p in self._latest
                if p.name == _name
            ]

        meter.create_observable_gauge(
            name=point.name, unit=point.unit or "1", callbacks=[callback]
        )
        self._registered.add(point.name)

    def emit(self, points: list[MetricPoint]) -> None:
        if self._provider is None:
            return
        self._latest = points
        meter = self._provider.get_meter("vllm_llmd_agent")
        for p in points:
            self._ensure_instrument(meter, p)

    def stop(self) -> None:
        if self._provider is not None:
            self._provider.shutdown()
