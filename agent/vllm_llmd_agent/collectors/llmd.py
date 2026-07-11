"""llm-d collector.

llm-d's inference scheduler / Endpoint Picker (EPP) and the KV-cache-aware
routing layer expose Prometheus metrics. Names vary across llm-d releases, so
we match a set of candidate names and take the first that resolves.

Candidate metric names:
  prefix cache hit rate : llmd_prefix_cache_hit_rate,
                          inference_extension_prefix_cache_hit_ratio
  kv cache utilisation  : llmd_kv_cache_usage_perc,
                          inference_pool_average_kv_cache_utilization
  routing decisions     : llmd_routing_decisions_total,
                          inference_extension_scheduler_decisions_total
  ready endpoints       : inference_pool_ready_pods,
                          llmd_pool_ready_endpoints
"""

from __future__ import annotations

import logging

import requests

from . import promparse
from ..schema import LlmdSample

logger = logging.getLogger("collector.llmd")


def _first_match(samples, names: list[str]) -> float:
    by_name: dict[str, float] = {}
    for s in samples:
        by_name[s.name] = by_name.get(s.name, 0.0) + s.value
    for n in names:
        if n in by_name:
            return by_name[n]
    return 0.0


_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"


def _read_sa_token() -> str:
    """In-cluster ServiceAccount token, used as the EPP metrics bearer token."""
    try:
        with open(_TOKEN_PATH, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


class LlmdCollector:
    def __init__(self, endpoints: list[str], timeout: float = 5.0) -> None:
        self.endpoints = [e.rstrip("/") for e in endpoints]
        self.timeout = timeout
        # EPP /metrics is protected by delegated K8s auth (TokenReview +
        # SubjectAccessReview). Present the pod's SA token when available.
        token = _read_sa_token()
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.errors = 0  # failed scrapes in the last collect()

    def collect(self) -> list[LlmdSample]:
        out: list[LlmdSample] = []
        self.errors = 0
        for base in self.endpoints:
            sample = self._scrape(base)
            if sample is not None:
                out.append(sample)
            else:
                self.errors += 1
        return out

    def _scrape(self, base: str) -> LlmdSample | None:
        url = f"{base}/metrics"
        try:
            resp = requests.get(url, timeout=self.timeout, headers=self._headers)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("llm-d scrape failed %s: %s", url, exc)
            return None

        samples = list(promparse.parse(resp.text))
        s = LlmdSample(endpoint=base)

        # Prefix cache: EPP exposes a hit-bytes histogram, not a ratio. Approximate
        # a hit rate as mean hit bytes per query, normalised is not meaningful, so
        # fall back to any explicit ratio metric if a future version adds one.
        s.prefix_cache_hit_rate = _first_match(
            samples,
            ["llmd_prefix_cache_hit_rate", "inference_extension_prefix_cache_hit_ratio"],
        )

        # KV cache utilisation is a 0-1 gauge -> percent.
        s.kv_cache_usage_percent = _first_match(
            samples,
            ["inference_pool_average_kv_cache_utilization", "llmd_kv_cache_usage_perc"],
        ) * 100.0

        # Scheduler attempts = routing decisions made by the EPP.
        s.routing_decisions_total = _first_match(
            samples,
            [
                "inference_extension_scheduler_attempts_total",
                "inference_extension_scheduler_decisions_total",
                "llmd_routing_decisions_total",
            ],
        )
        s.ready_endpoints = _first_match(
            samples,
            ["inference_pool_ready_pods", "llmd_pool_ready_endpoints"],
        )
        return s
