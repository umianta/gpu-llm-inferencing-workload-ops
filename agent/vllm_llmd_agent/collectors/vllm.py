"""vLLM /metrics collector.

Scrapes the Prometheus endpoint every vLLM server exposes and distills the
signals that matter for serving: queue depth, latency (TTFT / ITL), throughput,
KV-cache pressure and prefix-cache effectiveness.

vLLM metric names (v0.6+):
  vllm:num_requests_running / vllm:num_requests_waiting
  vllm:time_to_first_token_seconds_{sum,count}
  vllm:time_per_output_token_seconds_{sum,count}
  vllm:generation_tokens_total / vllm:prompt_tokens_total
  vllm:gpu_cache_usage_perc
  vllm:gpu_prefix_cache_hit_rate  (or *_hits_total / *_queries_total)
"""

from __future__ import annotations

import logging

import requests

from . import promparse
from ..schema import VllmSample

logger = logging.getLogger("collector.vllm")


def _sum_by_name(samples, name: str) -> float:
    return sum(s.value for s in samples if s.name == name)


def _first_model(samples) -> str | None:
    for s in samples:
        if "model_name" in s.labels:
            return s.labels["model_name"]
    return None


def _avg_from_histogram(samples, base: str) -> float:
    total = _sum_by_name(samples, f"{base}_sum")
    count = _sum_by_name(samples, f"{base}_count")
    return total / count if count > 0 else 0.0


class VllmCollector:
    def __init__(self, endpoints: list[str], timeout: float = 5.0) -> None:
        self.endpoints = [e.rstrip("/") for e in endpoints]
        self.timeout = timeout
        self.errors = 0  # failed scrapes in the last collect()

    def collect(self) -> list[VllmSample]:
        out: list[VllmSample] = []
        self.errors = 0
        for base in self.endpoints:
            sample = self._scrape(base)
            if sample is not None:
                out.append(sample)
            else:
                self.errors += 1
        return out

    def _scrape(self, base: str) -> VllmSample | None:
        url = f"{base}/metrics"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("vllm scrape failed %s: %s", url, exc)
            return None

        samples = list(promparse.parse(resp.text))
        s = VllmSample(endpoint=base)
        s.model_name = _first_model(samples)
        s.requests_running = _sum_by_name(samples, "vllm:num_requests_running")
        s.requests_waiting = _sum_by_name(samples, "vllm:num_requests_waiting")

        # Latency histograms (avg = _sum / _count). ITL is inter_token_latency;
        # fall back to the older time_per_output_token name.
        s.ttft_seconds = _avg_from_histogram(samples, "vllm:time_to_first_token_seconds")
        s.itl_seconds = _avg_from_histogram(samples, "vllm:inter_token_latency_seconds")
        if s.itl_seconds == 0.0:
            s.itl_seconds = _avg_from_histogram(samples, "vllm:time_per_output_token_seconds")
        s.e2e_latency_seconds = _avg_from_histogram(samples, "vllm:e2e_request_latency_seconds")
        s.queue_time_seconds = _avg_from_histogram(samples, "vllm:request_queue_time_seconds")
        s.inference_time_seconds = _avg_from_histogram(samples, "vllm:request_inference_time_seconds")
        s.prefill_time_seconds = _avg_from_histogram(samples, "vllm:request_prefill_time_seconds")
        s.decode_time_seconds = _avg_from_histogram(samples, "vllm:request_decode_time_seconds")

        s.tokens_generated_total = _sum_by_name(samples, "vllm:generation_tokens_total")
        s.tokens_prompt_total = _sum_by_name(samples, "vllm:prompt_tokens_total")
        s.tokens_prompt_cached_total = _sum_by_name(samples, "vllm:prompt_tokens_cached_total")
        s.requests_success_total = _sum_by_name(samples, "vllm:request_success_total")
        s.preemptions_total = _sum_by_name(samples, "vllm:num_preemptions_total")

        # KV-cache usage: newer vLLM uses vllm:kv_cache_usage_perc; older uses
        # vllm:gpu_cache_usage_perc. Both are fractions (0-1).
        kv = _sum_by_name(samples, "vllm:kv_cache_usage_perc")
        if kv == 0.0:
            kv = _sum_by_name(samples, "vllm:gpu_cache_usage_perc")
        s.kv_cache_usage_percent = kv * 100.0

        # Prefix cache: prefer a hit_rate gauge, else derive from hits/queries
        # counters. Metric names vary (with/without the gpu_ prefix) by version.
        hit_rate = _sum_by_name(samples, "vllm:gpu_prefix_cache_hit_rate")
        if hit_rate == 0.0:
            hits = _sum_by_name(samples, "vllm:prefix_cache_hits_total") or _sum_by_name(
                samples, "vllm:gpu_prefix_cache_hits_total"
            )
            queries = _sum_by_name(samples, "vllm:prefix_cache_queries_total") or _sum_by_name(
                samples, "vllm:gpu_prefix_cache_queries_total"
            )
            hit_rate = hits / queries if queries > 0 else 0.0
        s.prefix_cache_hit_rate = hit_rate
        return s
