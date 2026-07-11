"""Shared dataclasses (the vendor-neutral telemetry schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Attribution:
    """Who owns a GPU / metric sample."""

    gpu_index: Optional[int] = None
    gpu_uuid: Optional[str] = None
    node_name: Optional[str] = None
    pod_name: Optional[str] = None
    namespace: Optional[str] = None
    deployment: Optional[str] = None
    model_name: Optional[str] = None
    role: Optional[str] = None  # llm-d: "prefill" | "decode" | None

    def as_labels(self) -> dict[str, str]:
        raw = {
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
            "node": self.node_name,
            "pod": self.pod_name,
            "namespace": self.namespace,
            "deployment": self.deployment,
            "model": self.model_name,
            "role": self.role,
        }
        return {k: str(v) for k, v in raw.items() if v is not None}


@dataclass
class GpuSample:
    """One NVML snapshot for a single GPU."""

    index: int
    uuid: str
    model: str
    utilization: float = 0.0            # 0.0-1.0
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    memory_used_percent: float = 0.0    # 0.0-100.0
    temperature_c: int = 0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    ecc_uncorrected: int = 0
    throttle_active: int = 0            # 0/1
    compute_pids: list[int] = field(default_factory=list)
    # pid -> GPU memory bytes used by that process (for per-model VRAM)
    proc_mem: dict[int, int] = field(default_factory=dict)
    # pid -> SM utilization percent (for per-model GPU utilization)
    proc_util: dict[int, int] = field(default_factory=dict)


@dataclass
class VllmSample:
    """Parsed subset of a vLLM /metrics endpoint."""

    endpoint: str
    model_name: Optional[str] = None
    requests_running: float = 0.0
    requests_waiting: float = 0.0
    ttft_seconds: float = 0.0           # rolling avg from histogram
    itl_seconds: float = 0.0            # inter-token latency avg
    e2e_latency_seconds: float = 0.0    # end-to-end request latency avg
    queue_time_seconds: float = 0.0     # time spent waiting in queue avg
    inference_time_seconds: float = 0.0 # total inference time avg
    prefill_time_seconds: float = 0.0   # prefill phase time avg
    decode_time_seconds: float = 0.0    # decode phase time avg
    tokens_generated_total: float = 0.0
    tokens_prompt_total: float = 0.0
    tokens_prompt_cached_total: float = 0.0
    requests_success_total: float = 0.0
    preemptions_total: float = 0.0
    kv_cache_usage_percent: float = 0.0
    prefix_cache_hit_rate: float = 0.0


@dataclass
class LlmdSample:
    """Parsed subset of an llm-d EPP / gateway /metrics endpoint."""

    endpoint: str
    prefix_cache_hit_rate: float = 0.0
    kv_cache_usage_percent: float = 0.0
    routing_decisions_total: float = 0.0
    ready_endpoints: float = 0.0
