"""Agent configuration — CLI flags with env fallbacks."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class Config:
    cluster: str = "default"
    node_name: str = field(default_factory=lambda: os.environ.get("NODE_NAME", ""))
    interval: int = 60
    once: bool = False

    # collectors
    enable_nvml: bool = True
    vllm_endpoints: list[str] = field(default_factory=list)
    llmd_endpoints: list[str] = field(default_factory=list)

    # discovery — auto-find vLLM/EPP pods on THIS node (multi-node safe: each
    # pod is scraped by exactly one agent, so no duplicate series across nodes).
    enable_discovery: bool = True
    vllm_selector: str = ""             # optional label selector; "" = any model pod
    vllm_default_port: int = 8000
    llmd_default_port: int = 9090

    # attribution
    enable_k8s: bool = True

    # exporters
    sink: str = "prometheus"            # prometheus | otlp | both | stdout
    prometheus_port: int = 9835
    otlp_endpoint: str = field(
        default_factory=lambda: os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    )
    otlp_timeout: int = 10

    # cost model
    gpu_hourly_usd: float = 0.0
    # persist accrued GPU-seconds here so cost survives agent restarts.
    # Empty = in-memory only (resets on restart).
    cost_state_path: str = field(
        default_factory=lambda: os.environ.get("COST_STATE_PATH", "")
    )

    @staticmethod
    def from_args(argv: list[str] | None = None) -> "Config":
        p = argparse.ArgumentParser(
            prog="vllm-llmd-agent",
            description="Workload-aware GPU + vLLM + llm-d telemetry agent",
        )
        p.add_argument("--cluster", default=os.environ.get("CLUSTER", "default"))
        p.add_argument("--interval", type=int, default=int(os.environ.get("INTERVAL", "60")))
        p.add_argument("--once", action="store_true", help="collect one cycle and exit")
        p.add_argument("--no-nvml", dest="enable_nvml", action="store_false")
        p.add_argument(
            "--vllm-endpoints",
            default="",
            help="comma-separated vLLM /metrics base URLs (or VLLM_ENDPOINTS env)",
        )
        p.add_argument(
            "--llmd-endpoints",
            default="",
            help="comma-separated llm-d EPP/gateway base URLs (or LLMD_ENDPOINTS env)",
        )
        p.add_argument("--no-k8s", dest="enable_k8s", action="store_false")
        p.add_argument(
            "--no-discovery",
            dest="enable_discovery",
            action="store_false",
            help="disable per-node auto-discovery of vLLM/EPP pods",
        )
        p.add_argument(
            "--vllm-selector",
            default=os.environ.get("VLLM_SELECTOR", ""),
            help="label selector for vLLM pods (default: any pod that serves a model)",
        )
        p.add_argument(
            "--sink",
            choices=["prometheus", "otlp", "both", "stdout"],
            default=os.environ.get("SINK", "prometheus"),
        )
        p.add_argument("--prometheus-port", type=int, default=int(os.environ.get("PROM_PORT", "9835")))
        p.add_argument("--gpu-hourly-usd", type=float, default=float(os.environ.get("GPU_HOURLY_USD", "0")))
        p.add_argument(
            "--cost-state-path",
            default=os.environ.get("COST_STATE_PATH", ""),
            help="file to persist accrued GPU-seconds across restarts",
        )
        args = p.parse_args(argv)

        vllm = [x.strip() for x in args.vllm_endpoints.split(",") if x.strip()] or _env_list(
            "VLLM_ENDPOINTS"
        )
        llmd = [x.strip() for x in args.llmd_endpoints.split(",") if x.strip()] or _env_list(
            "LLMD_ENDPOINTS"
        )

        return Config(
            cluster=args.cluster,
            interval=args.interval,
            once=args.once,
            enable_nvml=args.enable_nvml,
            vllm_endpoints=vllm,
            llmd_endpoints=llmd,
            enable_discovery=args.enable_discovery,
            vllm_selector=args.vllm_selector,
            enable_k8s=args.enable_k8s,
            sink=args.sink,
            prometheus_port=args.prometheus_port,
            gpu_hourly_usd=args.gpu_hourly_usd,
            cost_state_path=args.cost_state_path,
        )
