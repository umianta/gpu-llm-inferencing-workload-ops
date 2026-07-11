"""Agent core — the collect → attribute → build → emit loop."""

from __future__ import annotations

import json as _json
import logging
import os
import signal
import time
import urllib.parse

from . import __version__
from .attribution.k8s import K8sAttributor, PodInfo
from .collectors.llmd import LlmdCollector
from .collectors.nvml import NvmlCollector
from .collectors.vllm import VllmCollector
from .config import Config
from .exporters.base import Exporter, MetricPoint
from .schema import Attribution, GpuSample, LlmdSample, VllmSample

logger = logging.getLogger("agent")


def _endpoint_meta(endpoint: str) -> tuple[str, str | None, str | None]:
    """Derive (service, namespace, role) from a K8s service DNS endpoint.

    e.g. http://vllm-prefill.llm-d:8000 -> ("vllm-prefill", "llm-d", "prefill")
         http://modelserver.llm-d-flow-control:8000 -> ("modelserver", ...)
    """
    host = urllib.parse.urlsplit(endpoint).hostname or endpoint
    parts = host.split(".")
    service = parts[0] if parts else host
    namespace = parts[1] if len(parts) > 1 else None
    low = service.lower()
    role: str | None = None
    if "prefill" in low:
        role = "prefill"
    elif "decode" in low:
        role = "decode"
    return service, namespace, role


class Agent:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        use_discovery = cfg.enable_discovery and cfg.enable_k8s
        self.nvml = NvmlCollector() if cfg.enable_nvml else None
        self.vllm = (
            VllmCollector(cfg.vllm_endpoints)
            if (cfg.vllm_endpoints or use_discovery)
            else None
        )
        self.llmd = (
            LlmdCollector(cfg.llmd_endpoints)
            if (cfg.llmd_endpoints or use_discovery)
            else None
        )
        self.attributor = K8sAttributor(cfg.node_name) if cfg.enable_k8s else None
        self.exporters: list[Exporter] = self._build_exporters()
        self._running = True
        # cost bookkeeping: gpu_uuid -> accumulated gpu-seconds
        self._gpu_seconds: dict[str, float] = {}
        # the single model running on this node (if unambiguous), used to label
        # GPU metrics when per-pod attribution can't pin one owner.
        self._node_model: str | None = None
        # url -> EndpointInfo for the current cycle (discovery attribution)
        self._ep_meta: dict[str, object] = {}
        # self-observability
        self._cycles = 0
        self._cycle_seconds = 0.0
        self._scrape_errors = 0
        # restore accrued cost from disk (survives restarts)
        self._load_cost()

    def _load_cost(self) -> None:
        path = self.cfg.cost_state_path
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            self._gpu_seconds = {str(k): float(v) for k, v in data.items()}
            logger.info("restored cost state for %d GPU(s) from %s", len(self._gpu_seconds), path)
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not restore cost state: %s", exc)

    def _save_cost(self) -> None:
        path = self.cfg.cost_state_path
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                _json.dump(self._gpu_seconds, fh)
            os.replace(tmp, path)  # atomic
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not persist cost state: %s", exc)

    def _build_exporters(self) -> list[Exporter]:
        # Imported lazily so Prometheus-only or stdout mode doesn't require the
        # OTLP libraries (and vice-versa).
        sink = self.cfg.sink
        out: list[Exporter] = []
        if sink in ("prometheus", "both"):
            from .exporters.prometheus import PrometheusExporter

            out.append(PrometheusExporter(self.cfg.prometheus_port))
        if sink in ("otlp", "both"):
            from .exporters.otlp import OtlpExporter

            out.append(OtlpExporter(self.cfg.otlp_endpoint, self.cfg.cluster, self.cfg.otlp_timeout))
        if sink == "stdout":
            from .exporters.stdout import StdoutExporter

            out.append(StdoutExporter())
        return out

    def start(self) -> None:
        if self.nvml:
            self.nvml.start()
        if self.attributor:
            self.attributor.start()
        for exp in self.exporters:
            exp.start()

    def stop(self) -> None:
        self._running = False
        if self.nvml:
            self.nvml.stop()
        for exp in self.exporters:
            exp.stop()

    def run(self) -> None:
        self.start()
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        while self._running:
            t0 = time.monotonic()
            try:
                points = self._cycle()
                self._cycle_seconds = time.monotonic() - t0
                self._cycles += 1
                points += self._self_points()
                for exp in self.exporters:
                    exp.emit(points)
                self._save_cost()
                logger.info("cycle: %d points", len(points))
            except Exception:  # noqa: BLE001 - never let one cycle kill the agent
                logger.exception("collection cycle failed")
            if self.cfg.once:
                break
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self.cfg.interval - elapsed))
        self.stop()

    # ---- the join ----------------------------------------------------------

    def _cycle(self) -> list[MetricPoint]:
        if self.attributor:
            self.attributor.refresh()
            if self.cfg.enable_discovery:
                self._apply_discovery()

        gpu_samples = self.nvml.collect() if self.nvml else []
        vllm_samples = self.vllm.collect() if self.vllm else []
        llmd_samples = self.llmd.collect() if self.llmd else []
        self._scrape_errors = (
            (self.vllm.errors if self.vllm else 0)
            + (self.llmd.errors if self.llmd else 0)
        )

        # If exactly one distinct model is being served on this node, use it to
        # label GPU metrics (single-GPU / time-sliced nodes serve one model).
        models = {v.model_name for v in vllm_samples if v.model_name}
        self._node_model = next(iter(models)) if len(models) == 1 else None

        points: list[MetricPoint] = []
        points += self._gpu_points(gpu_samples)
        points += self._vllm_points(vllm_samples)
        points += self._llmd_points(llmd_samples)
        return points

    def _self_points(self) -> list[MetricPoint]:
        """Observability of the observer — the agent's own health."""
        labels = {"cluster": self.cfg.cluster, "node": self.cfg.node_name or ""}
        n_targets = (len(self.vllm.endpoints) if self.vllm else 0) + (
            len(self.llmd.endpoints) if self.llmd else 0
        )
        return [
            MetricPoint("agent.up", 1, "1", "gauge", labels),
            MetricPoint("agent.cycles.total", self._cycles, "1", "counter", labels),
            MetricPoint("agent.cycle.duration.seconds", self._cycle_seconds, "s", "gauge", labels),
            MetricPoint("agent.scrape.errors", self._scrape_errors, "1", "gauge", labels),
            MetricPoint("agent.targets.discovered", n_targets, "1", "gauge", labels),
            MetricPoint(
                "agent.build.info", 1, "1", "gauge",
                {**labels, "version": __version__},
            ),
        ]

    def _apply_discovery(self) -> None:
        """Point the vLLM/EPP collectors at pods discovered on THIS node.

        Merged with any static endpoints from config. Because each pod runs on
        exactly one node, every target is scraped by exactly one agent — so
        multi-node deployments never double-count.
        """
        v_eps = self.attributor.local_vllm_endpoints()
        l_eps = self.attributor.local_llmd_endpoints()
        self._ep_meta = {e.url: e for e in (v_eps + l_eps)}
        if self.vllm is not None:
            self.vllm.endpoints = sorted(
                set(self.cfg.vllm_endpoints) | {e.url for e in v_eps}
            )
        if self.llmd is not None:
            self.llmd.endpoints = sorted(
                set(self.cfg.llmd_endpoints) | {e.url for e in l_eps}
            )

    def _endpoint_labels(self, endpoint: str) -> tuple[str, str, str]:
        """(service, namespace, role) for an endpoint — from discovery meta if
        available, else parsed from the URL (static/service-name endpoints)."""
        meta = self._ep_meta.get(endpoint)
        if meta is not None:
            return (meta.service or "unknown", meta.namespace or "", meta.role or "aggregated")
        service, namespace, role = _endpoint_meta(endpoint)
        return service, namespace or "", role or "aggregated"

    def _attr_for_gpu(self, gpu: GpuSample) -> Attribution:
        attr = Attribution(
            gpu_index=gpu.index,
            gpu_uuid=gpu.uuid,
            node_name=self.cfg.node_name or None,
        )
        pod: PodInfo | None = None
        if self.attributor:
            for pid in gpu.compute_pids:
                pod = self.attributor.pod_for_pid(pid)
                if pod:
                    break
            if pod is None:
                pods = self.attributor.pods()
                pod = pods[0] if len(pods) == 1 else None
        if pod:
            attr.pod_name = pod.name
            attr.namespace = pod.namespace
            attr.deployment = pod.deployment
            attr.model_name = pod.model_name
            attr.role = pod.role
        return attr

    def _gpu_points(self, samples: list[GpuSample]) -> list[MetricPoint]:
        out: list[MetricPoint] = []
        for g in samples:
            labels = self._attr_for_gpu(g).as_labels()
            labels["cluster"] = self.cfg.cluster
            labels["gpu_model"] = g.model
            # Always keep a stable `model` label key. Prefer per-pod attribution,
            # else the single node model, else "shared" (multiple models on GPU).
            if "model" not in labels:
                labels["model"] = self._node_model or "shared"
            out += [
                MetricPoint("gpu.utilization", g.utilization, "1", "gauge", labels),
                MetricPoint("gpu.memory.used.bytes", g.memory_used_bytes, "By", "gauge", labels),
                MetricPoint("gpu.memory.used.percent", g.memory_used_percent, "%", "gauge", labels),
                MetricPoint("gpu.temperature.celsius", g.temperature_c, "Cel", "gauge", labels),
                MetricPoint("gpu.power.draw.watts", g.power_draw_w, "W", "gauge", labels),
                MetricPoint("gpu.power.limit.watts", g.power_limit_w, "W", "gauge", labels),
                MetricPoint("gpu.ecc.errors.uncorrected", g.ecc_uncorrected, "1", "counter", labels),
                MetricPoint("gpu.throttle.active", g.throttle_active, "1", "gauge", labels),
            ]
            out += self._cost_points(g, labels)
            out += self._vram_by_model_points(g)
            out += self._util_by_model_points(g)
        return out

    def _model_for_pid(self, pid: int) -> str:
        if self.attributor:
            pod = self.attributor.pod_for_pid(pid)
            if pod and pod.model_name:
                return pod.model_name
        return "unattributed"

    def _vram_by_model_points(self, g: GpuSample) -> list[MetricPoint]:
        """Per-model VRAM: map each GPU compute process to its model and sum."""
        if not g.proc_mem:
            return []
        by_model: dict[str, int] = {}
        for pid, mem in g.proc_mem.items():
            model = self._model_for_pid(pid)
            by_model[model] = by_model.get(model, 0) + int(mem)
        out: list[MetricPoint] = []
        for model, mem in by_model.items():
            labels = {
                "cluster": self.cfg.cluster,
                "gpu_uuid": g.uuid,
                "node": self.cfg.node_name or "",
                "model": model,
            }
            out.append(MetricPoint("gpu.memory.model.bytes", mem, "By", "gauge", labels))
        return out

    def _util_by_model_points(self, g: GpuSample) -> list[MetricPoint]:
        """Per-model GPU utilization from NVML per-process SM samples (percent)."""
        if not g.proc_util:
            return []
        by_model: dict[str, int] = {}
        for pid, sm in g.proc_util.items():
            model = self._model_for_pid(pid)
            by_model[model] = by_model.get(model, 0) + int(sm)
        out: list[MetricPoint] = []
        for model, sm in by_model.items():
            labels = {
                "cluster": self.cfg.cluster,
                "gpu_uuid": g.uuid,
                "node": self.cfg.node_name or "",
                "model": model,
            }
            out.append(MetricPoint("gpu.utilization.model.percent", sm, "%", "gauge", labels))
        return out

    def _cost_points(self, g: GpuSample, labels: dict[str, str]) -> list[MetricPoint]:
        # Accumulate active GPU-seconds only while the GPU is doing work.
        prev = self._gpu_seconds.get(g.uuid, 0.0)
        if g.utilization > 0.01:
            prev += self.cfg.interval
        self._gpu_seconds[g.uuid] = prev
        points = [MetricPoint("cost.gpu.seconds", prev, "s", "counter", labels)]
        if self.cfg.gpu_hourly_usd > 0:
            usd = prev / 3600.0 * self.cfg.gpu_hourly_usd
            points.append(MetricPoint("cost.gpu.usd.estimate", usd, "1", "counter", labels))
        return points

    def _vllm_points(self, samples: list[VllmSample]) -> list[MetricPoint]:
        out: list[MetricPoint] = []
        for v in samples:
            service, namespace, role = self._endpoint_labels(v.endpoint)
            # Keep the label KEYS identical across every sample — Prometheus
            # requires consistent label sets per metric.
            labels = {
                "cluster": self.cfg.cluster,
                "endpoint": v.endpoint,
                "service": service,
                "model_namespace": namespace or "",
                "role": role or "aggregated",
                "model": v.model_name or "unknown",
            }
            out += [
                MetricPoint("vllm.requests.running", v.requests_running, "1", "gauge", labels),
                MetricPoint("vllm.requests.waiting", v.requests_waiting, "1", "gauge", labels),
                MetricPoint("vllm.requests.success.total", v.requests_success_total, "1", "counter", labels),
                MetricPoint("vllm.preemptions.total", v.preemptions_total, "1", "counter", labels),
                MetricPoint("vllm.ttft.seconds", v.ttft_seconds, "s", "gauge", labels),
                MetricPoint("vllm.itl.seconds", v.itl_seconds, "s", "gauge", labels),
                MetricPoint("vllm.e2e_latency.seconds", v.e2e_latency_seconds, "s", "gauge", labels),
                MetricPoint("vllm.queue_time.seconds", v.queue_time_seconds, "s", "gauge", labels),
                MetricPoint("vllm.inference_time.seconds", v.inference_time_seconds, "s", "gauge", labels),
                MetricPoint("vllm.prefill_time.seconds", v.prefill_time_seconds, "s", "gauge", labels),
                MetricPoint("vllm.decode_time.seconds", v.decode_time_seconds, "s", "gauge", labels),
                MetricPoint("vllm.tokens.generated.total", v.tokens_generated_total, "1", "counter", labels),
                MetricPoint("vllm.tokens.prompt.total", v.tokens_prompt_total, "1", "counter", labels),
                MetricPoint("vllm.tokens.prompt.cached.total", v.tokens_prompt_cached_total, "1", "counter", labels),
                MetricPoint("vllm.kv_cache.usage.percent", v.kv_cache_usage_percent, "%", "gauge", labels),
                MetricPoint("vllm.prefix_cache.hit_rate", v.prefix_cache_hit_rate, "1", "gauge", labels),
            ]
        return out

    def _llmd_points(self, samples: list[LlmdSample]) -> list[MetricPoint]:
        out: list[MetricPoint] = []
        for m in samples:
            service, namespace, role = self._endpoint_labels(m.endpoint)
            labels = {
                "cluster": self.cfg.cluster,
                "endpoint": m.endpoint,
                "service": service,
                "model_namespace": namespace or "",
            }
            out += [
                MetricPoint("llmd.prefix_cache.hit_rate", m.prefix_cache_hit_rate, "1", "gauge", labels),
                MetricPoint("llmd.kv_cache.usage.percent", m.kv_cache_usage_percent, "%", "gauge", labels),
                MetricPoint("llmd.routing.decisions.total", m.routing_decisions_total, "1", "counter", labels),
                MetricPoint("llmd.pool.ready_endpoints", m.ready_endpoints, "1", "gauge", labels),
            ]
        return out
