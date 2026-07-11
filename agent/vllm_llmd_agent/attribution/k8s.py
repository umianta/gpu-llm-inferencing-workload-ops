"""Kubernetes workload attribution.

Runs in-cluster (ServiceAccount token) and answers: for a pod on *this* node
that requested ``nvidia.com/gpu``, what is its namespace, deployment and
served model name?

Two joins are provided:

1. ``pod_for_pid`` — best-effort local join of an NVML compute PID to a pod by
   reading the cgroup of that PID (works when the agent shares the host PID
   namespace, as configured in the DaemonSet).
2. ``model_for_pod`` — parse ``--served-model-name`` (or ``--model``) out of the
   vLLM container args so GPU metrics can be labelled by model.

If the K8s API is unreachable (e.g. running on bare metal), every method
degrades to ``None`` and the agent still emits hardware + endpoint metrics.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("attribution.k8s")

try:
    from kubernetes import client, config  # type: ignore

    _HAVE_K8S = True
except Exception:  # pragma: no cover
    client = None  # type: ignore
    config = None  # type: ignore
    _HAVE_K8S = False


@dataclass
class PodInfo:
    name: str
    namespace: str
    deployment: Optional[str]
    model_name: Optional[str]
    role: Optional[str]  # prefill | decode | None
    uid: Optional[str] = None


@dataclass
class EndpointInfo:
    """A discovered scrape target with attribution metadata from the pod."""

    url: str
    service: Optional[str] = None
    namespace: Optional[str] = None
    role: Optional[str] = None
    model: Optional[str] = None


_MODEL_ARG = re.compile(r"--served-model-name(?:=|\s+)(\S+)")
_MODEL_FALLBACK = re.compile(r"--model(?:=|\s+)(\S+)")


def _norm_model(value: str) -> str:
    """Normalize a model arg to match vLLM's reported model_name.

    HF IDs (``Qwen/Qwen2.5-1.5B-Instruct``) are kept whole — vLLM reports them
    verbatim. Filesystem paths (``/models/gemma``) are reduced to their
    basename, since vLLM reports the served-model-name for those.
    """
    value = value.strip()
    if value.startswith("/"):
        return value.rstrip("/").rsplit("/", 1)[-1]
    return value


def _extract_model(args: list[str], cmd: Optional[list[str]]) -> Optional[str]:
    joined = " ".join((cmd or []) + (args or []))
    m = _MODEL_ARG.search(joined)
    if m:
        return _norm_model(m.group(1))
    m = _MODEL_FALLBACK.search(joined)
    if m:
        return _norm_model(m.group(1))
    # arg-list form: ["--served-model-name", "x"]
    tokens = (cmd or []) + (args or [])
    for flag in ("--served-model-name", "--model"):
        if flag in tokens:
            idx = tokens.index(flag)
            if idx + 1 < len(tokens):
                return _norm_model(tokens[idx + 1])
    # positional form: `vllm serve Qwen/Qwen2.5-1.5B-Instruct ...`
    if "serve" in tokens:
        idx = tokens.index("serve")
        for tok in tokens[idx + 1:]:
            if not tok.startswith("-"):
                return _norm_model(tok)
    return None


def _role_from(pod_name: str, labels: dict[str, str]) -> Optional[str]:
    hay = f"{pod_name} {labels.get('llm-d.ai/role', '')} {labels.get('app', '')}".lower()
    if "prefill" in hay:
        return "prefill"
    if "decode" in hay:
        return "decode"
    return None


class K8sAttributor:
    def __init__(self, node_name: str) -> None:
        self.node_name = node_name or os.environ.get("NODE_NAME", "")
        self._api: Optional["client.CoreV1Api"] = None
        self._pid_to_pod: dict[int, PodInfo] = {}
        self._pods: list[PodInfo] = []
        self._by_uid: dict[str, PodInfo] = {}
        self._vllm_eps: list[str] = []
        self._llmd_eps: list[str] = []

    def start(self) -> None:
        if not _HAVE_K8S:
            logger.warning("kubernetes client not installed; attribution disabled")
            return
        try:
            config.load_incluster_config()
        except Exception:
            try:
                config.load_kube_config()
            except Exception as exc:  # noqa: BLE001
                logger.warning("no kube config available: %s", exc)
                return
        self._api = client.CoreV1Api()
        logger.info("k8s attribution ready (node=%s)", self.node_name)

    def refresh(self) -> None:
        """Rebuild the pod cache for this node once per cycle.

        Also discovers vLLM and EPP metrics endpoints for pods **on this node**,
        by pod IP. Because each pod lives on exactly one node, this makes
        multi-node scraping duplication-free.
        """
        if self._api is None:
            return
        field = f"spec.nodeName={self.node_name}" if self.node_name else ""
        try:
            pods = self._api.list_pod_for_all_namespaces(
                field_selector=field or None
            ).items
        except Exception as exc:  # noqa: BLE001
            logger.warning("pod list failed: %s", exc)
            return

        infos: list[PodInfo] = []
        vllm_eps: list[EndpointInfo] = []
        llmd_eps: list[EndpointInfo] = []
        for pod in pods:
            if (pod.status.phase or "") != "Running":
                continue
            pod_ip = pod.status.pod_ip
            gpu_pod = False
            model = None
            vllm_port: Optional[int] = None
            epp_port: Optional[int] = None
            for c in pod.spec.containers or []:
                req = (c.resources.limits or {}) if c.resources else {}
                if any(k.endswith("nvidia.com/gpu") or k == "nvidia.com/gpu" for k in req):
                    gpu_pod = True
                for e in c.env or []:
                    if e.name == "NVIDIA_VISIBLE_DEVICES" and (e.value or "") not in ("", "void"):
                        gpu_pod = True
                args = list(c.args or [])
                cmd = list(c.command or [])
                m = _extract_model(args, cmd)
                if m:
                    model = model or m
                    vllm_port = vllm_port or _serving_port(c)
                # EPP: identifiable by the --pool-name flag.
                if any("--pool-name" in a for a in (args + cmd)):
                    epp_port = _metrics_port(c) or 9090
            if model:
                gpu_pod = True
            labels = pod.metadata.labels or {}
            deployment = _owner_deployment(pod)
            role = _role_from(pod.metadata.name, labels)
            # Build endpoints (only for pods on this node with an IP).
            if pod_ip and model:
                vllm_eps.append(
                    EndpointInfo(
                        url=f"http://{pod_ip}:{vllm_port or 8000}",
                        service=deployment or pod.metadata.name,
                        namespace=pod.metadata.namespace,
                        role=role,
                        model=model,
                    )
                )
            if pod_ip and epp_port:
                llmd_eps.append(
                    EndpointInfo(
                        url=f"http://{pod_ip}:{epp_port}",
                        service=deployment or pod.metadata.name,
                        namespace=pod.metadata.namespace,
                    )
                )
            if not gpu_pod:
                continue
            infos.append(
                PodInfo(
                    name=pod.metadata.name,
                    namespace=pod.metadata.namespace,
                    deployment=deployment,
                    model_name=model,
                    role=role,
                    uid=pod.metadata.uid,
                )
            )
        self._pods = infos
        self._by_uid = {p.uid: p for p in infos if p.uid}
        self._vllm_eps = vllm_eps
        self._llmd_eps = llmd_eps

    def local_vllm_endpoints(self) -> list[EndpointInfo]:
        """vLLM /metrics targets for model pods on this node."""
        return list(self._vllm_eps)

    def local_llmd_endpoints(self) -> list[EndpointInfo]:
        """EPP /metrics targets for llm-d pods on this node."""
        return list(self._llmd_eps)

    def pod_for_pid(self, pid: int) -> Optional[PodInfo]:
        """Map an NVML compute PID to a pod via its cgroup (host PID ns)."""
        pod_uid = _pod_uid_from_cgroup(pid)
        if pod_uid and pod_uid in self._by_uid:
            return self._by_uid[pod_uid]
        # Fallback: if there's only one model pod, it must be that one.
        return self._pods[0] if len(self._pods) == 1 else None

    def pods(self) -> list[PodInfo]:
        return list(self._pods)


def _serving_port(container) -> Optional[int]:  # noqa: ANN001
    """Best-guess vLLM serving/metrics port from a container's ports."""
    ports = container.ports or []
    for p in ports:
        if (p.name or "").lower() in ("http", "api", "openai"):
            return p.container_port
    return ports[0].container_port if ports else None


def _metrics_port(container) -> Optional[int]:  # noqa: ANN001
    """Metrics port for an EPP container (named 'metrics', else None)."""
    for p in container.ports or []:
        if (p.name or "").lower() == "metrics":
            return p.container_port
    return None


def _owner_deployment(pod) -> Optional[str]:  # noqa: ANN001
    for ref in pod.metadata.owner_references or []:
        if ref.kind == "ReplicaSet":
            # ReplicaSet name is <deployment>-<hash>
            return ref.name.rsplit("-", 1)[0]
        if ref.kind in ("Deployment", "StatefulSet", "DaemonSet"):
            return ref.name
    return None


def _pod_uid_from_cgroup(pid: int) -> Optional[str]:
    path = f"/proc/{pid}/cgroup"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = fh.read()
    except OSError:
        return None
    m = re.search(r"pod([0-9a-fA-F_\-]{36})", data)
    if m:
        return m.group(1).replace("_", "-")
    return None
