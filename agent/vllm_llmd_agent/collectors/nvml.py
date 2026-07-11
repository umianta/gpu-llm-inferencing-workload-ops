"""NVML GPU collector — reads NVIDIA hardware telemetry via pynvml.

Fail-soft: every NVML call is wrapped so one broken metric (or a MIG/GH200
quirk) never kills a collection cycle.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from ..schema import GpuSample

logger = logging.getLogger("collector.nvml")

T = TypeVar("T")

try:
    import pynvml  # type: ignore

    _HAVE_NVML = True
except Exception:  # pragma: no cover - import guard
    pynvml = None  # type: ignore
    _HAVE_NVML = False


def _safe(fn: Callable[[], T], default: T) -> T:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - fail-soft by design
        logger.debug("nvml call failed: %s", exc)
        return default


class NvmlCollector:
    def __init__(self) -> None:
        self._ready = False

    def start(self) -> None:
        if not _HAVE_NVML:
            logger.warning("pynvml not available; GPU hardware metrics disabled")
            return
        try:
            pynvml.nvmlInit()
            self._ready = True
            logger.info("NVML initialized: %d device(s)", pynvml.nvmlDeviceGetCount())
        except Exception as exc:  # noqa: BLE001
            logger.warning("nvmlInit failed: %s", exc)

    def stop(self) -> None:
        if self._ready:
            _safe(pynvml.nvmlShutdown, None)
            self._ready = False

    def collect(self) -> list[GpuSample]:
        if not self._ready:
            return []
        samples: list[GpuSample] = []
        count = _safe(pynvml.nvmlDeviceGetCount, 0)
        for i in range(count):
            handle = _safe(lambda i=i: pynvml.nvmlDeviceGetHandleByIndex(i), None)
            if handle is None:
                continue
            samples.append(self._read_device(i, handle))
        return samples

    def _read_device(self, index: int, handle) -> GpuSample:  # noqa: ANN001
        uuid = _safe(lambda: _decode(pynvml.nvmlDeviceGetUUID(handle)), f"unknown-{index}")
        model = _safe(lambda: _decode(pynvml.nvmlDeviceGetName(handle)), "unknown")

        sample = GpuSample(index=index, uuid=uuid, model=model)

        util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle), None)
        if util is not None:
            sample.utilization = util.gpu / 100.0

        mem = _safe(lambda: _memory_info(handle), None)
        if mem is not None:
            sample.memory_used_bytes = mem.used
            sample.memory_total_bytes = mem.total
            if mem.total > 0:
                sample.memory_used_percent = mem.used / mem.total * 100.0

        sample.temperature_c = _safe(
            lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU), 0
        )
        # NVML reports power in milliwatts
        sample.power_draw_w = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle), 0) / 1000.0
        sample.power_limit_w = (
            _safe(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle), 0) / 1000.0
        )
        sample.ecc_uncorrected = _safe(
            lambda: pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_VOLATILE_ECC,
            ),
            0,
        )
        sample.throttle_active = 1 if _throttled(handle) else 0
        procs = _safe(lambda: _compute_procs(handle), {})
        sample.proc_mem = procs
        sample.compute_pids = list(procs.keys())
        sample.proc_util = _safe(lambda: _compute_util(handle), {})
        return sample


def _decode(value) -> str:  # noqa: ANN001
    return value.decode() if isinstance(value, bytes) else str(value)


def _memory_info(handle):  # noqa: ANN001
    # Prefer v2 for GH200/GB200 unified memory; fall back to v1.
    try:
        return pynvml.nvmlDeviceGetMemoryInfo(handle, version=2)
    except (AttributeError, TypeError, Exception):  # noqa: BLE001
        return pynvml.nvmlDeviceGetMemoryInfo(handle)


def _throttled(handle) -> bool:  # noqa: ANN001
    reasons = _safe(lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle), 0)
    hw_slowdown = getattr(pynvml, "nvmlClocksThrottleReasonHwSlowdown", 0)
    sw_thermal = getattr(pynvml, "nvmlClocksThrottleReasonSwThermalSlowdown", 0)
    hw_thermal = getattr(pynvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0)
    mask = hw_slowdown | sw_thermal | hw_thermal
    return bool(reasons & mask)


def _compute_pids(handle) -> list[int]:  # noqa: ANN001
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return [p.pid for p in procs]


def _compute_procs(handle) -> dict[int, int]:  # noqa: ANN001
    """pid -> GPU memory bytes. usedGpuMemory may be None on some GPUs."""
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    out: dict[int, int] = {}
    for p in procs:
        mem = getattr(p, "usedGpuMemory", None)
        out[p.pid] = int(mem) if mem is not None else 0
    return out


def _compute_util(handle) -> dict[int, int]:  # noqa: ANN001
    """pid -> SM utilization percent, from recent NVML samples."""
    samples = pynvml.nvmlDeviceGetProcessUtilization(handle, 0)
    out: dict[int, int] = {}
    for s in samples:
        # keep the max sample per pid (samples can repeat within the window)
        out[s.pid] = max(out.get(s.pid, 0), int(s.smUtil))
    return out
