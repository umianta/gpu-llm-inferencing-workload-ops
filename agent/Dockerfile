FROM python:3.11-slim

# NVML is provided at runtime by the NVIDIA container runtime + driver on the
# host; we only need the Python bindings and app deps here.
WORKDIR /app

COPY pyproject.toml ./
COPY vllm_llmd_agent ./vllm_llmd_agent

RUN pip install --no-cache-dir .

# Non-root; still able to read /proc for cgroup-based PID->pod attribution
# because the DaemonSet shares the host PID namespace.
USER 65532:65532

EXPOSE 9835
ENTRYPOINT ["vllm-llmd-agent"]
CMD ["--sink", "both"]
