#!/usr/bin/env bash
# Port-forward the vLLM/llm-d/agent endpoints so they're reachable from your
# laptop over VS Code Remote-SSH. Binds 127.0.0.1 on the server; VS Code
# auto-forwards those ports to your laptop (open http://localhost:<port>).
#
# Set ADDRESS=0.0.0.0 to also expose on the server LAN IP (direct browser access
# without VS Code forwarding).
#
# Usage:
#   ./scripts/port-forward.sh                 # forward vLLM(8000) + agent(9835)
#   ADDRESS=0.0.0.0 ./scripts/port-forward.sh # also expose on LAN
#
# Ctrl-C stops all forwards.
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-d}"
MON_NAMESPACE="${MON_NAMESPACE:-monitoring}"
# Bind to all interfaces by default so the dashboard is reachable from your
# laptop via the server LAN IP (Remote-SSH localhost forwarding is flaky).
ADDRESS="${ADDRESS:-0.0.0.0}"

# vLLM target — defaults to the running llm-d model server (Qwen2.5-1.5B).
# Override with VLLM_NS / VLLM_SVC to point elsewhere.
VLLM_NS="${VLLM_NS:-llm-d-flow-control}"
VLLM_SVC="${VLLM_SVC:-svc/modelserver}"

# service:localPort:remotePort:namespace
FORWARDS=(
  "${VLLM_SVC}:8000:8000:${VLLM_NS}"                # vLLM /metrics + OpenAI API
  "svc/vllm-llmd-agent:9835:9835:${MON_NAMESPACE}"  # monitoring agent /metrics
  "svc/prometheus:9090:9090:${MON_NAMESPACE}"       # Prometheus UI
  "svc/grafana:3000:3000:${MON_NAMESPACE}"          # Grafana (admin/admin)
)

pids=()
cleanup() {
  echo; echo "stopping port-forwards..."
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Supervise a single forward: restart it whenever it dies (e.g. the target pod
# is rescheduled). This is why the dashboard stops "loading" after a pod
# restart — without supervision the forward exits and never comes back.
supervise() {
  local target="$1" local_port="$2" remote="$3" ns="$4"
  while true; do
    if kubectl get "$target" -n "$ns" >/dev/null 2>&1; then
      kubectl port-forward --address "$ADDRESS" -n "$ns" "$target" \
        "${local_port}:${remote}" >/tmp/pf-${local_port}.log 2>&1 || true
    fi
    # target missing or forward exited — wait and retry
    sleep 3
  done
}

echo "port-forwarding (address=${ADDRESS}, auto-restart on):"
for entry in "${FORWARDS[@]}"; do
  IFS=":" read -r target local remote ns <<<"$entry"
  echo "  ->    http://localhost:${local}  ($ns/$target :${remote})"
  supervise "$target" "$local" "$remote" "$ns" &
  pids+=("$!")
done

echo "ready. vLLM metrics: http://localhost:8000/metrics | Grafana :3000 | Prometheus :9090"
echo "Ctrl-C to stop."
wait
