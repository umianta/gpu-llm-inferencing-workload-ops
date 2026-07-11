#!/usr/bin/env bash
# Deploy the standalone Prometheus + Grafana monitoring stack and load the
# Grafana dashboards from dashboards/grafana/*.json.
#
# The repo dashboards are Grafana *export* format (they carry an __inputs block
# and a ${DS_PROMETHEUS} datasource variable for manual import). For file
# provisioning we resolve that variable to the provisioned datasource uid
# ("prometheus") and strip __inputs, so they load with zero clicks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=monitoring

echo "== applying Prometheus =="
kubectl apply -f "$ROOT/deploy/manifests/prometheus.yaml"

echo "== building dashboards ConfigMap =="
tmp="$(mktemp -d)"
for f in "$ROOT"/dashboards/grafana/*.json; do
  base="$(basename "$f")"
  sed 's/${DS_PROMETHEUS}/prometheus/g' "$f" \
    | python3 -c 'import sys,json;d=json.load(sys.stdin);d.pop("__inputs",None);json.dump(d,sys.stdout)' \
    > "$tmp/$base"
done
kubectl create configmap grafana-dashboards -n "$NS" \
  --from-file="$tmp" --dry-run=client -o yaml | kubectl apply -f -
rm -rf "$tmp"

echo "== applying Grafana =="
kubectl apply -f "$ROOT/deploy/manifests/grafana.yaml"

echo "== waiting for rollouts =="
kubectl rollout status deploy/prometheus -n "$NS" --timeout=120s
kubectl rollout status deploy/grafana -n "$NS" --timeout=120s

echo
echo "Grafana:    http://localhost:3000  (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Add these to the port-forward or run: kubectl port-forward -n $NS svc/grafana 3000:3000"
