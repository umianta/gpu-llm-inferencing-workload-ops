# Metrics catalog

All metrics carry attribution labels where available: `cluster`, `node`,
`gpu_uuid`, `gpu_index`, `pod`, `namespace`, `deployment`, `model`, `role`.

Names below are the OTLP (dotted) form. In Prometheus, dots become underscores
(e.g. `gpu.utilization` → `gpu_utilization`).

## GPU hardware (`gpu.*`)

| Metric | Unit | Kind | Meaning |
|---|---|---|---|
| `gpu.utilization` | 1 (0–1) | gauge | SM utilization fraction |
| `gpu.memory.used.bytes` | By | gauge | VRAM used |
| `gpu.memory.used.percent` | % | gauge | VRAM used / total |
| `gpu.temperature.celsius` | Cel | gauge | Core temperature |
| `gpu.power.draw.watts` | W | gauge | Instantaneous power draw |
| `gpu.power.limit.watts` | W | gauge | Enforced power cap |
| `gpu.ecc.errors.uncorrected` | 1 | counter | Volatile uncorrected ECC errors |
| `gpu.throttle.active` | 1 | gauge | 1 if thermal/power throttling |

## vLLM (`vllm.*`)

| Metric | Unit | Kind | Meaning |
|---|---|---|---|
| `vllm.requests.running` | 1 | gauge | In-flight requests |
| `vllm.requests.waiting` | 1 | gauge | Queued requests |
| `vllm.ttft.seconds` | s | gauge | Avg time-to-first-token (from histogram) |
| `vllm.itl.seconds` | s | gauge | Avg inter-token latency |
| `vllm.tokens.generated.total` | 1 | counter | Output tokens produced |
| `vllm.tokens.prompt.total` | 1 | counter | Prompt tokens processed |
| `vllm.kv_cache.usage.percent` | % | gauge | GPU KV-cache occupancy |
| `vllm.prefix_cache.hit_rate` | 1 (0–1) | gauge | Prefix-cache hit rate |

## llm-d (`llmd.*`)

| Metric | Unit | Kind | Meaning |
|---|---|---|---|
| `llmd.prefix_cache.hit_rate` | 1 (0–1) | gauge | Router prefix-cache hit rate |
| `llmd.kv_cache.usage.percent` | % | gauge | Pool avg KV-cache utilization |
| `llmd.routing.decisions.total` | 1 | counter | Scheduler routing decisions |
| `llmd.pool.ready_endpoints` | 1 | gauge | Ready backends in the pool |

## Cost (`cost.*`)

| Metric | Unit | Kind | Meaning |
|---|---|---|---|
| `cost.gpu.seconds` | s | counter | Accumulated *active* GPU-seconds per model/pod |
| `cost.gpu.usd.estimate` | 1 | counter | `gpu_seconds / 3600 * --gpu-hourly-usd` |

> Cost accrues only while `gpu.utilization > 0.01`, so idle allocation shows up
> as a gap between wall-clock time and `cost.gpu.seconds`.
