#!/usr/bin/env python3
"""Load generator for the vLLM / llm-d monitoring demo.

Sends chat-completion requests so the Grafana panels come alive: request rate,
running/waiting queue, TTFT/ITL, token throughput, KV-cache and prefix-cache.

Prefix-cache exercise: every request shares a long common SYSTEM prompt, so vLLM
(and the llm-d prefix indexer) get repeated identical prefixes -> the prefix
cache hit rate should climb after the first few requests.

Stdlib only (urllib + threads) — no pip installs. Target the port-forwarded
model server by default.

Examples:
  # 90s of load at concurrency 6 against the forwarded model server
  python3 scripts/load-test.py --duration 90 --concurrency 6

  # Point at the llm-d gateway instead
  python3 scripts/load-test.py --url http://localhost:8080 --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# A deliberately long shared prefix to make prefix caching effective.
SYSTEM_PROMPT = (
    "You are a meticulous senior site-reliability engineer embedded with a GPU "
    "inference platform team. You answer strictly, concisely, and technically. "
    "You always consider latency, throughput, KV-cache pressure, and cost. "
    "Follow these standing rules for every answer: (1) be correct before being "
    "brief; (2) prefer concrete numbers; (3) never invent metrics; (4) assume "
    "vLLM and llm-d on Kubernetes with NVIDIA GPUs; (5) keep answers under "
    "120 words unless asked otherwise. Begin now."
)

QUESTIONS = [
    "What causes high TTFT under load?",
    "How does prefix caching reduce prompt cost?",
    "When should I add a decode replica?",
    "Explain KV-cache eviction in one paragraph.",
    "What is a good gpu-memory-utilization for a 1.5B model?",
    "How does llm-d route requests across pods?",
    "Why would requests queue even at low GPU util?",
    "Summarize chunked prefill benefits.",
    "What metric signals under-provisioning first?",
    "How do I detect a throttling GPU?",
]

_lock = threading.Lock()
_stats = {"ok": 0, "err": 0, "prompt_tok": 0, "gen_tok": 0, "latency_sum": 0.0}


def one_request(url: str, model: str, max_tokens: int) -> None:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": random.choice(QUESTIONS)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        dt = time.monotonic() - t0
        usage = payload.get("usage", {})
        with _lock:
            _stats["ok"] += 1
            _stats["latency_sum"] += dt
            _stats["prompt_tok"] += usage.get("prompt_tokens", 0)
            _stats["gen_tok"] += usage.get("completion_tokens", 0)
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _stats["err"] += 1
        if _stats["err"] <= 3:
            print(f"  request error: {exc}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000", help="vLLM/gateway base URL")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--duration", type=int, default=90, help="seconds")
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    print(
        f"load: {args.url} model={args.model} concurrency={args.concurrency} "
        f"duration={args.duration}s"
    )
    stop = time.monotonic() + args.duration

    def worker() -> None:
        while time.monotonic() < stop:
            one_request(args.url, args.model, args.max_tokens)

    last = time.monotonic()
    last_ok = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(worker) for _ in range(args.concurrency)]
        while time.monotonic() < stop:
            time.sleep(5)
            with _lock:
                ok, err = _stats["ok"], _stats["err"]
                gen = _stats["gen_tok"]
                lat = _stats["latency_sum"]
            now = time.monotonic()
            rps = (ok - last_ok) / (now - last) if now > last else 0
            avg_lat = lat / ok if ok else 0
            print(
                f"  [{int(now - (stop - args.duration))}s] ok={ok} err={err} "
                f"~{rps:.1f} req/s avg_lat={avg_lat:.2f}s gen_tok={gen}"
            )
            last, last_ok = now, ok
        for f in futs:
            f.result()

    with _lock:
        ok, err = _stats["ok"], _stats["err"]
        avg_lat = _stats["latency_sum"] / ok if ok else 0
    print(
        f"\ndone: {ok} ok / {err} err | avg latency {avg_lat:.2f}s | "
        f"prompt_tok={_stats['prompt_tok']} gen_tok={_stats['gen_tok']}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
