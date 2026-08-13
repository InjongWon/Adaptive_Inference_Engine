# Adaptive LLM Serving Gateway

A production-style inference gateway built on top of **vLLM** that provides a unified OpenAI-compatible HTTP API for text generation, supports streaming and non-streaming inference, exposes Prometheus metrics for full observability, and includes a benchmarking framework for evaluating latency, throughput, and scaling behavior under concurrent workloads.

> **Status:** Gateway and observability stack are fully operational. Baseline throughput benchmarks are complete. KV cache, quantization, and tensor parallelism experiments are in progress.

---

## Why This Exists

Modern LLM inference servers like vLLM deliver exceptional throughput, but raw inference is only part of the problem. Production deployments require an additional layer:

- Request validation and error handling before traffic reaches the model
- Streaming token delivery over SSE without buffering entire responses
- Observable metrics that surface latency distributions, cache pressure, and GPU utilization in real time
- A benchmarking framework that isolates the effect of individual variables: concurrency, quantization, tensor parallelism, prompt length

This project implements that production layer — keeping the inference engine delegated to vLLM, while owning everything above it.

---

## Architecture

```
USER / BENCHMARK CLIENT
        │
        ▼
POST /generate (HTTP)
        │
        ▼
┌──────────────────────────────┐
│       FastAPI Gateway        │
│         app/api.py           │
│                              │
│  • Pydantic request validation│
│  • Stream / non-stream routing│
│  • Error translation         │
│  • Prometheus instrumentation│
└──────────────────────────────┘
        │
        ▼ OpenAI-compatible HTTP
┌──────────────────────────────┐
│         VLLMClient           │
│      app/vllm_client.py      │
│                              │
│  • Async httpx client        │
│  • SSE chunk forwarding      │
│  • Retry and timeout handling│
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│           vLLM               │
│                              │
│  • Continuous batching       │
│  • Request scheduler         │
│  • PagedAttention            │
│  • KV Cache management       │
│  • Transformer model         │
└──────────────────────────────┘
        │
        ▼
  Generated Tokens
```

### Request Flow — Non-Streaming

```
Client → POST /generate (stream=false)
       → Pydantic validation
       → vllm_client.complete()
       → OpenAI /v1/completions
       → Full response JSON returned
```

### Request Flow — Streaming

```
Client → POST /generate (stream=true)
       → Pydantic validation
       → vllm_client.stream()
       → SSE chunk generator
       → StreamingResponse (incremental tokens)
```

---

## Benchmark Results

### Baseline Throughput (FP16, TP=1)

All results are 100% success rate across all concurrency levels.

| Concurrency | Requests | Req/sec | Tokens/sec | Avg Latency | p50 Latency | p95 Latency | p99 Latency |
|:-----------:|:--------:|:-------:|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 1 (C=32) | 32 | 0.49 | 62.8 | 2.04s | 2.03s | 2.12s | 2.14s |
| 8 (C=64) | 64 | 3.69 | 472.2 | 2.16s | 2.15s | 2.31s | 2.31s |
| 32 (C=128) | 128 | 11.16 | 1,429.1 | 2.82s | 2.79s | 2.99s | 2.99s |

**Key observations:**

- **Throughput scales strongly with concurrency** — tokens/sec grows from 62.8 at C=32 to 1,429 at C=128, a **22.7× increase**, demonstrating vLLM's continuous batching efficiency
- **Latency remains bounded under load** — p99 latency increases from 2.14s to 2.99s across the full concurrency range, less than 1.5× degradation despite a 4× increase in request volume
- **100% success rate at all concurrency levels** — no dropped requests or timeout failures across all runs
- **p95/p99 spread is tight** — less than 50ms gap between p95 and p99 at all concurrency levels, indicating consistent scheduling without outlier stalls

### Throughput vs Concurrency

```
Tokens/sec
1,500 ┤                                    ●
1,400 ┤
1,300 ┤
1,200 ┤
1,100 ┤
1,000 ┤
  900 ┤
  800 ┤
  700 ┤
  600 ┤
  500 ┤                ●
  400 ┤
  300 ┤
  200 ┤
  100 ┤  ●
    0 └──────────────────────────────────────
       C=32          C=64          C=128
```

### Latency Distribution (p50 / p95 / p99)

```
Latency (s)
3.0 ┤                                 ● ● ●
2.9 ┤
2.8 ┤
2.7 ┤
2.6 ┤
2.5 ┤
2.4 ┤
2.3 ┤               ● ●
2.2 ┤             ●
2.1 ┤
2.0 ┤  ● ● ●
    └──────────────────────────────────────
       C=32          C=64          C=128

  ● p50   ● p95   ● p99
```
## Observability Stack

The gateway exports Prometheus metrics scraped at `/metrics`. Grafana dashboards surface real-time system state.

| Metric | Type | Description |
|--------|------|-------------|
| `gateway_requests_total` | Counter | Total requests by status |
| `gateway_inflight_requests` | Gauge | Active concurrent requests |
| `gateway_request_latency_seconds` | Histogram | End-to-end latency (p50/p95/p99) |

vLLM's native metrics — KV cache utilization, queue depth, token throughput — are also scraped directly and surfaced alongside gateway metrics in the same Grafana instance.

---

## Benchmarking Framework

The benchmark suite is designed to isolate the effect of individual variables rather than produce single aggregate numbers. Each experiment fixes all variables except the one under study.

**Variables swept:**

| Dimension | Values |
|-----------|--------|
| Concurrency | 1, 4, 8, 16, 32 |
| Prompt length | Short (64 tok), Medium (256 tok), Long (1024 tok) |
| Output length | 128, 512, 1024 tokens |
| Precision | FP16, BF16, AWQ, GPTQ |
| Tensor parallelism | TP=1, TP=2, TP=4 |
| Mode | Streaming, Non-streaming |

**Metrics collected per run:**

- Average, median, p95, p99 latency
- Time-to-first-token (TTFT)
- Tokens/second throughput
- Requests/second
- KV cache hit rate and memory pressure
- GPU utilization and memory consumption

---


### Upcoming Experiments

The following configurations are queued and results will be added as runs complete:

| Experiment | Status |
|------------|--------|
| Baseline FP16 — TTFT measurement | In progress |
| AWQ quantization vs FP16 | Queued |
| GPTQ quantization vs FP16 | Queued |
| Tensor parallel TP=2 | Queued |
| Tensor parallel TP=4 | Queued |
| KV cache block size sweep | Queued |
| Streaming latency (TTFT distribution) | Queued |

---

## Design Decisions and Tradeoffs

**Why a gateway in front of vLLM rather than vLLM directly?**

vLLM's built-in OpenAI-compatible server is production-grade but doesn't provide the observability or benchmarking surfaces needed for systematic experimentation. The gateway adds a thin instrumentation layer without interfering with the inference path — request overhead is sub-millisecond.

**Why async httpx over requests?**

Streaming responses require an async context to forward SSE chunks incrementally without buffering. Blocking I/O in the client would negate the latency benefit of streaming — the full response would be held in memory before delivery.

**Why Prometheus over structured logging for metrics?**

Prometheus enables real-time dashboards and percentile-accurate histograms across concurrent requests. Structured logs are retained for debugging but are not the primary observability signal — they don't aggregate well across concurrent requests at high throughput.

**KV cache block sizing:**

PagedAttention's performance is sensitive to block size. Smaller blocks reduce fragmentation but increase allocation overhead. Larger blocks improve throughput under long sequences but waste memory on short ones. The benchmark configs sweep block sizes alongside quantization to isolate this interaction — results pending.

**On the latency/throughput tradeoff observed in results:**

The benchmark shows that p99 latency increases by ~40% (2.14s → 2.99s) as concurrency grows from C=32 to C=128, while tokens/sec grows by 22.7×. This is the expected continuous batching tradeoff — the scheduler holds individual requests slightly longer to fill larger batches, which is the right call for throughput-optimized serving. A latency-optimized configuration would cap batch size and accept lower tokens/sec.

---

## Current Limitations

- Single vLLM backend — no load balancing across replicas
- No authentication or rate limiting on the gateway
- No autoscaling — static capacity provisioning
- TTFT not yet instrumented in baseline runs (null in current results)
- Quantization and tensor parallelism results pending

These are known constraints, not oversights. The project scope is infrastructure observability and systematic benchmarking, not production serving at scale.

---

## Running the Project

```bash
git clone <repo>
cd adaptive-llm-serving
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Start the full stack:

```bash
docker-compose up
```

Or run the gateway directly:

```bash
uvicorn app.api:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`
Metrics: `http://localhost:8000/metrics`
Grafana: `http://localhost:3000`

---

Streaming TTFT:

```bash
python benchmark/streaming_benchmark.py
```

Full experiment suite:

```bash
make benchmarks
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
