# Adaptive LLM Serving Gateway

A production-style inference gateway built on top of **vLLM** that provides a unified OpenAI-compatible HTTP API for text generation, supports streaming and non-streaming inference, exposes Prometheus metrics for full observability, and includes a benchmarking framework for evaluating latency, throughput, and scaling behavior under concurrent workloads.

> **Status:** Gateway and observability stack are fully operational. KV cache benchmarking experiments are in progress — results will be updated as experiments complete across quantization and tensor parallelism configurations.

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

## Repository Structure

```
adaptive-llm-serving/
├── app/
│   ├── api.py              # FastAPI routes, streaming, error handling
│   ├── config.py           # Environment and model configuration
│   ├── metrics.py          # Prometheus metric definitions
│   ├── schemas.py          # Pydantic request/response models
│   └── vllm_client.py      # Async vLLM HTTP client
│
├── benchmark/
│   ├── load_generator.py   # Concurrent request load driver
│   ├── metrics.py          # Benchmark result collection
│   ├── plot_results.py     # Latency/throughput visualization
│   ├── prometheus_snapshot.py  # Metric snapshots during runs
│   ├── streaming_benchmark.py  # TTFT and streaming latency
│   └── workloads.py        # Configurable prompt distributions
│
├── configs/
│   ├── baseline.yaml
│   ├── quantization.yaml       # AWQ / GPTQ configurations
│   ├── speculative_decoding.yaml
│   └── tensor_parallel.yaml    # TP=1,2,4 configurations
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboard-starter.json
│       └── provisioning/
│
├── tests/
│   ├── test_api.py
│   ├── test_metrics.py
│   ├── test_schemas.py
│   └── test_vllm_client.py
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

---

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

## Benchmark Results

> KV cache and quantization experiments are actively running. This section will be updated with full results including latency/throughput curves and per-configuration comparisons.

**Baseline (FP16, TP=1) — preliminary:**

| Concurrency | Avg Latency | p95 Latency | Throughput |
|:-----------:|:-----------:|:-----------:|:----------:|
| 1 | — | — | — |
| 8 | — | — | — |
| 32 | — | — | — |

*Full results including quantization comparisons (FP16 vs AWQ vs GPTQ) and tensor parallelism scaling (TP=1/2/4) to be published after experiment runs complete.*

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

---

## Current Limitations

- Single vLLM backend — no load balancing across replicas
- No authentication or rate limiting on the gateway
- No autoscaling — static capacity provisioning
- KV cache and quantization benchmark results pending hardware availability

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

## Running Benchmarks

Baseline throughput:

```bash
python benchmark/load_generator.py --concurrency 1 4 8 16 32
```

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