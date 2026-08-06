# Adaptive LLM Serving Gateway

A production-style inference gateway built on top of **vLLM** that provides a unified HTTP API for text generation, supports both streaming and non-streaming inference, exposes Prometheus metrics for observability, and includes a benchmarking framework for evaluating latency, throughput, and scaling behavior under concurrent workloads.

---

## Overview

Modern LLM inference servers such as vLLM provide extremely high throughput, but production deployments typically require additional infrastructure:

- Request validation
- Public API gateway
- Streaming support
- Error handling
- Metrics and monitoring
- Benchmarking
- Load testing

This project implements those production components while keeping the inference engine delegated to vLLM.

---

# Problem

Large Language Models require significantly more infrastructure than simply exposing an inference endpoint.

Production deployments must handle:

- HTTP request validation
- Streaming token delivery
- Error handling and retries
- Latency measurement
- Request accounting
- Concurrent workloads
- Benchmarking
- Production monitoring

The objective of this project is to build a lightweight inference gateway that sits in front of vLLM while exposing production-grade observability and benchmarking capabilities.

---

# System Architecture

```text
                Client

                   │
                   ▼

        FastAPI Gateway
        ──────────────────
        Request Validation
        Error Handling
        Streaming
        Metrics
        Benchmark API

                   │

          HTTP (OpenAI API)

                   │
                   ▼

              vLLM Server

                   │
                   ▼

         Transformer Model

                   │
                   ▼

            Generated Tokens
```

---

# Repository Structure

```text
adaptive-llm-serving/

app/
    api.py
    config.py
    metrics.py
    schemas.py
    vllm_client.py

benchmark/
    load_generator.py
    metrics.py
    plot_results.py
    prometheus_snapshot.py
    streaming_benchmark.py
    workloads.py

tests/

Dockerfile
docker-compose.yml
README.md
```

---

# Gateway Architecture

```text
Incoming HTTP Request

        │

        ▼

GenerateRequest
(Pydantic Validation)

        │

        ▼

Generate Endpoint

        │

 ┌───────────────┐
 │ stream=False  │──────────────┐
 └───────────────┘              │
                                │
                                ▼

                         client.complete()

                                │

                                ▼

                      JSON Response Returned


 ┌───────────────┐
 │ stream=True   │──────────────┐
 └───────────────┘              │
                                ▼

                         client.stream()

                                │

                        Async Generator

                                │

                       StreamingResponse

                                │

                          Token Chunks
```

---

# How the Gateway Works

The gateway provides a clean interface in front of the vLLM server.

Responsibilities include:

- Validating incoming requests
- Constructing OpenAI-compatible payloads
- Forwarding inference requests to vLLM
- Supporting both synchronous and streaming generation
- Translating upstream failures into HTTP errors
- Recording Prometheus metrics
- Returning standardized API responses

---

# Streaming Inference

Streaming uses Server-Sent Events (SSE).

Workflow:

```
Client

    │

POST /generate

    │

stream=true

    ▼

Gateway

    │

client.stream()

    │

Receive SSE chunks

    │

yield text

    │

StreamingResponse

    ▼

Client receives incremental tokens
```

Unlike the non-streaming endpoint, the gateway forwards generated text immediately as it arrives from vLLM without waiting for completion.

---

# Observability

The gateway exports Prometheus metrics.

Metrics include:

| Metric | Description |
|---------|-------------|
| gateway_requests_total | Total requests |
| gateway_inflight_requests | Active requests |
| gateway_request_latency_seconds | Request latency histogram |

Prometheus scrapes the `/metrics` endpoint while Grafana visualizes request rate, latency, and system health.

---

# Benchmark Framework

The benchmark suite evaluates inference performance under configurable workloads.

Measurements include:

- Request latency
- Throughput
- Success rate
- Time-to-first-token (TTFT)
- Streaming latency
- Concurrent request scaling

Benchmark modules:

```
benchmark/

load_generator.py
metrics.py
streaming_benchmark.py
workloads.py
```

---

# Benchmark Methodology

Experiments vary:

- concurrency
- prompt length
- output length
- streaming vs non-streaming
- model
- quantization
- tensor parallelism

Each benchmark records:

- average latency
- median latency
- p95 latency
- p99 latency
- throughput
- tokens/sec
- TTFT

---

# Hardware & Software

## Hardware

| Component | Value |
|-----------|-------|
| GPU | *(To be filled after experiments)* |
| CPU | *(To be filled)* |
| Memory | *(To be filled)* |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| FastAPI | Latest |
| httpx | Latest |
| Prometheus | Latest |
| Grafana | Latest |
| vLLM | Latest |

---

# Experimental Results

## Throughput

| Concurrency | Requests/sec | Tokens/sec |
|-------------|-------------:|-----------:|
| 1 | TBD | TBD |
| 4 | TBD | TBD |
| 8 | TBD | TBD |
| 16 | TBD | TBD |
| 32 | TBD | TBD |

---

## Latency

| Concurrency | Avg | p50 | p95 | p99 |
|-------------|----:|----:|----:|----:|
| 1 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD |

---

# Charts

Future benchmark results will include:

- Throughput vs Concurrency
- Latency vs Concurrency
- TTFT vs Concurrency
- GPU Utilization
- GPU Memory Usage

---

# Quantization Findings

*(To be completed after benchmarking.)*

Planned comparison:

- FP16/BF16
- AWQ
- GPTQ (if supported)

Metrics:

- throughput
- latency
- memory usage
- output quality

---

# Tensor Parallel Findings

*(To be completed after benchmarking.)*

Planned comparison:

- TP=1
- TP=2
- TP=4 (if hardware permits)

Metrics:

- throughput
- latency
- GPU utilization
- memory distribution

---

# Limitations

Current limitations include:

- Single vLLM backend instance
- No request batching inside the gateway
- No authentication or rate limiting
- No autoscaling
- No distributed load balancing
- Limited benchmark hardware until GPU experiments are completed

---

# Future Work

Potential improvements:

- Dynamic routing across multiple vLLM instances
- Request batching
- KV cache-aware scheduling
- Kubernetes deployment
- Distributed Prometheus/Grafana
- Multi-model routing
- Authentication
- Rate limiting
- Autoscaling

---

# Running the Project

```bash
git clone <repo>

cd adaptive-llm-serving

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

Start the gateway:

```bash
uvicorn app.api:app --reload
```

View documentation:

```
http://localhost:8000/docs
```

View metrics:

```
http://localhost:8000/metrics
```

---

# Running Benchmarks

```bash
python benchmark/load_generator.py
```

Streaming benchmark:

```bash
python benchmark/streaming_benchmark.py
```

---

# Running Tests

```bash
pytest
```

---

# License

MIT