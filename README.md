# Adaptive LLM Serving

A learning-first, production-style project for understanding and benchmarking modern LLM inference with **vLLM**.

The repository intentionally separates:

1. **vLLM engine** — model execution, scheduling, continuous batching, PagedAttention/KV-cache management, quantization, speculative decoding, tensor parallelism.
2. **Learning gateway** — a small FastAPI layer where you implement timeouts, cancellation, admission control, metrics, and error handling.
3. **Benchmark harness** — reproducible workloads for TTFT, TPOT/ITL, throughput, tail latency, memory pressure, and scheduler experiments.
4. **Observability** — Prometheus/Grafana scaffolding and raw metric snapshots.

## Architecture

```text
Async clients / load generator
             |
             v
FastAPI learning gateway :8080 ----> gateway /metrics
             |
             v
vLLM OpenAI-compatible server :8000 ----> vLLM /metrics
             |
     scheduler + continuous batching
             |
       paged KV-cache blocks
             |
 GPU worker(s): BF16 / INT4 / speculative / TP
```

## Hardware

- Most development and unit tests run without a GPU.
- Real inference benchmarks require a supported accelerator environment.
- Tensor parallel experiments require at least two visible GPUs.
- Start with a 0.6B–1.7B model if GPU memory is limited.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Install vLLM separately following the instructions for your CUDA and PyTorch environment. Pin the version you actually benchmark in `BENCHMARKS.md`.

## First run

Terminal 1:

```bash
./scripts/serve_baseline.sh
```

Terminal 2:

```bash
./scripts/run_gateway.sh
```

Terminal 3:

```bash
python -m benchmark.load_generator --workload smoke
```

Direct vLLM check:

```bash
curl http://localhost:8000/v1/completions \
  -H 'Authorization: Bearer local-token' \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-1.7B","prompt":"Explain KV cache.","max_tokens":64}'
```

## High Level Design 

- Run the baseline vLLM server.
- Trace one request through tokenizer, prefill, decode, and streaming.
- Complete gateway integration tests.
- Record model, GPU, driver, CUDA, PyTorch, and vLLM versions.

### sampling and streaming

- Implement accurate token counting.
- Implement streaming TTFT and TPOT measurement.
- Compare greedy, low-temperature, and high-temperature generation.
- Explain why sampling changes quality but not the model logits themselves.

### Scheduler and continuous batching

- Implement mixed-length workloads.
- Sweep `max-num-seqs` and `max-num-batched-tokens`.
- Measure throughput, queue depth, P95/P99 latency, and fairness.
- Add one admission-control strategy to the gateway.

### KV cache and PagedAttention

- Sweep prompt length, output length, concurrency, and maximum model length.
- Capture vLLM cache metrics before/during/after each experiment.
- Estimate KV-cache bytes analytically and compare with observed capacity.
- Trigger and explain preemption or OOM safely.

### quantization and speculative decoding

- Select a compatible pre-quantized model.
- Compare BF16/FP16 against AWQ or GPTQ.
- Compare normal decoding against a draft-model configuration.
- Report the operating region where speculation helps and where overhead wins.

### tensor parallelism and profiling

- Compare TP=1 and TP=2 using identical workloads.
- Measure per-GPU memory, throughput, latency, and communication overhead.
- Capture one Nsight Systems trace.
- Determine whether the workload is compute-, bandwidth-, capacity-, or scheduler-bound.

### Some report

- Add plots and a Grafana dashboard.
- Finish `BENCHMARKS.md` and `docs/INTERVIEW_NOTES.md`.
- Add a reproducibility command for every headline result.
- Use only measured numbers in resume bullets.

```

High-value tasks:

- Accurate output-token accounting
- TTFT and token-level TPOT
- Mixed-length and bursty request generation
- Queue-time correlation with vLLM metrics
- Gateway cancellation and bounded admission control
- Automated experiment matrix and environment capture
- Quality comparison for quantized outputs
- Speculative token acceptance-rate PromQL
- TP scaling-efficiency calculation

## Required experiment table

For every run, record:

| Field | Example |
|---|---|
| Engine version | vLLM x.y.z |
| Model and revision | exact repository + commit |
| GPU | A10G 24 GB |
| Precision | BF16 / AWQ INT4 |
| TP size | 1 / 2 |
| Prompt distribution | 128–2048 tokens |
| Output distribution | 32–256 tokens |
| Concurrency | 1 / 8 / 16 / 32 |
| Scheduler flags | max sequences, token budget |
| TTFT | mean + P95 |
| TPOT/ITL | mean + P95 |
| Throughput | requests/s and output tokens/s |
| Tail latency | P95/P99 |
| GPU memory/utilization | peak and average |
| KV-cache utilization | peak and preemption count |

## Resume standard

A strong claim has all four parts:

```text
optimization + workload + hardware + measured outcome
```

Example only after measurement:

```text
Benchmarked continuous batching on Qwen3-1.7B/A10G across 32 concurrent mixed-length requests, sustaining X output tokens/s while holding P95 TTFT below Y ms.
```

## Useful commands

```bash
make test
make lint
make monitor
make smoke
make bench
python -m benchmark.prometheus_snapshot --output results/before.prom
```

Grafana: `http://localhost:3000` (`admin` / `admin`)
Prometheus: `http://localhost:9090`
