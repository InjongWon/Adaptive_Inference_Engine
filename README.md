# Adaptive LLM Serving

An LLM inference systems project built around two pieces:

- a production-style serving stack running **Qwen3-1.7B with vLLM on NVIDIA GPUs**
- a small inference engine built from scratch to understand and experiment with **scheduling, continuous batching, KV caching, prefill/decode, and paged memory management**

I started by treating vLLM as a black box: drive it with controlled workloads, measure throughput and latency, and understand how serving behavior changes under concurrency. From there, I began rebuilding the inference path from first principles in mini_vllm—request lifecycle, prefill vs. decode, KV cache reuse, scheduling, continuous batching, sampling, and paged memory management.

## Architecture

### Production serving path

```text
                         ┌─────────────────────┐
                         │  Benchmark Clients  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   FastAPI Gateway   │
                         │                     │
                         │ validation          │
                         │ streaming / SSE     │
                         │ metrics             │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │        vLLM         │
                         │    Qwen3-1.7B       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              NVIDIA B200
```

The gateway exposes streaming and non-streaming generation and forwards requests to an OpenAI-compatible vLLM server running remotely on a B200.

The benchmark client generates concurrent workloads and records request latency, throughput, and streaming metrics.

### `mini_vllm`

```text
                         Incoming Requests
                                │
                                ▼
                        ┌───────────────┐
                        │   Scheduler   │
                        │ waiting/running│
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  ModelRunner  │
                        │ prefill/decode│
                        └───────┬───────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
           ┌─────────────┐             ┌─────────────┐
           │  KV Cache   │             │ Transformer │
           │Block Manager│             │   Model     │
           └─────────────┘             └──────┬──────┘
                                              │
                                              ▼
                                           Logits
                                              │
                                              ▼
                                        ┌─────────┐
                                        │ Sampler │
                                        └────┬────┘
                                             │
                                             ▼
                                         Next Token
                                             │
                                             └──→ next decode step
```

`mini_vllm` implements the serving logic around the transformer rather than reimplementing the transformer or CUDA kernels themselves.

The main pieces are:

- request lifecycle (`WAITING → RUNNING → FINISHED`)
- variable-length request batching
- prefill and token-by-token decode
- KV-cache reuse
- sampling
- request scheduling
- continuous batching
- block-based KV-cache allocation

The model execution layer uses PyTorch/Hugging Face.

## Current progress

```text
Production serving
├── [x] vLLM deployment
├── [x] Qwen3-1.7B on NVIDIA B200
├── [x] FastAPI gateway
├── [x] streaming generation
├── [x] concurrent load generator
├── [x] latency / throughput benchmarks
└── [x] full TTFT / TPOT experiment matrix

mini_vllm
├── [x] request lifecycle
├── [x] variable-length batching
├── [x] attention masks / padding
├── [x] batched transformer forward
├── [x] real Hugging Face model execution
├── [x] prefill + KV-cached decode
├── [x] sampler
├── [x] scheduler
├── [x ] continuous batching
├── [x] KV-cache manager
├── [x] paged block manager
└── [x] full generation loop
```

## Baseline results

Initial concurrency benchmark against the B200-backed vLLM deployment:

| Concurrency | Requests | Req/s | Output tok/s | Mean latency | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 0.49 | 62.2 | 2.06s | 2.17s | 2.27s |
| 8 | 64 | 3.53 | 451.4 | 2.26s | 2.46s | 2.51s |
| 32 | 128 | 11.68 | 1,495.2 | 2.64s | 2.85s | 2.86s |

Increasing concurrency from 1 to 32 increased output throughput from **62 to 1,495 tokens/sec**, while mean request latency increased from **2.06s to 2.64s**.

These numbers are measurements of the vLLM deployment, not `mini_vllm`.

## Why build `mini_vllm`?

The first version of this project mostly looked at vLLM from the outside:

```text
send workload → vLLM → measure performance
```

That's useful for understanding serving behavior, but it doesn't expose why the engine behaves that way.

`mini_vllm` moves the experiments inside the inference loop:

```text
request arrives
      ↓
scheduler chooses work
      ↓
prefill or decode
      ↓
reuse / allocate KV cache
      ↓
run model
      ↓
sample token
      ↓
update request
      ↓
schedule again
```




## Experiments

The next set of experiments will look at:

**Serving configuration**
- H100 vs H200 vs B200
- FP16/BF16 vs AWQ/GPTQ
- tensor parallelism
- speculative decoding

**Engine behavior**
- concurrency and batch size
- prompt vs decode-heavy workloads
- scheduler policies
- KV-cache pressure
- block allocation / fragmentation
- continuous batching


## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the gateway:

```bash
uvicorn app.api:app --reload --port 8080
```

Run the load benchmark:

```bash
python benchmark/load_generator.py
```

Run the streaming benchmark:

```bash
python benchmark/streaming_benchmark.py
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Stack

**Python · PyTorch · Hugging Face Transformers · vLLM · FastAPI · httpx · Modal · Prometheus · Grafana**

## License

MIT