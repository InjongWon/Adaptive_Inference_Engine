# Adaptive LLM Serving

Adaptive LLM Serving is an inference-systems project for understanding and experimenting with the mechanisms that determine LLM serving performance: **request scheduling, continuous batching, prefill/decode execution, KV-cache management, paged memory allocation, and sampling**.

The project approaches inference from two directions:

1. **Production serving** — Qwen3-8B served with vLLM on NVIDIA B200 infrastructure, with an OpenAI-compatible gateway, streaming, observability, and a workload benchmarking harness.
2. **Inference engine internals** — `mini_vllm`, a PyTorch-based inference engine built from first principles to make scheduling, batching, KV-cache reuse, and memory-management decisions explicit and experimentally measurable.

The goal is not to reimplement transformer kernels or CUDA primitives. It is to understand the systems layer between an incoming generation request and model execution, and connect those design decisions to **TTFT, TPOT, throughput, tail latency, and memory utilization**.

---

## Architecture

## Inference Engine

`mini_vllm` implements the core control plane of autoregressive inference around a Hugging Face/PyTorch transformer.

```text
                         Incoming Requests
                                │
                                ▼
                         ┌──────────────┐
                         │  Scheduler   │
                         └──────┬───────┘
                                │
                       continuous batch
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                 PREFILL                  DECODE
                    │                       │
                    └───────────┬───────────┘
                                ▼
                         ┌──────────────┐
                         │ ModelRunner  │
                         └──────┬───────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
          ┌──────────────┐             ┌──────────────┐
          │   KV Cache   │             │ Transformer  │
          │Block Manager │             │    Model     │
          └──────────────┘             └──────┬───────┘
                                              │
                                           logits
                                              │
                                              ▼
                                        ┌─────────┐
                                        │ Sampler │
                                        └────┬────┘
                                             │
                                         next token
                                             │
                                             ▼
                                      request state
                                             │
                                             └──→ schedule again
```

## Production serving
```text

                         ┌─────────────────────┐
                         │  Benchmark Clients  │
                         │                     │
                         │ concurrency         │
                         │ workload shapes     │
                         │ streaming metrics   │
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
                         │    Qwen3-8B         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              NVIDIA B200
```

The production path provides a controlled baseline against a mature inference runtime. The gateway exposes streaming and non-streaming generation while the benchmark harness drives concurrent workloads and measures end-to-end serving behavior.

This gives the project an external view of inference performance before examining the same problems inside the engine.

---


```

A request progresses through:

```text
WAITING → RUNNING → PREFILL → DECODE → FINISHED
```

Each scheduling iteration determines which requests execute and how much work they are allowed to perform. Newly admitted requests execute prefill work, while active generations execute incremental decode using previously cached attention state.

Finished requests immediately release their resources, allowing waiting requests to enter the active batch without waiting for every sequence in the previous batch to finish.

---

## Scheduling and Continuous Batching

Static batching couples the lifetime of every request in a batch. A short generation can finish early but its capacity remains unused until the longest sequence completes.

`mini_vllm` instead continuously reconstructs the active batch:

```text
Step 1:  [A, B, C]
Step 2:  [A, B, C]
              ↓ B finishes
Step 3:  [A, D, C]
Step 4:  [A, D, C]
         ↓ A finishes
Step 5:  [E, D, C]
```

The scheduler maintains waiting and running request state and performs admission under both **sequence capacity and token-budget constraints**.

Scheduling operates on work rather than only request count:

```text
iteration token budget
        │
        ├── decode tokens for active generations
        │
        └── remaining budget → prefill work
```

This makes long prompts chunkable instead of allowing one large prefill to monopolize an execution iteration.

The scheduler is intentionally separated from memory management. It decides **what should execute**; the KV/block manager determines whether the required memory resources can be allocated.

This separation makes scheduling policies independently replaceable and allows experiments with FCFS, decode prioritization, prefill chunking, admission control, and memory-pressure behavior.

---

## Prefill and Decode

Autoregressive inference has two computationally different phases.

### Prefill

The prompt is processed in parallel under causal attention:

```text
prompt
[t0, t1, t2, ... tN]
        │
        ▼
Transformer
        │
        ├── next-token logits
        └── K/V state for every attention layer
```

Prefill performs substantial matrix computation across the prompt and populates the KV cache required for subsequent generation.

### Decode

After prefill, previously processed tokens are not passed through the transformer again.

For each new token:

```text
new token
    │
    ├── compute new Q
    ├── compute new K/V
    │
    ▼
Q_new × K_cached
    │
    ▼
weighted V_cached
    │
    ▼
next-token logits
```

The newly generated K/V state is appended to the cache and reused by the following decode iteration.

This reduces repeated computation but causes KV memory consumption to grow with active sequence length. As a result, decode performance increasingly depends on memory movement and cache-management efficiency.

---

## KV-Cache and Paged Memory Management

KV caching exchanges recomputation for memory.

Conceptually, cache usage grows with:

```text
active sequences
× sequence length
× transformer layers
× KV heads
× head dimension
```

Allocating one contiguous region for every possible sequence length wastes memory because request lengths are unknown and generations finish at different times.

`mini_vllm` therefore manages KV capacity using fixed-size blocks.

```text
Logical sequence

Request A
[0][1][2][3][4][5][6][7][8][9]
 │           │           │
 ▼           ▼           ▼
block 0     block 1     block 2


Physical KV blocks

┌─────┬─────┬─────┬─────┬─────┐
│ B7  │ B2  │free │ B9  │free │
└─────┴─────┴─────┴─────┴─────┘
   ▲     ▲           ▲
   └──── Request A ──┘
```

Logical sequence position is therefore decoupled from physical KV placement.

The block manager owns allocation, mapping, reclamation, and free-space accounting. When a request finishes, its physical blocks return to the free pool and can immediately be reused by another request.

This design avoids requiring a sequence's KV state to occupy one large contiguous allocation and provides an explicit mechanism for studying fragmentation, cache pressure, admission, and reclamation.

---

## Sampling

Model execution produces one vector of next-token logits per active request:

```text
[batch_size, vocabulary_size]
```

The sampling pipeline supports:

```text
logits
  │
  ├── greedy decoding
  │
  └── temperature scaling
          │
          ▼
        top-k
          │
          ▼
       softmax
          │
          ▼
        top-p
          │
          ▼
     multinomial
          │
          ▼
one token ID / request
```

Top-k retains an exact set of candidate token IDs rather than filtering only by the kth logit value, avoiding incorrect behavior when multiple candidates have tied scores.

Top-p operates on cumulative probability mass and retains the smallest highest-probability candidate set whose cumulative mass reaches the configured threshold.

---

## Benchmarking

The same workload methodology is used to reason about both production serving behavior and engine-level design decisions.

The benchmark harness measures:

- **TTFT** — time from request submission to first generated token
- **TPOT** — average time per output token after the first token
- **request latency**
- **p50 / p95 / p99 latency**
- **requests/sec**
- **output tokens/sec**
- **KV-cache utilization**
- **active and waiting request counts**

Workloads vary concurrency, prompt length, generation length, scheduling policy, and memory pressure to separate throughput effects from user-visible latency.

---

## vLLM Baseline

Initial concurrency measurements against Qwen3-8B on a single NVIDIA B200:

| Concurrency | Requests | Req/s | Output tok/s | Mean latency | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 0.49 | 62.2 | 2.06s | 2.17s | 2.27s |
| 8 | 64 | 3.53 | 451.4 | 2.26s | 2.46s | 2.51s |
| 32 | 128 | 11.68 | 1,495.2 | 2.64s | 2.85s | 2.86s |

Increasing concurrency from 1 to 32 increased measured output throughput from **62.2 to 1,495.2 tokens/sec (~24×)** while mean request latency increased from **2.06s to 2.64s (~28%)**.

The important result is not simply that higher concurrency is faster. Increasing concurrency gives the inference runtime more independent work to batch together, improving accelerator utilization and amortizing model execution overhead. The tradeoff is increased per-request latency as requests share execution capacity.

These measurements describe the production vLLM deployment, not `mini_vllm`.

---

## Experiments

The completed experiment suite studies inference performance along several dimensions.

### Workload shape

Concurrency, prompt length, and output length are varied independently to distinguish prefill-heavy workloads from decode-heavy workloads.

### Scheduling

Alternative scheduling and admission policies are compared using throughput, TTFT, TPOT, and tail latency rather than optimizing only aggregate throughput.

### KV-cache pressure

Long-context and high-concurrency workloads deliberately constrain available KV capacity to measure block utilization, allocation behavior, fragmentation, reclamation, and request admission.

### Hardware

Equivalent workloads are executed across NVIDIA H100, H200, and B200 configurations while holding model and serving parameters constant.

### Numerical representation

FP16/BF16 and supported quantized configurations are evaluated for their effects on memory footprint, throughput, and latency.

### Parallelism

Tensor-parallel configurations are evaluated separately from single-GPU concurrency experiments so communication overhead can be distinguished from batching and model-execution effects.

---

## Reflections
The project reinforced several properties of production LLM inference that are easy to miss when interacting with a model only through an API.

**Throughput and latency are coupled through scheduling.** Increasing the amount of concurrent work available to the engine can dramatically increase throughput, but scheduling policy determines how that capacity translates into TTFT and tail latency for individual requests.

**Prefill and decode are different workloads.** Prefill exposes substantial parallel computation across prompt tokens, while autoregressive decode repeatedly performs small incremental computations while reading growing model and KV state. Treating them identically leads to poor scheduling decisions.

**KV cache turns inference into a memory-management problem.** Avoiding recomputation makes generation practical, but cache capacity becomes a first-class scheduling resource as concurrency and context length increase.

**Continuous batching is a scheduling problem, not simply a larger batch.** The active batch changes at token boundaries as requests arrive, finish, or become blocked by resource constraints.

**Paged KV allocation separates logical sequence growth from physical memory placement.** This makes memory reusable at block granularity and gives the scheduler and memory manager a cleaner interface for admission and reclamation.

**Benchmark results need workload context.** A throughput number without concurrency, prompt/output lengths, hardware, precision, and latency characteristics says very little about the actual serving system.

---

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Start the gateway:

```bash
uvicorn app.api:app --reload --port 8080
```

Run the concurrent workload benchmark:

```bash
python benchmark/load_generator.py
```

Run the streaming TTFT/TPOT benchmark:

```bash
python benchmark/streaming_benchmark.py
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

---

## Stack

**Python · PyTorch · Hugging Face Transformers · vLLM · FastAPI · httpx · Modal · Prometheus · Grafana**

## License

MIT