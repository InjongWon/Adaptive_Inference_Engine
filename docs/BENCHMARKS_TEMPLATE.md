# Benchmark Report

## Environment

- Date:
- Commit:
- vLLM:
- PyTorch/CUDA:
- GPU(s):
- Model/revision:
- Driver:

## Methodology

Describe warm-up, number of repetitions, prompt/output token distributions, concurrency, sampling settings, and whether prefix caching was controlled.

## Baseline

| Concurrency | Prompt tokens | Output tokens | TTFT P50/P95 | TPOT P50/P95 | Output tok/s | P99 latency |
|---:|---:|---:|---:|---:|---:|---:|

## Continuous batching and scheduler sweep

Record `max-num-seqs`, `max-num-batched-tokens`, queue depth, cache use, and fairness observations.

## Quantization

Compare model-weight memory, total memory, startup, TTFT, TPOT, throughput, and a small output-quality test.

## Speculative decoding

Record target/draft models, speculative-token count, acceptance rate, QPS, and the break-even workload.

## Tensor parallelism

Compute scaling efficiency:

```text
(TP=2 throughput / TP=1 throughput) / 2
```

Explain communication overhead and whether TP was needed for capacity or speed.

## Profiling diagnosis

State whether each workload is primarily compute-bound, memory-bandwidth-bound, memory-capacity-bound, communication-bound, or scheduler/queue-bound. Include evidence.

## Headline results

Only place reproducible measured results here.
