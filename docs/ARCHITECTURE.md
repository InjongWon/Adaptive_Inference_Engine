# Architecture Decisions

## Why a thin gateway?

vLLM already exposes an OpenAI-compatible server. The gateway exists to demonstrate application-level concerns without hiding engine behavior:

- overload protection and admission control
- cancellation and timeout propagation
- request validation
- experiment tagging
- application metrics
- authentication/routing extensions

## Request lifecycle to explain in interviews

1. Client creates a completion request.
2. Gateway validates it and forwards it asynchronously.
3. vLLM tokenizes the prompt.
4. Scheduler admits prefill/decode work under token and sequence budgets.
5. Prefill computes prompt attention and initializes KV cache.
6. Decode emits one or more tokens per engine iteration.
7. Continuous batching lets the active batch change as requests arrive or finish.
8. Paged KV-cache allocation avoids requiring one contiguous region per sequence.
9. Streaming returns chunks while metrics track queueing and execution.

## Extensions to implement

- semaphore or token-budget admission controller
- priority classes and starvation test
- request cancellation propagation
- retry policy for transient network failures
- model router for baseline/quantized endpoints
- experiment ID propagated into logs and metrics
