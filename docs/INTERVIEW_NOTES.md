# Interview Notes

## Continuous batching

Explain how the set of active sequences can change between engine iterations. Contrast this with static batching and discuss throughput versus tail-latency trade-offs.

## KV cache

Be able to derive how cache memory scales with layers, KV heads, head dimension, sequence length, batch/concurrency, and bytes per element.

## PagedAttention

Explain the virtual-memory analogy, block allocation, fragmentation reduction, and why it helps dynamic workloads. Do not say it reduces the mathematical cost of attention.

## Scheduler

Discuss sequence and token budgets, prefill versus decode work, fairness, preemption, starvation, and overload control.

## Quantization

Separate model-weight memory from total GPU memory. Discuss bandwidth savings, kernel support, dequantization overhead, accuracy, and hardware dependence.

## Speculative decoding

A draft proposes tokens and the target verifies them without changing the output distribution when implemented correctly. It helps only when accepted work saves more time than proposal/verification overhead.

## Tensor parallelism

Weights and matrix operations are sharded within layers. It improves capacity but adds collective communication, so two GPUs do not imply 2x speed.

## Profiling

Start with workload and user-visible metrics, then inspect queueing, GPU utilization, memory, bandwidth, kernel timelines, and communication. High GPU utilization alone is not proof of a good system.
